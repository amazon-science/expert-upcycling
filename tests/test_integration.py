"""End-to-end integration test for expert upcycling.

Builds a minimal mock model that replicates the Megatron-LM MoE weight
storage pattern, then runs the full upcycling pipeline and verifies
correctness.  No GPU or Megatron-LM installation required.
"""

import sys
import types
import torch
import torch.nn as nn
import pytest


# ======================================================================
# Mock classes that replicate Megatron-LM's MoE weight storage pattern
# ======================================================================

class MockGroupedLinear(nn.Module):
    """Mimics TE GroupedLinear: stores per-expert weights as weight0, weight1, ..."""

    def __init__(self, num_experts: int, in_features: int, out_features: int):
        super().__init__()
        self.num_gemms = num_experts
        self.num_experts = num_experts
        for i in range(num_experts):
            setattr(self, f"weight{i}", nn.Parameter(torch.randn(out_features, in_features)))


class MockTransformerConfig:
    """Minimal config object with the fields router/experts read."""

    def __init__(self, num_experts, hidden_size):
        self.num_moe_experts = num_experts
        self.hidden_size = hidden_size


class MockRouter(nn.Module):
    """Mimics TopKRouter: weight [E, H], optional expert_bias buffer."""

    def __init__(self, num_experts: int, hidden_size: int, enable_expert_bias: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.config = MockTransformerConfig(num_experts, hidden_size)
        self.layer_number = 0
        self.enable_expert_bias = enable_expert_bias
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size))
        if enable_expert_bias:
            self.register_buffer("expert_bias", torch.zeros(num_experts))
            self.register_buffer("local_tokens_per_expert", torch.zeros(num_experts))
        else:
            self.expert_bias = None
            self.local_tokens_per_expert = None


class MockExperts(nn.Module):
    """Mimics TEGroupedMLP: linear_fc1 and linear_fc2 with per-expert weights."""

    def __init__(self, num_experts: int, hidden: int, ffn: int):
        super().__init__()
        self.num_local_experts = num_experts
        self.linear_fc1 = MockGroupedLinear(num_experts, hidden, ffn)
        self.linear_fc2 = MockGroupedLinear(num_experts, ffn, hidden)


class MockMoELayer(nn.Module):
    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.experts = MockExperts(num_experts, hidden, ffn)
        self.router = MockRouter(num_experts, hidden)


class MockMLP(nn.Module):
    """Wraps experts + router like MoELayer.mlp"""

    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.experts = MockExperts(num_experts, hidden, ffn)
        self.router = MockRouter(num_experts, hidden)


class MockTransformerLayer(nn.Module):
    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.mlp = MockMLP(num_experts, hidden, ffn)


class MockDecoder(nn.Module):
    def __init__(self, num_layers, num_experts, hidden, ffn):
        super().__init__()
        self.layers = nn.ModuleList(
            [MockTransformerLayer(num_experts, hidden, ffn) for _ in range(num_layers)]
        )


class MockModel(nn.Module):
    """Top-level model with .module.decoder.layers structure."""

    def __init__(self, num_layers=2, num_experts=4, hidden=32, ffn=64):
        super().__init__()
        self.module = nn.Module()
        self.module.decoder = MockDecoder(num_layers, num_experts, hidden, ffn)


# ======================================================================
# Patch the mock classes into megatron's namespace so patch.py finds them
# ======================================================================

def _install_mock_megatron():
    """Create fake megatron.core.transformer.moe.{experts,router} modules."""
    # Build the module hierarchy
    for mod_path in [
        "megatron",
        "megatron.core",
        "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.experts",
        "megatron.core.transformer.moe.router",
    ]:
        if mod_path not in sys.modules:
            sys.modules[mod_path] = types.ModuleType(mod_path)

    experts_mod = sys.modules["megatron.core.transformer.moe.experts"]
    router_mod = sys.modules["megatron.core.transformer.moe.router"]

    # Register our mock classes as the "real" ones
    experts_mod.TEGroupedMLP = MockExperts
    experts_mod.GroupedMLP = MockExperts  # alias
    router_mod.TopKRouter = MockRouter


# ======================================================================
# Tests
# ======================================================================

class TestEndToEnd:
    """Full pipeline integration tests."""

    @classmethod
    def setup_class(cls):
        _install_mock_megatron()

    def _make_model_and_optimizer(self, num_experts=4, hidden=32, ffn=64, num_layers=2):
        model = MockModel(num_layers, num_experts, hidden, ffn)
        all_params = list(model.parameters())
        optimizer = torch.optim.Adam(all_params, lr=1e-3)
        # Do a fake step to populate optimizer state
        loss = sum(p.pow(2).sum() for p in all_params)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return model, optimizer

    def test_heuristic_copy_upcycling(self):
        """Test basic copy-paste upcycling through the full pipeline."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)
        orig_param_count = sum(1 for _ in model.parameters())

        from expert_upcycling.patch import apply_patches
        apply_patches()
        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={"method": "copy", "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"},
        )

        # Verify expert count doubled in every layer
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 8
            assert layer.mlp.experts.linear_fc1.num_gemms == 8
            assert layer.mlp.experts.linear_fc2.num_gemms == 8
            # Verify new weight attributes exist
            for i in range(8):
                assert hasattr(layer.mlp.experts.linear_fc1, f"weight{i}")
                assert hasattr(layer.mlp.experts.linear_fc2, f"weight{i}")
            # Verify router expanded
            assert layer.mlp.router.num_experts == 8
            assert layer.mlp.router.weight.shape[0] == 8

        # Verify new params were added to optimizer
        new_param_count = sum(len(pg["params"]) for pg in optimizer.param_groups)
        assert new_param_count > orig_param_count
        print(f"  params: {orig_param_count} -> {new_param_count}")

    def test_copy_preserves_weights(self):
        """COPY method should produce exact duplicates."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)

        # Save original weights
        orig_fc1_w0 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight0.data.clone()

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={"method": "copy"},
            router_cfg={"method": "copy"},
        )

        # weight4 should be exact copy of weight0
        new_w4 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4.data
        assert torch.allclose(orig_fc1_w0, new_w4), "COPY should produce exact duplicates"

    def test_utility_based_upcycling(self):
        """Test utility-based (principled) upcycling."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)

        # Make expert 3 have largest weights so it gets selected first
        with torch.no_grad():
            for layer in model.module.decoder.layers:
                layer.mlp.experts.linear_fc1.weight3.data *= 10.0
                layer.mlp.experts.linear_fc2.weight3.data *= 10.0

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={
                "usefulness_metric": "weight_norm",
                "selection_strategy": "greedy",
                "max_duplicates_per_expert": 3,
                "optimizer_state_strategy": "copy",
            },
            router_cfg={"method": "bias_only", "bias_noise_scale": 0.01},
        )

        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 8
            assert layer.mlp.router.num_experts == 8

    def test_noise_method_differs_from_original(self):
        """COPY_NOISE should produce different weights than the source."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)
        orig_w0 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight0.data.clone()

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={"method": "copy_noise", "noise_lambda": 0.1},
            router_cfg={"method": "copy"},
        )

        new_w4 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4.data
        assert not torch.allclose(orig_w0, new_w4, atol=1e-6), "COPY_NOISE should differ"

    def test_router_bias_only_preserves_weights(self):
        """BIAS_ONLY router method should keep weights identical."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)
        orig_router_w = model.module.decoder.layers[0].mlp.router.weight.data.clone()

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={"method": "copy"},
            router_cfg={"method": "bias_only", "bias_noise_scale": 0.0, "bias_shift": 0.0},
        )

        new_router_w = model.module.decoder.layers[0].mlp.router.weight.data
        # First E rows should be unchanged
        assert torch.allclose(orig_router_w, new_router_w[:4])
        # Duplicate rows should match originals (bias_only doesn't change weights)
        assert torch.allclose(orig_router_w, new_router_w[4:])

    def test_optimizer_state_copied(self):
        """New expert params should have optimizer state after COPY strategy."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(
            model, optimizer,
            expert_cfg={"method": "copy", "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"},
        )

        # Check that new params have optimizer state
        new_p = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4
        found = new_p in optimizer.state
        assert found, "New param should have optimizer state after COPY strategy"

    def test_2x_upcycling(self):
        """Test doubling twice: 4 -> 8 -> 16 experts."""
        model, optimizer = self._make_model_and_optimizer(num_experts=4)

        from expert_upcycling.upcycle_model import perform_expert_upcycling

        perform_expert_upcycling(model, optimizer,
                                 expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        perform_expert_upcycling(model, optimizer,
                                 expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})

        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 16
            assert layer.mlp.router.num_experts == 16
            assert layer.mlp.router.weight.shape[0] == 16
            for i in range(16):
                assert hasattr(layer.mlp.experts.linear_fc1, f"weight{i}")


# ======================================================================
# Run
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
