"""Comprehensive integration tests for expert-upcycling package.

Tests every code path: all 10 expert methods, all 10 router methods,
utility-based selection with all 4 metrics × 2 strategies,
optimizer state handling (4 strategies), chained upcycling,
edge cases, and the full pipeline.
"""

import sys
import types
import copy
import torch
import torch.nn as nn
import numpy as np

# ======================================================================
# Mock Megatron-LM classes
# ======================================================================

class MockGroupedLinear(nn.Module):
    def __init__(self, num_experts, in_f, out_f):
        super().__init__()
        self.num_gemms = num_experts
        self.num_experts = num_experts
        for i in range(num_experts):
            setattr(self, f"weight{i}", nn.Parameter(torch.randn(out_f, in_f)))

class MockTransformerConfig:
    def __init__(self, num_experts, hidden_size):
        self.num_moe_experts = num_experts
        self.hidden_size = hidden_size

class MockRouter(nn.Module):
    def __init__(self, num_experts, hidden_size, enable_expert_bias=True):
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
    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.num_local_experts = num_experts
        self.linear_fc1 = MockGroupedLinear(num_experts, hidden, ffn)
        self.linear_fc2 = MockGroupedLinear(num_experts, ffn, hidden)

class MockMLP(nn.Module):
    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.experts = MockExperts(num_experts, hidden, ffn)
        self.router = MockRouter(num_experts, hidden)

class MockTransformerLayer(nn.Module):
    def __init__(self, num_experts, hidden, ffn):
        super().__init__()
        self.mlp = MockMLP(num_experts, hidden, ffn)

class MockDenseLayer(nn.Module):
    """A layer without MoE — should be skipped."""
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Linear(hidden, hidden)

class MockDecoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

class MockModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.module = nn.Module()
        self.module.decoder = MockDecoder(layers)


def install_mock_megatron():
    for p in [
        "megatron", "megatron.core", "megatron.core.transformer",
        "megatron.core.transformer.moe",
        "megatron.core.transformer.moe.experts",
        "megatron.core.transformer.moe.router",
    ]:
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)
    sys.modules["megatron.core.transformer.moe.experts"].TEGroupedMLP = MockExperts
    sys.modules["megatron.core.transformer.moe.experts"].GroupedMLP = MockExperts
    sys.modules["megatron.core.transformer.moe.router"].TopKRouter = MockRouter

install_mock_megatron()

# Now safe to import our package
from expert_upcycling.config import *
from expert_upcycling.expert_upcycler import ExpertUpcycler
from expert_upcycling.expert_selector import (
    ExpertUsefulnessEvaluator, ExpertSelector, PrincipledExpertUpcycler,
)
from expert_upcycling.router_upcycler import RouterUpcycler
from expert_upcycling.optimizer_utils import OptimizerStateHandler
from expert_upcycling.patch import apply_patches, _te_upcycle_experts, _topk_upcycle_router
from expert_upcycling.upcycle_model import perform_expert_upcycling


def make_model_and_opt(num_experts=4, hidden=32, ffn=64, num_layers=2):
    layers = [MockTransformerLayer(num_experts, hidden, ffn) for _ in range(num_layers)]
    model = MockModel(layers)
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    loss = sum(p.pow(2).sum() for p in params)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return model, opt


# ======================================================================
# Counters
# ======================================================================
_passed = 0
_failed = 0
_total = 0

def run_test(name, fn):
    global _passed, _failed, _total
    _total += 1
    print(f"  [{_total:2d}] {name:60s}", end="", flush=True)
    try:
        fn()
        _passed += 1
        print(" PASS")
    except Exception as e:
        _failed += 1
        print(f" FAIL: {e}")
        import traceback; traceback.print_exc()


# ======================================================================
# SECTION 1: Expert Upcycler — all 10 methods
# ======================================================================

def test_all_expert_methods():
    w = torch.randn(64, 128)
    others = [torch.randn(64, 128) for _ in range(4)]
    for method in UpcycleMethod:
        def _t(m=method):
            cfg = UpcycleConfig(method=m)
            out = ExpertUpcycler(cfg).upcycle_param(w, 0, 4, others)
            assert out.shape == w.shape, f"shape mismatch: {out.shape}"
            assert out.dtype == w.dtype, f"dtype mismatch: {out.dtype}"
            if m == UpcycleMethod.COPY:
                assert torch.allclose(out, w), "COPY must be exact"
            elif m == UpcycleMethod.SCALED_COPY:
                assert torch.allclose(out, w * 0.95, atol=1e-6)
            elif m != UpcycleMethod.COPY:
                assert not torch.allclose(out, w, atol=1e-6), f"{m} should differ from original"
        run_test(f"expert_method_{method.value}", _t)

def test_expert_methods_bfloat16():
    """Verify methods work with bfloat16 tensors."""
    w = torch.randn(32, 64, dtype=torch.bfloat16)
    others = [torch.randn(32, 64, dtype=torch.bfloat16) for _ in range(4)]
    for method in [UpcycleMethod.COPY, UpcycleMethod.SVD_PERTURB, UpcycleMethod.ORTHOGONAL]:
        def _t(m=method):
            out = ExpertUpcycler(UpcycleConfig(method=m)).upcycle_param(w, 0, 4, others)
            assert out.dtype == torch.bfloat16
        run_test(f"expert_bf16_{method.value}", _t)

def test_expert_interpolate_no_others():
    """INTERPOLATE without other_params should fall back to copy."""
    def _t():
        w = torch.randn(32, 64)
        out = ExpertUpcycler(UpcycleConfig(method=UpcycleMethod.INTERPOLATE)).upcycle_param(w, 0, 1, None)
        assert torch.allclose(out, w)
    run_test("expert_interpolate_no_others_fallback", _t)

def test_expert_drop_upcycle_init_methods():
    """Test all drop_upcycle init methods."""
    for init in ["xavier", "kaiming", "normal"]:
        def _t(i=init):
            w = torch.randn(64, 128)
            cfg = UpcycleConfig(method=UpcycleMethod.DROP_UPCYCLE, drop_ratio=0.5, drop_init_method=i)
            out = ExpertUpcycler(cfg).upcycle_param(w)
            assert out.shape == w.shape
        run_test(f"expert_drop_init_{init}", _t)


# ======================================================================
# SECTION 2: Router Upcycler — all 10 methods
# ======================================================================

def test_all_router_methods():
    E, H = 8, 64
    w = torch.randn(E, H)
    bias = torch.randn(E)
    order = list(range(E))
    for method in RouterUpcycleMethod:
        def _t(m=method):
            cfg = RouterUpcycleConfig(method=m)
            nw, nb = RouterUpcycler(cfg, order).upcycle_router_weights(w, bias)
            assert nw.shape == (2*E, H), f"weight shape: {nw.shape}"
            if nb is not None:
                assert nb.shape == (2*E,), f"bias shape: {nb.shape}"
        run_test(f"router_method_{method.value}", _t)

def test_router_no_bias():
    """Router methods should handle None bias."""
    def _t():
        w = torch.randn(4, 32)
        order = list(range(4))
        for m in RouterUpcycleMethod:
            if m == RouterUpcycleMethod.BIAS_ONLY:
                continue  # bias_only with None bias returns None
            nw, nb = RouterUpcycler(RouterUpcycleConfig(method=m), order).upcycle_router_weights(w, None)
            assert nw.shape == (8, 32)
            assert nb is None
    run_test("router_no_bias", _t)

def test_router_nonuniform_order():
    """Non-uniform duplication order (utility-based)."""
    def _t():
        w = torch.randn(4, 16)
        order = [3, 3, 2, 2]  # duplicate experts 3 and 2 twice each
        nw, _ = RouterUpcycler(RouterUpcycleConfig(), order).upcycle_router_weights(w, None)
        assert torch.allclose(nw[4], w[3]), "First dup should be expert 3"
        assert torch.allclose(nw[5], w[3]), "Second dup should be expert 3"
        assert torch.allclose(nw[6], w[2])
        assert torch.allclose(nw[7], w[2])
    run_test("router_nonuniform_order", _t)


# ======================================================================
# SECTION 3: Expert Selector — all metrics × strategies
# ======================================================================

def test_all_usefulness_metrics():
    params = [torch.randn(32, 64, requires_grad=True) for _ in range(8)]
    # Give gradients
    for p in params:
        p.grad = torch.randn_like(p)
        p.main_grad = p.grad  # Megatron convention
    for metric in UsefulnessMetric:
        def _t(m=metric):
            if m == UsefulnessMetric.APPROX_FISHER:
                return  # needs optimizer state, tested separately
            scores = ExpertUsefulnessEvaluator.evaluate(params, m)
            assert len(scores) == 8
            assert all(s >= 0 for s in scores)
        run_test(f"usefulness_metric_{metric.value}", _t)

def test_approx_fisher_metric():
    """APPROX_FISHER needs optimizer with exp_avg_sq."""
    def _t():
        params = [nn.Parameter(torch.randn(16, 16)) for _ in range(4)]
        opt = torch.optim.Adam(params, lr=0.01)
        loss = sum(p.pow(2).sum() for p in params)
        loss.backward()
        opt.step()
        for p in params:
            p.main_grad = p.grad
        scores = ExpertUsefulnessEvaluator.evaluate(params, UsefulnessMetric.APPROX_FISHER, opt)
        assert len(scores) == 4
        assert all(s >= 0 for s in scores)
    run_test("usefulness_approx_fisher", _t)

def test_greedy_selection():
    def _t():
        scores = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
        sel = ExpertSelector.greedy(scores, n=5, max_dup=2)
        assert len(sel) == 5
        assert sel[0] == 3  # highest score
        assert sel[1] == 3  # second dup of highest
        assert sel[2] == 1  # next highest
    run_test("greedy_selection", _t)

def test_greedy_max_dup_cycling():
    """When all experts hit max_dup, counts should reset."""
    def _t():
        scores = np.array([1.0, 2.0])
        sel = ExpertSelector.greedy(scores, n=6, max_dup=2)
        assert len(sel) == 6
    run_test("greedy_max_dup_cycling", _t)

def test_weighted_sampling():
    def _t():
        scores = np.array([0.0, 0.0, 0.0, 100.0])  # heavily biased
        sel = ExpertSelector.weighted_sampling(scores, n=10, max_dup=5, temperature=0.1)
        assert len(sel) == 10
        assert sel.count(3) >= 5  # expert 3 should dominate
    run_test("weighted_sampling_biased", _t)

def test_principled_all_layer_selections():
    fc1 = [torch.randn(32, 64) * (i+1) for i in range(4)]
    fc2 = [torch.randn(64, 32) * (i+1) for i in range(4)]
    for ls in LayerSelection:
        def _t(l=ls):
            cfg = PrincipledUpcycleConfig(layer_selection=l)
            sel = PrincipledExpertUpcycler(cfg).select_experts_to_duplicate(fc1, fc2, 4)
            assert len(sel) == 4
        run_test(f"principled_layer_sel_{ls.value}", _t)


# ======================================================================
# SECTION 4: Optimizer State Handler — all 4 strategies
# ======================================================================

def test_optimizer_reset():
    def _t():
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.Adam([p1], lr=0.01)
        (p1**2).sum().backward(); opt.step()
        opt.add_param_group({"params": [p2]})
        OptimizerStateHandler.copy_optimizer_state(opt, p1, p2)
        assert p2 in opt.state
        OptimizerStateHandler.reset_optimizer_state(opt, p2)
        assert p2 not in opt.state
    run_test("optimizer_reset", _t)

def test_optimizer_copy():
    def _t():
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.Adam([p1], lr=0.01)
        (p1**2).sum().backward(); opt.step()
        opt.add_param_group({"params": [p2]})
        OptimizerStateHandler.copy_optimizer_state(opt, p1, p2)
        assert torch.allclose(opt.state[p1]["exp_avg"], opt.state[p2]["exp_avg"])
        assert torch.allclose(opt.state[p1]["exp_avg_sq"], opt.state[p2]["exp_avg_sq"])
        # Verify deep copy (not shared)
        opt.state[p1]["exp_avg"].zero_()
        assert not torch.allclose(opt.state[p1]["exp_avg"], opt.state[p2]["exp_avg"])
    run_test("optimizer_copy_deep", _t)

def test_optimizer_scale():
    def _t():
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.Adam([p1], lr=0.01)
        (p1**2).sum().backward(); opt.step()
        opt.add_param_group({"params": [p2]})
        OptimizerStateHandler.scale_optimizer_state(opt, p1, p2, p1.data, p2.data, 0.5, 0.5)
        # Scaled state should be smaller than original
        assert opt.state[p2]["exp_avg"].abs().sum() <= opt.state[p1]["exp_avg"].abs().sum() + 1e-6
    run_test("optimizer_scale", _t)

def test_optimizer_interpolate():
    def _t():
        p1 = nn.Parameter(torch.randn(4, 4))
        p2 = nn.Parameter(torch.randn(4, 4))
        p3 = nn.Parameter(torch.randn(4, 4))
        opt = torch.optim.Adam([p1, p2], lr=0.01)
        (p1**2 + p2**2).sum().backward(); opt.step()
        opt.add_param_group({"params": [p3]})
        OptimizerStateHandler.interpolate_optimizer_state(opt, p1, p3, p2, alpha=0.5)
        expected = 0.5 * opt.state[p1]["exp_avg"] + 0.5 * opt.state[p2]["exp_avg"]
        assert torch.allclose(opt.state[p3]["exp_avg"], expected, atol=1e-6)
    run_test("optimizer_interpolate", _t)


# ======================================================================
# SECTION 5: Full pipeline integration tests
# ======================================================================

def test_pipeline_heuristic_copy():
    def _t():
        apply_patches()
        model, opt = make_model_and_opt(num_experts=4)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy", "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"})
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 8
            assert layer.mlp.router.num_experts == 8
            assert layer.mlp.router.weight.shape[0] == 8
            for i in range(8):
                assert hasattr(layer.mlp.experts.linear_fc1, f"weight{i}")
                assert hasattr(layer.mlp.experts.linear_fc2, f"weight{i}")
    run_test("pipeline_heuristic_copy", _t)

def test_pipeline_all_heuristic_methods():
    """Run full pipeline with every heuristic method."""
    for method in UpcycleMethod:
        def _t(m=method):
            model, opt = make_model_and_opt(num_experts=4)
            perform_expert_upcycling(model, opt,
                expert_cfg={"method": m.value, "optimizer_state_strategy": "copy"},
                router_cfg={"method": "copy"})
            for layer in model.module.decoder.layers:
                assert layer.mlp.experts.num_local_experts == 8
        run_test(f"pipeline_heuristic_{method.value}", _t)

def test_pipeline_all_router_methods():
    """Run full pipeline with every router method."""
    for method in RouterUpcycleMethod:
        def _t(m=method):
            model, opt = make_model_and_opt(num_experts=4)
            perform_expert_upcycling(model, opt,
                expert_cfg={"method": "copy"},
                router_cfg={"method": m.value})
            for layer in model.module.decoder.layers:
                assert layer.mlp.router.num_experts == 8
                assert layer.mlp.router.weight.shape[0] == 8
        run_test(f"pipeline_router_{method.value}", _t)

def test_pipeline_utility_all_metrics():
    """Run full pipeline with every usefulness metric."""
    for metric in UsefulnessMetric:
        def _t(m=metric):
            model, opt = make_model_and_opt(num_experts=4)
            perform_expert_upcycling(model, opt,
                expert_cfg={"usefulness_metric": m.value, "selection_strategy": "greedy",
                            "optimizer_state_strategy": "copy"},
                router_cfg={"method": "copy"})
            for layer in model.module.decoder.layers:
                assert layer.mlp.experts.num_local_experts == 8
        run_test(f"pipeline_utility_{metric.value}", _t)

def test_pipeline_utility_weighted_sampling():
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        perform_expert_upcycling(model, opt,
            expert_cfg={"usefulness_metric": "weight_norm", "selection_strategy": "weighted_sampling",
                        "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"})
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 8
    run_test("pipeline_utility_weighted_sampling", _t)

def test_pipeline_all_optimizer_strategies():
    """Test every optimizer state strategy through the full pipeline."""
    for strat in OptimizerStateStrategy:
        def _t(s=strat):
            model, opt = make_model_and_opt(num_experts=4)
            perform_expert_upcycling(model, opt,
                expert_cfg={"method": "copy", "optimizer_state_strategy": s.value},
                router_cfg={"method": "copy"})
            for layer in model.module.decoder.layers:
                assert layer.mlp.experts.num_local_experts == 8
        run_test(f"pipeline_opt_strategy_{strat.value}", _t)

def test_pipeline_copy_preserves_weights():
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        orig = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight0.data.clone()
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        new = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4.data
        assert torch.allclose(orig, new), "COPY should produce exact duplicates"
    run_test("pipeline_copy_exact_duplicate", _t)

def test_pipeline_noise_differs():
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        orig = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight0.data.clone()
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy_noise", "noise_lambda": 0.1},
            router_cfg={"method": "copy"})
        new = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4.data
        assert not torch.allclose(orig, new, atol=1e-6)
    run_test("pipeline_noise_differs", _t)

def test_pipeline_chained_4_to_16():
    """4 -> 8 -> 16 experts via two upcycling rounds."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 16
            assert layer.mlp.router.num_experts == 16
            assert layer.mlp.router.weight.shape[0] == 16
            for i in range(16):
                assert hasattr(layer.mlp.experts.linear_fc1, f"weight{i}")
    run_test("pipeline_chained_4_to_16", _t)

def test_pipeline_chained_4_to_32():
    """4 -> 8 -> 16 -> 32 experts via three rounds."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        for _ in range(3):
            perform_expert_upcycling(model, opt,
                expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 32
            assert layer.mlp.router.num_experts == 32
    run_test("pipeline_chained_4_to_32", _t)

def test_pipeline_utility_selects_strongest():
    """Utility-based should preferentially duplicate the strongest expert."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        # Make expert 3 much larger
        with torch.no_grad():
            for layer in model.module.decoder.layers:
                layer.mlp.experts.linear_fc1.weight3.data *= 100.0
                layer.mlp.experts.linear_fc2.weight3.data *= 100.0
        perform_expert_upcycling(model, opt,
            expert_cfg={"usefulness_metric": "weight_norm", "selection_strategy": "greedy",
                        "max_duplicates_per_expert": 3},
            router_cfg={"method": "copy"})
        # weight4 should be a copy of weight3 (strongest)
        w3 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight3.data
        w4 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4.data
        assert torch.allclose(w3, w4), "First duplicate should be strongest expert"
    run_test("pipeline_utility_selects_strongest", _t)

def test_pipeline_router_bias_only_preserves_weights():
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        orig_w = model.module.decoder.layers[0].mlp.router.weight.data.clone()
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"},
            router_cfg={"method": "bias_only", "bias_noise_scale": 0.0, "bias_shift": 0.0})
        new_w = model.module.decoder.layers[0].mlp.router.weight.data
        assert torch.allclose(orig_w, new_w[:4])
        assert torch.allclose(orig_w, new_w[4:])
    run_test("pipeline_router_bias_only_preserves", _t)

def test_pipeline_router_buffers_expanded():
    """expert_bias and local_tokens_per_expert should double."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        r = model.module.decoder.layers[0].mlp.router
        if r.expert_bias is not None:
            assert r.expert_bias.shape[0] == 8
        if r.local_tokens_per_expert is not None:
            assert r.local_tokens_per_expert.shape[0] == 8
    run_test("pipeline_router_buffers_expanded", _t)

def test_pipeline_optimizer_has_new_params():
    """After upcycling, optimizer should contain new expert params."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        orig_n = sum(len(pg["params"]) for pg in opt.param_groups)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy", "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"})
        new_n = sum(len(pg["params"]) for pg in opt.param_groups)
        # Each layer adds 4 new fc1 + 4 new fc2 = 8 params, 2 layers = 16
        assert new_n == orig_n + 16, f"Expected {orig_n+16} params, got {new_n}"
    run_test("pipeline_optimizer_new_params", _t)

def test_pipeline_optimizer_state_exists():
    """New params should have optimizer state with COPY strategy."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy", "optimizer_state_strategy": "copy"},
            router_cfg={"method": "copy"})
        w4 = model.module.decoder.layers[0].mlp.experts.linear_fc1.weight4
        assert w4 in opt.state, "New param should have optimizer state"
        assert "exp_avg" in opt.state[w4]
        assert "exp_avg_sq" in opt.state[w4]
    run_test("pipeline_optimizer_state_exists", _t)

def test_pipeline_mixed_layers():
    """Model with both MoE and dense layers — dense should be skipped."""
    def _t():
        layers = [
            MockTransformerLayer(4, 32, 64),
            MockDenseLayer(32),
            MockTransformerLayer(4, 32, 64),
        ]
        model = MockModel(layers)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        (sum(p.pow(2).sum() for p in params)).backward(); opt.step(); opt.zero_grad()
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        # MoE layers should be upcycled
        assert model.module.decoder.layers[0].mlp.experts.num_local_experts == 8
        assert model.module.decoder.layers[2].mlp.experts.num_local_experts == 8
        # Dense layer should be unchanged
        assert isinstance(model.module.decoder.layers[1].mlp, nn.Linear)
    run_test("pipeline_mixed_layers", _t)

def test_pipeline_single_expert():
    """Edge case: 1 expert -> 2 experts."""
    def _t():
        model, opt = make_model_and_opt(num_experts=1)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        for layer in model.module.decoder.layers:
            assert layer.mlp.experts.num_local_experts == 2
            assert layer.mlp.router.num_experts == 2
    run_test("pipeline_single_expert", _t)

def test_pipeline_large_expert_count():
    """32 -> 64 experts."""
    def _t():
        model, opt = make_model_and_opt(num_experts=32, hidden=16, ffn=32, num_layers=1)
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        assert model.module.decoder.layers[0].mlp.experts.num_local_experts == 64
        assert model.module.decoder.layers[0].mlp.router.weight.shape[0] == 64
    run_test("pipeline_32_to_64", _t)

def test_config_dict_conversion():
    """Verify dict configs are properly converted to dataclasses."""
    def _t():
        model, opt = make_model_and_opt(num_experts=4)
        # These are dicts, not dataclasses — patch.py should convert them
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "svd_perturb", "svd_perturb_singular_values": 0.2,
                        "optimizer_state_strategy": "reset"},
            router_cfg={"method": "svd_perturb", "svd_perturb_singular_values": 0.1})
        assert model.module.decoder.layers[0].mlp.experts.num_local_experts == 8
    run_test("config_dict_conversion", _t)

def test_router_no_expert_bias():
    """Router without expert_bias should still work."""
    def _t():
        layers = []
        for _ in range(1):
            layer = MockTransformerLayer(4, 32, 64)
            layer.mlp.router = MockRouter(4, 32, enable_expert_bias=False)
            layers.append(layer)
        model = MockModel(layers)
        params = list(model.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        (sum(p.pow(2).sum() for p in params)).backward(); opt.step(); opt.zero_grad()
        perform_expert_upcycling(model, opt,
            expert_cfg={"method": "copy"}, router_cfg={"method": "copy"})
        assert model.module.decoder.layers[0].mlp.router.num_experts == 8
    run_test("router_no_expert_bias", _t)


# ======================================================================
# RUN ALL
# ======================================================================

if __name__ == "__main__":
    print("=" * 72)
    print("COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 72)

    print("\n--- Section 1: Expert Upcycler (all methods) ---")
    test_all_expert_methods()
    test_expert_methods_bfloat16()
    test_expert_interpolate_no_others()
    test_expert_drop_upcycle_init_methods()

    print("\n--- Section 2: Router Upcycler (all methods) ---")
    test_all_router_methods()
    test_router_no_bias()
    test_router_nonuniform_order()

    print("\n--- Section 3: Expert Selector (metrics × strategies) ---")
    test_all_usefulness_metrics()
    test_approx_fisher_metric()
    test_greedy_selection()
    test_greedy_max_dup_cycling()
    test_weighted_sampling()
    test_principled_all_layer_selections()

    print("\n--- Section 4: Optimizer State Handler ---")
    test_optimizer_reset()
    test_optimizer_copy()
    test_optimizer_scale()
    test_optimizer_interpolate()

    print("\n--- Section 5: Full Pipeline Integration ---")
    test_pipeline_heuristic_copy()
    test_pipeline_all_heuristic_methods()
    test_pipeline_all_router_methods()
    test_pipeline_utility_all_metrics()
    test_pipeline_utility_weighted_sampling()
    test_pipeline_all_optimizer_strategies()
    test_pipeline_copy_preserves_weights()
    test_pipeline_noise_differs()
    test_pipeline_chained_4_to_16()
    test_pipeline_chained_4_to_32()
    test_pipeline_utility_selects_strongest()
    test_pipeline_router_bias_only_preserves_weights()
    test_pipeline_router_buffers_expanded()
    test_pipeline_optimizer_has_new_params()
    test_pipeline_optimizer_state_exists()
    test_pipeline_mixed_layers()
    test_pipeline_single_expert()
    test_pipeline_large_expert_count()
    test_config_dict_conversion()
    test_router_no_expert_bias()

    print("\n" + "=" * 72)
    print(f"RESULTS: {_passed} passed, {_failed} failed, {_total} total")
    print("=" * 72)
    if _failed:
        sys.exit(1)
