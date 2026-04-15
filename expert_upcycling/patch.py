"""Monkey-patch upcycling methods onto Megatron-LM MoE classes.

Usage::

    import expert_upcycling
    expert_upcycling.apply_patches()

    # Now TEGroupedMLP instances have .upcycle_experts()
    # and TopKRouter instances have .upcycle_router()
"""

import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn

from expert_upcycling.config import (
    OptimizerStateStrategy,
    PrincipledUpcycleConfig,
    RouterUpcycleConfig,
    RouterUpcycleMethod,
    UpcycleConfig,
    UpcycleMethod,
)
from expert_upcycling.expert_selector import PrincipledExpertUpcycler
from expert_upcycling.expert_upcycler import ExpertUpcycler
from expert_upcycling.optimizer_utils import OptimizerStateHandler
from expert_upcycling.router_upcycler import RouterUpcycler

logger = logging.getLogger(__name__)

_PATCHED = False


# ======================================================================
# Helpers shared by both upcycling paths
# ======================================================================

def _get_per_expert_params(linear_mod, num_experts: int) -> list:
    """Retrieve per-expert weight parameters from a TE GroupedLinear module."""
    params = []
    for i in range(num_experts):
        p = getattr(linear_mod, f"weight{i}", None)
        if p is None:
            raise RuntimeError(
                f"Cannot find weight{i} in {linear_mod.__class__.__name__}. "
                "Ensure you are using TEGroupedMLP with Transformer Engine."
            )
        params.append(p)
    return params


def _copy_param_data(src: nn.Parameter, tgt: nn.Parameter, data: torch.Tensor):
    """Copy *data* into *tgt*, syncing ``main_param`` if present."""
    with torch.no_grad():
        tgt.data.copy_(data)
    sp = getattr(src, "main_param", None)
    tp = getattr(tgt, "main_param", None)
    if sp is not None and tp is not None:
        with torch.no_grad():
            tp.data.copy_(data)


def _handle_optimizer_state(
    strategy: OptimizerStateStrategy,
    handler: OptimizerStateHandler,
    optimizer,
    src_p, tgt_p,
    src_data=None, tgt_data=None,
    momentum_scale=0.5, variance_scale=0.5,
    interp_other=None, interp_alpha=0.5,
):
    if strategy == OptimizerStateStrategy.RESET:
        handler.reset_optimizer_state(optimizer, tgt_p)
    elif strategy == OptimizerStateStrategy.COPY:
        handler.copy_optimizer_state(optimizer, src_p, tgt_p)
    elif strategy == OptimizerStateStrategy.SCALE:
        handler.scale_optimizer_state(
            optimizer, src_p, tgt_p, src_data, tgt_data, momentum_scale, variance_scale
        )
    elif strategy == OptimizerStateStrategy.INTERPOLATE:
        if interp_other is not None:
            handler.interpolate_optimizer_state(optimizer, src_p, tgt_p, interp_other, interp_alpha)
        else:
            handler.reset_optimizer_state(optimizer, tgt_p)


# ======================================================================
# Methods that will be bound to TEGroupedMLP
# ======================================================================

def _te_upcycle_experts(
    self,
    optimizer: torch.optim.Optimizer,
    current_layer_index: int,
    upcycle_config=None,
) -> List[int]:
    """Dispatch to utility-based or heuristic upcycling."""
    if upcycle_config is None:
        upcycle_config = UpcycleConfig()
    if isinstance(upcycle_config, dict):
        if "usefulness_metric" in upcycle_config:
            return _te_upcycle_experts_utility(self, optimizer, current_layer_index, upcycle_config)
        return _te_upcycle_experts_heuristic(self, optimizer, current_layer_index, upcycle_config)
    if isinstance(upcycle_config, PrincipledUpcycleConfig):
        return _te_upcycle_experts_utility(self, optimizer, current_layer_index, upcycle_config)
    return _te_upcycle_experts_heuristic(self, optimizer, current_layer_index, upcycle_config)


def _te_upcycle_experts_utility(
    self,
    optimizer: torch.optim.Optimizer,
    current_layer_index: int,
    config=None,
) -> List[int]:
    """Utility-based expert upcycling for TEGroupedMLP."""
    if isinstance(config, dict):
        from dacite import from_dict, Config as DaciteConfig
        from expert_upcycling.config import UsefulnessMetric, SelectionStrategy, LayerSelection
        config = from_dict(
            data_class=PrincipledUpcycleConfig, data=config,
            config=DaciteConfig(cast=[UsefulnessMetric, SelectionStrategy, OptimizerStateStrategy, LayerSelection]),
        )
    if config is None:
        config = PrincipledUpcycleConfig()

    upcycler = PrincipledExpertUpcycler(config)
    handler = OptimizerStateHandler()
    orig_n = self.num_local_experts

    fc1_params = _get_per_expert_params(self.linear_fc1, orig_n)
    fc2_params = _get_per_expert_params(self.linear_fc2, orig_n)

    selected = upcycler.select_experts_to_duplicate(fc1_params, fc2_params, orig_n, optimizer)
    logger.info(
        "[Utility Upcycle] layer %d: %d -> %d experts, selected=%s",
        current_layer_index, orig_n, orig_n * 2, selected,
    )

    for new_idx, src_idx in enumerate(selected):
        tgt_idx = orig_n + new_idx
        for linear, params in ((self.linear_fc1, fc1_params), (self.linear_fc2, fc2_params)):
            src_p = params[src_idx]
            tgt_p = nn.Parameter(torch.empty_like(src_p.data))
            setattr(linear, f"weight{tgt_idx}", tgt_p)
            _copy_param_data(src_p, tgt_p, src_p.data.clone())
            optimizer.add_param_group({"params": [tgt_p]})
            _handle_optimizer_state(
                config.optimizer_state_strategy, handler, optimizer, src_p, tgt_p,
                src_p.data, tgt_p.data, config.momentum_scale, config.variance_scale,
            )

    new_total = orig_n * 2
    self.num_local_experts = new_total
    self.linear_fc1.num_gemms = new_total
    self.linear_fc2.num_gemms = new_total
    for attr in ("num_experts",):
        if hasattr(self.linear_fc1, attr):
            setattr(self.linear_fc1, attr, new_total)
        if hasattr(self.linear_fc2, attr):
            setattr(self.linear_fc2, attr, new_total)

    logger.info("[Utility Upcycle] layer %d done, total experts=%d", current_layer_index, new_total)
    return selected


def _te_upcycle_experts_heuristic(
    self,
    optimizer: torch.optim.Optimizer,
    current_layer_index: int,
    config=None,
) -> List[int]:
    """Heuristic expert upcycling for TEGroupedMLP."""
    if isinstance(config, dict):
        from dacite import from_dict, Config as DaciteConfig
        config = from_dict(
            data_class=UpcycleConfig, data=config,
            config=DaciteConfig(cast=[UpcycleMethod, OptimizerStateStrategy]),
        )
    if config is None:
        config = UpcycleConfig()

    eu = ExpertUpcycler(config)
    handler = OptimizerStateHandler()
    orig_n = self.num_local_experts

    fc1_params = _get_per_expert_params(self.linear_fc1, orig_n)
    fc2_params = _get_per_expert_params(self.linear_fc2, orig_n)

    fc1_list = [p.data for p in fc1_params] if config.method == UpcycleMethod.INTERPOLATE else None
    fc2_list = [p.data for p in fc2_params] if config.method == UpcycleMethod.INTERPOLATE else None

    logger.info(
        "[Heuristic Upcycle][%s] layer %d: %d -> %d experts",
        config.method.value, current_layer_index, orig_n, orig_n * 2,
    )

    for i in range(orig_n):
        tgt_idx = i + orig_n
        for linear, params, other_list in (
            (self.linear_fc1, fc1_params, fc1_list),
            (self.linear_fc2, fc2_params, fc2_list),
        ):
            src_p = params[i]
            new_data = eu.upcycle_param(src_p.data, i, orig_n, other_list)
            tgt_p = nn.Parameter(torch.empty_like(src_p.data))
            setattr(linear, f"weight{tgt_idx}", tgt_p)
            _copy_param_data(src_p, tgt_p, new_data)
            optimizer.add_param_group({"params": [tgt_p]})

            interp_other = params[(i + 1) % len(params)] if (
                config.optimizer_state_strategy == OptimizerStateStrategy.INTERPOLATE
                and other_list and len(other_list) > 1
            ) else None
            _handle_optimizer_state(
                config.optimizer_state_strategy, handler, optimizer, src_p, tgt_p,
                src_p.data, new_data, config.momentum_scale, config.variance_scale,
                interp_other, config.interp_alpha,
            )

    new_total = orig_n * 2
    self.num_local_experts = new_total
    self.linear_fc1.num_gemms = new_total
    self.linear_fc2.num_gemms = new_total
    for attr in ("num_experts",):
        if hasattr(self.linear_fc1, attr):
            setattr(self.linear_fc1, attr, new_total)
        if hasattr(self.linear_fc2, attr):
            setattr(self.linear_fc2, attr, new_total)

    logger.info("[Heuristic Upcycle] layer %d done, total experts=%d", current_layer_index, new_total)
    return list(range(orig_n))


# ======================================================================
# Method that will be bound to TopKRouter
# ======================================================================

def _topk_upcycle_router(
    self,
    upcycle_config=None,
    selected_expert_indices: Optional[List[int]] = None,
):
    """Expand router from E to 2E experts."""
    if isinstance(upcycle_config, dict):
        from dacite import from_dict, Config as DaciteConfig
        upcycle_config = from_dict(
            data_class=RouterUpcycleConfig, data=upcycle_config,
            config=DaciteConfig(cast=[RouterUpcycleMethod]),
        )
    if upcycle_config is None:
        upcycle_config = RouterUpcycleConfig()

    if selected_expert_indices is None:
        selected_expert_indices = list(range(self.num_experts))

    ru = RouterUpcycler(upcycle_config, selected_expert_indices)

    with torch.no_grad():
        orig_n = self.num_experts
        new_n = orig_n * 2

        # Get bias — latest Megatron uses self.bias (Parameter or None),
        # older versions use self.expert_bias (buffer).
        orig_bias = None
        if hasattr(self, "bias") and self.bias is not None:
            orig_bias = self.bias.data
        elif hasattr(self, "expert_bias") and self.expert_bias is not None:
            orig_bias = self.expert_bias.data

        new_w, new_b = ru.upcycle_router_weights(self.weight.data, orig_bias)

        # Replace weight parameter
        new_weight_param = torch.nn.Parameter(new_w)
        if hasattr(self.weight, "sequence_parallel"):
            setattr(new_weight_param, "sequence_parallel", self.weight.sequence_parallel)
        self.weight = new_weight_param

        # Replace bias
        if new_b is not None:
            if hasattr(self, "bias") and self.bias is not None:
                self.bias = torch.nn.Parameter(new_b)
            elif hasattr(self, "expert_bias"):
                self.register_buffer("expert_bias", new_b)

        # Expand loss-free load-balancing buffers
        if hasattr(self, "enable_expert_bias") and self.enable_expert_bias:
            if hasattr(self, "expert_bias") and self.expert_bias is not None and self.expert_bias.shape[0] == orig_n:
                self.register_buffer("expert_bias", torch.cat([
                    self.expert_bias, torch.zeros_like(self.expert_bias)
                ]))
            if hasattr(self, "local_tokens_per_expert") and self.local_tokens_per_expert is not None:
                self.register_buffer(
                    "local_tokens_per_expert",
                    torch.cat([self.local_tokens_per_expert, torch.zeros_like(self.local_tokens_per_expert)]),
                    persistent=False,
                )
        if hasattr(self, "global_tokens_per_expert") and self.global_tokens_per_expert is not None:
            self.register_buffer(
                "global_tokens_per_expert",
                torch.cat([self.global_tokens_per_expert, torch.zeros_like(self.global_tokens_per_expert)]),
                persistent=False,
            )

        self.config.num_moe_experts = new_n
        self.num_experts = new_n

        logger.info(
            "[Router Upcycle][%s] layer %s: %d -> %d experts, weight=%s",
            upcycle_config.method.value, self.layer_number, orig_n, new_n, tuple(self.weight.shape),
        )


# ======================================================================
# Patch application
# ======================================================================

def apply_patches():
    """Attach upcycling methods to Megatron-LM MoE classes."""
    global _PATCHED
    if _PATCHED:
        return

    try:
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError as e:
        raise ImportError(
            "Cannot import Megatron-LM MoE classes. "
            "Ensure megatron-core is installed."
        ) from e

    TEGroupedMLP.upcycle_experts = _te_upcycle_experts
    TopKRouter.upcycle_router = _topk_upcycle_router

    _PATCHED = True
    logger.info("expert_upcycling: patched TEGroupedMLP.upcycle_experts and TopKRouter.upcycle_router")
