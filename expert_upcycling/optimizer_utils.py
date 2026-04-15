"""Optimizer state handling for upcycled expert parameters."""

import copy
import logging

import torch

logger = logging.getLogger(__name__)


class OptimizerStateHandler:
    """Handles optimizer state transfer for newly created expert parameters."""

    @staticmethod
    def _find_state(optimizer, param):
        """Locate the optimizer state entry for *param* (or its ``main_param``)."""
        key = getattr(param, "main_param", param)
        for state_key, state_dict in optimizer.state.items():
            actual_key = state_key
            opt_idx = None
            if isinstance(state_key, tuple) and len(state_key) > 1:
                opt_idx, actual_key = state_key[0], state_key[1]
            if actual_key is key:
                return opt_idx, state_dict
        return None, None

    @staticmethod
    def reset_optimizer_state(optimizer, param):
        """Remove any existing optimizer state so the param is treated as new."""
        key = getattr(param, "main_param", param)
        if key in optimizer.state:
            del optimizer.state[key]
        logger.info("[Upcycle] Reset optimizer state for param id %s", id(param))

    @staticmethod
    def copy_optimizer_state(optimizer, src_param, tgt_param):
        """Deep-copy optimizer state from *src_param* to *tgt_param*."""
        src_key = getattr(src_param, "main_param", src_param)
        tgt_key = getattr(tgt_param, "main_param", tgt_param)

        opt_idx, src_state = OptimizerStateHandler._find_state(optimizer, src_param)
        if src_state is None:
            logger.warning("[Upcycle] No optimizer state found for param id %s", id(src_param))
            return

        tgt_state = {}
        for k, v in src_state.items():
            tgt_state[k] = v.clone() if torch.is_tensor(v) else copy.deepcopy(v)

        tgt_state_key = (opt_idx, tgt_key) if opt_idx is not None else tgt_key
        optimizer.state[tgt_state_key] = tgt_state
        logger.info("[Upcycle] Copied optimizer state %s -> %s", id(src_param), id(tgt_param))

    @staticmethod
    def scale_optimizer_state(
        optimizer,
        src_param,
        tgt_param,
        src_data: torch.Tensor,
        tgt_data: torch.Tensor,
        momentum_scale: float = 0.5,
        variance_scale: float = 0.5,
    ):
        """Copy optimizer state and scale momentum / variance estimates."""
        tgt_key = getattr(tgt_param, "main_param", tgt_param)
        opt_idx, src_state = OptimizerStateHandler._find_state(optimizer, src_param)
        if src_state is None:
            logger.warning("[Upcycle] No optimizer state found for param id %s", id(src_param))
            return

        with torch.no_grad():
            change = (tgt_data - src_data).abs().mean() / (src_data.abs().mean() + 1e-8)
            adaptive = torch.clamp(1.0 - change, 0.1, 1.0).item()

        tgt_state = {}
        for k, v in src_state.items():
            if torch.is_tensor(v):
                if k in ("exp_avg", "momentum_buffer"):
                    tgt_state[k] = v.clone() * momentum_scale * adaptive
                elif k == "exp_avg_sq":
                    tgt_state[k] = v.clone() * variance_scale * adaptive
                elif k == "step":
                    tgt_state[k] = torch.tensor(0, dtype=v.dtype, device=v.device)
                else:
                    tgt_state[k] = v.clone()
            else:
                tgt_state[k] = 0 if k == "step" else copy.deepcopy(v)

        tgt_state_key = (opt_idx, tgt_key) if opt_idx is not None else tgt_key
        optimizer.state[tgt_state_key] = tgt_state
        logger.info(
            "[Upcycle] Scaled optimizer state %s -> %s (adaptive=%.3f)",
            id(src_param), id(tgt_param), adaptive,
        )

    @staticmethod
    def interpolate_optimizer_state(
        optimizer, src_param, tgt_param, other_param, alpha: float = 0.5
    ):
        """Interpolate optimizer states between *src_param* and *other_param*."""
        tgt_key = getattr(tgt_param, "main_param", tgt_param)
        opt_idx, src_state = OptimizerStateHandler._find_state(optimizer, src_param)
        _, other_state = OptimizerStateHandler._find_state(optimizer, other_param)

        if src_state is None or other_state is None:
            logger.warning("[Upcycle] Cannot interpolate optimizer state — missing source(s)")
            return

        tgt_state = {}
        for k in src_state:
            if k in other_state and torch.is_tensor(src_state[k]) and torch.is_tensor(other_state[k]):
                tgt_state[k] = (1 - alpha) * src_state[k].clone() + alpha * other_state[k].clone()
            else:
                tgt_state[k] = copy.deepcopy(src_state[k])

        tgt_state_key = (opt_idx, tgt_key) if opt_idx is not None else tgt_key
        optimizer.state[tgt_state_key] = tgt_state
        logger.info("[Upcycle] Interpolated optimizer state %s -> %s", id(src_param), id(tgt_param))
