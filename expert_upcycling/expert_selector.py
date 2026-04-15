"""Utility-based expert selection for principled upcycling.

Evaluates expert importance using gradient-based metrics and selects
which experts to duplicate.
"""

import copy
import logging
from typing import List, Optional

import numpy as np
import torch

from expert_upcycling.config import (
    LayerSelection,
    PrincipledUpcycleConfig,
    SelectionStrategy,
    UsefulnessMetric,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Usefulness evaluation
# ---------------------------------------------------------------------------

class ExpertUsefulnessEvaluator:
    """Computes per-expert importance scores."""

    @staticmethod
    def _weight_norm(param: torch.Tensor) -> float:
        with torch.no_grad():
            return param.norm(p=2).item()

    @staticmethod
    def _get_grad(param: torch.Tensor):
        """Retrieve gradient, preferring main_grad (Megatron convention)."""
        g = getattr(param, "main_grad", None)
        if g is not None:
            return g
        return getattr(param, "grad", None)

    @staticmethod
    def _weight_grad_product(param: torch.Tensor) -> float:
        grad = ExpertUsefulnessEvaluator._get_grad(param)
        if grad is None:
            return ExpertUsefulnessEvaluator._weight_norm(param)
        with torch.no_grad():
            return param.norm(p=2).item() * grad.norm(p=2).item()

    @staticmethod
    def _gradient_squared(param: torch.Tensor) -> float:
        grad = ExpertUsefulnessEvaluator._get_grad(param)
        if grad is None:
            return 0.0
        with torch.no_grad():
            return grad.norm(p=2).item() ** 2

    @staticmethod
    def _approx_fisher(param: torch.Tensor, optimizer, eps: float) -> float:
        grad = ExpertUsefulnessEvaluator._get_grad(param)
        if grad is None:
            return 0.0
        key = getattr(param, "main_param", param)
        state = None
        for sk, sd in optimizer.state.items():
            ak = sk
            if isinstance(sk, tuple) and len(sk) > 1:
                ak = sk[1]
            if ak is key:
                state = sd
                break
        if state is None or "exp_avg_sq" not in state:
            return ExpertUsefulnessEvaluator._gradient_squared(param)
        with torch.no_grad():
            g2 = grad.pow(2).sum().item()
            vn = state["exp_avg_sq"].norm(p=2).item()
            return g2 / (vn + eps)

    @staticmethod
    def evaluate(
        params: List[torch.Tensor],
        metric: UsefulnessMetric,
        optimizer=None,
        eps: float = 1e-8,
    ) -> np.ndarray:
        scores = []
        for p in params:
            if metric == UsefulnessMetric.WEIGHT_NORM:
                s = ExpertUsefulnessEvaluator._weight_norm(p)
            elif metric == UsefulnessMetric.WEIGHT_GRAD_PRODUCT:
                s = ExpertUsefulnessEvaluator._weight_grad_product(p)
            elif metric == UsefulnessMetric.GRADIENT_SQUARED:
                s = ExpertUsefulnessEvaluator._gradient_squared(p)
            elif metric == UsefulnessMetric.APPROX_FISHER:
                s = ExpertUsefulnessEvaluator._approx_fisher(p, optimizer, eps)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            scores.append(s)
        return np.array(scores)


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

class ExpertSelector:
    """Chooses which experts to duplicate given usefulness scores."""

    @staticmethod
    def greedy(scores: np.ndarray, n: int, max_dup: int) -> List[int]:
        order = np.argsort(-scores)
        counts = np.zeros(len(scores), dtype=int)
        selected = []
        for _ in range(n):
            chosen = None
            for idx in order:
                if counts[idx] < max_dup:
                    chosen = idx
                    break
            if chosen is None:
                counts[:] = 0
                chosen = order[0]
            selected.append(int(chosen))
            counts[chosen] += 1
        return selected

    @staticmethod
    def weighted_sampling(
        scores: np.ndarray, n: int, max_dup: int, temperature: float = 1.0
    ) -> List[int]:
        s = scores / max(temperature, 1e-10)
        s = s - s.max()
        probs = np.exp(s)
        probs = probs / (probs.sum() + 1e-30)  # avoid div-by-zero
        # Replace any remaining NaN with uniform
        if np.any(np.isnan(probs)):
            probs = np.ones_like(probs) / len(probs)
        counts = np.zeros(len(scores), dtype=int)
        selected = []
        for _ in range(n):
            mask = counts < max_dup
            if not mask.any():
                counts[:] = 0
                mask = np.ones(len(scores), dtype=bool)
            p = probs * mask
            psum = p.sum()
            if psum < 1e-30:
                p = mask.astype(float)
                psum = p.sum()
            p = p / psum
            idx = int(np.random.choice(len(scores), p=p))
            selected.append(idx)
            counts[idx] += 1
        return selected


# ---------------------------------------------------------------------------
# Principled upcycler (combines evaluation + selection)
# ---------------------------------------------------------------------------

class PrincipledExpertUpcycler:
    """Selects experts to duplicate based on importance scores."""

    def __init__(self, config: PrincipledUpcycleConfig):
        self.config = config

    @staticmethod
    def _combine(fc1: np.ndarray, fc2: np.ndarray, method: LayerSelection) -> np.ndarray:
        if method == LayerSelection.FC1_ONLY:
            return fc1
        if method == LayerSelection.FC2_ONLY:
            return fc2
        if method == LayerSelection.BOTH_AVERAGE:
            return (fc1 + fc2) / 2.0
        if method == LayerSelection.BOTH_PRODUCT:
            f1 = fc1 / (fc1.sum() + 1e-10)
            f2 = fc2 / (fc2.sum() + 1e-10)
            return f1 * f2
        if method == LayerSelection.BOTH_MAX:
            return np.maximum(fc1, fc2)
        if method == LayerSelection.BOTH_MIN:
            return np.minimum(fc1, fc2)
        raise ValueError(f"Unknown layer selection: {method}")

    def select_experts_to_duplicate(
        self,
        fc1_params: List[torch.Tensor],
        fc2_params: List[torch.Tensor],
        num_new: int,
        optimizer=None,
    ) -> List[int]:
        cfg = self.config
        fc1_scores = ExpertUsefulnessEvaluator.evaluate(
            fc1_params, cfg.usefulness_metric, optimizer, cfg.fisher_epsilon
        )
        logger.info("[Principled] FC1 scores (%s): %s", cfg.usefulness_metric.value, fc1_scores)

        if cfg.layer_selection != LayerSelection.FC1_ONLY:
            fc2_scores = ExpertUsefulnessEvaluator.evaluate(
                fc2_params, cfg.usefulness_metric, optimizer, cfg.fisher_epsilon
            )
            scores = self._combine(fc1_scores, fc2_scores, cfg.layer_selection)
        else:
            scores = fc1_scores

        if cfg.selection_strategy == SelectionStrategy.GREEDY:
            sel = ExpertSelector.greedy(scores, num_new, cfg.max_duplicates_per_expert)
        elif cfg.selection_strategy == SelectionStrategy.WEIGHTED_SAMPLING:
            sel = ExpertSelector.weighted_sampling(
                scores, num_new, cfg.max_duplicates_per_expert, cfg.temperature
            )
        else:
            raise ValueError(f"Unknown strategy: {cfg.selection_strategy}")

        logger.info("[Principled] Selected experts: %s", sel)
        return sel
