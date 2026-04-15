"""Router weight duplication strategies.

Each method takes the original router weight matrix ``[E, hidden]`` and
returns an expanded matrix ``[2E, hidden]``.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from expert_upcycling.config import RouterUpcycleConfig, RouterUpcycleMethod

logger = logging.getLogger(__name__)


class RouterUpcycler:
    """Applies a configured strategy to expand router weights from E to 2E experts."""

    def __init__(self, config: RouterUpcycleConfig, expert_duplication_order: List[int]):
        self.config = config
        self.expert_duplication_order = expert_duplication_order
        self.duplicate_map: Dict[int, int] = {
            d_id: o_id for o_id, d_id in enumerate(expert_duplication_order)
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def upcycle_router_weights(
        self,
        original_weights: torch.Tensor,
        original_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(new_weights, new_bias)`` with shape ``[2E, ...]``."""
        m = self.config.method
        order = self.expert_duplication_order

        # Default bias handling: duplicate according to order
        def _dup_bias(b):
            if b is None:
                return None
            return torch.cat([b, b[order].clone()], dim=0)

        if m == RouterUpcycleMethod.COPY:
            w = torch.cat([original_weights, original_weights[order].clone()], dim=0)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.COPY_NOISE:
            w = self._copy_noise(original_weights, order, self.config.noise_lambda)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.INTERPOLATE:
            w = self._interpolate(original_weights, order, self.config.interp_alpha, self.config.interp_circular)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.BIAS_ONLY:
            return self._bias_only(
                original_weights, original_bias, order,
                self.config.bias_noise_scale, self.config.bias_shift, self.config.noise_to_original,
            )

        if m == RouterUpcycleMethod.SCALED_COPY:
            dup = original_weights[order] * self.config.scale_factor
            return torch.cat([original_weights, dup], dim=0), _dup_bias(original_bias)

        if m == RouterUpcycleMethod.PERTURB_NEW_ONLY:
            w = self._perturb_new_only(original_weights, order, self.config.new_expert_noise)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.ORTHOGONAL:
            w = self._orthogonal(original_weights, order)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.ADVERSARIAL:
            w = self._adversarial(original_weights, order, self.config.adversarial_strength)
            return w, _dup_bias(original_bias)

        if m == RouterUpcycleMethod.TEMPERATURE_SCALED:
            dup = original_weights[order] * self.config.temperature_factor
            return torch.cat([original_weights, dup], dim=0), _dup_bias(original_bias)

        if m == RouterUpcycleMethod.SVD_PERTURB:
            w = self._svd_perturb(
                original_weights, order,
                self.config.svd_perturb_singular_values, self.config.svd_perturb_vectors,
            )
            return w, _dup_bias(original_bias)

        raise ValueError(f"Unknown router upcycle method: {m}")

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_noise(w: torch.Tensor, order: List[int], lam: float) -> torch.Tensor:
        dup = w[order].clone()
        dup.add_(torch.randn_like(dup) * w.std() * lam)
        return torch.cat([w, dup], dim=0)

    @staticmethod
    def _interpolate(w: torch.Tensor, order: List[int], alpha: float, circular: bool) -> torch.Tensor:
        E = w.shape[0]
        dups = []
        for i in order:
            nxt = (i + 1) % E if circular else min(i + 1, E - 1)
            dups.append((1 - alpha) * w[i] + alpha * w[nxt])
        return torch.cat([w, torch.stack(dups)], dim=0)

    @staticmethod
    def _bias_only(
        w: torch.Tensor, bias: Optional[torch.Tensor], order: List[int],
        noise_scale: float, shift: float, noise_to_original: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        new_w = torch.cat([w, w[order].clone()], dim=0)
        if bias is None:
            return new_w, None
        orig_bias = bias.clone() if noise_to_original else bias
        dup_bias = bias[order].clone()
        if noise_scale > 0:
            dup_bias.add_(torch.randn_like(dup_bias) * dup_bias.std() * noise_scale)
            if noise_to_original:
                orig_bias = orig_bias * (1 + orig_bias.std() * noise_scale)
        if shift != 0:
            dup_bias.add_(shift)
        return new_w, torch.cat([orig_bias, dup_bias], dim=0)

    @staticmethod
    def _perturb_new_only(w: torch.Tensor, order: List[int], noise: float) -> torch.Tensor:
        dup = w[order].clone()
        dup.add_(torch.randn_like(dup) * dup.std() * noise)
        return torch.cat([w, dup], dim=0)

    def _orthogonal(self, w: torch.Tensor, order: List[int], eps: float = 1e-6) -> torch.Tensor:
        dup = w[order].clone()
        dup.add_(torch.randn_like(dup) * w.std() * 0.1)
        for i in range(dup.shape[0]):
            orig = w[self.duplicate_map[i]]
            proj = (torch.dot(dup[i], orig) / (torch.dot(orig, orig) + eps)) * orig
            dup[i] = dup[i] - proj
            dup[i] = dup[i] * (orig.norm() / (dup[i].norm() + eps))
        return torch.cat([w, dup], dim=0)

    def _adversarial(self, w: torch.Tensor, order: List[int], strength: float) -> torch.Tensor:
        dup = w[order].clone()
        mean = w.mean(dim=0, keepdim=True)
        direction = w - mean
        for i in range(dup.shape[0]):
            dup[i] = dup[i] - direction[self.duplicate_map[i]] * strength
        return torch.cat([w, dup], dim=0)

    @staticmethod
    def _svd_perturb(
        w: torch.Tensor, order: List[int], pv: float, pvec: float
    ) -> torch.Tensor:
        orig_dtype = w.dtype
        wf = w.float() if w.dtype in (torch.bfloat16, torch.float16) else w
        U, S, Vh = torch.linalg.svd(wf, full_matrices=False)
        S = (S + torch.randn_like(S) * S * pv).clamp(min=1e-8)
        if pvec > 0:
            U = U + torch.randn_like(U) * pvec
            U, _ = torch.linalg.qr(U)
            Vh2 = Vh + torch.randn_like(Vh) * pvec
            Vh2, _ = torch.linalg.qr(Vh2.T)
            Vh = Vh2.T
        dup = (U @ torch.diag(S) @ Vh).to(orig_dtype)
        return torch.cat([w, dup[order]], dim=0)
