"""Heuristic expert weight duplication strategies.

Each method takes a source parameter tensor and returns a new tensor
for the duplicated expert.
"""

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from expert_upcycling.config import UpcycleConfig, UpcycleMethod

logger = logging.getLogger(__name__)


class ExpertUpcycler:
    """Applies a configured heuristic to produce a duplicated expert's weights."""

    def __init__(self, config: UpcycleConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def upcycle_param(
        self,
        src_param: torch.Tensor,
        expert_idx: int = 0,
        total_experts: int = 1,
        other_params: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        m = self.config.method
        if m == UpcycleMethod.COPY:
            return src_param.clone()
        if m == UpcycleMethod.COPY_NOISE:
            return self._copy_noise(src_param, self.config.noise_lambda)
        if m == UpcycleMethod.DROP_UPCYCLE:
            return self._drop_upcycle(src_param, self.config.drop_ratio, self.config.drop_init_method)
        if m == UpcycleMethod.SHUFFLE_COLUMNS:
            return self._shuffle_columns(src_param)
        if m == UpcycleMethod.INTERPOLATE:
            if other_params and len(other_params) > 0:
                next_idx = (expert_idx + 1) % len(other_params)
                return self._interpolate(src_param, other_params[next_idx], self.config.interp_alpha)
            return src_param.clone()
        if m == UpcycleMethod.ORTHOGONAL:
            return self._orthogonalize(src_param, self.config.orthogonal_epsilon)
        if m == UpcycleMethod.SCALED_COPY:
            return src_param * self.config.scale_factor
        if m == UpcycleMethod.SVD_PERTURB:
            return self._svd_perturb(
                src_param,
                self.config.svd_perturb_singular_values,
                self.config.svd_perturb_vectors,
                self.config.svd_drop_components,
            )
        if m == UpcycleMethod.SVD_MIX:
            return self._svd_mix(src_param, other_params, self.config.svd_mix_ratio)
        if m == UpcycleMethod.SPARSE_CODE_MIX:
            return self._sparse_code_mix(
                src_param, other_params,
                self.config.sparse_dict_size, self.config.sparse_sparsity,
                self.config.sparse_mix_ratio, self.config.sparse_n_iter,
            )
        raise ValueError(f"Unknown upcycle method: {m}")

    # ------------------------------------------------------------------
    # Individual strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_noise(src: torch.Tensor, noise_lambda: float) -> torch.Tensor:
        out = src.clone()
        out.add_(torch.randn_like(out) * src.std() * noise_lambda)
        return out

    @staticmethod
    def _drop_upcycle(src: torch.Tensor, drop_ratio: float, init_method: str) -> torch.Tensor:
        out = src.clone()
        num_cols = out.shape[-1]
        num_drop = int(num_cols * drop_ratio)
        if num_drop == 0:
            return out
        idx = torch.randperm(num_cols)[:num_drop]
        with torch.no_grad():
            cols = out[..., idx]
            if init_method == "xavier":
                nn.init.xavier_uniform_(cols)
            elif init_method == "kaiming":
                nn.init.kaiming_uniform_(cols)
            else:
                nn.init.normal_(cols, mean=0.0, std=src.std().item())
            out[..., idx] = cols
        return out

    @staticmethod
    def _shuffle_columns(src: torch.Tensor) -> torch.Tensor:
        return src[..., torch.randperm(src.shape[-1])].clone()

    @staticmethod
    def _interpolate(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
        return (1 - alpha) * a + alpha * b

    @staticmethod
    def _orthogonalize(src: torch.Tensor, eps: float) -> torch.Tensor:
        orig_dtype = src.dtype
        w = src.float() if src.dtype in (torch.bfloat16, torch.float16) else src.clone()
        w.add_(torch.randn_like(w) * w.std() * 0.1)
        if w.ndim == 2:
            transposed = w.shape[0] < w.shape[1]
            if transposed:
                w = w.t()
            q, _ = torch.linalg.qr(w)
            q = q * (src.float().norm() / (q.norm() + eps))
            if transposed:
                q = q.t()
            w = q
        return w.to(orig_dtype)

    @staticmethod
    def _svd_perturb(
        src: torch.Tensor, pv: float = 0.1, pvec: float = 0.05, drop: float = 0.0
    ) -> torch.Tensor:
        orig_dtype, orig_shape = src.dtype, src.shape
        w = src.reshape(src.shape[0], -1) if src.ndim > 2 else src
        if w.dtype in (torch.bfloat16, torch.float16):
            w = w.float()
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)
        S = S + torch.randn_like(S) * S * pv
        S = S.clamp(min=1e-8)
        if drop > 0:
            nd = int(len(S) * drop)
            if nd:
                S[-nd:] = torch.randn(nd, device=S.device, dtype=S.dtype) * S[0] * 0.01
        if pvec > 0:
            U = U + torch.randn_like(U) * pvec
            U, _ = torch.linalg.qr(U)
            Vh2 = Vh + torch.randn_like(Vh) * pvec
            Vh2, _ = torch.linalg.qr(Vh2.T)
            Vh = Vh2.T
        out = U @ torch.diag(S) @ Vh
        return out.to(orig_dtype).reshape(orig_shape)

    def _svd_mix(
        self, src: torch.Tensor, others: Optional[List[torch.Tensor]], ratio: float
    ) -> torch.Tensor:
        if not others:
            return self._svd_perturb(src)
        orig_dtype, orig_shape = src.dtype, src.shape
        w = src.reshape(src.shape[0], -1) if src.ndim > 2 else src
        cast = w.dtype in (torch.bfloat16, torch.float16)
        if cast:
            w = w.float()
        U_s, S_s, Vh_s = torch.linalg.svd(w, full_matrices=False)
        oi = torch.randint(0, len(others), (1,)).item()
        o = others[oi]
        o = o.reshape(o.shape[0], -1) if o.ndim > 2 else o
        if cast:
            o = o.float()
        U_o, _, Vh_o = torch.linalg.svd(o, full_matrices=False)
        n = len(S_s)
        nm = int(n * ratio)
        if nm > 0:
            idx = torch.randperm(n)[:nm]
            U_s = U_s.clone()
            Vh_s = Vh_s.clone()
            U_s[:, idx] = U_o[:, idx]
            Vh_s[idx, :] = Vh_o[idx, :]
        out = U_s @ torch.diag(S_s) @ Vh_s
        return out.to(orig_dtype).reshape(orig_shape)

    def _sparse_code_mix(
        self, src: torch.Tensor, others: Optional[List[torch.Tensor]],
        dict_size: int, sparsity: float, mix_ratio: float, n_iter: int,
    ) -> torch.Tensor:
        orig_dtype, orig_shape = src.dtype, src.shape
        w = src.reshape(src.shape[0], -1) if src.ndim > 2 else src
        cast = w.dtype in (torch.bfloat16, torch.float16)
        if cast:
            w = w.float()
        nf, ns = w.shape
        ds = min(dict_size, nf)
        D = torch.randn(nf, ds, device=w.device, dtype=w.dtype)
        D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
        C_src = self._ista(w, D, sparsity, n_iter)
        C_mixed = C_src
        if others and len(others) > 0:
            oi = torch.randint(0, len(others), (1,)).item()
            o = others[oi]
            o = o.reshape(o.shape[0], -1) if o.ndim > 2 else o
            if cast:
                o = o.float()
            C_o = self._ista(o, D, sparsity, n_iter)
            nm = int(C_src.shape[0] * mix_ratio)
            if nm > 0:
                idx = torch.randperm(C_src.shape[0])[:nm]
                C_mixed = C_src.clone()
                C_mixed[idx] = C_o[idx]
        out = D @ C_mixed
        return out.to(orig_dtype).reshape(orig_shape)

    @staticmethod
    def _ista(X: torch.Tensor, D: torch.Tensor, sparsity: float, n_iter: int) -> torch.Tensor:
        ds = D.shape[1]
        C = torch.zeros(ds, X.shape[1], device=X.device, dtype=X.dtype)
        L = torch.norm(D.T @ D, p=2).item()
        step = 1.0 / (L + 1e-8)
        lam = sparsity * X.norm().item() / X.shape[1]
        for _ in range(n_iter):
            C = C - step * (D.T @ (D @ C - X))
            thr = step * lam
            C = C.sign() * (C.abs() - thr).clamp(min=0)
        return C
