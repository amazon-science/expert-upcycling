"""Walk a Megatron-LM model and apply expert + router upcycling to all MoE layers."""

import logging
from typing import Optional

from expert_upcycling.patch import apply_patches

logger = logging.getLogger(__name__)


def perform_expert_upcycling(model, optimizer, expert_cfg=None, router_cfg=None):
    """Traverse *model*, find MoE layers, and upcycle experts + router.

    Args:
        model: The top-level model (may be wrapped in DDP / FP16 wrappers).
            Expected structure: ``model.module.decoder.layers[i].mlp.{experts, router}``.
        optimizer: The optimizer holding parameter states.
        expert_cfg: An :class:`UpcycleConfig`, :class:`PrincipledUpcycleConfig`,
            or a plain ``dict`` that will be auto-converted.
        router_cfg: A :class:`RouterUpcycleConfig` or ``dict``.
    """
    # Ensure methods are patched onto the classes
    apply_patches()

    from megatron.core.transformer.moe.experts import TEGroupedMLP, GroupedMLP
    from megatron.core.transformer.moe.router import TopKRouter

    # Unwrap common wrappers to reach the decoder
    inner = model
    for attr in ("module",):
        if hasattr(inner, attr):
            inner = getattr(inner, attr)

    if not hasattr(inner, "decoder") or not hasattr(inner.decoder, "layers"):
        logger.warning("Model structure not as expected — cannot find decoder.layers")
        return

    for i, layer in enumerate(inner.decoder.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            logger.info("Layer %d: no mlp attribute, skipping", i)
            continue

        experts = getattr(mlp, "experts", None)
        router = getattr(mlp, "router", None)

        selected_expert_indices = None

        if isinstance(experts, (TEGroupedMLP, GroupedMLP)):
            selected_expert_indices = experts.upcycle_experts(optimizer, i, expert_cfg)

        if isinstance(router, TopKRouter):
            router.upcycle_router(router_cfg, selected_expert_indices)

        if experts is None:
            logger.info("Layer %d: dense layer, skipping upcycling", i)

    logger.info("Expert upcycling complete.")
