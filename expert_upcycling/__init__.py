"""Expert Upcycling: Capacity expansion for Mixture-of-Experts models.

Implements MoE -> larger MoE growth during continued pre-training by
duplicating experts and expanding the router while preserving top-K routing.

Reference:
    Dwivedi et al., "Expert Upcycling: Shifting the Compute-Efficient
    Frontier of Mixture-of-Experts", NeurIPS 2025.
"""

__version__ = "0.1.0"

from expert_upcycling.config import (
    UpcycleConfig,
    UpcycleMethod,
    RouterUpcycleConfig,
    RouterUpcycleMethod,
    PrincipledUpcycleConfig,
    UsefulnessMetric,
    SelectionStrategy,
    LayerSelection,
    OptimizerStateStrategy,
)


def apply_patches():
    """Monkey-patch upcycling methods onto Megatron-LM MoE classes."""
    from expert_upcycling.patch import apply_patches as _apply
    _apply()


def perform_expert_upcycling(model, optimizer, expert_cfg=None, router_cfg=None):
    """Walk model layers and apply expert + router upcycling."""
    from expert_upcycling.upcycle_model import perform_expert_upcycling as _upcycle
    return _upcycle(model, optimizer, expert_cfg, router_cfg)
