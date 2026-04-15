"""Configuration dataclasses and enums for expert upcycling."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


# ---------------------------------------------------------------------------
# Expert upcycling (heuristic methods)
# ---------------------------------------------------------------------------

class UpcycleMethod(Enum):
    """Heuristic expert duplication strategies."""
    COPY = "copy"
    COPY_NOISE = "copy_noise"
    DROP_UPCYCLE = "drop_upcycle"
    SHUFFLE_COLUMNS = "shuffle_columns"
    INTERPOLATE = "interpolate"
    ORTHOGONAL = "orthogonal"
    SCALED_COPY = "scaled_copy"
    SVD_PERTURB = "svd_perturb"
    SVD_MIX = "svd_mix"
    SPARSE_CODE_MIX = "sparse_code_mix"


class OptimizerStateStrategy(Enum):
    """How to handle optimizer states for upcycled experts."""
    RESET = "reset"
    COPY = "copy"
    SCALE = "scale"
    INTERPOLATE = "interpolate"


@dataclass
class UpcycleConfig:
    """Configuration for heuristic expert upcycling."""
    method: UpcycleMethod = UpcycleMethod.COPY
    optimizer_state_strategy: OptimizerStateStrategy = OptimizerStateStrategy.COPY

    # COPY_NOISE
    noise_lambda: float = 0.01

    # DROP_UPCYCLE
    drop_ratio: float = 0.5
    drop_init_method: Literal["xavier", "kaiming", "normal"] = "xavier"

    # INTERPOLATE
    interp_alpha: float = 0.5

    # SCALED_COPY
    scale_factor: float = 0.95

    # ORTHOGONAL
    orthogonal_epsilon: float = 1e-6

    # SVD_PERTURB
    svd_perturb_singular_values: float = 0.1
    svd_perturb_vectors: float = 0.05
    svd_drop_components: float = 0.0

    # SVD_MIX
    svd_mix_ratio: float = 0.3

    # SPARSE_CODE_MIX
    sparse_dict_size: int = 512
    sparse_sparsity: float = 0.1
    sparse_mix_ratio: float = 0.3
    sparse_n_iter: int = 100

    # Optimizer state scaling
    momentum_scale: float = 0.5
    variance_scale: float = 0.5


# ---------------------------------------------------------------------------
# Router upcycling
# ---------------------------------------------------------------------------

class RouterUpcycleMethod(Enum):
    """Router weight duplication strategies."""
    COPY = "copy"
    COPY_NOISE = "copy_noise"
    INTERPOLATE = "interpolate"
    BIAS_ONLY = "bias_only"
    SCALED_COPY = "scaled_copy"
    PERTURB_NEW_ONLY = "perturb_new_only"
    ORTHOGONAL = "orthogonal"
    ADVERSARIAL = "adversarial"
    TEMPERATURE_SCALED = "temperature_scaled"
    SVD_PERTURB = "svd_perturb"


@dataclass
class RouterUpcycleConfig:
    """Configuration for router upcycling."""
    method: RouterUpcycleMethod = RouterUpcycleMethod.COPY

    # COPY_NOISE
    noise_lambda: float = 0.01

    # INTERPOLATE
    interp_alpha: float = 0.5
    interp_circular: bool = True

    # BIAS_ONLY
    bias_noise_scale: float = 0.1
    bias_shift: float = 0.0
    noise_to_original: bool = False

    # SCALED_COPY
    scale_factor: float = 0.95

    # PERTURB_NEW_ONLY
    new_expert_noise: float = 0.02

    # ADVERSARIAL
    adversarial_strength: float = 0.1

    # TEMPERATURE_SCALED
    temperature_factor: float = 0.9

    # SVD_PERTURB
    svd_perturb_singular_values: float = 0.1
    svd_perturb_vectors: float = 0.05


# ---------------------------------------------------------------------------
# Utility-based (principled) expert upcycling
# ---------------------------------------------------------------------------

class UsefulnessMetric(Enum):
    """Metrics for evaluating expert importance."""
    WEIGHT_NORM = "weight_norm"            # ||w||_2
    WEIGHT_GRAD_PRODUCT = "weight_grad_product"  # ||w||_2 * ||g||_2  (saliency)
    GRADIENT_SQUARED = "gradient_squared"  # ||g||_2^2
    APPROX_FISHER = "approx_fisher"        # ||g||^2 / (||v|| + eps)


class SelectionStrategy(Enum):
    """Strategy for choosing which experts to duplicate."""
    GREEDY = "greedy"
    WEIGHTED_SAMPLING = "weighted_sampling"


class LayerSelection(Enum):
    """Which MLP sub-layers to use for usefulness evaluation."""
    FC1_ONLY = "fc1_only"
    FC2_ONLY = "fc2_only"
    BOTH_AVERAGE = "both_average"
    BOTH_PRODUCT = "both_product"
    BOTH_MAX = "both_max"
    BOTH_MIN = "both_min"


@dataclass
class PrincipledUpcycleConfig:
    """Configuration for utility-based expert upcycling."""
    usefulness_metric: UsefulnessMetric = UsefulnessMetric.WEIGHT_NORM
    selection_strategy: SelectionStrategy = SelectionStrategy.GREEDY
    layer_selection: LayerSelection = LayerSelection.FC1_ONLY
    max_duplicates_per_expert: int = 3
    fisher_epsilon: float = 1e-8
    temperature: float = 1.0
    optimizer_state_strategy: OptimizerStateStrategy = OptimizerStateStrategy.COPY
    momentum_scale: float = 0.5
    variance_scale: float = 0.5
