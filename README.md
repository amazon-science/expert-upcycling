# Expert Upcycling

**Capacity expansion for Mixture-of-Experts models during continued pre-training.**

> Dwivedi et al., *"Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts"* (preprint).

Scaling laws show that MoE quality improves predictably with total expert count at fixed active computation, but training large MoEs from scratch is expensive — memory, gradients, and all-to-all communication all scale with total parameters. Expert upcycling sidesteps this by starting training with a smaller E-expert model and expanding to mE experts mid-training via the upcycling operator:

1. **Expert replication** — each expert is duplicated (high-utility experts receive more copies via gradient-based importance scores).
2. **Router extension** — router weights are copied to new slots with small bias perturbations to seed routing diversity.
3. **Continued pre-training (CPT)** — stochastic gradient diversity and loss-free load balancing break symmetry among duplicates, driving specialization.

Top-K routing is held fixed throughout, so per-token inference cost is unchanged.

![Expert Upcycling](assets/figure_optE_hires.png)
*Figure 1: Overview of the expert upcycling procedure.*

**Key results** on a 7B→13B total parameter (1B active) interleaved MoE, pre-trained on 380B tokens:
- The upcycled model (32→64 experts) matches the fixed-size 64-expert baseline across 11 downstream benchmarks (56.4 vs. 56.7 avg accuracy) and validation loss (1.263 vs. 1.267).
- Training cost is reduced by **~32% of GPU hours** (27,888 vs. 41,328 hours). When a pre-trained checkpoint already exists (e.g., from a prior training run or a public release), the pre-training cost is already paid and only the CPT phase is needed, bringing savings to **~67%**.
- Results generalize to full MoE architectures (256→512 experts, TopK=8) with 93–95% gap closure across scales from 154M to 1B total parameters.

![Results](assets/barplot.JPG)
*Figure 2: GPU hours, validation loss, and downstream accuracy for the 7B→13B upcycled model vs. baselines.*

## Installation

### Recommended: NeMo 2.x container

Start from the official NeMo container — PyTorch, Megatron-LM, Transformer Engine, NeMo, Lightning, and omegaconf are all pre-installed.

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/expert-upcycling:/workspace/expert-upcycling \
    -it nvcr.io/nvidia/nemo:24.09 bash

# Inside the container:
cd /workspace/expert-upcycling
pip install -e .
pip install dacite
```

> **Do not** use `pip install -e ".[nemo]"` inside the container — it would conflict with the container's pre-installed NeMo.

### From scratch (no NeMo container)

Install dependencies manually, then install the package with the relevant extras:

```bash
# Core only (torch + numpy):
pip install -e .
pip install dacite

# With Megatron-LM integration:
pip install -e ".[megatron]"

# Full NeMo entrypoint (installs NeMo, Lightning, omegaconf):
pip install -e ".[nemo]"
```

## Quick Start

### Option A: NeMo entrypoint (recommended)

Edit `configs/upcycle.yaml` to set your model dimensions, then run from the repo root:

```bash
# Single GPU
cd /workspace/expert-upcycling
python -m expert_upcycling.entrypoint \
    --config-path=configs --config-name=upcycle \
    resume.restore_config.path=/path/to/base/checkpoint

# Multi-GPU (e.g. 8 GPUs with tensor parallelism)
torchrun --nproc_per_node=8 -m expert_upcycling.entrypoint \
    --config-path=configs --config-name=upcycle \
    resume.restore_config.path=/path/to/base/checkpoint \
    strategy.tensor_model_parallel_size=8
```

The callback fires on the first optimizer step, doubles the expert count, saves the upcycled checkpoint, and exits. The output path defaults to `<input_checkpoint>-upcycled`.

### Option B: Patch existing training script

```python
import expert_upcycling
expert_upcycling.apply_patches()

# Now TEGroupedMLP has .upcycle_experts() and TopKRouter has .upcycle_router()
# Call them during training at the desired transition point.
# Note: model is typically wrapped — unwrap to reach the decoder:
inner = model
for attr in ("module", "module"):
    if hasattr(inner, attr):
        inner = getattr(inner, attr)

for i, layer in enumerate(inner.decoder.layers):
    if hasattr(layer.mlp, 'experts'):
        selected = layer.mlp.experts.upcycle_experts(optimizer, i, expert_cfg)
    if hasattr(layer.mlp, 'router'):
        layer.mlp.router.upcycle_router(router_cfg, selected)
```

### Option C: Use the model-level API

```python
from expert_upcycling import perform_expert_upcycling

perform_expert_upcycling(
    model, optimizer,
    expert_cfg={"usefulness_metric": "gradient_norm", "selection_strategy": "greedy"},
    router_cfg={"method": "bias_only", "bias_noise_scale": 0.01},
)
```

## Upcycling Strategies

### Expert duplication

| Strategy | Description |
|---|---|
| **Utility-based** (recommended) | Duplicate high-importance experts using gradient-based scores (weight norm, saliency, gradient squared, approx Fisher) |
| `copy` | Exact duplication (baseline) |
| `copy_noise` | Duplication + Gaussian noise |
| `drop_upcycle` | Re-initialize a fraction of columns |
| `svd_perturb` | SVD decomposition + perturbation |
| + 6 more | See `expert_upcycling.config.UpcycleMethod` |

### Router expansion

| Strategy | Description |
|---|---|
| `bias_only` (recommended) | Keep weights identical, add noise to bias |
| `copy` | Exact duplication |
| `copy_noise` | Duplication + noise |
| + 7 more | See `expert_upcycling.config.RouterUpcycleMethod` |

## Architecture

This package treats Megatron-LM and NeMo as **third-party dependencies** — no fork required. Upcycling methods are injected at runtime via monkey-patching:

```
expert-upcycling/          # pip install -e .
├── expert_upcycling/
│   ├── config.py          # All enums + dataclasses (no deps)
│   ├── expert_upcycler.py # Heuristic strategies (torch only)
│   ├── expert_selector.py # Utility-based selection (torch + numpy)
│   ├── router_upcycler.py # Router strategies (torch only)
│   ├── optimizer_utils.py # Optimizer state handling (torch only)
│   ├── patch.py           # Monkey-patches onto Megatron-LM classes
│   ├── upcycle_model.py   # Model traversal
│   └── entrypoint.py      # NeMo launch script
├── configs/
│   └── upcycle.yaml       # Example config
└── scripts/
    └── run_upcycle.sh     # Example launch script
```

## Running Tests

```bash
# CPU tests (no GPU, no Megatron install required)
python tests/test_comprehensive.py          # 91 tests: all methods, all strategies
pytest tests/test_integration.py -v        # 7 end-to-end integration tests

# GPU test (requires NeMo container + GPU)
python tests/test_entrypoint_gpu.py        # real TEGroupedMLP + TopKRouter, 32->64 experts
```

## Citation

```bibtex
@article{dwivedi2025expertupcycling,
  title={Expert Upcycling: Shifting the Compute-Efficient Frontier of Mixture-of-Experts},
  author={Dwivedi, Chaitanya and Gupta, Himanshu and Varshney, Neeraj and Jayarao, Pratik and Yin, Bing and Chilimbi, Trishul and Huang, Binxuan},
  year={2026}
}
```

## License

CC-BY-NC-4.0

This code is being released solely for academic and scientific reproducibility purposes, in support of the methods and findings described in the associated publication. Pull requests are not being accepted in order to maintain the code exactly as it was used in the paper.