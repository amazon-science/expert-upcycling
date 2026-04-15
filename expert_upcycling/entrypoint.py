"""Standalone NeMo 2.x entrypoint for expert upcycling.

Loads a trained MoE checkpoint, doubles the expert count via upcycling,
saves the expanded checkpoint, and exits.

Usage::

    torchrun --nproc_per_node=<N> -m expert_upcycling.entrypoint \\
        --config-path=configs --config-name=upcycle
"""

import logging
import re
import sys
from copy import deepcopy

import numpy as np
import torch.multiprocessing as mp
try:
    import lightning.pytorch as L
    from lightning.pytorch.callbacks.callback import Callback
except ImportError:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks.callback import Callback
from omegaconf import OmegaConf

from nemo import lightning as nl
from nemo.collections import llm
from nemo.core.config import hydra_runner

from expert_upcycling.upcycle_model import perform_expert_upcycling

logger = logging.getLogger(__name__)


# ======================================================================
# Upcycling callback
# ======================================================================

class UpcycleCallback(Callback):
    """PyTorch Lightning callback that upcycles experts on the first optimizer step."""

    def __init__(self, cfg: dict):
        self._output_ckpt_location = cfg.get("output_ckpt_location", "")
        self._expert_multiplier = float(cfg.get("expert_multiplier", 2))
        self._expert_upcycle_strategy = cfg.get("expert_upcycle_strategy")
        self._router_upcycle_strategy = cfg.get("router_upcycle_strategy")

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        logger.info("---------- Starting Expert Upcycling ----------")
        if not trainer.optimizers:
            raise RuntimeError("No optimizers found in trainer.")

        n_iters = int(np.log2(self._expert_multiplier))
        for it in range(n_iters):
            logger.info("Upcycling iteration %d / %d", it + 1, n_iters)
            # Unwrap to reach the Megatron model
            inner = pl_module
            for attr in ("module", "module"):
                if hasattr(inner, attr):
                    inner = getattr(inner, attr)
            perform_expert_upcycling(
                inner, optimizer,
                self._expert_upcycle_strategy,
                self._router_upcycle_strategy,
            )

        # Determine output path
        output_path = self._output_ckpt_location
        if not output_path:
            ckpts = sorted(
                [
                    d for d in trainer.checkpoint_callback.dirpath.glob("*")
                    if re.search(r"step=(\d+)", d.name)
                ],
                key=lambda p: (
                    float(re.search(r"step=(\d+)", p.name).group(1)),
                    float(m.group(1)) if (m := re.search(r".0-v(\d+)", p.name)) else 0,
                ),
                reverse=True,
            )
            if ckpts:
                output_path = str(ckpts[0]) + "-upcycled"
            else:
                output_path = "upcycled_checkpoint"

        logger.info("Saving upcycled checkpoint to %s", output_path)
        trainer.save_checkpoint(output_path)
        logger.info("---------- Upcycling complete. Exiting. ----------")
        sys.exit(0)


# ======================================================================
# Helpers
# ======================================================================

def nemo1_no_weight_decay_cond(name, param):
    """NeMo 1.x backward-compat: skip weight decay on biases."""
    return name.endswith(".bias")


# ======================================================================
# Main
# ======================================================================

@hydra_runner(config_path="configs", config_name="upcycle")
def main(cfg) -> None:
    mp.set_start_method("fork", force=True)

    cfg["strategy"]["ckpt_load_optimizer"] = True
    cfg["strategy"]["ckpt_save_optimizer"] = False

    trainer = nl.Trainer(
        **cfg["trainer"],
        strategy=cfg["strategy"],
        plugins=nl.MegatronMixedPrecision(**cfg["mixed_precision"]),
        callbacks=[UpcycleCallback(cfg["upcycle"])],
    )

    model = llm.GPTModel(config=cfg["model"], tokenizer=cfg["tokenizer"])

    optimizer = nl.MegatronOptimizerModule(
        config=cfg["optim"],
        lr_scheduler=cfg["lr_scheduler"],
        no_weight_decay_cond=(
            nemo1_no_weight_decay_cond
            if cfg.get("use_nemo1_no_weight_decay_cond", True)
            else None
        ),
    )

    data_module = llm.PreTrainingDataModule(**cfg["data"], tokenizer=cfg["tokenizer"])

    llm.pretrain(
        model=model,
        data=data_module,
        trainer=trainer,
        log=cfg.get("logger"),
        resume=cfg.get("resume"),
        optim=optimizer,
    )


if __name__ == "__main__":
    main()
