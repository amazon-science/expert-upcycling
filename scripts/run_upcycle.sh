#!/bin/bash
# Example launch script for expert upcycling.
# Adjust NNODES, NPROC, and config as needed.

set -euo pipefail

NNODES=${NNODES:-1}
NPROC=${NPROC:-8}

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    -m expert_upcycling.entrypoint \
    --config-path=configs \
    --config-name=upcycle \
    "$@"
