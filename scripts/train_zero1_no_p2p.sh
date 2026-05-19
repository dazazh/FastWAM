#!/usr/bin/env bash
# Wrapper around train_zero1.sh with NCCL P2P disabled.
# Use this when encountering "Invalid access of peer GPU memory over nvlink" errors.
# Performance impact: step time ~5-10% slower, loss convergence unaffected.
set -euo pipefail

export NCCL_P2P_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_zero1.sh" "$@"
