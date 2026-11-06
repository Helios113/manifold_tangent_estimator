#!/bin/bash
#SBATCH --partition=cpu   # adjust to your cluster
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_tangent_compare.out
# usage: sbatch diagnostics/tangent.sbatch <run folder> [function ...]
cd "$(dirname "$0")/.."  # repo root
uv run python diagnostics/tangent.py "$@"
