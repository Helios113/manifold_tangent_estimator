#!/bin/bash
#SBATCH --partition=cpu-only
#SBATCH -w ngongotaha
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_tangent_compare.out
# usage: sbatch diagnostics/tangent.sbatch <run folder> [function ...]
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python diagnostics/tangent.py "$@"
