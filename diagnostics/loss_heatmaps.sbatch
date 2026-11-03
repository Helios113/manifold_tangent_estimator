#!/bin/bash
#SBATCH --partition=cpu-only
#SBATCH -w ngongotaha
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j_loss_heatmaps.out
# usage: sbatch diagnostics/loss_heatmaps.sbatch <sweep folder>
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python diagnostics/loss_heatmaps.py "$@"
