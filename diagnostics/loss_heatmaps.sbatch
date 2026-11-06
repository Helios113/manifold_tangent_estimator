#!/bin/bash
#SBATCH --partition=cpu   # adjust to your cluster
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%j_loss_heatmaps.out
# usage: sbatch diagnostics/loss_heatmaps.sbatch <sweep folder>
cd "$(dirname "$0")/.."  # repo root
uv run python diagnostics/loss_heatmaps.py "$@"
