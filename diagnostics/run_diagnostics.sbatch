#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_run_diagnostics.out
# usage: sbatch diagnostics/run_diagnostics.sbatch <run> [<run> ...]      (writes into results/autoencoder/<run>/eval/)
cd "$(dirname "$0")/.."  # repo root
uv run python diagnostics/run_diagnostics.py "$@"
