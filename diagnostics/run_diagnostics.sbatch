#!/bin/bash
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_run_diagnostics.out
# usage: sbatch diagnostics/run_diagnostics.sbatch <run> [<run> ...]      (writes into results/autoencoder/<run>/eval/)
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python diagnostics/run_diagnostics.py "$@"
