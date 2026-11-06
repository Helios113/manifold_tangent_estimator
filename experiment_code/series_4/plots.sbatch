#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/%j_series_4_plots.out
cd "$(dirname "$0")/.."  # repo root
uv run python experiment_code/series_4/bandwidth_ramp.py plot
