#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_series_1.out
cd "$(dirname "$0")/.."  # repo root
uv run --with pytest pytest experiment_code/series_1 -s -v
uv run python experiment_code/series_1/make_plots.py
