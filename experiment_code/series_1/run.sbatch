#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_series_1.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run --with pytest pytest experiment_code/series_1 -s -v
uv run python experiment_code/series_1/make_plots.py
