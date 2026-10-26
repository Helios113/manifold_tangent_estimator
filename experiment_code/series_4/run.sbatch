#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_series_4.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_4/bandwidth_ramp.py compute
