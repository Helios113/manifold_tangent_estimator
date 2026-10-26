#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_s2_rate_check.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_2/rate_check.py
