#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_s2_validate_generators.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_2/validate_generators.py
