#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_s8_sphere_cells.out
# usage: sbatch experiment_code/series_8/run_cells.sbatch
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_8/run_cells.py 0 1
