#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_s6_cells_shard%a.out
# usage: sbatch --array=0-19 experiment_code/series_6/run_cells.sbatch
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_6/run_cells.py ${SLURM_ARRAY_TASK_ID:-0} ${SLURM_ARRAY_TASK_COUNT:-1} ${ONLY:-}
