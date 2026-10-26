#!/bin/bash
#SBATCH --partition=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%j_s2_cells_shard%a.out
# usage: sbatch --array=0-3 experiment_code/series_2/run_cells.sbatch   (shard = array index, number of shards = SLURM_ARRAY_TASK_COUNT)
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_2/run_cells.py ${SLURM_ARRAY_TASK_ID:-0} ${SLURM_ARRAY_TASK_COUNT:-1} ${ONLY:-}
