#!/bin/bash
#SBATCH --partition=cpu   # adjust to your cluster
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a_tangent_all.out
# usage: sbatch --array=0-35 diagnostics/tangent_all.sbatch      (task i takes the i-th folder of results/autoencoder/run*; all functions of tangent.py)
cd "$(dirname "$0")/.."  # repo root
runs=(results/autoencoder/run*)
echo "${runs[$SLURM_ARRAY_TASK_ID]}"
uv run python diagnostics/tangent.py "${runs[$SLURM_ARRAY_TASK_ID]}"
