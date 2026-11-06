#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_s7_guidotti_D.out
# usage: sbatch --array=0-6 experiment_code/series_7/run.sbatch    (array index -> D in 10 30 100 300 1000 3000 10000)
cd "$(dirname "$0")/.."  # repo root
DS=(10 30 100 300 1000 3000 10000)
uv run python experiment_code/series_7/run.py ${DS[$SLURM_ARRAY_TASK_ID]}
