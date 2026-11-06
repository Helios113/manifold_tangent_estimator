#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_s8_aggregate.out
# usage: sbatch --dependency=afterany:<cell job id> experiment_code/series_8/aggregate.sbatch
cd "$(dirname "$0")/.."  # repo root
uv run python experiment_code/series_8/aggregate.py
