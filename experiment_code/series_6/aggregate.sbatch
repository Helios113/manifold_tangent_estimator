#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_s6_aggregate.out
# usage: sbatch --dependency=afterany:<array job id> experiment_code/series_6/aggregate.sbatch
cd "$(dirname "$0")/.."  # repo root
uv run python experiment_code/series_6/aggregate.py
