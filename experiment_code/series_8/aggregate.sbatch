#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_s8_aggregate.out
# usage: sbatch --dependency=afterany:<cell job id> experiment_code/series_8/aggregate.sbatch
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_8/aggregate.py
