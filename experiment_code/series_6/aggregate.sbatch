#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_s6_aggregate.out
# usage: sbatch --dependency=afterany:<array job id> experiment_code/series_6/aggregate.sbatch
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python experiment_code/series_6/aggregate.py
