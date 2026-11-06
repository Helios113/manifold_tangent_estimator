#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/%j_s2_validate_generators.out
cd "$(dirname "$0")/.."  # repo root
uv run python experiment_code/series_2/validate_generators.py
