#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j_s2_rate_check.out
cd "$(dirname "$0")/.."  # repo root
uv run python experiment_code/series_2/rate_check.py
