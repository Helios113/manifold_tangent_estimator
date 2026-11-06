#!/bin/bash
#SBATCH --partition=cpu   # adjust to your cluster
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_sphere.out
cd "$(dirname "$0")/.."  # repo root
uv run python auto_encoder/sphere.py
