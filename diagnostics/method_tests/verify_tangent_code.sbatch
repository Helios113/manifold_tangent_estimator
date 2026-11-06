#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_verify_tangent_code.out
cd "$(dirname "$0")/.."  # repo root
uv run python diagnostics/method_tests/verify_tangent_code.py
