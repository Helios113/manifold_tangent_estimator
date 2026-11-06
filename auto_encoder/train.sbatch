#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_autoencoder_sphere.out
cd "$(dirname "$0")/.."  # repo root
uv run python auto_encoder/train.py "$@"
