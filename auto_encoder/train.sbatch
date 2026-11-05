#!/bin/bash
#SBATCH --partition=hopper
#SBATCH -w ruapehu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_autoencoder_sphere.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python auto_encoder/train.py "$@"
