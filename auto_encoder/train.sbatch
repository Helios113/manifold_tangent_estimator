#!/bin/bash
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j_autoencoder_sphere.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python auto_encoder/train.py "$@"
