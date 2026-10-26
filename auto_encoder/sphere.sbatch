#!/bin/bash
#SBATCH --partition=cpu-only
#SBATCH -w ngongotaha
#SBATCH --time=00:10:00
#SBATCH --output=logs/%j_sphere.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python auto_encoder/sphere.py
