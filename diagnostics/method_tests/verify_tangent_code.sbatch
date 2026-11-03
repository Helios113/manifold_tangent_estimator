#!/bin/bash
#SBATCH --partition=turing
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=logs/%j_verify_tangent_code.out
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
uv run python diagnostics/method_tests/verify_tangent_code.py
