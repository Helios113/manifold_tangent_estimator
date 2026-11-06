#!/bin/bash
#SBATCH --partition=gpu   # adjust to your cluster
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%A_%a_sweep_diagnostics.out
# Plan: all diagnostics of run_diagnostics.py (tangent, reconstruction_error, lpca_lego, tangent_methods, activation_distributions, singular_values_over_training, fold_determinant_stages) on every run of results/sweep_new;
# data and plots go to <run>/eval/; eval/tangent/*.npz is deleted once tangent, reconstruction_error and tangent_methods are done; <run>/eval/status.json says what ran.
# usage: sbatch --array=0-N diagnostics/sweep_diagnostics.sbatch <list file>      (task i takes line i + 1: a run folder)
cd "$(dirname "$0")/.."  # repo root
run=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
uv run python diagnostics/run_diagnostics.py --delete-tangent-npz "$run"
