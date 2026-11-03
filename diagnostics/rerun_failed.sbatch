#!/bin/bash
#SBATCH --partition=turing
#SBATCH -w ngongotaha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%A_%a_rerun_failed.out
# Plan: rerun the diagnostics that failed in sweep_diagnostics.sbatch job 74485 (lpca_lego: GPU LU solve crashed, now Cholesky; tangent_methods: needed lpca_lego; fold_determinant_stages: CUDA context broken by lpca_lego;
# singular_values_over_training: now skipped without checkpoints) on the runs of the list; the other entries of <run>/eval/status.json are kept; eval/tangent/*.npz is deleted afterwards as in the sweep.
# usage: sbatch --array=0-N diagnostics/rerun_failed.sbatch <list file>      (task i takes line i + 1: a run folder)
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
run=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
uv run python diagnostics/run_diagnostics.py --delete-tangent-npz --only lpca_lego,tangent_methods,singular_values_over_training,fold_determinant_stages "$run"
