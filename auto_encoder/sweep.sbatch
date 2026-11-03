#!/bin/bash
#SBATCH --output=logs/%j_autoencoder_sweep.out
# one task of a sweep array: $1 is a text file with one line per task, the task takes line SLURM_ARRAY_TASK_ID + 1: one or more run folders, all trained at the same time on the task's GPU (output lines are prefixed by the run name)
cd /nfs-share/pa511/code_bases/manifold_tangent_estimator
for run in $(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1"); do
    (uv run python auto_encoder/train.py "$run" 2>&1 | sed -u "s#^#[$(basename "$run")] #") &
done
wait
