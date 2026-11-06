#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%j_cpu_test.out
# Plan: do the diagnostics need a GPU? Run run_diagnostics.py with no GPU (CUDA_VISIBLE_DEVICES empty, no --gres) on one large and one small run on each node, in the normal partitions.
# Each node works on its own copy of the two runs (results/sweep_new/cpu_test/<node>/), so nothing collides with the real sweep.
# Measured: seconds of every diagnostic (<run>/eval/status.json), job elapsed time and peak memory (sacct MaxRSS) -> collected by diagnostics/cpu_test.py.
# usage: sbatch --partition=<p> -w <node> diagnostics/cpu_test.sbatch
cd "$(dirname "$0")/.."  # repo root
export CUDA_VISIBLE_DEVICES=""
node=$(hostname -s); dest=results/sweep_new/cpu_test/$node; rm -rf "$dest"; mkdir -p "$dest"
for name in intrinsic_dim30_ambient_dim1000_embedded_extra1_layers3_noise0.0_radius_head_onFalse intrinsic_dim8_ambient_dim50_embedded_extra1_layers7_noise0.1_radius_head_onTrue; do
  mkdir "$dest/$name"; find results/sweep_new/runs/$name -maxdepth 1 -type f -exec cp {} "$dest/$name/" \;
  uv run python diagnostics/run_diagnostics.py --delete-tangent-npz "$PWD/$dest/$name"
done
