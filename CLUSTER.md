# Cluster

Submit with `sbatch`, do not wait on the job. Use a free node; if a job fits on ngongotaha, run it there first.

| node | partition | GPUs | GPU memory | CPUs | RAM | notes |
|---|---|---|---|---|---|---|
| ngongotaha (ngog) | `turing` (default) | 7 × RTX 2080 Ti | 11 GB each | 56 | 258 GB | run here first if the job fits |
| mauao | `ampere` | 8 × A40 | 48 GB each | 88 | 774 GB | next choice for larger jobs |
| ruapehu | `hopper` | 2 × H100 + 2 × H100 (single/PCIe) | 80 GB each | 192 | 774 GB | largest jobs |
| gpu-vm | `priority`, `interactive`, `cpu-only` | 2 × V100 | 32 GB each | 48 | 193 GB | |

`sbatch --partition=<partition> --gres=gpu:1 <script>`. Check which node is free with `sinfo -N`.
