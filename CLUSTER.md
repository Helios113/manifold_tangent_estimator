# Cluster

Submit with `sbatch`, do not wait on the job. See [SLURM.md](SLURM.md) for the everyday commands.

The `.sbatch` scripts in this repo use generic partition names (`gpu`, `cpu`) — adjust them to your cluster, e.g.:

```bash
sbatch --partition=<partition> --gres=gpu:1 <script>
```

Check which nodes are free with `sinfo -N`, and pin a job to a node with `#SBATCH -w <node>` if you need to.
