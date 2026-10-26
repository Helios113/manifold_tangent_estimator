# Autoencoder on the sphere

A tied-weight autoencoder is trained on a k-sphere embedded in R^D by f(p) = Q [p ; p₁² ; 0 …], then its latent space and tangent spaces are analysed.
Python 3.12, dependencies with [uv](https://docs.astral.sh/uv/): `uv sync`. Run everything from the repo root.

## Train one run

A run is a folder holding `config.toml`; every result is written into that folder. Keys left out take their defaults (`DEFAULTS` in `train.py`).

```toml
intrinsic_dim = 2      # k of the sphere
embedded_dim = 3       # latent dimension
hidden_dims = [6]
optimizer = "adam"    # or "sgd" (plain SGD: no momentum)
lr = 0.0002
orth_lambda = 0.0      # weight of the orthogonality penalty, 0 = off
plane_lambda = 0.0     # weight of the distance of the latent codes from a plane, 0 = off
plane_normal = [0.0, 0.2, 1.0]   # normal of the plane in the latent space (tilt); empty = the last latent axis
plane_offset = 0.0     # distance of the plane from the origin
total_samples = 5000000
```

```bash
mkdir -p results/autoencoder/my_run && $EDITOR results/autoencoder/my_run/config.toml
sbatch auto_encoder/train.sbatch results/autoencoder/my_run        # logs/<jobid>_autoencoder_sphere.out
```

Outputs: `history.csv`, `metrics.json`, `best.pt`, `checkpoints/`, `loss.png`, `orth_{euclidian,spectral}.png`, `latent.png`, `latent_3d.html` (open in a browser), `gram*.png`.
A folder that already has a `history.csv` is not overwritten.

## Plots of a finished run

```bash
uv run python auto_encoder/plots.py results/autoencoder/my_run [plot_loss plot_orth plot_latent plot_latent_3d plot_gram plot_gram_composed]
```

## Tangent spaces

`tangent.py` works on a trained run (it needs `best.pt` and `config.toml`). It compares five tangent spaces at 2000 fresh points:

| name | what |
|---|---|
| true input | analytic tangent space of the embedded sphere |
| true latent | the true tangent pushed through the encoder |
| true reconstructed | the true tangent pushed through the autoencoder |
| estimated latent | local PCA of the 30 nearest encoded points (`N_NEIGHBOURS` in `tangent.py`) |
| estimated reconstructed | the estimated latent tangent pushed through the decoder |

```bash
uv run python auto_encoder/tangent.py results/autoencoder/my_run                    # all three steps, in order
uv run python auto_encoder/tangent.py results/autoencoder/my_run compare            # or only the named steps
```

| step | writes | needs |
|---|---|---|
| `pushforward` | `tangent.npz`: the three true tangents | `best.pt` |
| `sample_latent_tangent` | `tangent_sampled.npz`: the two estimated tangents | `best.pt`, `tangent.npz` |
| `compare` | `tangent_compare.png` (histograms of the largest principal angle, in degrees, for four comparisons), `tangent_compare.json` (summary, and the singular values of the pushforwards), `tangent_compare.npz` (per-point values) | `tangent.npz`, `tangent_sampled.npz` |

An angle of 0 means two spaces are the same. A pushforward with a very small second singular value is almost degenerate, so its angles are poorly conditioned.
On the cluster: `sbatch --partition=cpu-only -w ngongotaha --output=logs/%j_tangent.out --wrap="cd $PWD && uv run python auto_encoder/tangent.py results/autoencoder/my_run"`.

## Hyper-parameter sweep

A sweep folder holds `sweep.toml`; one slurm job is submitted per grid point.

```toml
sbatch_args = ["--time=04:00:00"]   # optional, added to every sbatch call
[base]
total_samples = 5000000
[grid]
embedded_dim = [2, 3, 4]
hidden_dims = [[6], [8, 6], [10, 8, 6]]   # network depth: hidden widths per run, i.e. 2, 3 and 4 weight matrices
lr = [1e-3, 2e-4]
orth_lambda = [0.0, 0.1, 1.0]
```

```bash
uv run python auto_encoder/sweep.py results/autoencoder_sweeps/my_sweep submit      # runs/<name>/, jobs.csv
uv run python auto_encoder/sweep.py results/autoencoder_sweeps/my_sweep summarise   # summary.csv, summary.png
```

## Cluster

`train.sbatch` and `sphere.sbatch` hard-code the repo path and a node; edit the `cd` line and `-w`/`--partition`, or override on the command line (`sbatch -w <node> --partition=<p> …`). See [`../CLUSTER.md`](../CLUSTER.md) for the nodes.
