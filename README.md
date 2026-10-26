# Autoencoders on manifolds and their tangent spaces

The main code is in [`auto_encoder/`](auto_encoder/README.md): a tied-weight autoencoder trained on a sphere embedded in a higher-dimensional space, with plots of its latent space, a hyper-parameter sweep, and an analysis of how well the tangent space is reconstructed.

```bash
uv sync                                                   # Python 3.12
sbatch auto_encoder/train.sbatch results/autoencoder/my_run    # needs my_run/config.toml, see auto_encoder/README.md
```

Training outputs (`results/`, `logs/`, checkpoints) are not tracked by git.

## Other parts

| folder / file | what it is |
|---|---|
| [`tangent_torch/`](tangent_torch/README.md) | GPU implementations of tangent-space estimators (LPCA, LDLE, LEGO, Guidotti, ...) |
| [`tangent_est/`](tangent_est/README.md) | manifold generators with analytic tangents and the estimator benchmark |
| [`experiment_code/`](experiment_code/README.md) | scripts of the benchmark campaigns, `series_1` ... `series_8` |
| [`CLUSTER.md`](CLUSTER.md) | the slurm nodes and how to submit |
| `plan.md` | protocol of the estimator benchmark |
