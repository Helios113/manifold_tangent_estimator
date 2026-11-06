# auto_encoder

A tied-weight autoencoder trained on a manifold (sphere or torus, see `sphere.py`) embedded in a higher-dimensional space (`quadratic` or `fourier` embedding), with plots of the latent space, a hyper-parameter sweep and diagnostics of how well the tangent space is reconstructed.

| file | what it is |
|---|---|
| `model.py` | `DeepAutoEncoder`: activations with exact inverses, optional bottleneck RMS norm, tied/untied weights, orthogonality penalty |
| `sphere.py` | the sphere / torus with analytic tangent spaces, and the two embeddings |
| `train.py` | one training run: `python train.py <run folder>` (the folder needs a `config.toml`, see `DEFAULTS` in `train.py`) |
| `plots.py` | all figures of a finished run: `python plots.py <run folder>` |
| `sweep.py` | hyper-parameter sweep over a grid of configs, submitted as slurm job arrays |
| `*.sbatch` | slurm launchers for the above |

Training outputs (`results/`, `logs/`, checkpoints) are not tracked by git.
