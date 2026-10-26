# experiment_code

Scripts of the estimator benchmark campaigns, one folder per series; each has `.sbatch` launchers. Outputs go to `results/series_*/` and logs to `logs/` (neither is tracked). The protocol is in [`../plan.md`](../plan.md).

| series | what |
|---|---|
| `series_1` | tests of the estimator implementations against known results from the papers |
| `series_2` | small intrinsic dimension: cells over n, D, sampling and noise; generator validation and rate check |
| `series_3` | large intrinsic dimension (k = 50, 100, 300), near-flat manifolds |
| `series_4` | Guidotti bandwidth ramp on a 100-dimensional curved patch |
| `series_5` | easy-setting pre-study: every method on k = 3, D = 100 |
| `series_6` | series 5 repeated over intrinsic dimension k = 3 ... 300, D = 1000 |
| `series_7` | scaling of Guidotti's signature function with the ambient dimension |
| `series_8` | the 100-sphere of the autoencoder experiments with analytic tangent spaces |

Typical use: `run_cells.py` (via `run_cells.sbatch`) computes the cells, `aggregate.py` collects them.
