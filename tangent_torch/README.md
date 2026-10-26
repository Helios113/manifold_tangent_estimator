# tangent_torch

Torch (GPU) implementations of tangent-space estimators, batched over points. Float32 unless stated.

| file | estimator |
|---|---|
| `lpca.py` | local PCA |
| `msvd.py` | multiscale-SVD local PCA |
| `lpoly.py` | local polynomial estimator of Aamari & Levrard (2019) |
| `ldle.py` | tangent spaces from the local-parameterisation step of LDLE |
| `lego.py` | LEGO, Laplacian Eigenvector Gradient Orthogonalization |
| `guidotti.py` | Guidotti's signature function |
| `common.py` | shared GPU helpers |

Imported by `tangent_est/` and `experiment_code/`.
