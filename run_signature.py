"""Benchmark Guidotti's signature-function tangent estimator on the same datasets as Exp A (noise sweep, N=20, n=2000).
Output: results/results_sig.csv with the same columns as benchmark.py (method = sig_shape / sig_perturb)."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import time, itertools, numpy as np, pandas as pd
from multiprocessing import Pool
from tangent_est import SignatureFunction, signature_tangent, sample_embedded, tangent_errors, MANIFOLDS

GRID = [dict(bw_mult=b, alpha=a) for b in (0.7, 1.5, 3.0) for a in (1e-6, 1e-3, 1e-2, 1e-1)]
DEFAULT = dict(bw_mult=1.5, alpha=1e-3)


def job(a):
    name, sigma, n, N, seed = a
    X, Q, d = sample_embedded(name, n, N, sigma, noise="ambient", seed=seed)
    rows = []
    for g in GRID:
        t0 = time.time(); sig = SignatureFunction(X, k_bw=10, **g); tfit = time.time() - t0
        for method in ("shape", "perturb"):
            t = time.time(); Qe = signature_tangent(X, d, method=method, sig=sig); dt = time.time() - t + tfit
            e = tangent_errors(Qe, Q)
            rows.append(dict(exp="A_noise_ambient", manifold=name, noise="ambient", sigma=sigma, n=n, N=N, seed=seed,
                             method="sig_" + method, params=str(g), is_default=(g == DEFAULT), time=dt,
                             mean_max=e["max_angle"].mean(), med_max=np.median(e["max_angle"]),
                             p90_max=np.percentile(e["max_angle"], 90), mean_proj=e["proj_err"].mean()))
    return rows


if __name__ == "__main__":
    jobs = [(m, s, 2000, 20, sd) for m in MANIFOLDS for s in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05) for sd in range(4)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    pd.DataFrame(rows).to_csv("results/results_sig.csv", index=False); print("done")
