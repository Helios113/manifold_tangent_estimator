"""Intrinsic-dimension estimation benchmark: LPCA vs LDLE-spectrum vs LEGO.

Per point, d_hat = smallest d' whose top-d' squared singular values explain >= f_var of the total (the rule
in LEGO's Algorithm 1; applied identically to all three spectra), plus a parameter-free 'largest ratio' rule.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import time, sys, numpy as np, pandas as pd
from multiprocessing import Pool
from tangent_est import lpca, lego, ldle_spectrum, sample_embedded, MANIFOLDS
from tangent_est.graph import knn_graph

CFG = dict(lpca=dict(k=20), ldle=dict(k=25, klv=20), lego=dict(k=15, m=20, m0=60))
FVARS = (0.9, 0.95, 0.99)


def dhat_fvar(S, f):
    e = np.cumsum(S ** 2, 1) / np.sum(S ** 2, 1, keepdims=True)
    return (e >= f).argmax(1) + 1


def dhat_ratio(S, dmax=6):
    S = S[:, : dmax + 1] + 1e-300
    return (S[:, :-1] / S[:, 1:]).argmax(1) + 1


def job(a):
    name, sigma, N, n, seed = a
    X, Q, d = sample_embedded(name, n, N, sigma, noise="ambient", seed=seed)
    dist, idx = knn_graph(X, 40)
    out = []
    for meth in ("lpca", "ldle", "lego"):
        c = CFG[meth]; t = time.time()
        if meth == "lpca":
            _, S = lpca(X, d, idx=idx, return_sv=True, **c)
        elif meth == "ldle":
            S = ldle_spectrum(X, idx=idx[:, : 26], dist=dist[:, : 26], **c)
        else:
            _, info = lego(X, d, idx=idx[:, : c["k"] + 1], dist=dist[:, : c["k"] + 1], return_info=True, **c)
            S = info["sv"]
        dt = time.time() - t
        for rule, dh in [(f"fvar={f}", dhat_fvar(S, f)) for f in FVARS] + [("ratio", dhat_ratio(S))]:
            out.append(dict(manifold=name, d=d, sigma=sigma, N=N, n=n, seed=seed, method=meth, rule=rule,
                            acc=(dh == d).mean(), mean_dhat=dh.mean(), time=dt))
    return out


if __name__ == "__main__":
    jobs = [(m, s, 20, 2000, sd) for m in MANIFOLDS for s in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05) for sd in range(4)]
    jobs += [(m, 0.09 / np.sqrt(1000), 1000, 2000, sd) for m in MANIFOLDS for sd in range(2)]
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        rows = sum(p.map(job, jobs, chunksize=1), [])
    pd.DataFrame(rows).to_csv("results/dim.csv", index=False); print("done")
