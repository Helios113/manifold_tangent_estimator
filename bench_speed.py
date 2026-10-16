"""Clean (sequential, single-thread) wall-clock timing of each method vs n and vs ambient dim N."""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import time, numpy as np, pandas as pd
from tangent_est import lpca, ldle, lego, sample_embedded, tangent_errors
from tangent_est.graph import knn_graph

CFG = dict(lpca=dict(k=20), ldle=dict(k=25, klv=20), lego=dict(k=15, m=20, m0=60))
rows = []
for name in ("sphere S2", "swiss roll"):
    for n, N in [(n, 20) for n in (500, 1000, 2000, 4000, 8000, 16000)] + [(2000, N) for N in (50, 200, 1000)]:
        X, Q, d = sample_embedded(name, n, N, 0.02 if N == 20 else 0.09 / np.sqrt(N), seed=0)
        t = time.time(); dist, idx = knn_graph(X, 40); tk = time.time() - t
        for meth, f in (("lpca", lpca), ("ldle", ldle), ("lego", lego)):
            c = CFG[meth]; kn = max(c["k"], c.get("klv", 0))
            t = time.time(); Qe = f(X, d, idx=idx[:, : kn + 1], dist=dist[:, : kn + 1], **c); dt = time.time() - t
            e = tangent_errors(Qe, Q)
            rows.append(dict(manifold=name, n=n, N=N, method=meth, time=dt, knn_time=tk, mean_max=e['max_angle'].mean(), med_max=np.median(e['max_angle']), mean_proj=e['proj_err'].mean())); print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("results/speed.csv", index=False)
