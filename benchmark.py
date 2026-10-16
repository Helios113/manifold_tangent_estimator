"""Benchmark LPCA vs LDLE vs LEGO tangent-space estimation on manifolds embedded in R^N.

Usage: python benchmark.py [--quick] [--out results.csv]
Every (experiment, manifold, noise, n, seed) dataset is evaluated with a grid of hyper-parameters per method;
raw results are written to CSV so `analyze.py` can report both fixed-default and best-over-grid numbers.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import argparse, itertools, time
import numpy as np, pandas as pd
from multiprocessing import Pool
from tangent_est import lpca, ldle, lego, sample_embedded, tangent_errors, MANIFOLDS
from tangent_est.graph import knn_graph

GRIDS = {
    "lpca": [dict(k=k) for k in (8, 12, 18, 25, 40, 60, 90)],
    "ldle": [dict(k=k, klv=klv) for k in (15, 25) for klv in (12, 20, 35)],
    "lego": [dict(k=k, m=m, m0=3 * m) for k in (10, 15, 25) for m in (10, 20, 40)],
}
DEFAULTS = {"lpca": dict(k=20), "ldle": dict(k=25, klv=20), "lego": dict(k=15, m=20, m0=60)}
FUNCS = {"lpca": lpca, "ldle": ldle, "lego": lego}
# make sure defaults are in the grid
for m_, dflt in DEFAULTS.items():
    if dflt not in GRIDS[m_]:
        GRIDS[m_].append(dflt)


def run_job(job):
    exp, name, noise, sigma, n, N, seed = job
    X, Q, d = sample_embedded(name, n, N, sigma, noise=noise, seed=seed)
    kmax = 90
    dist, idx = knn_graph(X, kmax)
    rows = []
    for meth, grid in GRIDS.items():
        for kw in grid:
            t = time.time()
            try:
                if meth == "lpca":
                    Qe = lpca(X, d, idx=idx, **kw)
                else:
                    kneed = max(kw["k"], kw.get("klv", 0))
                    Qe = FUNCS[meth](X, d, idx=idx[:, : kneed + 1], dist=dist[:, : kneed + 1], **kw)
                e = tangent_errors(Qe, Q)
                rec = dict(mean_max=e["max_angle"].mean(), med_max=np.median(e["max_angle"]),
                           p90_max=np.percentile(e["max_angle"], 90), mean_proj=e["proj_err"].mean())
            except Exception as ex:  # keep the sweep going, record failure
                rec = dict(mean_max=np.nan, med_max=np.nan, p90_max=np.nan, mean_proj=np.nan, err=repr(ex))
            rows.append(dict(exp=exp, manifold=name, noise=noise, sigma=sigma, n=n, N=N, seed=seed, method=meth,
                             params=str(kw), is_default=(kw == DEFAULTS[meth]), time=time.time() - t, **rec))
    return rows


def make_jobs(quick):
    seeds = range(2 if quick else 4)
    sig = [0.0, 0.01, 0.03] if quick else [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
    mf = list(MANIFOLDS)
    jobs = []
    # A: noise sweep, isotropic ambient noise, all manifolds
    jobs += [("A_noise_ambient", m, "ambient", s, 2000, 20, sd) for m in mf for s in sig for sd in seeds]
    # B: noise sweep, tubular (normal-only) noise = LEGO's model
    jobs += [("B_noise_normal", m, "normal", s, 2000, 20, sd) for m in ("sphere S2", "swiss roll", "clifford T2")
             for s in sig for sd in seeds]
    # C: sample-size sweep at moderate noise
    jobs += [("C_n_sweep", m, "ambient", 0.02, n, 20, sd) for m in ("sphere S2", "swiss roll", "clifford T2")
             for n in ((500, 1000, 2000) if quick else (500, 1000, 2000, 4000)) for sd in seeds]
    # D: ambient-dimension sweep at fixed absolute noise energy: sigma*sqrt(N) = 0.09
    jobs += [("D_ambient_dim", m, "ambient", 0.09 / np.sqrt(Nn), 2000, Nn, sd) for m in ("sphere S2", "swiss roll")
             for Nn in (5, 10, 20, 50, 100) for sd in seeds]
    # E: very high ambient dimension N=1000 (noise energy sigma*sqrt(N) in {0, 0.09, 0.18})
    jobs += [("E_N1000", m, "ambient", e / np.sqrt(1000), 2000, 1000, sd) for m in mf for e in (0.0, 0.09, 0.18)
             for sd in range(2 if quick else 3)]
    return jobs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--exps", default="A,B,C,D,E")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    jobs = [j for j in make_jobs(a.quick) if j[0][0] in a.exps.split(',')]
    jobs.sort(key=lambda j: -j[4])           # big first
    print(f"{len(jobs)} datasets", flush=True)
    t0 = time.time(); allrows = []
    with Pool(a.procs) as pool:
        for i, rows in enumerate(pool.imap_unordered(run_job, jobs), 1):
            allrows += rows
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(allrows).to_csv(a.out, index=False)
    pd.DataFrame(allrows).to_csv(a.out, index=False)
    print("done", time.time() - t0)
