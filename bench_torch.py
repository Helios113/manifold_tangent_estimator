"""GPU benchmark of LPCA / LDLE / LEGO / signature tangent estimation.

  python bench_torch.py noise --shard i --nshards S   accuracy vs noise, in-sample and on new points
  python bench_torch.py scale                         time and memory vs sample size, in-sample and on new points

New points: LPCA and the signature function evaluate at arbitrary points. LEGO and LDLE (as published) only give tangents for
the points of the cloud they are run on, so new points are appended to the cloud and the method is rerun.
"""
import argparse, sys, time, numpy as np, pandas as pd, torch
from tangent_est import sample_embedded, tangent_errors, MANIFOLDS
from tangent_torch import LPCA, LDLE, LEGO, Signature, to_t, knn_train, MemTimer
from tangent_torch.common import sync, nbytes


class Watch:
    """Wall-clock stopwatch for one stage (does not touch the GPU peak-memory counter, so it nests inside MemTimer)."""
    def __enter__(self):
        sync(); self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        sync(); self.time = time.perf_counter() - self.t0


DEFAULTS = {"lpca": dict(k=20), "ldle": dict(k=25, klv=20), "lego": dict(k=15, m=20, m0=60), "sig": dict(bw_mult=1.5, alpha=1e-3)}
GRIDS = {
    "lpca": [dict(k=k) for k in (8, 12, 18, 25, 40, 60, 90)],
    "ldle": [dict(k=k, klv=klv) for k in (15, 25) for klv in (12, 20, 35)],
    "lego": [dict(k=k, m=m, m0=3 * m) for k in (10, 15, 25) for m in (10, 20, 40)],
    "sig": [dict(bw_mult=b, alpha=a) for b in (0.7, 1.5, 3.0) for a in (1e-6, 1e-3, 1e-2, 1e-1)],
}
CLS = {"lpca": LPCA, "ldle": LDLE, "lego": LEGO, "sig": Signature}
for m_, dflt in DEFAULTS.items():
    if dflt not in GRIDS[m_]:
        GRIDS[m_].append(dflt)


def errs(Q, Qtrue):
    e = tangent_errors(Q.cpu().numpy().astype(np.float64), Qtrue)["max_angle"]
    return float(e.mean()), float(np.median(e))


def noise(shard, nshards, out):
    n, q, N = 2000, 200, 20
    jobs = [(m, s, sd) for m in MANIFOLDS for s in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05) for sd in range(3)][shard::nshards]
    rows = []
    for (name, sigma, seed) in jobs:
        Xn, Qn, d = sample_embedded(name, n + q, N, sigma, seed=seed)
        X, Y = to_t(Xn[:n]), to_t(Xn[n:])
        nn = knn_train(X, 90)
        for meth, grid in GRIDS.items():
            cache, cache2 = {}, {}
            for kw in grid:
                rec = dict(manifold=name, sigma=sigma, seed=seed, n=n, q=q, N=N, method=meth, params=str(kw),
                           is_default=(kw == DEFAULTS[meth]))
                try:
                    m = CLS[meth](d, **kw).fit(X, nn=nn, cache=cache)
                    rec["in_mean"], rec["in_med"] = errs(m.tangents_train(), Qn[:n])
                    rec["oos_mean"], rec["oos_med"] = errs(m.tangents_query(Y, cache=cache2), Qn[n:])
                except Exception as ex:
                    rec["err"] = repr(ex)[:200]; torch.cuda.empty_cache()
                rows.append(rec)
        print(name, sigma, seed, flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)


def scale(out):
    N, sigma = 20, 0.02
    warm = to_t(sample_embedded("sphere S2", 500, N, sigma, seed=0)[0])          # CUDA / cuSOLVER warm-up
    for c in CLS.values():
        m = c(2, **DEFAULTS[[k for k, v in CLS.items() if v is c][0]]).fit(warm); m.tangents_train(); m.tangents_query(warm[:50])
    rows = []
    for name in ("sphere S2", "swiss roll"):
        for n in (1000, 2000, 4000, 8000, 16000, 32000):
            Xn, Qn, d = sample_embedded(name, n + 1000, N, sigma, seed=0)
            X, Yall = to_t(Xn[:n]), to_t(Xn[n:])
            for meth, kw in DEFAULTS.items():
                rec = dict(manifold=name, n=n, N=N, sigma=sigma, method=meth)
                try:
                    torch.cuda.empty_cache()
                    with MemTimer() as total:
                        with Watch() as tk:
                            nn = knn_train(X, 90 if meth == "lpca" else 40)
                        with Watch() as tf:
                            m = CLS[meth](d, **kw).fit(X, nn=nn)
                        with Watch() as tt:
                            Q = m.tangents_train()
                    rec.update(t_total=total.time, t_knn=tk.time, t_fit=tf.time, t_tangent=tt.time, peak_extra_bytes=total.peak,
                               state_bytes=m.state_nbytes(), data_bytes=nbytes(X), output_bytes=nbytes(Q),
                               in_mean=errs(Q, Qn[:n])[0])
                    del Q, nn
                    for qn in (1, 100):                                       # new points
                        Y = Yall[:qn]
                        with MemTimer() as tq:
                            Qq = m.tangents_query(Y)
                        rec[f"oos{qn}_time"] = tq.time; rec[f"oos{qn}_peak_extra_bytes"] = tq.peak
                        rec[f"oos{qn}_mean"] = errs(Qq, Qn[n : n + qn])[0]
                        del Qq
                except Exception as ex:
                    rec["err"] = repr(ex)[:200]
                rows.append(rec); print(rec, flush=True)
                del m
                torch.cuda.empty_cache()
                pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noise", "scale"])
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.mode == "noise":
        noise(a.shard, a.nshards, a.out or f"results/torch_noise_{a.shard}.csv")
    else:
        scale(a.out or "results/torch_scale.csv")
