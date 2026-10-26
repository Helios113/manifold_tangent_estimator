"""Plan.md step 6, 'rate check for AL (main validation of the implementation)': noise-free data, families A and B, k in {1, 2, 5}, D = 300, uniform sampling,
n in {1e3, 2e3, 4e3, 8e3, 1.6e4, 3.2e4} (1.5 decades).  For every n and order m in {2, 3, 4}: the ball radius h (target median count, grid of al_counts) is chosen on a
validation draw (lowest max-over-points angle); the error reported is on 3 independent test draws: max over the evaluation points of the largest principal angle
and, for m >= 3, the curvature error ||II o pi_T - T2 o pi_hat||_op.  Slopes of log(error) vs log(n) are fitted by aggregate.py next to the predicted -(m-1)/k and -(m-2)/k.
Results: results/series_2/rate_check.csv"""
import sys, json, pathlib, numpy as np, pandas as pd, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tangent_est.bench import make_draw, al_counts, principal, curvature_error, NEV, F64, gc_cuda
from tangent_torch import LocalPoly
from tangent_torch.common import DEV, knn_train
from cells import cell

SIZES = (1000, 2000, 4000, 8000, 16000, 32000)


def one(c, seed, m, counts_by_m):
    M, ds, meta = make_draw(c, seed)
    n, k = c["n"], c["k"]
    X = torch.as_tensor(ds["X"], dtype=F64, device=DEV); th = ds["theta"]
    ev = np.random.default_rng(seed).permutation(n)[:NEV]; ev_t = torch.as_tensor(ev, device=DEV)
    Etrue = M.tangent(th[ev])
    cs = counts_by_m[m]
    kmax = int(min(n - 1, max(2 * x for x in cs)))
    dist, idx = knn_train(X, kmax)
    out = []
    for cnt in cs:
        h = float(dist[:, cnt - 1].median().item()); cap = int(min(kmax, 2 * cnt))
        est = LocalPoly(k, k=cap, order=m, radius=h).fit(X, nn=(dist, idx))
        r = dict(count=cnt, h=h)
        if m == 2:
            T = est.tangents_at(ev_t).cpu().numpy()
        else:
            T, T2 = est.tangents_at(ev_t, want_T2=True); T, T2 = T.cpu().numpy(), T2.cpu().numpy()
            E_, II = M.second_form(th[ev])
            ce = curvature_error(E_, II, T, T2); r.update(curv_max=float(ce.max()), curv_med=float(np.median(ce)))
        ang = principal(T, Etrue).max(1)
        r.update(max_ang=float(ang.max()), med_ang=float(np.median(ang)))
        out.append(r); del est; gc_cuda()
    return out


if __name__ == "__main__":
    rows = []
    for fam in ("A", "B"):
        for k in (1, 2, 5):
            for n in SIZES:
                c = cell(fam, {}, k, 300, n, "uniform", "none", 0.0)
                for m in (2, 3, 4):
                    counts_by_m = {mm: al_counts(k, mm, n)[0] for mm in (2, 3, 4)}
                    if not counts_by_m[m]:
                        rows.append(dict(family=fam, k=k, n=n, m=m, status="skipped", reason="infeasible (ball count)")); continue
                    val = one(c, 1000, m, counts_by_m)
                    best = min(val, key=lambda r: r["max_ang"])["count"]
                    for seed in range(3):
                        for r in one(c, seed, m, counts_by_m):
                            rows.append(dict(family=fam, k=k, n=n, m=m, seed=seed, status="ok", chosen=(r["count"] == best), **r))
                    pd.DataFrame(rows).to_csv(ROOT / "results" / "series_2" / "rate_check.csv", index=False)
                    print(fam, k, n, m, "best count", best, flush=True)
