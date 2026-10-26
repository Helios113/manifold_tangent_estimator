"""Generator validation (plan.md step 2): immersion, bottleneck, reach, non-triviality (global PCA must fail on families B and D at small k); records the values
and rejects the manifold on failure.  Used by experiment_code/series_*/validate_generators.py."""
import json, numpy as np, pandas as pd
from .generators import make_manifold, eps_for_diam_over_tau
from .bench import principal, estimate_tau


def validate_one(spec, seeds=(0, 1, 2), n=2000):
    rows = []
    for seed in seeds:
        rng = np.random.default_rng([seed, 5, spec["k"], spec["D"]])
        M = make_manifold(spec["family"], spec["k"], spec["D"], seed, **spec.get("gen", {}))
        if spec["family"] == "D" and spec.get("diam_over_tau") is not None:
            M.set_eps(eps_for_diam_over_tau(M, spec["diam_over_tau"], rng))
        th, info = M.sample_theta(n, rng)
        minsv = np.inf; maxsv = 0.0
        for s in range(0, n, 200):
            sv = np.linalg.svd(M.J(th[s:s + 200]), compute_uv=False); minsv = min(minsv, sv.min()); maxsv = max(maxsv, sv.max())
        t = estimate_tau(M, rng)
        X = M.F(th); Xc = X - X.mean(0)
        sq = ((X[:500, None, :] - X[None, :, :]) ** 2).sum(2); sq[np.arange(500), np.arange(500)] = np.inf
        nn_med = float(np.sqrt(sq.min(1)).mean())
        V = np.linalg.eigh(Xc.T @ Xc)[1][:, -spec["k"]:]
        ev = np.arange(300); Et = M.tangent(th[ev])
        ang = principal(np.repeat(V[None], len(ev), 0), Et)
        pca_med = float(np.median(ang.max(1)))
        nu_max = 0.5 * min(t["tau"], t["diam"])
        r = dict(**{k_: spec[k_] for k_ in ("family", "k", "D")}, base=spec.get("gen", {}).get("base", ""), diam_over_tau_target=spec.get("diam_over_tau"), seed=seed,
                 eps=getattr(M, "eps", None), min_sv_J=float(minsv), max_sv_J=float(maxsv), tau=t["tau"], tau_exact=bool(getattr(M, "reach_exact", None)), kappa_max=t["kappa_max"],
                 bottleneck=t["bottleneck"], diam=t["diam"], diam_over_tau=t["diam"] / t["tau"], bottleneck_over_noise_max=t["bottleneck"] / nu_max, nn_distance_mean=nn_med, bottleneck_over_nn=t["bottleneck"] / nn_med,
                 global_pca_median_max_angle=pca_med, sampling_acceptance=info["acceptance"])
        r["immersion_ok"] = bool(minsv > 1e-6 * maxsv)
        r["bottleneck_ok"] = bool(t["bottleneck"] > 3 * nn_med)       # pairs far apart in parameter space must be much farther than the typical neighbour distance
        r["nontrivial_ok"] = bool(pca_med > 20.0) if (spec["family"] in ("B", "D") or spec.get("gen", {}).get("base") == "B") and spec["k"] <= 10 else None
        r["accepted"] = bool(r["immersion_ok"] and r["bottleneck_ok"] and r["nontrivial_ok"] is not False)
        rows.append(r); print(json.dumps(r, default=str), flush=True)
    return rows


def validate(specs, out_csv):
    rows = []
    for sp in specs:
        rows += validate_one(sp)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return pd.DataFrame(rows)
