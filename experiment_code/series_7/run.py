"""Series 7: how Guidotti's signature function scales with the ambient dimension D.  python run.py D  ->  results/series_7/cells/D<D>.csv
Family A (product of circles), k = 2, n = 2000, no noise, uniform sampling; Guidotti with the true k, bw_mult 1.5, alpha 1e-3, float64, default number of perturbations 2(D-k).
Per seed: fit time and peak GPU memory, time and peak memory of the tangent estimate at N_EV points, error at those points (principal angles to the true tangent space)."""
import sys, pathlib, json
import numpy as np, pandas as pd, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est.generators import make_manifold, make_dataset, tangent_frame
from tangent_est.bench import principal
from tangent_torch import Guidotti
from tangent_torch.common import DEV, MemTimer

K, N, N_EV, SEEDS = 2, 2000, 20, (0, 1, 2)

if __name__ == "__main__":
    D = int(sys.argv[1])
    out = ROOT / "results" / "series_7" / "cells"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng([seed, 7, K, D, N])
        M = make_manifold("A", K, D, seed)
        ds = make_dataset(M, N, rng, uniform=True)
        X = torch.tensor(ds["X"], dtype=torch.float64, device=DEV)
        ev = torch.arange(N_EV, device=DEV)
        E = tangent_frame(M.J(ds["theta"][:N_EV]))[0]
        with MemTimer() as tf: est = Guidotti(K, bw_mult=1.5, alpha=1e-3, dtype=torch.float64).fit(X)
        with MemTimer() as tt: T = est.tangents_at(ev).cpu().numpy()
        ang = principal(T, E)
        mx = ang.max(1)
        row = dict(D=D, seed=seed, n=N, k=K, n_ev=N_EV, n_perturb=est._n_perturb(), time_fit=tf.time, peak_fit=tf.peak, time_tangent=tt.time, time_per_point=tt.time / N_EV,
                   peak_tangent=tt.peak, ang_med=float(np.median(mx)), ang_p90=float(np.percentile(mx, 90)), ang_max=float(mx.max()), ang_mean_all=float(ang.mean()))
        print(row, flush=True)
        rows.append(row)
        pd.DataFrame(rows).to_csv(out / f"D{D}.csv", index=False)
        del est, X; torch.cuda.empty_cache()
