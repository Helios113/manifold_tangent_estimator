"""Generator check for family S, run before the cells: analytic tangent frame orthonormal, tangent_vectors = J v are tangent (compare with finite differences of F along the sphere)."""
import sys, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent; sys.path.insert(0, str(ROOT))
from tangent_est.generators import make_manifold
M = make_manifold("S", 100, 1000, 0); rng = np.random.default_rng(0)
th, _ = M.sample_theta(50, rng); E = M.tangent(th)
res = dict(unit_norm=np.abs(np.linalg.norm(th, axis=1) - 1).max(), orthonormal=np.abs(np.swapaxes(E, 1, 2) @ E - np.eye(100)).max())
v = rng.normal(size=th.shape); v -= (v * th).sum(1, keepdims=True) * th; v /= np.linalg.norm(v, axis=1, keepdims=True)      # tangent direction on the sphere
h = 1e-6; th2 = th + h * v; th2 /= np.linalg.norm(th2, axis=1, keepdims=True)
fd = (M.F(th2) - M.F(th)) / h
res["fd_in_span_residual"] = np.linalg.norm(fd - np.einsum("nDk,nk->nD", E, np.einsum("nDk,nD->nk", E, fd)), axis=1).max() / np.linalg.norm(fd, axis=1).mean()
res["min_sv_J"] = np.linalg.svd(M.J(th), compute_uv=False).min()
print(res)
(ROOT / "results" / "series_8").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "series_8" / "generator_check.txt").write_text("\n".join(f"{k}: {v:.3e}" for k, v in res.items()) + "\n")
