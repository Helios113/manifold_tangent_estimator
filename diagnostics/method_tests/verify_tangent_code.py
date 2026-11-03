"""Verification of the tangent-space comparison code. Method test (checks the code, not a run). Writes results/method_tests/verify_tangent_code/{results.json,summary.txt,verify.png}.

Plan (every test states what must come out; a test passes when the worst error is below its tolerance)
  A. Identical spaces, large dimension: D in {1000, 2000, 5000}, k = 100. A random orthonormal basis against (A1) itself, (A2) the same space in another basis (A Rot, Rot a random orthogonal k x k),
     (A3) the same space with a non-orthonormal basis (A M, M random invertible). Every principal angle must be 0. Float64 and float32 (the precision of the estimators).
  B. Known angles: A = first k columns of a random orthonormal Q, E = the next k columns, B_i = cos(t_i) a_i + sin(t_i) e_i; the principal angles of (A, B) are exactly t_i. D in {1000, 2000, 5000}, k = 100.
     Angle profiles: all 10 deg; linspace(0, 90); one angle 45 and the rest 0 (checks which end is "largest"); all 90 (orthogonal spaces); all 0.5 (small angles). The sorted angles and
     the largest / smallest / mean must equal the profile. Also symmetry: angles(A, B) == angles(B, A).
  C. Known offset through the estimator: noiseless points on the k-dim linear space B (tilted from A by a known profile), D = 1000, k = 100; LPCA (ambient dim + 1 neighbours) estimates the
     tangent, compared with A: must give the profile. Float32 on the GPU.
  D. The true tangent of the embedded sphere (Sphere.embed_tangent_basis with the run36 embedding): (D1) orthonormal columns; (D2) the finite-difference velocity of the embedded curve
     gamma(t) = (p + t b) / |p + t b| (b a tangent vector of the sphere at p) lies in the span, with the residual shrinking like t^2."""
import json, sys, tomllib
from pathlib import Path
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "auto_encoder")); sys.path.insert(0, str(ROOT / "diagnostics"))
from tangent import principal_angles                                        # the function under test, as used by compare()
from sphere import Sphere, EMBEDDINGS
from tangent_torch import LPCA, DEV

OUT = ROOT / "results" / "method_tests" / "verify_tangent_code"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
K, N = 100, 20
TOL = {np.float64: 1e-4, np.float32: 0.2}                                    # degrees; arccos near 1 loses precision (float32: cos error 1e-7 -> ~0.03 deg)
rows = []


def record(test, case, D, dtype, err, tol):
    rows.append(dict(test=test, case=case, D=D, dtype=np.dtype(dtype).name, max_abs_error_deg=float(err), tol=tol, passed=bool(err < tol)))
    print(f"{test:3s} {case:42s} D={D:5d} {np.dtype(dtype).name:8s} worst error {err:.3g} deg  {'PASS' if err < tol else 'FAIL'}", flush=True)


def orthonormal(D, m, n=N):
    return np.linalg.qr(rng.standard_normal((n, D, m)))[0]


# ---- A: identical spaces
for D in (1000, 2000, 5000):
    A = orthonormal(D, K)
    Rot = np.linalg.qr(rng.standard_normal((N, K, K)))[0]
    M = rng.standard_normal((N, K, K)) + 5 * np.eye(K)
    for dt in (np.float64, np.float32):
        for case, B in (("A1 itself", A), ("A2 other orthonormal basis", A @ Rot), ("A3 non-orthonormal basis", A @ M)):
            record("A", case, D, dt, principal_angles(A.astype(dt), B.astype(dt)).max(), TOL[dt])

# ---- B: known angles
profiles = {"all 10 deg": np.full(K, 10.0), "linspace(0, 90)": np.linspace(0, 90, K), "one 45, rest 0": np.r_[np.zeros(K - 1), 45.0], "all 90": np.full(K, 90.0), "all 0.5": np.full(K, 0.5), "all 0": np.zeros(K)}
scatter = []                                                                # (expected, measured) for the plot
for D in (1000, 2000, 5000):
    Q = orthonormal(D, 2 * K)
    A, E = Q[..., :K], Q[..., K:]
    for name, prof in profiles.items():
        t = np.radians(prof)
        B = A * np.cos(t) + E * np.sin(t)
        ang = principal_angles(A, B)
        record("B", name, D, np.float64, np.abs(ang - np.sort(prof)).max(), TOL[np.float64])
        record("B", name + " (B, A) swapped", D, np.float64, np.abs(principal_angles(B, A) - np.sort(prof)).max(), TOL[np.float64])
        scatter.append((np.sort(prof), ang.mean(0)))
        s = dict(largest=ang[:, -1], smallest=ang[:, 0], mean=ang.mean(1))
        record("B", name + " largest/smallest/mean", D, np.float64, max(np.abs(s["largest"] - prof.max()).max(), np.abs(s["smallest"] - prof.min()).max(), np.abs(s["mean"] - prof.mean()).max()), TOL[np.float64])

# ---- C: known offset through LPCA
D, n_pts, n_eval = 1000, 3000, 100
for name in ("all 30 deg", "linspace(1, 89)"):
    prof = np.full(K, 30.0) if name.startswith("all") else np.linspace(1, 89, K)
    Q = orthonormal(D, 2 * K, 1)[0]
    A, E = Q[:, :K], Q[:, K:]
    t = np.radians(prof)
    B = A * np.cos(t) + E * np.sin(t)                                       # the plane the points really lie on
    X = (rng.standard_normal((n_pts, K)) @ B.T).astype(np.float32)          # noiseless points of B
    est = LPCA(K, k=D + 1).fit(torch.tensor(X, device=DEV)).tangents_at(torch.arange(n_eval, device=DEV)).cpu().numpy().astype(np.float64)
    ang = principal_angles(est, np.broadcast_to(A, (n_eval, D, K)).copy())
    record("C", "LPCA on plane tilted by " + name, D, np.float32, np.abs(ang - np.sort(prof)).max(), TOL[np.float32])
    scatter.append((np.sort(prof), ang.mean(0)))

# ---- D: true tangent of the embedded sphere
cfg = tomllib.load(open(ROOT / "results" / "autoencoder" / "run36" / "config.toml", "rb"))
S = Sphere(cfg["intrinsic_dim"])
emb = EMBEDDINGS[cfg["embedding"]](cfg["ambient_dim"], torch.Generator().manual_seed(cfg["data_seed"]), in_dim=S.D)
emb.Q = emb.Q.double()
p = S.sample(50, torch.Generator().manual_seed(7), dtype=torch.float64)
Bs = S.tangent_basis(p)                                                     # sphere tangent basis in R^101
T = S.embed_tangent_basis(p, Bs, emb)                                       # (50, 1000, 100)
record("D", "D1 orthonormal columns |T^T T - I|", cfg["ambient_dim"], np.float64, (T.transpose(1, 2) @ T - torch.eye(S.k, dtype=T.dtype)).abs().max().item(), 1e-10)
record("D", "D1 tangent basis perpendicular to p", S.D, np.float64, (Bs.transpose(1, 2) @ p[:, :, None]).abs().max().item(), 1e-10)
b = (Bs @ torch.randn(50, S.k, 1, dtype=torch.float64))[..., 0]; b = b / b.norm(dim=1, keepdim=True)
resid = {}
for h in (1e-1, 1e-2, 1e-3):
    gam = lambda s: emb(torch.nn.functional.normalize(p + s * h * b, dim=1))
    v = (gam(1) - gam(-1)) / (2 * h)                                       # velocity of the embedded curve at p
    r = v - (T @ (T.transpose(1, 2) @ v[:, :, None]))[..., 0]               # part outside span(T)
    resid[h] = (r.norm(dim=1) / v.norm(dim=1)).max().item()
    print(f"    finite difference step {h}: relative residual outside span(T) {resid[h]:.3g}", flush=True)
record("D", "D2 velocity in span(T), step 1e-3 (residual ~ t^2)", cfg["ambient_dim"], np.float64, resid[1e-3], 1e-4)
rate = np.log10(resid[1e-2] / resid[1e-3])
record("D", "D2 residual shrinks ~100x per 10x step (log10 ratio - 2)", cfg["ambient_dim"], np.float64, abs(rate - 2), 0.3)

# ---- outputs
json.dump(rows, open(OUT / "results.json", "w"), indent=1)
bad = [r for r in rows if not r["passed"]]
open(OUT / "summary.txt", "w").write(f"{len(rows) - len(bad)} / {len(rows)} passed\n" + "".join(f"FAIL {r}\n" for r in bad))
print(f"\n{len(rows) - len(bad)} / {len(rows)} passed", flush=True)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
for e, m in scatter: a1.plot(e, m, ".", ms=3, alpha=.5)
a1.plot([0, 90], [0, 90], "k--", lw=1, label="measured = expected"); a1.set_xlabel("expected principal angle (degrees)"); a1.set_ylabel("measured, mean over samples (degrees)")
a1.set_title("B, C: known angles (D 1000-5000, k 100, incl. LPCA)"); a1.legend()
for dt, c in (("float64", "C0"), ("float32", "C1")):
    for i, D in enumerate((1000, 2000, 5000)):
        v = max(r["max_abs_error_deg"] for r in rows if r["test"] == "A" and r["D"] == D and r["dtype"] == dt)
        a2.bar(i + (0.2 if dt == "float32" else -0.2), max(v, 1e-12), 0.4, color=c, label=dt if i == 0 else None)
a2.set_yscale("log"); a2.set_xticks(range(3)); a2.set_xticklabels(["D=1000", "D=2000", "D=5000"]); a2.set_ylabel("worst principal angle between identical spaces (degrees)")
a2.set_title("A: identical spaces (must be 0)"); a2.legend()
fig.tight_layout(); fig.savefig(OUT / "verify.png", dpi=150)
