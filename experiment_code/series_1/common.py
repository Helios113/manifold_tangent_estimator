"""Shared helpers for the known-result tests: data generators from the papers' experiments and the accuracy measures."""
import pathlib, numpy as np
from tangent_torch import to_t

RESULTS = pathlib.Path(__file__).resolve().parent.parent.parent / "results" / "series_1"

# Acceptance criteria, shared by the tests (assert) and by make_plots.py (the target regions drawn in the figures).
CRIT = dict(
    lpca_numpy_deg=0.2,       # max angle between torch LPCA and an independent numpy LPCA (fp32 round-off only)
    lpca_plane_deg=0.2,       # max angle on a noise-free plane (exact answer)
    lpoly_slope2=(-0.65, -0.35),  # log-log slope of the median error vs n, order 2, fixed k, d = 2 (Theorem 2-3: (1/n)^{1/d} = n^-0.5)
    lpoly_curv_slope=-0.35,   # log-log slope of the curvature error vs n, order 3 (Theorem 4: (1/n)^{(m-2)/d} = n^-0.5)
    lpoly_curv_final=0.1,     # curvature error at the largest n (|| II ||_op = 1 for the unit sphere)
    lpoly_slope3=-0.8,        # same, order 3 (theory: (1/n)^{2/d} = n^-1)
    ldle_grid_deg=0.1,        # median interior angle error, flat square grid (exact answer: whole plane)
    ldle_torus_deg=1.0,       # median angle error, flat torus in R^4 (analytic tangents)
    lego_ratio=0.5,           # median D_j(LEGO) must be below this fraction of median D_j(LPCA) under normal noise
    guidotti_numpy_deg=1e-4,       # max angle between torch and independent numpy implied normals (fp64)
    guidotti_numpy_u=1e-8,         # max |u_torch - u_numpy|
    guidotti_sphere_u=2.25e-3,     # max |u - 1| at held-out sphere points, in every one of 20 draws (100 x the paper's single draw: 2.25e-5)
    guidotti_sphere_deg=0.873,     # max angle between implied and exact normal, in every draw (100 x the paper's 0.00873 deg)
    guidotti_line_ratio=1e-3,      # straight line: sv3/sv2 of the perturbed normals (rank 2)
    guidotti_line_along=1e-3,      # straight line: normals orthogonal to the line
    guidotti_surface_sv1=0.01,     # Table 6.8.1, surface: |sv1 - sqrt(15)| (paper: 3.87 = sqrt(15) at all four points)
    guidotti_surface_sv2=0.3,      # surface: sv2 (paper: <= 0.22)
    guidotti_curve_deg=2.0,        # median tangent angle error, closed curve (d=1)
    guidotti_surface_deg=2.0,      # median tangent angle error, extruded surface (d=2)
)
PAPER = dict(guidotti_sphere_u=2.25e-5, guidotti_sphere_deg=0.00873)


def save(name, _file="data", **arrays):
    """Store the raw results of one test in results/<name>/<_file>.npz (read by make_plots.py)."""
    d = RESULTS / name; d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{_file}.npz", **arrays)


def discrepancy(Q, Qtrue):
    """LEGO paper Eq. 44: D_j = sum_i (1 - cos(theta_i)) over the principal angles between span(Q_j) and span(Q*_j)."""
    Q = np.asarray(Q, dtype=np.float64)
    c = np.linalg.svd(np.einsum("npd,npe->nde", Q, Qtrue), compute_uv=False).clip(0, 1)
    return (1 - c).sum(1)


def max_angle_deg(Q, Qtrue):
    """Largest principal angle (degrees) between the estimated and true tangent spaces, per point."""
    Q = np.asarray(Q, dtype=np.float64)
    c = np.linalg.svd(np.einsum("npd,npe->nde", Q, Qtrue), compute_uv=False).clip(0, 1)
    return np.degrees(np.arccos(c.min(1)))


def estimate(method, X, dtype=None):
    """Fit a torch method on the rows of X (n,p) and return its tangent bases as a float64 numpy array (n,p,d)."""
    return method.fit(to_t(X) if dtype is None else to_t(X, dtype)).tangents_train().cpu().numpy().astype(np.float64)


def orth(J):
    """Orthonormal basis of the column span of J (n,p,d) -- batched."""
    return np.linalg.qr(J)[0]


def swiss_roll(n, rng, h_max=0.5):
    """High-aspect-ratio Swiss roll (LEGO Sec. 5.1): Y = s (t cos t, h/s, t sin t), t in [1.5pi, 4.5pi] sampled uniformly w.r.t.
    area, scaled by s = 1/(4.5pi) so the roll has radius <= 1; h uniform in [0, h_max]. Returns Y, true tangents, unit normals."""
    s = 1 / (4.5 * np.pi)
    g = np.linspace(1.5 * np.pi, 4.5 * np.pi, 20001)
    cdf = np.cumsum(np.sqrt(1 + g ** 2)); cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    t = np.interp(rng.uniform(size=n), cdf, g)
    h = rng.uniform(0, h_max, n)
    Y = np.c_[s * t * np.cos(t), h, s * t * np.sin(t)]
    Tt = np.c_[s * (np.cos(t) - t * np.sin(t)), np.zeros(n), s * (np.sin(t) + t * np.cos(t))]
    Th = np.c_[np.zeros(n), np.ones(n), np.zeros(n)]
    T = orth(np.stack([Tt, Th], 2))
    return Y, T, np.cross(T[:, :, 0], T[:, :, 1])


def truncated_torus(n, rng, R=0.3, r=0.12, u_max=1.5 * np.pi):
    """Truncated torus (LEGO Sec. 5.1): u in [0, u_max) uniform, v in [0, 2pi) uniform w.r.t. area (rejection).
    Returns Y, true tangents, unit normals, u."""
    u = rng.uniform(0, u_max, n)
    v = np.empty(0)
    while len(v) < n:
        c = rng.uniform(0, 2 * np.pi, 2 * n)
        v = np.r_[v, c[rng.uniform(0, R + r, 2 * n) < R + r * np.cos(c)]]
    v = v[:n]
    Y = np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
    Tu = np.c_[-(R + r * np.cos(v)) * np.sin(u), (R + r * np.cos(v)) * np.cos(u), np.zeros(n)]
    Tv = np.c_[-r * np.sin(v) * np.cos(u), -r * np.sin(v) * np.sin(u), r * np.cos(v)]
    T = orth(np.stack([Tu, Tv], 2))
    return Y, T, np.cross(T[:, :, 0], T[:, :, 1]), u
