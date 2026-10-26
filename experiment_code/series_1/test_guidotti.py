"""Guidotti, 'Extracting manifold information from point clouds' (arXiv:2404.00427), all in float64 (the paper's regime: alpha = 1e-10 cannot be factorised in fp32). The paper's kernel is exp(-|x|^2) (or the
regularised Laplace kernel exp(-sqrt(|x|^2+r)), r=1) with no bandwidth, so every test uses bw=1 (Guidotti(bw=1)).
 * vs_numpy:   the torch code equals a from-scratch numpy implementation of Eqs. 3.3, 3.5 and the gradient/normal formulas of Sec. 5.
 * sphere:     Sec. 6.6, 80 random points on the unit sphere, normals at 32 other points (paper, one draw: max|u-1| = 2.25e-5, 0.00873 deg).
 * line:        Sec. 4 / 6.8, a straight line in R^3: the normals at perturbed points are radial, i.e. rank 2, and orthogonal to the line.
 * table_6_8_1: Sec. 6.8, singular values of the 3x15 matrix of normals at 15 perturbations (distance <= 0.01) of points of a closed curve and
                of the surface obtained by extruding it. The paper's sample of the curve is not specified (only 'segments and semicircles').
 * curve_surface: the tangent estimate of the library (smallest singular directions of the perturbed normals) on that curve and surface."""
import numpy as np, pytest, torch
from tangent_torch import Guidotti, to_t
from tangent_torch.common import sqdist
from experiment_code.series_1.common import estimate, max_angle_deg, save, CRIT


def reference(X, alpha, kernel, r=1.0):
    """From-scratch numpy (float64) signature function: solve (alpha I + K/m) Lambda = 1 (Eq. 3.3), lambda = Lambda/m, u = sum lambda_j K(x-x_j) (Eq. 3.5);
    returns u(x) and the implied normal nu = -grad u / |grad u| (Sec. 5), with grad u = -2 sum lambda_j e^{-|x-x_j|^2}(x-x_j) (Gauss)
    or -sum lambda_j e^{-q} (x-x_j)/q, q = sqrt(|x-x_j|^2+r) (regularised Laplace)."""
    m = len(X)
    D2 = ((X[:, None] - X[None]) ** 2).sum(2)
    K = np.exp(-D2) if kernel == "gauss" else np.exp(-np.sqrt(D2 + r))
    lam = np.linalg.solve(alpha * np.eye(m) + K / m, np.ones(m)) / m

    def u(x):
        d2 = ((x - X) ** 2).sum(1)
        return lam @ (np.exp(-d2) if kernel == "gauss" else np.exp(-np.sqrt(d2 + r)))

    def nu(x):
        dd = x - X; d2 = (dd ** 2).sum(1)
        g = -2 * (lam * np.exp(-d2)) @ dd if kernel == "gauss" else -((lam * np.exp(-np.sqrt(d2 + r)) / np.sqrt(d2 + r)) @ dd)
        return -g / np.linalg.norm(g)
    return u, nu


def _sphere_pts(n, rng):
    z = rng.uniform(-1, 1, n); th = rng.uniform(0, 2 * np.pi, n)
    return np.c_[np.sqrt(1 - z ** 2) * np.cos(th), np.sqrt(1 - z ** 2) * np.sin(th), z]


def _signature(sig, Yt):
    return (sig._k(sqdist((Yt - sig.mu0) / sig.bw, sig.Xs)) @ sig.lam / sig.n).cpu().numpy().astype(np.float64)


@pytest.mark.parametrize("kernel", ["gauss", "lap"])
def test_matches_independent_numpy(kernel):
    rng = np.random.default_rng(0)
    X, Y = _sphere_pts(80, rng), _sphere_pts(32, rng)
    alpha = 1e-8
    sig = Guidotti(2, kernel=kernel, r=1.0, alpha=alpha, bw=1.0, dtype=torch.float64).fit(to_t(X, torch.float64))
    u_ref, nu_ref = reference(X, alpha, kernel)
    nu = sig._normals(to_t(Y, torch.float64)).cpu().numpy()
    nur = np.array([nu_ref(y) for y in Y]); ur = np.array([u_ref(y) for y in Y])
    ang = np.degrees(np.arccos(np.clip((nu * nur).sum(1), -1, 1)))
    du = np.abs(_signature(sig, to_t(Y, torch.float64)) - ur)
    save("guidotti_vs_numpy", _file=f"data_{kernel}", ang=ang, du=du)
    print(f"guidotti vs numpy ({kernel}): max normal angle={ang.max():.2e} deg, max |u - u_ref|={du.max():.2e}")
    assert ang.max() < CRIT["guidotti_numpy_deg"] and du.max() < CRIT["guidotti_numpy_u"]


def test_sphere_80_points_normals():
    dt = torch.float64
    alpha = 0.0
    udev, angs = [], []
    for seed in range(20):                                                   # the paper reports one random draw; we show the spread over 20
        rng = np.random.default_rng(seed)
        X, Y = _sphere_pts(80, rng), _sphere_pts(32, rng)
        sig = Guidotti(2, alpha=alpha, bw=1.0, dtype=dt).fit(to_t(X, dt))
        Yt = to_t(Y, dt)
        nu = sig._normals(Yt).cpu().numpy().astype(np.float64)
        udev.append(np.abs(_signature(sig, Yt) - 1).max())
        angs.append(np.degrees(np.arccos(np.clip((nu * Y).sum(1), -1, 1))).max())
    udev, angs = np.array(udev), np.array(angs)
    save("guidotti_sphere", _file="data", u_dev=udev, ang=angs)
    print(f"guidotti sphere (20 draws, max over 32 points): |u-1| median={np.median(udev):.2e} [{udev.min():.1e},{udev.max():.1e}] (paper 2.25e-5);  "
          f"angle median={np.median(angs):.4f} [{angs.min():.4f},{angs.max():.4f}] deg (paper 0.00873)")
    assert udev.max() < CRIT["guidotti_sphere_u"] and angs.max() < CRIT["guidotti_sphere_deg"]


def _perturbed_svals(sig, pts, rng, n=15, eps=0.01):
    """Singular values of the (3 x n) matrix of implied normals at n random perturbations of each row of pts (random direction, distance U[0, eps])."""
    out = []
    for p in pts:
        dirs = rng.normal(size=(n, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        Q = p + dirs * rng.uniform(0, eps, (n, 1))
        N = sig._normals(to_t(Q, sig.dt)).cpu().numpy().astype(np.float64)
        out.append(np.linalg.svd(N.T, compute_uv=False))
    return np.array(out)


@pytest.mark.parametrize("kernel", ["gauss", "lap"])
def test_straight_line_normals_are_radial(kernel):
    t = np.linspace(-6, 6, 61); P = np.c_[t, 0 * t, 0 * t]
    sig = Guidotti(1, kernel=kernel, r=1.0, alpha=1e-8, bw=1.0, dtype=torch.float64).fit(to_t(P, torch.float64))
    rng = np.random.default_rng(1)
    sv = _perturbed_svals(sig, [np.zeros(3)], rng)[0]
    dirs = rng.normal(size=(15, 3)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    N = sig._normals(to_t(dirs * rng.uniform(0, 0.01, (15, 1)), torch.float64)).cpu().numpy()
    along = np.abs(N[:, 0]).max()
    save("guidotti_line", _file=f"data_{kernel}", sv=sv, along=along)
    print(f"guidotti line ({kernel}): singular values {np.round(sv, 4)}, max |normal component along the line|={along:.1e}")
    assert sv[2] / sv[1] < CRIT["guidotti_line_ratio"] and along < CRIT["guidotti_line_along"]


def _stadium(m):
    """Closed curve in the x-z plane: two unit segments joined by two semicircles of radius 1/2; m points, with unit tangents."""
    Lseg, rad = 2.0, 0.5
    per = 2 * Lseg + 2 * np.pi * rad
    s = np.arange(m) / m * per
    P = np.zeros((m, 3)); T = np.zeros((m, 3))
    for i, si in enumerate(s):
        if si < Lseg:                      P[i] = (-1 + si, 0, rad);                          T[i] = (1, 0, 0)
        elif si < Lseg + np.pi * rad:      a = (si - Lseg) / rad; P[i] = (1 + rad * np.sin(a), 0, rad * np.cos(a)); T[i] = (np.cos(a), 0, -np.sin(a))
        elif si < 2 * Lseg + np.pi * rad:  P[i] = (1 - (si - Lseg - np.pi * rad), 0, -rad);   T[i] = (-1, 0, 0)
        else:                              a = (si - 2 * Lseg - np.pi * rad) / rad; P[i] = (-1 - rad * np.sin(a), 0, -rad * np.cos(a)); T[i] = (-np.cos(a), 0, np.sin(a))
    return P, T


def _stadium_data(m=300):
    P, T = _stadium(m)
    ys = np.linspace(-0.5, 0.5, 11)
    S = np.concatenate([P + np.array([0, y, 0]) for y in ys])
    TS = np.stack([np.tile(T, (11, 1)), np.tile([0, 1, 0], (len(S), 1))], 2)
    return P, T, S, TS


def test_table_6_8_1_singular_values():
    dt = torch.float64
    P, T, S, TS = _stadium_data()
    kw = dict(kernel="lap", r=1.0, alpha=1e-10, bw=1.0, dtype=dt)   # Sec. 6.7: regularised Laplace kernel, r = 1, alpha = 1e-10
    ic = Guidotti(1, **kw).fit(to_t(P, dt)); is_ = Guidotti(2, **kw).fit(to_t(S, dt))
    pick = np.arange(0, len(P), len(P) // 4)[:4]                              # four points of the curve; the same points at y = 0 of the surface
    rng = np.random.default_rng(1)
    sv_c = _perturbed_svals(ic, P[pick], rng)
    sv_s = _perturbed_svals(is_, S[5 * len(P) + pick], rng)
    save("guidotti_table_6_8_1", _file="data", sv_c=sv_c, sv_s=sv_s)
    print(f"guidotti table 6.8.1: curve sv=\n{np.round(sv_c, 4)}\nsurface sv=\n{np.round(sv_s, 4)}")
    assert np.all(np.abs(sv_s[:, 0] - np.sqrt(15)) < CRIT["guidotti_surface_sv1"]) and np.all(sv_s[:, 1] < CRIT["guidotti_surface_sv2"])


def test_curve_and_surface_tangent():
    dt = torch.float64
    P, T, S, TS = _stadium_data()
    kw = dict(kernel="lap", r=1.0, alpha=1e-10, bw=1.0, rho=0.01, dtype=dt)   # paper's kernel and perturbation size (distance <= 0.01)
    ang_c = max_angle_deg(estimate(Guidotti(1, **kw), P, dt), T[:, :, None])
    mid = np.arange(len(S)) // len(P) == 5                                    # the points at y = 0, away from the boundary
    ang_s = max_angle_deg(estimate(Guidotti(2, **kw), S, dt)[mid], TS[mid])
    save("guidotti_curve_surface", _file="data", ang_c=ang_c, ang_s=ang_s)
    print(f"guidotti curve (d=1): median angle={np.median(ang_c):.2f} max={ang_c.max():.2f} deg;  "
          f"surface (d=2, y=0): median={np.median(ang_s):.2f} max={ang_s.max():.2f} deg")
    assert np.median(ang_c) < CRIT["guidotti_curve_deg"] and np.median(ang_s) < CRIT["guidotti_surface_deg"]
