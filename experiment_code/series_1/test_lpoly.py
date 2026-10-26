"""Local polynomial estimator (Aamari & Levrard 2019, eq. (2)): order 2 is local PCA (checked against an independent numpy implementation), it is exact
on a noise-free plane, and on a curved surface the error decreases with the sample size at the rates of Theorems 2-3 -- (1/n)^{(k-1)/d} for order k, i.e. n^-1/2
for order 2 and n^-1 for order 3 when d = 2 (fixed number of neighbours, so that h ~ n^{-1/d}). The rate test runs in float64: in float32 the angle has a floor of about 0.01 degrees (arccos of rounded bases), which order 3 reaches."""
import numpy as np, torch
from scipy.spatial import cKDTree
from tangent_torch import LocalPoly
from tangent_torch.common import DEV
from experiment_code.series_1.common import estimate, max_angle_deg, save, CRIT

RATE_SIZES = (1000, 2000, 4000, 8000, 16000, 32000)


def test_order2_matches_independent_numpy_implementation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5)); X /= np.linalg.norm(X, axis=1, keepdims=True)    # S^4 in R^5, d = 4
    X += 0.01 * rng.normal(size=X.shape)
    d, k = 3, 20
    _, nb = cKDTree(X).query(X, k + 1)                                             # each point + its k neighbours (the point itself adds a zero row)
    ref = np.empty((len(X), 5, d))
    for i in range(len(X)):
        Z = X[nb[i, 1:]] - X[i]                                                    # non-centred: the paper's k = 2 problem is min over pi of sum ||Z - pi Z||^2
        ref[i] = np.linalg.svd(Z, full_matrices=False)[2][:d].T
    ang = max_angle_deg(estimate(LocalPoly(d, k=k, order=2), X), ref)
    save("lpoly_vs_numpy", ang=ang)
    print(f"LocalPoly(order 2) vs numpy: max principal angle  max={ang.max():.4f} deg  mean={ang.mean():.5f} deg")
    assert ang.max() < CRIT["lpca_numpy_deg"]


def test_exact_on_noise_free_plane():
    rng = np.random.default_rng(1)
    E = np.linalg.qr(rng.normal(size=(10, 2)))[0]
    X = rng.uniform(-1, 1, (400, 2)) @ E.T
    Qtrue = np.broadcast_to(E, (400, 10, 2))
    ang = max_angle_deg(estimate(LocalPoly(2, k=30, order=3), X), Qtrue)
    save("lpoly_plane", ang=ang)
    print(f"LocalPoly(order 3) on a plane in R^10: max angle {ang.max():.5f} deg")
    assert ang.max() < CRIT["lpca_plane_deg"]


def sphere(n, rng, E):
    X = rng.normal(size=(n, 3)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = np.linalg.svd(np.eye(3) - X[:, :, None] * X[:, None, :])[0][:, :, :2]      # orthonormal basis of x^perp
    return X @ E.T, np.einsum("pq,nqd->npd", E, U)


def test_convergence_rate_on_the_sphere():
    E = np.linalg.qr(np.random.default_rng(2).normal(size=(10, 3)))[0]
    err = {o: np.zeros((3, len(RATE_SIZES))) for o in (2, 3)}
    for s in range(3):
        for j, n in enumerate(RATE_SIZES):
            X, Q = sphere(n, np.random.default_rng(100 * s + j), E)
            for o in (2, 3):
                err[o][s, j] = np.median(max_angle_deg(estimate(LocalPoly(2, k=30, order=o), X, dtype=torch.float64), Q))
    e2, e3 = err[2].mean(0), err[3].mean(0)
    sl = {o: np.polyfit(np.log(RATE_SIZES), np.log(e), 1)[0] for o, e in ((2, e2), (3, e3))}
    save("lpoly_rate", n=np.array(RATE_SIZES), err2=err[2], err3=err[3], slope2=sl[2], slope3=sl[3])
    print(f"sphere S^2 in R^10, k=30: median error order 2 {np.round(e2, 3)}, order 3 {np.round(e3, 4)}; slopes {sl[2]:.2f} (theory -0.5), {sl[3]:.2f} (theory -1)")
    assert CRIT["lpoly_slope2"][0] < sl[2] < CRIT["lpoly_slope2"][1]
    assert sl[3] < CRIT["lpoly_slope3"]


def test_curvature_on_the_sphere():
    """The second-order tensor T_2 of the order-3 estimator estimates the second fundamental form (Theorem 4: error (1/n)^{(m-2)/k} = n^-1/2 for m = 3, d = 2).
    Unit sphere S^2 in R^10: II(u, v) = -<u, v> N with N = E x the unit normal, so II_ab = -delta_ab N in an orthonormal tangent frame; the error is the operator norm
    sup ||II o pi_T (x, y) - T2 o pi_hat (x, y)|| (|| II ||_op = 1)."""
    from tangent_est.bench import curvature_error
    E = np.linalg.qr(np.random.default_rng(2).normal(size=(10, 3)))[0]
    errs = np.zeros((3, len(RATE_SIZES)))
    for s in range(3):
        for j, n in enumerate(RATE_SIZES):
            X, Q = sphere(n, np.random.default_rng(100 * s + j), E)
            ev = np.arange(200)
            est = LocalPoly(2, k=30, order=3).fit(torch.as_tensor(X, dtype=torch.float64, device=DEV))
            T, T2 = est.tangents_at(torch.as_tensor(ev, device=DEV), want_T2=True)
            II = -np.einsum("nD,ab->nDab", X[ev], np.eye(2))                      # the unit normal of the unit sphere is the point itself
            errs[s, j] = np.median(curvature_error(Q[ev], II, T.cpu().numpy(), T2.cpu().numpy()))
    e = errs.mean(0)
    slope = np.polyfit(np.log(RATE_SIZES), np.log(e), 1)[0]
    save("lpoly_curvature", n=np.array(RATE_SIZES), err=errs, slope=slope)
    print(f"sphere S^2 in R^10, k=30, order 3: median curvature error {np.round(e, 4)}; slope {slope:.2f} (theory -0.5)")
    assert slope < CRIT["lpoly_curv_slope"] and e[-1] < CRIT["lpoly_curv_final"]
