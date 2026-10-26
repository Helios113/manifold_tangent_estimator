"""LPCA is textbook: compare against an independent numpy/scipy implementation, and check it is exact on a noise-free plane."""
import numpy as np
from scipy.spatial import cKDTree
from tangent_torch import LPCA
from experiment_code.series_1.common import estimate, max_angle_deg, save, CRIT


def test_matches_independent_numpy_implementation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 5)); X /= np.linalg.norm(X, axis=1, keepdims=True)    # S^4 in R^5, d = 4
    X += 0.01 * rng.normal(size=X.shape)
    d, k = 3, 20
    _, nb = cKDTree(X).query(X, k + 1)                                             # each point + its k neighbours
    ref = np.empty((len(X), 5, d))
    for i in range(len(X)):
        P = X[nb[i]] - X[nb[i]].mean(0)
        ref[i] = np.linalg.svd(P, full_matrices=False)[2][:d].T
    ang = max_angle_deg(estimate(LPCA(d, k=k), X), ref)
    save("lpca_vs_numpy", ang=ang)
    print(f"LPCA vs numpy: max principal angle  max={ang.max():.4f} deg  mean={ang.mean():.5f} deg")
    assert ang.max() < CRIT["lpca_numpy_deg"]


def test_exact_on_noise_free_plane():
    rng = np.random.default_rng(1)
    E = np.linalg.qr(rng.normal(size=(10, 2)))[0]
    X = rng.uniform(-1, 1, (400, 2)) @ E.T
    Qtrue = np.broadcast_to(E, (400, 10, 2))
    ang = max_angle_deg(estimate(LPCA(2, k=15), X), Qtrue)
    save("lpca_plane", ang=ang)
    print(f"LPCA on a plane in R^10: max angle {ang.max():.5f} deg")
    assert ang.max() < CRIT["lpca_plane_deg"]
