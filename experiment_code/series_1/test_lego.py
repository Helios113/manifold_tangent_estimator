"""LEGO paper (arXiv:2510.02308) Sec. 5.1, Figs. 3-4: high-aspect-ratio Swiss roll and truncated torus with noise added in the
direction normal to the tangent space, tangent error measured by the discrepancy D_j (Eq. 44), LPCA vs LEGO with the
hyperparameters of the figure captions.  The claim being verified: LEGO is much more accurate than LPCA under this noise.
The paper does not state the exact geometry (scale, aspect ratio, torus radii); see common.py for what is used here."""
import numpy as np
from tangent_torch import LPCA, LEGO
from experiment_code.series_1.common import estimate, discrepancy, swiss_roll, truncated_torus, save, CRIT


def _run(X, Qtrue, knn, m, label, name):
    D_lpca = discrepancy(estimate(LPCA(2, k=knn), X), Qtrue)
    D_lego = discrepancy(estimate(LEGO(2, k=knn, m=m, m0=100, beta=0.5), X), Qtrue)
    save(name, D_lpca=D_lpca, D_lego=D_lego)
    print(f"{label}: D_j median  LPCA={np.median(D_lpca):.2e}  LEGO={np.median(D_lego):.2e}   "
          f"mean  LPCA={D_lpca.mean():.2e}  LEGO={D_lego.mean():.2e}")
    return D_lpca, D_lego


def test_swiss_roll_fig3():
    rng = np.random.default_rng(0)
    Y, T, nu = swiss_roll(10700, rng)
    X = Y + rng.uniform(-0.0175, 0.0175, (len(Y), 1)) * nu
    D_lpca, D_lego = _run(X, T, knn=9, m=40, label="Swiss roll (Fig. 3)", name="lego_swiss_roll")
    assert np.median(D_lego) < CRIT["lego_ratio"] * np.median(D_lpca)


def test_truncated_torus_fig4():
    rng = np.random.default_rng(0)
    Y, T, nu, u = truncated_torus(3617, rng)
    eps = 1e-2 + 2.5e-3 * (1 + np.cos(2 * u))
    X = Y + (rng.uniform(-1, 1, len(Y)) * eps)[:, None] * nu
    D_lpca, D_lego = _run(X, T, knn=14, m=20, label="Truncated torus (Fig. 4)", name="lego_truncated_torus")
    assert np.median(D_lego) < CRIT["lego_ratio"] * np.median(D_lpca)
