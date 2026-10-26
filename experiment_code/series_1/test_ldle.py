"""LDLE paper (arXiv:2101.11055).  Hyperparameters are the paper's defaults (Table 1): knn=49, ktune=7, N=100, klv=25, p=0.99.
 * Running example (Sec. 3.2): the square [0,1]^2 on a grid of spacing 0.01, embedded isometrically in R^3 by a random orthonormal
   frame, so the true tangent space of every point is the span of the frame.
 * Flat torus (Appendix, Eq. 58-59): X = (4 cos th, 4 sin th, cos ph, sin ph)/4pi, th = 0.01 i pi (i<200), ph = 0.04 j pi (j<50),
   a closed 2-manifold in R^4 (10000 points) with analytic tangents d/dth and d/dph."""
import numpy as np
from tangent_torch import LDLE
from experiment_code.series_1.common import estimate, max_angle_deg, save, CRIT

PAPER_KW = dict(k=49, k_tune=7, klv=25, N=100, p_chi=0.99)


def test_square_grid_flat_sheet():
    rng = np.random.default_rng(0)
    g = np.linspace(0, 1, 101)
    xy = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    E = np.linalg.qr(rng.normal(size=(3, 2)))[0]
    Q = estimate(LDLE(2, **PAPER_KW), xy @ E.T)
    ang = max_angle_deg(Q, np.broadcast_to(E, (len(xy), 3, 2)))
    interior = (np.abs(xy - 0.5) < 0.4).all(1)
    save("ldle_square_grid", xy=xy, ang=ang, interior=interior)
    print(f"LDLE square grid: interior median={np.median(ang[interior]):.3f} max={ang[interior].max():.3f}  "
          f"all median={np.median(ang):.3f} max={ang.max():.3f} deg")
    assert np.median(ang[interior]) < CRIT["ldle_grid_deg"]


def test_flat_torus_R4():
    i, j = np.meshgrid(np.arange(200), np.arange(50), indexing="ij")
    th, ph = 0.01 * np.pi * i.ravel(), 0.04 * np.pi * j.ravel()
    X = np.c_[4 * np.cos(th), 4 * np.sin(th), np.cos(ph), np.sin(ph)] / (4 * np.pi)
    z = np.zeros_like(th)
    T = np.stack([np.c_[-np.sin(th), np.cos(th), z, z], np.c_[z, z, -np.sin(ph), np.cos(ph)]], 2)   # orthonormal already
    ang = max_angle_deg(estimate(LDLE(2, **PAPER_KW), X), T)
    save("ldle_flat_torus", th=th, ph=ph, ang=ang)
    print(f"LDLE flat torus R^4: angle median={np.median(ang):.3f} 95%={np.percentile(ang, 95):.3f} max={ang.max():.3f} deg")
    assert np.median(ang) < CRIT["ldle_torus_deg"]
