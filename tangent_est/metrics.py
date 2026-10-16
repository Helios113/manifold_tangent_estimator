import numpy as np


def tangent_errors(Q_est, Q_true):
    """Per-point principal-angle statistics between estimated and true tangent spaces.

    Both are (n, p, d) with orthonormal columns. Returns dict of per-point arrays (angles in degrees):
      max_angle : largest principal angle (the "worst direction" error)
      mean_angle: mean principal angle
      proj_err  : ||P_est - P_true||_F / sqrt(2)  (= sqrt(sum sin^2 theta))
    """
    M = np.einsum("npd,npe->nde", Q_est, Q_true)
    s = np.clip(np.linalg.svd(M, compute_uv=False), -1, 1)
    ang = np.degrees(np.arccos(s))               # (n, d)
    return dict(max_angle=ang.max(1), mean_angle=ang.mean(1), proj_err=np.sqrt((np.sin(np.radians(ang)) ** 2).sum(1)))
