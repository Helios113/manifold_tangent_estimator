"""Tangent spaces from LDLE (Kohli, Cloninger, Mishne, 'LDLE: Low Distortion Local Eigenmaps').

LDLE builds, for every point x_k, a d-dimensional local parameterisation Phi_k = (gamma_ki phi_i)_{i in I_k}
from d low-frequency Laplacian eigenvectors chosen so that their gradients at x_k are (a) nearly orthogonal
and (b) of similar scale (Algorithm 2, BiLipschitz-Local-Parameterization). To a first order Phi_k is a
projection onto the tangent space, so the tangent space is the span of the chosen eigenvectors' gradients.

Implementation notes
 * The selection (Eqs. 26-45) uses the finite-sum local-correlation estimate  A_kij = G_k^T((phi_i-phi_ik)(phi_j-phi_jk)) / 2t_k.
 * The *ambient* gradient of phi_i is obtained from the same estimator applied to the coordinate
   functions (Remark 3): (grad phi_i)_a = A_k(phi_i, x^a) = (2 t_k)^-1 sum_k' G_kk' (phi_i(k')-phi_i(k)) (x_k'-x_k)_a.
 * The final basis is the orthonormalised (QR) set of the d selected ambient gradients.
 * Distortion post-processing / clustering / global alignment of LDLE are *not* needed for tangent spaces.
"""
import numpy as np
from scipy.stats import chi2
from .graph import knn_graph, self_tuned_kernel, laplacian_eigs


def ldle(X, d, k=20, k_tune=None, N=100, klv=25, p_chi=0.99, tau=50.0, delta=0.9,
         idx=None, dist=None, return_info=False):
    n, p = X.shape
    if idx is None:
        dist, idx = knn_graph(X, max(k, klv))
    k_tune = k_tune or max(2, k // 2)
    kk = min(k, idx.shape[1] - 1)
    K = self_tuned_kernel(dist[:, : kk + 1], idx[:, : kk + 1], min(k_tune, kk))
    N = min(N, n - 2)
    evals, Phi = laplacian_eigs(K, N + 1, kind="unnormalized")
    Phi, evals = Phi[:, 1:], evals[1:]            # drop the constant eigenvector -> (n, N), ascending eval
    klv = min(klv, idx.shape[1] - 1)

    chi = chi2.ppf(p_chi, d)
    Q = np.zeros((n, p, d))
    for k_ in range(n):
        nb = idx[k_, : klv + 1]                   # U_k : x_k and its klv nearest neighbours
        eps = dist[k_, klv]
        tk = eps ** 2 / (2 * chi)
        w = np.exp(-dist[k_, : klv + 1] ** 2 / (4 * tk))
        w = w / w.sum()
        w[0] = 0.0                                # G_kk = 0 (Eq. 27); denominator still includes x_k
        # note: w was normalised including x_k, exactly as Eq. (27)
        D = Phi[nb] - Phi[k_]                     # (|U|, N)
        A = (D * w[:, None]).T @ D / (2 * tk)     # (N, N)  A_kij
        gamma = 1.0 / np.sqrt((Phi[nb] ** 2).mean(0))
        diagA = np.diag(A)
        S = np.where(diagA >= np.percentile(diagA, tau))[0]   # candidate set S_k (Eq. 31)
        # steps 2-3: first eigenvector
        r1 = S[0]                                 # lowest-frequency candidate (evals ascending)
        a1 = np.abs(gamma[S] * A[S, r1])
        i1 = S[np.argmax(a1 >= delta * a1.max())]
        sel = [i1]
        for s in range(2, d + 1):                 # steps 4-5
            V = A[np.ix_(sel, sel)]
            C = A[sel][:, S]                      # (s-1, |S|)
            H = A[np.ix_(S, S)] - C.T @ np.linalg.solve(V + 1e-12 * np.eye(len(sel)), C)   # H^s_kij on S
            Hd = np.diag(H)
            theta = np.percentile(Hd, tau)
            cand = np.where(Hd >= theta)[0]
            rs = cand[0]
            a = np.abs(gamma[S] * H[:, rs])
            ok = np.where(a >= delta * a.max())[0]
            sel.append(S[ok[0]])
        # ambient gradients of the selected eigenvectors
        Xd = X[nb] - X[k_]                        # (|U|, p)
        Gsel = (D[:, sel] * w[:, None]).T @ Xd / (2 * tk)      # (d, p)
        q, _ = np.linalg.qr(Gsel.T)               # (p, d)
        Q[k_] = q
    return (Q, dict(evals=evals)) if return_info else Q


def ldle_spectrum(X, k=20, k_tune=None, N=100, klv=25, p_chi=0.99, d_nominal=3, idx=None, dist=None):
    """Per-point singular values of the gamma-scaled ambient gradients of the N lowest LDLE eigenvectors.

    LDLE itself takes d as an input (it only appears in chi2inv(p, d) and in the number of selected
    eigenvectors); this is the natural parameter-free analogue used for *intrinsic dimension estimation*:
    the gradients of Laplacian eigenfunctions span the tangent space, so the numerical rank of
    [gamma_ki grad phi_i(x_k)]_i is the local dimension. d_nominal only sets the neighbourhood bandwidth t_k.
    """
    n, p = X.shape
    if idx is None:
        dist, idx = knn_graph(X, max(k, klv))
    k_tune = k_tune or max(2, k // 2)
    kk = min(k, idx.shape[1] - 1)
    K = self_tuned_kernel(dist[:, : kk + 1], idx[:, : kk + 1], min(k_tune, kk))
    N = min(N, n - 2)
    _, Phi = laplacian_eigs(K, N + 1, kind="unnormalized")
    Phi = Phi[:, 1:]
    klv = min(klv, idx.shape[1] - 1)
    chi = chi2.ppf(p_chi, d_nominal)
    r = min(N, p, klv)
    SV = np.zeros((n, r))
    for k_ in range(n):
        nb = idx[k_, : klv + 1]
        tk = dist[k_, klv] ** 2 / (2 * chi)
        w = np.exp(-dist[k_, : klv + 1] ** 2 / (4 * tk)); w /= w.sum(); w[0] = 0.0
        D = Phi[nb] - Phi[k_]
        gamma = 1.0 / np.sqrt((Phi[nb] ** 2).mean(0))
        Gall = (D * w[:, None] * gamma).T @ (X[nb] - X[k_]) / (2 * tk)    # (N, p)
        SV[k_] = np.linalg.svd(Gall, compute_uv=False)[:r]
    return SV
