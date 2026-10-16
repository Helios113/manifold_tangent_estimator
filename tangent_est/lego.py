"""LEGO: Laplacian Eigenvector Gradient Orthogonalization (Kohli, Robertson, Mishne, Cloninger).

Algorithm 1 of arXiv:2510.02308: estimate gradients of the first m low-frequency eigenvectors of a graph
Laplacian by local regression (Tikhonov-regularised), low-pass them by projecting each gradient
component onto span{phi_1..phi_m0}, then take the top-d left singular vectors of the p x m gradient matrix.
"""
import numpy as np
from .graph import knn_graph, self_tuned_kernel, laplacian_eigs


def lego(X, d, k=20, m=20, m0=60, beta=0.5, k_tune=None, laplacian="random_walk",
         idx=None, dist=None, return_info=False):
    n, p = X.shape
    if idx is None:
        dist, idx = knn_graph(X, k)
    else:
        idx = idx[:, : k + 1]
    k_tune = k_tune or max(2, k // 2)

    # Scale so that diam(X) = 2 max_i ||X_i - mean|| < 1 (required by the eta_j rule, Eq. 5). Tangent
    # spaces are scale invariant.
    Xs = X - X.mean(0)
    Xs = Xs * (0.99 / (2 * np.linalg.norm(Xs, axis=1).max()))
    dist_s = dist * (0.99 / (2 * np.linalg.norm(X - X.mean(0), axis=1).max()))

    k = idx.shape[1] - 1
    K = self_tuned_kernel(dist_s, idx, min(k_tune, idx.shape[1] - 1))
    m0 = min(m0, n - 1)
    m = min(m, m0 - 1)
    evals, Phi = laplacian_eigs(K, m0, kind=laplacian)
    UPhi, _ = np.linalg.qr(Phi)                                   # (n, m0) orthonormal basis of range(Phi)

    # local regression: grad_i(X_j) = (Xj^T Xj + eta_j I)^-1 Xj^T (phi_i(N_j) - phi_i(X_j))
    Xc = Xs[idx[:, 1:]] - Xs[:, None, :]                          # (n, k, p) centred neighbours
    eta = (np.linalg.norm(Xc, axis=2) ** (2 * (1 + beta))).sum(1) # (n,)
    Phi_g = Phi[:, 1 : m + 1]                                     # skip the constant eigenvector
    dphi = Phi_g[idx[:, 1:]] - Phi_g[:, None, :]                  # (n, k, m)
    if k < p:   # dual form X^T (X X^T + eta I_k)^-1 : identical to (X^T X + eta I_p)^-1 X^T, cheaper when k < p
        Kg = np.einsum("nkp,njp->nkj", Xc, Xc) + eta[:, None, None] * np.eye(Xc.shape[1])
        G = np.einsum("nkp,nkm->npm", Xc, np.linalg.solve(Kg, dphi))
    else:
        Gram = np.einsum("nkp,nkq->npq", Xc, Xc) + eta[:, None, None] * np.eye(p)
        G = np.linalg.solve(Gram, np.einsum("nkp,nkm->npm", Xc, dphi))   # (n, p, m) raw local gradients

    # low-pass: project each gradient component (a function on the data) onto span(UPhi)
    Gf = G.transpose(1, 2, 0).reshape(p * m, n)                   # (p*m, n)
    Gf = (Gf @ UPhi) @ UPhi.T
    G = Gf.reshape(p, m, n).transpose(2, 0, 1)                    # (n, p, m)

    U, S, _ = np.linalg.svd(G, full_matrices=False)
    Q = U[:, :, :d]
    return (Q, dict(sv=S, evals=evals)) if return_info else Q
