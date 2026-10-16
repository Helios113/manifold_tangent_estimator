"""Shared graph utilities: kNN search and self-tuned graph Laplacians."""
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors


def knn_graph(X, k):
    """Indices (n, k+1) and distances of the k nearest neighbours; column 0 is the point itself."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, idx = nn.kneighbors(X)
    return dist, idx


def self_tuned_kernel(dist, idx, k_tune):
    """LDLE Algorithm 1 kernel: K_kk' = exp(-d^2 / (sigma_k sigma_k')), sigma_k = dist to k_tune-th NN.
    Returns a symmetric sparse (n, n) matrix with zero diagonal."""
    n, kp1 = idx.shape
    sigma = dist[:, k_tune]
    sigma = np.maximum(sigma, 1e-12)
    rows = np.repeat(np.arange(n), kp1 - 1)
    cols = idx[:, 1:].ravel()
    d = dist[:, 1:].ravel()
    vals = np.exp(-(d ** 2) / (sigma[rows] * sigma[cols]))
    K = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return K.maximum(K.T)  # symmetrise (union of kNN graphs)


def laplacian_eigs(K, n_eigs, kind="unnormalized"):
    """Smallest `n_eigs` eigenpairs of the graph Laplacian built from kernel K.

    kind='unnormalized': L = D - K                (used by LDLE)
    kind='random_walk' : L = I - D^-1 K           (eigenvectors phi = D^-1/2 u, u eig of sym. normalised)
    Returns (evals, evecs) with evecs[:, 0] the constant vector (eval ~ 0).
    """
    n = K.shape[0]
    deg = np.asarray(K.sum(1)).ravel()
    dense = n <= 2500
    if kind == "unnormalized":
        L = sp.diags(deg) - K
        if dense:
            w, V = sla.eigh(L.toarray(), subset_by_index=[0, n_eigs - 1])
        else:
            w, V = spla.eigsh(L.tocsc(), k=n_eigs, sigma=-1e-4, which="LM")
    elif kind == "random_walk":
        dm = 1.0 / np.sqrt(deg)
        S = sp.diags(dm) @ K @ sp.diags(dm)
        if dense:
            w, U = sla.eigh(np.eye(n) - S.toarray(), subset_by_index=[0, n_eigs - 1])
        else:
            mu, U = spla.eigsh(S.tocsc(), k=n_eigs, which="LA")
            w = 1 - mu
        V = U * dm[:, None]
    else:
        raise ValueError(kind)
    o = np.argsort(w)
    return w[o], V[:, o]
