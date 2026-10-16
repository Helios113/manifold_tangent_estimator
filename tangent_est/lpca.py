"""Local PCA tangent space estimation."""
import numpy as np
from .graph import knn_graph


def lpca(X, d, k=20, idx=None, dist=None, return_sv=False):
    """Tangent bases (n, p, d) from the top-d principal directions of each point's k-NN patch."""
    if idx is None:
        _, idx = knn_graph(X, k)
    else:
        idx = idx[:, : k + 1]
    P = X[idx]                                   # (n, k+1, p)
    P = P - P.mean(axis=1, keepdims=True)
    # right singular vectors of the patch = eigenvectors of the local covariance
    _, S, Vt = np.linalg.svd(P, full_matrices=False)
    Q = np.transpose(Vt[:, :d, :], (0, 2, 1))
    return (Q, S) if return_sv else Q
