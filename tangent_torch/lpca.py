"""Local PCA (GPU)."""
import torch
from .common import *


class LPCA:
    name = "lpca"

    def __init__(self, d, k=20):
        self.d, self.k = d, k

    def fit(self, X, nn=None, cache=None):
        self.X = X
        self.nn = nn if nn is not None else knn_train(X, self.k)
        return self

    def _tangent(self, C, idx):
        """C: (B,p) centres, idx: (B,k) training neighbours -> (B,p,d)."""
        P = torch.cat([C[:, None, :], self.X[idx]], 1)         # patch = centre + k neighbours
        P = P - P.mean(1, keepdim=True)
        return top_left(P.transpose(1, 2), self.d)

    def tangents_train(self):
        n, p = self.X.shape
        out = torch.empty(n, p, self.d, dtype=DT, device=DEV)
        ch = auto_chunk((self.k + 1) * p * 8 * 4 + p * p * 8 * 3)
        for s in chunks(n, ch):
            out[s] = self._tangent(self.X[s], self.nn[1][s, : self.k])
        return out

    def tangents_query(self, Y, cache=None):
        _, idx = knn_query(self.X, Y, self.k)
        out = torch.empty(len(Y), Y.shape[1], self.d, dtype=DT, device=DEV)
        ch = auto_chunk((self.k + 1) * Y.shape[1] * 8 * 4 + Y.shape[1] ** 2 * 8 * 3)
        for s in chunks(len(Y), ch):
            out[s] = self._tangent(Y[s], idx[s])
        return out

    def state_nbytes(self):
        return 0           # nothing beyond the data (a spatial index could be added; brute-force kNN is used here)
