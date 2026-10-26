"""Multiscale-SVD local PCA (in the spirit of Little, Maggioni & Rosasco, "Multiscale geometric methods for data sets I"): at every point the local covariance
is computed for a geometric list of neighbourhood sizes (scales); in the clean regime its d leading eigenvalues grow like r^2 (tangent), the others like r^4
(curvature) or stay at the noise floor, so the ratio lambda_d / lambda_{d+1} signals a scale at which the tangent space is resolved.  The scale used at a
point is the smallest one at which the ratio reaches `gap`; if none does, the scale with the largest ratio.  The tangent space is the top-d eigenspace
of the (mean-centred) patch at that scale.  `scales` are neighbour counts (each must exceed d + 1); output has the dtype of the data."""
import torch
from .common import *


class MultiscaleSVD:
    name = "msvd"

    def __init__(self, d, scales, gap=5.0):
        self.d, self.scales, self.gap = d, sorted(scales), gap
        self.k = max(self.scales)
        self.last_scale = None

    def fit(self, X, nn=None):
        self.X = X
        self.nn = nn if nn is not None else knn_train(X, self.k)
        return self

    def _eig(self, P):
        """P: (B,m,p) centred patches -> top-d basis (B,p,d), lambda_d, lambda_{d+1} (eigenvalues of P^T P)."""
        B, m, p = P.shape
        d = self.d
        if m <= p:
            w, V = torch.linalg.eigh(P @ P.transpose(1, 2))
            Vt = V[..., -d:].flip(-1); wt = w[..., -d:].flip(-1)
            return P.transpose(1, 2) @ Vt / wt.clamp_min(1e-300).sqrt().unsqueeze(1), w[:, -d], w[:, -d - 1].clamp_min(1e-300)
        w, V = torch.linalg.eigh(P.transpose(1, 2) @ P)
        return V[..., -d:].flip(-1), w[:, -d], w[:, -d - 1].clamp_min(1e-300)

    def _tangent(self, C, idx):
        Cd = C.double()
        bases, ratios = [], []
        for s in self.scales:
            P = torch.cat([Cd[:, None, :], self.X[idx[:, :s]].double()], 1)
            P = P - P.mean(1, keepdim=True)
            T, l1, l2 = self._eig(P)
            bases.append(T); ratios.append(l1 / l2)
        R = torch.stack(ratios, 1)                                           # (B, n_scales)
        ok = R >= self.gap
        first = torch.where(ok.any(1), ok.float().argmax(1), R.argmax(1))
        self.last_scale = first
        out = torch.stack(bases, 1)[torch.arange(len(C), device=C.device), first]
        return out.to(self.X.dtype)

    def _run(self, C, idx):
        n, p = C.shape
        out = torch.empty(n, p, self.d, dtype=self.X.dtype, device=DEV)
        sc = []
        ch = auto_chunk((self.k + 1) * p * 8 * 6 + min(self.k + 1, p) ** 2 * 8 * 4 * len(self.scales) + p * self.d * 8 * len(self.scales), lo=1)
        for s in chunks(n, ch):
            out[s] = self._tangent(C[s], idx[s]); sc.append(self.last_scale)
        self.last_scale = torch.cat(sc)
        return out

    def tangents_train(self):
        return self._run(self.X, self.nn[1][:, : self.k])

    def tangents_at(self, ids):
        return self._run(self.X[ids], self.nn[1][ids, : self.k])

    def tangents_query(self, Y):
        _, idx = knn_query(self.X, Y, self.k)
        return self._run(Y, idx)

    def state_nbytes(self):
        return 0
