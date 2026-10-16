"""Guidotti's signature function (arXiv:2404.00427) on the GPU: dense collocation system solved by Cholesky, shape-operator
tangent extraction in batches. Works at arbitrary query points (the signature is a function), so in- and out-of-sample
use the same code."""
import torch
from .common import *


class Signature:
    name = "sig"

    def __init__(self, d, bw_mult=1.5, alpha=1e-3, k_bw=10, kernel="gauss", r=1.0):
        self.d, self.bw_mult, self.alpha, self.k_bw, self.kernel, self.r = d, bw_mult, alpha, k_bw, kernel, r

    def fit(self, X, nn=None, cache=None):
        n, p = X.shape
        self.n, self.p = n, p
        dk = (nn[0][:, self.k_bw - 1] if nn is not None else knn_train(X, self.k_bw)[0][:, -1])
        self.bw = float(self.bw_mult * dk.mean())
        self.X, self.mu0 = X, X.mean(0)
        self.Xs = (X - self.mu0) / self.bw
        S = sqdist(self.Xs, self.Xs)
        K = self._k(S); del S
        K.diagonal().add_(n * self.alpha)
        Lc = torch.linalg.cholesky(K); del K
        self.lam = torch.cholesky_solve(n * torch.ones(n, 1, dtype=DT, device=DEV), Lc)[:, 0]
        del Lc
        return self

    def _k(self, s):
        return torch.exp(-s) if self.kernel == "gauss" else torch.exp(-torch.sqrt(s + self.r))

    def _derivs(self, s):
        if self.kernel == "gauss":
            K = torch.exp(-s); return K, -K, K
        q = torch.sqrt(s + self.r); K = torch.exp(-q)
        return K, -K / (2 * q), K / (4 * q ** 2) + K / (4 * q ** 3)

    def _tangent(self, Y):
        Ys = (Y - self.mu0) / self.bw
        B, p = Ys.shape
        _, K1, K2 = self._derivs(sqdist(Ys, self.Xs))                  # (B,n)
        W1 = K1 * self.lam
        g = 2 * (Ys * W1.sum(1, keepdim=True) - W1 @ self.Xs) / self.n  # grad u (scaled coordinates)
        gn = g.norm(dim=1, keepdim=True).clamp_min(1e-30)
        nu = -g / gn
        Z = Ys[:, None, :] - self.Xs[None]                              # (B,n,p)
        b = self.lam * K2
        H = 2 * W1.sum(1)[:, None, None] * torch.eye(p, dtype=DT, device=DEV) + 4 * (Z * b[:, :, None]).transpose(1, 2) @ Z
        H = H / self.n
        Pm = torch.eye(p, dtype=DT, device=DEV) - nu[:, :, None] * nu[:, None, :]
        Sop = Pm @ H @ Pm / gn[:, :, None]
        w, V = torch.linalg.eigh(Sop)
        absw = w.abs()
        drop = (V * nu[:, :, None]).sum(1).abs().argmax(1)              # the eigenvector along nu (eigenvalue 0)
        absw[torch.arange(B, device=DEV), drop] = float("inf")
        pick = absw.topk(self.d, dim=1, largest=False).indices          # d flattest directions
        return torch.gather(V, 2, pick[:, None, :].expand(-1, p, -1))

    def _run(self, Y):
        out = torch.empty(len(Y), self.p, self.d, dtype=DT, device=DEV)
        ch = auto_chunk(self.n * self.p * 8 * 3 + self.n * 8 * 8 + self.p ** 2 * 8 * 8, budget=3e9, lo=1, hi=1024)
        for s in chunks(len(Y), ch):
            out[s] = self._tangent(Y[s])
        return out

    def tangents_train(self):
        return self._run(self.X)

    def tangents_query(self, Y, cache=None):
        return self._run(Y)

    def state_nbytes(self):
        return nbytes(self.lam, self.mu0)
