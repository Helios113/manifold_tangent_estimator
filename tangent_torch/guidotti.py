"""Guidotti's signature function (arXiv:2404.00427) on the GPU: dense collocation system solved by Cholesky; tangent space from the
implied normals at random perturbations of each point (Sec. 6.8). Works at arbitrary query points (the signature is a function), so in- and out-of-sample
use the same code."""
import torch
from .common import *


class Guidotti:
    name = "guidotti"

    def __init__(self, d=None, bw_mult=1.5, alpha=1e-3, k_bw=10, kernel="gauss", r=1.0, n_perturb=None, rho=0.3, seed=0, dtype=None, bw_rms=None, bw=None):
        self.d, self.bw_mult, self.alpha, self.k_bw, self.kernel, self.r = d, bw_mult, alpha, k_bw, kernel, r
        self.n_perturb, self.rho, self.seed = n_perturb, rho, seed
        self.bw_rms = bw_rms                                 # if set: bandwidth = bw_rms x rms distance of the points from their mean (the paper's kernel has width 1 on unit-size data)
        self.bw_fixed = bw                                   # if set: bandwidth used as is (bw=1 is the paper's kernel exp(-|x|^2))
        self.dt = dtype or DT                                # float64 for the paper-faithful runs (tiny alpha)

    def fit(self, X, nn=None):
        X = X.to(self.dt)
        n, p = X.shape
        self.n, self.p = n, p
        dk = (nn[0][:, self.k_bw - 1] if nn is not None else knn_train(X, self.k_bw)[0][:, -1])
        self.bw = (float(self.bw_fixed) if self.bw_fixed is not None else float(self.bw_mult * dk.mean()) if self.bw_rms is None
                   else float(self.bw_rms * (X - X.mean(0)).pow(2).sum(1).mean().sqrt()))
        self.X, self.mu0 = X, X.mean(0)
        self.Xs = (X - self.mu0) / self.bw
        S = sqdist(self.Xs, self.Xs)
        K = self._k(S); del S
        K.diagonal().add_(n * self.alpha)
        Lc = torch.linalg.cholesky(K); del K
        self.lam = torch.cholesky_solve(n * torch.ones(n, 1, dtype=self.dt, device=DEV), Lc)[:, 0]
        del Lc
        self.d_given = self.d is not None
        if not self.d_given:                                 # d unknown: estimated from the gap in the normal spectrum (see _estimate_d)
            self.d = self._estimate_d()
        return self

    def _n_perturb(self):
        return self.n_perturb or (max(2 * (self.p - self.d), 4 * self.d + 10) if self.d_given else 2 * self.p)

    def _estimate_d(self, n_sub=100):
        """Intrinsic dimension from the spectrum of the perturbed normals (Sec. 6.7-6.8): at each of n_sub random training points the
        number of small eigenvalues below the largest log-gap of the ascending spectrum; the median over the points."""
        gen = torch.Generator(device=DEV).manual_seed(self.seed)
        Y = self.X[torch.randperm(self.n, generator=gen, device=DEV)[:n_sub]] if self.n > n_sub else self.X
        P = self._n_perturb()
        ch = auto_chunk(P * (self.n * 8 * 3 + self.p * 8 * 3), budget=3e9, lo=1, hi=1024)
        ds = []
        for s in chunks(len(Y), ch):
            w = self._eig(Y[s], gen)[0]
            lw = torch.log(w.clamp_min(w[:, -1:] * 1e-30))
            ds.append((lw[:, 1:] - lw[:, :-1]).argmax(1) + 1)
        return int(torch.cat(ds).double().median().item())

    def _k(self, s):
        return torch.exp(-s) if self.kernel == "gauss" else torch.exp(-torch.sqrt(s + self.r))

    def _derivs(self, s):
        if self.kernel == "gauss":
            K = torch.exp(-s); return K, -K, K
        q = torch.sqrt(s + self.r); K = torch.exp(-q)
        return K, -K / (2 * q), K / (4 * q ** 2) + K / (4 * q ** 3)

    def _normals(self, Y):
        """Implied unit normal nu = -grad u / |grad u| of the level set of the signature function at each row of Y (B,p)."""
        Ys = (Y - self.mu0) / self.bw
        _, K1, _ = self._derivs(sqdist(Ys, self.Xs))                    # (B,n)
        W1 = K1 * self.lam
        g = 2 * (Ys * W1.sum(1, keepdim=True) - W1 @ self.Xs) / self.n  # grad u (scaled coordinates)
        return -g / g.norm(dim=1, keepdim=True).clamp_min(1e-30)

    def _tangent(self, Y, gen):
        """Sec. 6.8 of the paper: the normals at P random perturbations of each point span the normal space of the manifold
        (they turn quickly in normal directions, hardly at all in tangent directions); the tangent space is the span of
        the d directions least covered by them (d smallest singular directions of the p x P matrix of normals)."""
        return self._eig(Y, gen)[1][..., : self.d]                      # smallest singular values = least spanned

    def _eig(self, Y, gen):
        B, p = Y.shape
        P = self._n_perturb()
        dirs = torch.randn(B, P, p, dtype=self.dt, device=DEV, generator=gen)
        dirs = dirs / dirs.norm(dim=2, keepdim=True)
        dist = torch.rand(B, P, 1, dtype=self.dt, device=DEV, generator=gen) * (self.rho * self.bw)
        nu = self._normals((Y[:, None, :] + dist * dirs).reshape(B * P, p)).reshape(B, P, p)
        return torch.linalg.eigh(nu.transpose(1, 2) @ nu)               # left singular vectors of the (p,P) normal matrix

    def _run(self, Y):
        Y = Y.to(self.dt)
        gen = torch.Generator(device=DEV).manual_seed(self.seed)
        out = torch.empty(len(Y), self.p, self.d, dtype=self.dt, device=DEV)
        P = self._n_perturb()
        ch = auto_chunk(P * (self.n * 8 * 3 + self.p * 8 * 3), budget=3e9, lo=1, hi=1024)
        for s in chunks(len(Y), ch):
            out[s] = self._tangent(Y[s], gen)
        return out

    def tangents_train(self):
        return self._run(self.X)

    def tangents_at(self, ids):
        return self._run(self.X[ids])

    def tangents_query(self, Y):
        return self._run(Y)

    def state_nbytes(self):
        return nbytes(self.lam, self.mu0)
