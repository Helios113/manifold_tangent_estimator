"""LEGO: Laplacian Eigenvector Gradient Orthogonalization (arXiv:2510.02308) on the GPU.

Eigenvectors Phi (n x m0) of the random-walk Laplacian; local Tikhonov regression of the gradient of the first m
non-constant eigenvectors at every point; low-pass projection onto span(Phi), summarised by the coefficient tensor
C (p x m x m0):  grad_i(x_j) = C_i UPhi[j]^T  with Phi = UPhi R (QR).  Tangent = top-d left singular vectors."""
import torch
from .common import *


class LEGO:
    name = "lego"

    def __init__(self, d, k=15, m=20, m0=60, beta=0.5, k_tune=None):
        self.d, self.k, self.m, self.m0, self.beta = d, k, m, m0, beta
        self.k_tune = min(k_tune or max(2, k // 2), k)

    def fit(self, X, nn=None):
        n, p = X.shape
        m0 = min(self.m0, n - 1); m = min(self.m, m0 - 1)
        self.n, self.p, self.m_eff = n, p, m
        self.X = X
        self.mean = X.mean(0)
        Xc0 = X - self.mean
        self.scale = 0.99 / (2 * Xc0.norm(dim=1).max())           # diam < 1 normalisation (needed for eta_j)
        self.Xs = Xc0 * self.scale
        dist, idx = nn if nn is not None else knn_train(X, self.k)
        dist, idx = dist[:, : self.k], idx[:, : self.k]
        dist_s = dist * self.scale

        K, sigma = self_tuned_dense(dist_s, idx, self.k_tune, n)
        deg = K.sum(1); dm = deg.rsqrt()
        K.mul_(dm[:, None]).mul_(dm[None, :])                      # normalised adjacency S = D^-1/2 K D^-1/2
        mu, U = smallest_eigs(K, m0, largest=True)                 # largest mu = smallest Laplacian eigenvalue
        psi = U * dm[:, None]
        del K, U
        self.sigma, self.mu, self.Phi = sigma, mu, psi
        self.evals = 1 - mu
        self.UPhi, self.R = torch.linalg.qr(self.Phi)

        # local regression of the gradient of eigenvectors 1..m (skipping the constant)
        Phi_g = self.Phi[:, 1 : m + 1]
        Graw = torch.empty(p, m, n, dtype=DT, device=DEV)
        k = self.k
        ch = auto_chunk((k * p + k * k + p * m + k * m) * 8 * 3)
        eye_k = torch.eye(k, dtype=DT, device=DEV)
        for s in chunks(n, ch):
            Xc = self.Xs[idx[s]] - self.Xs[s][:, None, :]          # (B,k,p)
            eta = (Xc.norm(dim=2) ** (2 * (1 + self.beta))).sum(1)
            Kg = Xc @ Xc.transpose(1, 2) + eta[:, None, None] * eye_k
            dphi = Phi_g[idx[s]] - Phi_g[s][:, None, :]            # (B,k,m)
            G = Xc.transpose(1, 2) @ torch.linalg.solve(Kg, dphi)  # dual-form ridge solution (B,p,m)
            Graw[:, :, s] = G.permute(1, 2, 0)
        # low-pass: project every gradient component onto span(UPhi); only the (p, m, m0) coefficients are kept
        self.C = (Graw.reshape(p * m, n) @ self.UPhi).reshape(p, m, m0)
        del Graw
        return self

    def _tangent_from_UPhi(self, U):
        """U: (B, m0) rows of UPhi(x) -> tangent bases (B,p,d)."""
        G = torch.einsum("pmj,bj->bpm", self.C, U)
        return top_left(G, self.d)

    def tangents_train(self):
        out = torch.empty(self.n, self.p, self.d, dtype=DT, device=DEV)
        ch = auto_chunk(self.p * self.m_eff * 8 * 6)
        for s in chunks(self.n, ch):
            out[s] = self._tangent_from_UPhi(self.UPhi[s])
        return out

    def tangents_at(self, ids):
        return self._tangent_from_UPhi(self.UPhi[ids])

    def tangents_query(self, Y):
        """LEGO as published only yields tangents for the points of the cloud it is run on: new points are appended to
        the cloud and the whole method is rerun. Returns the tangents of the new points."""
        m = LEGO(self.d, k=self.k, m=self.m, m0=self.m0, beta=self.beta, k_tune=self.k_tune)
        m.fit(torch.cat([self.X, Y]))
        return m.tangents_train()[-len(Y):]

    def state_nbytes(self):
        return nbytes(self.Phi, self.mu, self.sigma, self.UPhi, self.C, self.mean)   # fit artefacts (not reusable for new points)
