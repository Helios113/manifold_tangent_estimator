"""Tangent spaces from LDLE's local-parameterisation step (arXiv:2101.11055) on the GPU, batched over points.
See tangent_est/ldle.py for the algorithmic notes (finite-sum local-correlation estimate, ambient gradients, QR)."""
import torch
from scipy.stats import chi2
from .common import *


class LDLE:
    name = "ldle"

    def __init__(self, d, k=25, klv=20, N=100, k_tune=None, p_chi=0.99, tau=50.0, delta=0.9):
        self.d, self.k, self.klv, self.N = d, k, klv, N
        self.k_tune = min(k_tune or max(2, k // 2), k)
        self.chi = float(chi2.ppf(p_chi, d)); self.tau, self.delta = tau, delta

    def fit(self, X, nn=None, cache=None):
        n, p = X.shape
        self.X, self.n, self.p = X, n, p
        kk = max(self.k, self.klv)
        dist, idx = nn if nn is not None else knn_train(X, kk)
        self.nn = (dist[:, :kk], idx[:, :kk])
        N = min(self.N, n - 2); self.N_eff = N
        key = ("ldle", self.k, self.k_tune, N)
        if cache is not None and key in cache:
            sigma, evals, Phi = cache[key]
        else:
            K, sigma = self_tuned_dense(dist[:, : self.k], idx[:, : self.k], self.k_tune, n)
            L = torch.diag(K.sum(1)) - K
            del K
            w, V = smallest_eigs(L, N + 1)
            del L
            evals, Phi = w[1:], V[:, 1:]                              # drop the constant eigenvector
            if cache is not None:
                cache[key] = (sigma, evals, Phi)
        self.sigma, self.evals, self.Phi = sigma, evals, Phi
        return self

    def _tangent(self, C, phi_c, dist, idx):
        """C (B,p) centres, phi_c (B,N) eigenvector values at centres, dist/idx (B,>=klv) training neighbours."""
        klv, d, N = self.klv, self.d, self.N_eff
        dist, idx = dist[:, :klv], idx[:, :klv]
        B = len(C)
        t = dist[:, -1] ** 2 / (2 * self.chi)                          # t_k
        w = torch.exp(-dist ** 2 / (4 * t[:, None]))
        w = w / (1 + w.sum(1, keepdim=True))                           # normalised incl. the centre, centre weight 0
        Pn = self.Phi[idx]                                             # (B,klv,N)
        D = Pn - phi_c[:, None, :]
        A = (D * w[:, :, None]).transpose(1, 2) @ D / (2 * t)[:, None, None]    # (B,N,N)
        gamma = 1.0 / torch.sqrt((Pn ** 2).sum(1) / (klv + 1) + (phi_c ** 2) / (klv + 1))
        diag = A.diagonal(dim1=1, dim2=2)
        q = self.tau / 100.0
        S = diag >= torch.quantile(diag, q, dim=1, keepdim=True)
        ar = torch.arange(B, device=DEV)
        first = lambda mask: mask.to(torch.int8).argmax(1)

        def masked_max_pick(col):                                      # i = first idx in S with gamma|col| >= delta*max
            a = (gamma * col.abs()).masked_fill(~S, 0.0)
            return first(S & (a >= self.delta * a.max(1, keepdim=True).values))

        r1 = first(S)
        sel = [masked_max_pick(A[ar, :, r1])]
        for s in range(2, d + 1):
            idxs = torch.stack(sel, 1)                                 # (B,s-1)
            Asel = A[ar[:, None], idxs, :]                             # (B,s-1,N)
            V = torch.gather(Asel, 2, idxs[:, None, :].expand(-1, s - 1, -1))        # (B,s-1,s-1) = A[sel,sel]
            V = V + 1e-12 * torch.eye(s - 1, dtype=DT, device=DEV)
            H = A - Asel.transpose(1, 2) @ torch.linalg.solve(V, Asel)
            Hd = H.diagonal(dim1=1, dim2=2)
            theta = torch.nanquantile(Hd.masked_fill(~S, float("nan")), q, dim=1, keepdim=True)
            rs = first(S & (Hd >= theta))
            sel.append(masked_max_pick(H[ar, :, rs]))
        sel = torch.stack(sel, 1)                                      # (B,d)
        Dsel = torch.gather(D, 2, sel[:, None, :].expand(-1, klv, -1))             # (B,klv,d)
        Xd = self.X[idx] - C[:, None, :]
        Gsel = (Dsel * w[:, :, None]).transpose(1, 2) @ Xd / (2 * t)[:, None, None]  # (B,d,p)
        return torch.linalg.qr(Gsel.transpose(1, 2)).Q

    def _chunk(self):
        return auto_chunk((self.N_eff ** 2 * 6 + self.klv * self.N_eff * 4 + self.klv * self.p * 2) * 8, budget=4e9)

    def tangents_train(self):
        out = torch.empty(self.n, self.p, self.d, dtype=DT, device=DEV)
        dist, idx = self.nn
        for s in chunks(self.n, self._chunk()):
            out[s] = self._tangent(self.X[s], self.Phi[s], dist[s], idx[s])
        return out

    def tangents_query(self, Y, cache=None):
        """LDLE as published only yields tangents for the points of the cloud it is run on: new points are appended to
        the cloud and the whole method is rerun. Returns the tangents of the new points."""
        m = LDLE(self.d, k=self.k, klv=self.klv, N=self.N, k_tune=self.k_tune)
        m.chi, m.tau, m.delta = self.chi, self.tau, self.delta
        m.fit(torch.cat([self.X, Y]), cache=cache)
        return m.tangents_train()[-len(Y):]

    def state_nbytes(self):
        return nbytes(self.Phi, self.evals, self.sigma)   # fit artefacts (not reusable for new points)
