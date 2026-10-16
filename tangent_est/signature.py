"""Signature (approximate defining) function of a point cloud, after P. Guidotti,
'Extracting manifold information from point clouds' (arXiv:2404.00427).

Given X = {x_1..x_m} in R^p the density Lambda solves the (ridge-regularised) collocation system (Eq. 3.3)

        (m*alpha*I + K) Lambda = m * 1 ,      K_ik = kappa(x_i - x_k)

and the signature function is   u(x) = (1/m) sum_k Lambda_k kappa(x - x_k)           (Eq. 3.5)

so that u ~ 1 on X (exactly for alpha = 0). Its level set {u = 1} is a smooth hypersurface interpolating the
cloud; its unit normal is nu = -grad u/|grad u| and the Jacobian of the normal field (shape operator) is
D nu = (I - nu nu^T) D^2 u (I - nu nu^T) / |grad u|  (Sec. 5).

Kernels (as functions of s = |z|^2 / bandwidth^2):
    'gauss'  : kappa = exp(-s)
    'laplace': kappa = exp(-sqrt(s + r))   (smoothed Laplace kernel L_r of Sec. 5; r > 0)
The bandwidth is the kernel scaling kappa(delta .) of Remark 3.6 (the paper's examples have it fixed to 1 by the
scale of the data).
"""
import numpy as np
import scipy.linalg as sla
from sklearn.neighbors import NearestNeighbors


class SignatureFunction:
    def __init__(self, X, bandwidth=None, k_bw=10, bw_mult=1.0, alpha=1e-6, kernel="gauss", r=1.0):
        self.X = np.asarray(X, float)
        m, p = self.X.shape
        if bandwidth is None:   # data-driven: bw_mult * mean distance to the k_bw-th nearest neighbour
            d = NearestNeighbors(n_neighbors=k_bw + 1).fit(self.X).kneighbors(self.X)[0][:, -1]
            bandwidth = bw_mult * d.mean()
        self.bw, self.alpha, self.kernel, self.r = float(bandwidth), alpha, kernel, r
        self.Xs = self.X / self.bw                          # work in units of the bandwidth
        self.sq = (self.Xs ** 2).sum(1)
        S = self._sqdist(self.Xs, self.Xs)
        K = self._k(S)
        K[np.diag_indices(m)] += m * alpha
        self.lam = sla.solve(K, m * np.ones(m), assume_a="pos")    # Lambda
        self.m = m

    # ---- kernel and its derivatives wrt s = |z|^2
    def _k(self, s):
        return np.exp(-s) if self.kernel == "gauss" else np.exp(-np.sqrt(s + self.r))

    def _k_derivs(self, s):
        """Return K, K', K'' (derivatives wrt s)."""
        if self.kernel == "gauss":
            K = np.exp(-s)
            return K, -K, K
        q = np.sqrt(s + self.r)
        K = np.exp(-q)
        return K, -K / (2 * q), K / (4 * q ** 2) + K / (4 * q ** 3)

    def _sqdist(self, A, B):
        S = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.maximum(S, 0.0)

    # ---- evaluation, in chunks of query points
    def u(self, Y, chunk=2000):
        Ys = np.asarray(Y, float) / self.bw
        out = np.empty(len(Ys))
        for i in range(0, len(Ys), chunk):
            out[i : i + chunk] = self._k(self._sqdist(Ys[i : i + chunk], self.Xs)) @ self.lam / self.m
        return out

    def grad(self, Y, chunk=2000):
        """grad u at the rows of Y (gradient wrt the original, unscaled coordinates)."""
        Ys = np.asarray(Y, float) / self.bw
        G = np.empty_like(Ys)
        for i in range(0, len(Ys), chunk):
            y = Ys[i : i + chunk]
            _, K1, _ = self._k_derivs(self._sqdist(y, self.Xs))
            W = K1 * self.lam[None, :]                          # lambda_k K'(s_ik)
            # sum_k W_ik (y_i - x_k) = y_i * sum_k W_ik - W X
            G[i : i + chunk] = 2 * (y * W.sum(1, keepdims=True) - W @ self.Xs) / self.m
        return G / self.bw

    def normal(self, Y, chunk=2000):
        """Unit normal nu = -grad u/|grad u| of the level set through each row of Y."""
        G = self.grad(Y, chunk)
        n = np.linalg.norm(G, axis=1, keepdims=True)
        return -G / np.maximum(n, 1e-300)

    def hessian(self, y):
        """D^2 u at one point y (in original coordinates), a (p, p) matrix."""
        ys = np.asarray(y, float) / self.bw
        s = ((ys[None, :] - self.Xs) ** 2).sum(1)
        _, K1, K2 = self._k_derivs(s)
        Z = ys[None, :] - self.Xs
        b = self.lam * K2
        H = 2 * (self.lam * K1).sum() * np.eye(len(ys)) + 4 * (Z * b[:, None]).T @ Z
        return H / self.m / self.bw ** 2

    def shape_operator(self, y):
        """(I - nu nu^T) D^2u (I - nu nu^T) / |grad u| at y; returns (S, nu). Symmetric, nu in its kernel."""
        g = self.grad(np.atleast_2d(y))[0]
        ng = np.linalg.norm(g)
        nu = -g / max(ng, 1e-300)
        P = np.eye(len(g)) - np.outer(nu, nu)
        return P @ self.hessian(y) @ P / max(ng, 1e-300), nu
