"""Local polynomial tangent estimator of Aamari & Levrard (Ann. Stat. 2019, eq. (2)) on the GPU.

Around a point y, with Z_i = X_i - y for the training points in the ball B(y, h) (at most `k` nearest), find a rank-d orthogonal projector pi and symmetric tensors
T_2..T_{order-1} minimising
    sum_i || Z_i - pi Z_i - sum_s T_s((pi Z_i)^{(x)s}) ||^2 ,
and return im(pi); the second-order tensor T_2 (m >= 3) estimates the second fundamental form composed with the tangent projection.  order = 2 has no
tensors: this is (non-centred) local PCA.  The non-convex problem is solved in the paper's two-stage way: local PCA gives a basis B, then `n_iter` rounds of
    U = Z B,  N = Z - U B^T,  ridge regression of N on the monomials of U of degree 1..order-1,  B <- orth(B + G^T)   (G = fitted linear term: the tilt).
Coordinates are scaled by h (the ball radius, or the distance to the farthest neighbour), so the ridge `reg` (relative to the mean diagonal of the Gram
matrix) is dimensionless; it replaces the paper's constraint ||T_s||_op <= t.  The regression is solved in the primal (monomial features, weights such that
the Gram matrix is sum_s (u.u')^s) when there are fewer monomials than neighbours, and in the dual (m x m kernel, cost independent of d) otherwise.
Output has the dtype of the data."""
import itertools, math
import torch
from .common import *


def n_monomials(d, order):
    """Number of coefficients per output coordinate of the tensors T_2..T_{order-1}: sum_{s=2}^{order-1} C(d+s-1, s)  (the paper's p(k, m))."""
    return sum(math.comb(d + s - 1, s) for s in range(2, order))


class LocalPoly:
    name = "lpoly"

    def __init__(self, d, k=30, order=3, n_iter=3, reg=1e-6, radius=None):
        self.d, self.k, self.order, self.n_iter, self.reg, self.radius = d, k, order, n_iter, reg, radius
        self.q = d + n_monomials(d, order) if order > 2 else 0
        self._mono = None

    def fit(self, X, nn=None):
        self.X = X
        self.nn = nn if nn is not None else knn_train(X, self.k)
        return self

    # ------------------------------------------------------------------------------------------------------------------ monomials
    def _monomials(self, dev):
        if self._mono is None:
            self._mono = []
            for s in range(1, self.order):
                idx = list(itertools.combinations_with_replacement(range(self.d), s))
                w = [math.sqrt(math.factorial(s) / math.prod(math.factorial(i_.count(a)) for a in set(i_))) for i_ in idx]
                self._mono.append((s, torch.tensor(idx, device=dev), torch.tensor(w, dtype=torch.float64, device=dev), idx))
        return self._mono

    def _features(self, Us):
        """(B,m,d) -> (B,m,q) monomials of degree 1..order-1 with weights sqrt(multinomial), so that F F^T = sum_s (Us Us^T)^s."""
        cols = []
        for s, idx, w, _ in self._monomials(Us.device):
            cols.append(Us[:, :, idx].prod(-1) * w)
        return torch.cat(cols, -1)

    # ------------------------------------------------------------------------------------------------------------------ regression
    def _regress(self, Us, N):
        """Ridge regression of N (B,m,p) on the monomials of Us (B,m,d). Returns (W, alpha): the primal coefficients W (B,q,p) or the dual weights alpha (B,m,p)."""
        m = Us.shape[1]
        if self.q < 0.7 * m:
            F = self._features(Us)
            A = F.transpose(1, 2) @ F
            lam = self.reg * A.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] + 1e-300
            w, V = torch.linalg.eigh(A)
            W = V @ ((V.transpose(1, 2) @ (F.transpose(1, 2) @ N)) / (w.clamp_min(0).unsqueeze(2) + lam).clamp_min(1e-300))
            return W, None
        S = Us @ Us.transpose(1, 2)
        K = sum(S ** i for i in range(1, self.order))
        lam = self.reg * K.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] + 1e-300
        w, V = torch.linalg.eigh(K)
        return None, V @ ((V.transpose(1, 2) @ N) / (w.clamp_min(0).unsqueeze(2) + lam).clamp_min(1e-300))

    def _tilt_and_T2(self, Us, h, W, alpha, want_T2):
        """Tilt G (B,d,p) (original units) and, if wanted, T_2 (B,p,d,d) from the regression result."""
        B, m, d = Us.shape
        if W is not None:
            G = W[:, :d, :] / h
            T2 = None
            if want_T2 and self.order > 2:
                _, _, w2, lst = self._monomials(Us.device)[1]
                T2 = torch.zeros(B, W.shape[2], d, d, dtype=W.dtype, device=W.device)
                for c, (a, b) in enumerate(lst):
                    v = W[:, d + c, :] * w2[c] / (1.0 if a == b else 2.0) / h[:, :, 0] ** 2          # sum_{a,b} T_ab u_a u_b: off-diagonal entries are counted twice
                    T2[:, :, a, b] += v
                    if a != b:
                        T2[:, :, b, a] += v
            return G, T2
        G = (Us.transpose(1, 2) @ alpha) / h
        T2 = torch.einsum("bmp,bma,bmc->bpac", alpha, Us, Us) / h[:, :, :, None] ** 2 if want_T2 and self.order > 2 else None
        return G, T2

    # ------------------------------------------------------------------------------------------------------------------ estimator
    def _tangent(self, C, dist, idx, want_T2=False):
        """C: (B,p) centres, dist/idx: (B,k) neighbours -> bases (B,p,d) [, T_2 (B,p,d,d)]."""
        Z = (self.X[idx] - C[:, None, :]).double()                            # (B,k,p), centred at the query point (the paper's model has no intercept)
        if self.radius is not None:                                           # ball of radius h; points whose ball holds fewer than max(2d+2, #monomials+1) points use the radius of that many neighbours
            mincnt = min(self.k, max(2 * self.d + 2, self.q + 1))
            r = torch.clamp(dist[:, mincnt - 1], min=float(self.radius))
            Z = Z * (dist <= r[:, None]).unsqueeze(2)
            h = r.double()[:, None, None]
        else:
            h = Z.norm(dim=2).amax(1).clamp_min(1e-30)[:, None, None]
        Bm = top_left(Z.transpose(1, 2), self.d)                              # non-centred local PCA
        T2 = None
        if self.order > 2:
            for it in range(self.n_iter + (1 if want_T2 else 0)):
                U = Z @ Bm
                N = Z - U @ Bm.transpose(1, 2)
                Us = U / h
                W, alpha = self._regress(Us, N)
                G, T2 = self._tilt_and_T2(Us, h, W, alpha, want_T2 and it == self.n_iter)
                if it < self.n_iter:
                    M = Bm + G.transpose(1, 2)                                # orthonormalise via the d x d Gram matrix (batched QR is not available on every GPU build)
                    w, V = torch.linalg.eigh(M.transpose(1, 2) @ M)
                    Bm = M @ V / w.clamp_min(1e-300).sqrt().unsqueeze(1)
        Bm = Bm.to(self.X.dtype)
        return (Bm, T2) if want_T2 else Bm

    def _chunk(self):
        m, p = self.k, self.X.shape[1]
        return auto_chunk(m * p * 8 * 8 + max(m * m, self.q * m + self.q * self.q) * 8 * 6 + p * self.d * self.d * 8 * 2, lo=1)

    def _run(self, C, dist, idx, want_T2=False):
        n, p = C.shape
        out = torch.empty(n, p, self.d, dtype=self.X.dtype, device=DEV)
        out2 = torch.empty(n, p, self.d, self.d, dtype=self.X.dtype, device=DEV) if want_T2 and self.order > 2 else None
        for s in chunks(n, self._chunk()):
            r = self._tangent(C[s], dist[s], idx[s], want_T2=out2 is not None)
            if out2 is None:
                out[s] = r
            else:
                out[s], out2[s] = r[0], r[1].to(out2.dtype)
        return (out, out2) if want_T2 else out

    def tangents_train(self):
        return self._run(self.X, self.nn[0][:, : self.k], self.nn[1][:, : self.k])

    def tangents_at(self, ids, want_T2=False):
        """Tangents (and T_2) at the training points with indices ids, using the same neighbourhoods as tangents_train."""
        return self._run(self.X[ids], self.nn[0][ids, : self.k], self.nn[1][ids, : self.k], want_T2)

    def ball_counts(self, ids):
        """Number of training points in the ball B(X_j, radius) for j in ids (capped at k)."""
        return (self.nn[0][ids, : self.k] <= self.radius).sum(1)

    def tangents_query(self, Y):
        dist, idx = knn_query(self.X, Y, self.k)
        return self._run(Y, dist, idx)

    def state_nbytes(self):
        return 0
