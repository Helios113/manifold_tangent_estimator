"""Manifold generators for the tangent-space benchmark protocol (plan.md, step 1): every manifold is F: box -> R^D with analytic Jacobian J(theta) (n, D, k),
analytic second derivatives, a sampler (uniform with respect to the volume measure, or with a known non-uniform density), and the three noise models.

  A  product of circles       F = Qa (r cos t_1, r sin t_1, ..., r cos t_k, r sin t_k),  Qa D x 2k orthonormal columns; reach = r
  B  Fourier embedding        F = A phi(theta), phi_j = (cos <w_j,theta>, sin <w_j,theta>), A = Q diag(c) (D x 2J), theta on the k-torus; w_j integer vectors
  C  with boundary            A or B restricted to the box theta in [0, pi]^k
  D  near-flat                F = Q [theta ; eps g(theta)],  theta in [0,1]^k,  g = C phi(2 pi f.theta) low-frequency Fourier features; eps is the curvature knob

All of A, B and D are in float64 numpy.  The tangent space at theta is col(J(theta)).  second fundamental form: II(v, w) = P_N D^2F[v, w]."""
import numpy as np


# ------------------------------------------------------------------------------------------------------------------------------ helpers
def rand_orth_cols(D, m, rng):
    """D x m with orthonormal columns (Haar)."""
    return np.linalg.qr(rng.normal(size=(D, m)))[0]


def tangent_frame(J):
    """(n,D,k) -> orthonormal tangent frames E (n,D,k) and R (n,k,k) with J = E R."""
    E, R = np.linalg.qr(J)
    return E, R


class Manifold:
    family = "?"
    k = D = None
    lo = hi = None                       # box of the parameters
    periodic = False

    # --- to be provided by the families
    def F(self, th): raise NotImplementedError
    def J(self, th): raise NotImplementedError
    def hess(self, th): raise NotImplementedError          # (n, D, k, k): D^2 F
    def d2(self, th, V): raise NotImplementedError         # (n, D): D^2 F[v, v] for directions V (n, k)
    def log_density_weight(self, th): return np.zeros(len(th))   # log sqrt(det J^T J) minus its max bound: set by sample_theta

    # --- common
    def volume_density(self, th):
        """sqrt(det(J^T J)) at th (chunked)."""
        out = np.empty(len(th))
        for s in range(0, len(th), 500):
            J = self.J(th[s:s + 500]); out[s:s + 500] = np.sqrt(np.linalg.det(np.swapaxes(J, 1, 2) @ J))
        return out

    def tilt(self, th):
        """Known non-uniform density factor (>0, <= 1.5): 1 + 0.5 cos(2 pi (th_1 - lo_1) / (hi_1 - lo_1))."""
        return 1.0 + 0.5 * np.cos(2 * np.pi * (th[:, 0] - self.lo[0]) / (self.hi[0] - self.lo[0]))

    def sample_theta(self, n, rng, uniform=True):
        """Parameters of n points with density proportional to sqrt(det J^T J) (uniform on M) -- or times `tilt` (known non-uniform) -- by rejection.
        Returns (theta, info) with info = dict(sampling=..., acceptance=...)."""
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        const_metric = getattr(self, "const_metric", False)
        if const_metric:
            bound_v = 1.0
        else:                                           # bound of the volume density from a pilot sample
            pil = lo + (hi - lo) * rng.uniform(size=(2000, self.k))
            vd = self.volume_density(pil); bound_v = 1.3 * vd.max()
        out, tried, acc = [], 0, 0
        while acc < n:
            m = max(2 * (n - acc), 200)
            th = lo + (hi - lo) * rng.uniform(size=(m, self.k))
            w = np.ones(m) if const_metric else self.volume_density(th) / bound_v
            if not uniform:
                w = w * self.tilt(th) / 1.5
            keep = rng.uniform(size=m) < w
            out.append(th[keep]); acc += keep.sum(); tried += m
        th = np.concatenate(out)[:n]
        return th, dict(sampling="uniform" if uniform else "nonuniform", acceptance=acc / tried)

    def boundary_mask(self, th, frac=0.05):
        """True where the point is within frac of the box extent of a facet (family C only)."""
        lo, hi = np.asarray(self.lo), np.asarray(self.hi)
        t = (th - lo) / (hi - lo)
        return ((t < frac) | (t > 1 - frac)).any(1)

    def has_boundary(self): return False

    # --- ground truth
    def tangent(self, th):
        """Orthonormal tangent bases (n, D, k)."""
        out = np.empty((len(th), self.D, self.k))
        for s in range(0, len(th), 500):
            out[s:s + 500] = tangent_frame(self.J(th[s:s + 500]))[0]
        return out

    def second_form(self, th):
        """II in the orthonormal tangent frame: (n, D, k, k), normal-valued: II(e_a, e_b) = P_N D^2F[R^-1 e_a, R^-1 e_b]. Returns (E, II)."""
        J = self.J(th); E, R = tangent_frame(J); H = self.hess(th)
        # project each H[:, :, i, j] onto the normal space
        EtH = np.einsum("nDa,nDij->naij", E, H)
        H = H - np.einsum("nDa,naij->nDij", E, EtH)
        Ri = np.linalg.inv(R)                                          # (n,k,k): v_a = Ri[:, :, a]
        II = np.einsum("nDij,nia,njb->nDab", H, Ri, Ri)
        return E, II

    def max_curvature(self, th, rng, n_dirs=200):
        """Largest normal curvature ||P_N D^2F[v,v]|| / ||J v||^2 over the points and n_dirs random directions per point (lower bound of the true maximum)."""
        J = self.J(th); E, R = tangent_frame(J)
        best = 0.0
        for _ in range(n_dirs // 10):
            u = rng.normal(size=(len(th), self.k, 10)); u /= np.linalg.norm(u, axis=1, keepdims=True)      # unit tangent coordinates
            v = np.linalg.solve(R, u)                                                                   # parameter directions with ||J v|| = 1
            for s_ in range(10):
                h = self.d2(th, v[:, :, s_])                                                            # (n,D)
                h = h - np.einsum("nDa,na->nD", E, np.einsum("nDa,nD->na", E, h))
                best = max(best, float(np.linalg.norm(h, axis=1).max()))
        return best

    def reach_estimate(self, th, X=None, chunk=64):
        """Federer's formula tau = inf_{p != q} ||p - q||^2 / (2 ||P_{N_p}(q - p)||) over the pairs of the sample th (an upper bound of the reach: covers the local
        curvature and the bottleneck without any threshold). float64, on the GPU if there is one."""
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = self.F(th) if X is None else X
        E = torch.as_tensor(self.tangent(th), device=dev); Xt = torch.as_tensor(X, device=dev)
        best = float("inf")
        for s in range(0, len(th), chunk):
            d = Xt[None, :, :] - Xt[s:s + chunk, None, :]                         # (c, n, D): q - p
            c = torch.einsum("cDk,cnD->cnk", E[s:s + chunk], d)
            nt2 = (d * d).sum(2) - (c * c).sum(2)                                   # ||P_N (q-p)||^2
            dn = nt2.clamp_min(0).sqrt(); d2 = (d * d).sum(2)
            r = d2 / (2 * dn)
            r[d2 < 1e-24] = float("inf")
            best = min(best, float(r.min()))
        return best

    def bottleneck(self, th, F=None, far_frac=None):
        """min ||F(t) - F(t')|| over sampled pairs whose parameters are far apart (periodic wrap-around respected): upper bound of the true bottleneck."""
        X = self.F(th) if F is None else F
        lo, hi = np.asarray(self.lo), np.asarray(self.hi); ext = hi - lo
        far_frac = far_frac or (0.25 if self.periodic else 0.4)
        best = np.inf
        for s in range(0, len(th), 500):
            d = np.abs(th[s:s + 500, None, :] - th[None, :, :])
            if self.periodic: d = np.minimum(d, ext - d)
            far = (d / ext).max(2) > far_frac                                              # far in at least one coordinate
            sq = ((X[s:s + 500, None, :] - X[None, :, :]) ** 2).sum(2)
            best = min(best, float(np.sqrt(np.where(far, sq, np.inf).min())))
        return best

    def diameter(self, th, F=None):
        X = self.F(th) if F is None else F
        m = min(len(X), 1500)
        d2 = ((X[:m, None, :] - X[None, :m, :]) ** 2).sum(2)
        return float(np.sqrt(d2.max()))


# ------------------------------------------------------------------------------------------------------------------------------ A
class ProductCircles(Manifold):
    family = "A"
    const_metric = True
    periodic = True

    def __init__(self, k, D, seed, r=1.0, hi=2 * np.pi):
        assert 2 * k <= D
        rng = np.random.default_rng(seed)
        self.k, self.D, self.r = k, D, r
        self.Qa = rand_orth_cols(D, 2 * k, rng)
        self.lo, self.hi = np.zeros(k), np.full(k, hi)
        self.reach_exact = r
        self.params = dict(r=r)

    def F(self, th):
        c = np.empty((len(th), 2 * self.k)); c[:, 0::2], c[:, 1::2] = self.r * np.cos(th), self.r * np.sin(th)
        return c @ self.Qa.T

    def J(self, th):
        n = len(th); dc = np.zeros((n, 2 * self.k, self.k)); j = np.arange(self.k)
        dc[:, 2 * j, j], dc[:, 2 * j + 1, j] = -self.r * np.sin(th), self.r * np.cos(th)
        return np.einsum("Dc,nck->nDk", self.Qa, dc)

    def hess(self, th):
        n = len(th); h = np.zeros((n, 2 * self.k, self.k, self.k)); j = np.arange(self.k)
        h[:, 2 * j, j, j], h[:, 2 * j + 1, j, j] = -self.r * np.cos(th), -self.r * np.sin(th)
        return np.einsum("Dc,ncij->nDij", self.Qa, h)

    def d2(self, th, V):
        c = np.empty((len(th), 2 * self.k)); c[:, 0::2], c[:, 1::2] = -self.r * np.cos(th) * V ** 2, -self.r * np.sin(th) * V ** 2
        return c @ self.Qa.T


# ------------------------------------------------------------------------------------------------------------------------------ B
class FourierEmbedding(Manifold):
    family = "B"
    const_metric = True
    periodic = True

    def __init__(self, k, D, seed, w=3, J=None, hi=2 * np.pi):
        rng = np.random.default_rng(seed)
        Jn = J or min(D // 2, 10 * k + 10)
        self.k, self.D, self.Jn = k, D, Jn
        om = np.zeros((Jn, k), dtype=int)
        om[:k] = np.eye(k, dtype=int)                                  # frequency-1 on every coordinate: makes the torus embedding injective
        for j in range(k, Jn):
            while True:
                v = rng.integers(-w, w + 1, size=k)
                if v.any(): break
            om[j] = v
        self.om = om.astype(float)
        c = np.full(Jn, 1 / np.sqrt(Jn))
        self.Qc = rand_orth_cols(D, 2 * Jn, rng) * np.concatenate([c, c])    # D x 2J
        self.lo, self.hi = np.zeros(k), np.full(k, hi)
        self.params = dict(w=w, J=Jn)
        G = (self.om.T * c ** 2) @ self.om                                # J^T J (constant)
        self.metric = G

    def _cs(self, th):
        a = th @ self.om.T
        return np.cos(a), np.sin(a)

    def F(self, th):
        c, s = self._cs(th)
        return np.concatenate([c, s], 1) @ self.Qc.T

    def J(self, th):
        c, s = self._cs(th)
        n = len(th); dphi = np.concatenate([-s[:, :, None] * self.om[None], c[:, :, None] * self.om[None]], 1)   # (n,2J,k)
        return np.einsum("Dc,nck->nDk", self.Qc, dphi)

    def hess(self, th):
        c, s = self._cs(th)
        oo = self.om[:, :, None] * self.om[:, None, :]                  # (J,k,k)
        h = np.concatenate([-c[:, :, None, None] * oo[None], -s[:, :, None, None] * oo[None]], 1)        # (n,2J,k,k)
        return np.einsum("Dc,ncij->nDij", self.Qc, h)

    def d2(self, th, V):
        c, s = self._cs(th); a = (V @ self.om.T) ** 2
        return np.concatenate([-c * a, -s * a], 1) @ self.Qc.T


# ------------------------------------------------------------------------------------------------------------------------------ C
class WithBoundary(Manifold):
    """Family A or B restricted to theta in [0, hi]^k (hi = pi by default): a patch with boundary."""
    family = "C"

    def __init__(self, base, hi=np.pi):
        self.base = base
        self.k, self.D = base.k, base.D
        self.lo, self.hi = np.zeros(base.k), np.full(base.k, hi)
        self.periodic = False
        self.const_metric = base.const_metric
        self.params = dict(base=base.family, **base.params)
        self.reach_exact = getattr(base, "reach_exact", None)

    def F(self, th): return self.base.F(th)
    def J(self, th): return self.base.J(th)
    def hess(self, th): return self.base.hess(th)
    def d2(self, th, V): return self.base.d2(th, V)
    def has_boundary(self): return True


# ------------------------------------------------------------------------------------------------------------------------------ D
class NearFlat(Manifold):
    family = "D"

    def __init__(self, k, D, seed, eps=0.1, n_freq=8):
        rng = np.random.default_rng(seed)
        self.k, self.D, self.eps, self.nf = k, D, eps, n_freq
        nq = min(2 * n_freq, D - k)                                   # dimension of the span of the normal part (D - k is small only in the D = 3 sanity case)
        self.Q = rand_orth_cols(D, k + nq, rng)
        self.Q1, Q2 = self.Q[:, :k], self.Q[:, k:]
        Cm = rng.normal(size=(nq, 2 * n_freq)) / np.sqrt(nq)
        self.M = Q2 @ Cm                                              # D x 2F: eps * M phi is the normal part (g = C phi in a random frame)
        f = np.zeros((n_freq, k))
        for j in range(n_freq):                                       # low frequencies: integer vectors in {-2..2}^k with at most 3 non-zeros
            while True:
                nz = rng.choice(k, size=min(3, k), replace=False)
                f[j, nz] = rng.integers(-2, 3, size=len(nz))
                if f[j].any(): break
                f[j] = 0
        self.f = f
        self.lo, self.hi = np.zeros(k), np.ones(k)
        self.params = dict(eps=eps, n_freq=n_freq)
        self.const_metric = False

    def set_eps(self, eps):
        self.eps = eps; self.params["eps"] = eps

    def _cs(self, th):
        a = 2 * np.pi * th @ self.f.T
        return np.cos(a), np.sin(a)

    def F(self, th):
        c, s = self._cs(th)
        return th @ self.Q1.T + self.eps * np.concatenate([c, s], 1) @ self.M.T

    def J(self, th):
        c, s = self._cs(th)
        n = len(th); dphi = 2 * np.pi * np.concatenate([-s[:, :, None] * self.f[None], c[:, :, None] * self.f[None]], 1)     # (n,2F,k)
        return self.Q1[None] + self.eps * np.einsum("Dc,nck->nDk", self.M, dphi)

    def hess(self, th):
        c, s = self._cs(th)
        ff = (2 * np.pi) ** 2 * self.f[:, :, None] * self.f[:, None, :]
        h = np.concatenate([-c[:, :, None, None] * ff[None], -s[:, :, None, None] * ff[None]], 1)
        return self.eps * np.einsum("Dc,ncij->nDij", self.M, h)

    def d2(self, th, V):
        c, s = self._cs(th); a = (2 * np.pi * V @ self.f.T) ** 2
        return self.eps * np.concatenate([-c * a, -s * a], 1) @ self.M.T

    def volume_density(self, th):
        """sqrt(det(J^T J)): J = Q1 + eps M dphi with Q1 orthogonal to M, so J^T J = I + eps^2 dphi^T (M^T M) dphi and det = det(I_2F + eps^2 (M^T M) dphi dphi^T)."""
        c, s = self._cs(th)
        dphi = 2 * np.pi * np.concatenate([-s[:, :, None] * self.f[None], c[:, :, None] * self.f[None]], 1)
        S = self.M.T @ self.M
        A = np.eye(2 * self.nf)[None] + self.eps ** 2 * (S[None] @ dphi) @ dphi.transpose(0, 2, 1)
        return np.sqrt(np.linalg.det(A))


# ------------------------------------------------------------------------------------------------------------------------------ S
class SphereAE(Manifold):
    """Family S (series 8): the unit k-sphere S^k in R^(k+1), embedded into R^D by the random Fourier-feature embedding of the autoencoder experiments
    (auto_encoder/sphere.py: f(p) = Q [sin(Wp + b); cos(Wp + b)]).  The 'parameter' th of a point is the point p of the sphere itself, shape (n, k+1).
    The tangent space at f(p) is col(J_f(p) B(p)), B(p) an orthonormal basis of p^perp: exact, analytic ground truth.  Uniform sampling only, no noise models,
    no second fundamental form / curvature (those entries of the diagnostics are nan)."""
    family = "S"
    reach_exact = None

    def __init__(self, k, D, seed):
        import importlib.util, pathlib, torch
        spec = importlib.util.spec_from_file_location("ae_sphere", pathlib.Path(__file__).resolve().parent.parent / "auto_encoder" / "sphere.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        self.k, self.D, self._torch = k, D, torch
        self.S = mod.Sphere(k + 1)
        self.emb = mod.FourierEmbedding(D, torch.Generator().manual_seed(seed), dtype=torch.float64, in_dim=k + 1)
        self.params = dict(embedding="fourier", freq=1.0)

    def _t(self, th): return self._torch.as_tensor(th, dtype=self._torch.float64)

    def F(self, th): return self.emb(self._t(th)).numpy()

    def J(self, th):
        p = self._t(th)
        return (self.emb.jacobian(p) @ self.S.tangent_basis(p)).numpy()

    def tangent(self, th):
        out = np.empty((len(th), self.D, self.k))
        for s in range(0, len(th), 500):
            p = self._t(th[s:s + 500])
            out[s:s + 500] = self.S.embed_tangent_basis(p, self.S.tangent_basis(p), self.emb).numpy()
        return out

    def sample_theta(self, n, rng, uniform=True):
        assert uniform, "family S: uniform sampling only"
        g = rng.normal(size=(n, self.k + 1))
        return g / np.linalg.norm(g, axis=1, keepdims=True), dict(sampling="uniform", acceptance=1.0)

    def max_curvature(self, th, rng, n_dirs=200): return float("nan")
    def bottleneck(self, th, F=None, far_frac=None): return float("nan")


def make_manifold(family, k, D, seed, **kw):
    """family in A, B, C (C: base = 'A' or 'B', default A), D."""
    if family == "A": return ProductCircles(k, D, seed, **{a: b for a, b in kw.items() if a in ("r",)})
    if family == "B": return FourierEmbedding(k, D, seed, **{a: b for a, b in kw.items() if a in ("w", "J")})
    if family == "C":
        base = kw.get("base", "A")
        return WithBoundary(make_manifold(base, k, D, seed, **kw))
    if family == "S": return SphereAE(k, D, seed)
    if family == "D": return NearFlat(k, D, seed, **{a: b for a, b in kw.items() if a in ("eps", "n_freq")})
    raise ValueError(family)


def eps_for_diam_over_tau(M, target, rng, n_pilot=300, iters=16):
    """Family D: choose eps by bisection (on log eps) so that the measured diam / tau_est equals target (tau_est = Federer's formula on one fixed pilot sample).
    diam / tau increases with eps."""
    th, _ = M.sample_theta(n_pilot, rng)
    e0 = M.eps

    def ratio(eps):
        M.set_eps(eps)
        X = M.F(th)
        return M.diameter(th, X) / M.reach_estimate(th, X)
    lo, hi = np.log(1e-7), np.log(20.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if ratio(np.exp(mid)) < target: lo = mid
        else: hi = mid
    eps = float(np.exp(0.5 * (lo + hi)))
    M.set_eps(e0)
    return eps


# ------------------------------------------------------------------------------------------------------------------------------ data + noise
def make_dataset(M, n, rng, uniform=True, noise="none", nu=0.0, chunk=500):
    """Sample n points, return dict(theta, X_clean, X (noisy), sampling info). nu = the norm of the noise: iid Gaussian with per-coordinate std nu/sqrt(D) ('iso'),
    uniform in the ball of radius nu of the normal space ('normal') or of the tangent space ('tangent'), always evaluated against the tangent at the clean point."""
    th, info = M.sample_theta(n, rng, uniform)
    Xc = M.F(th)
    X = Xc.copy()
    if noise == "iso" and nu > 0:
        X += rng.normal(size=X.shape) * (nu / np.sqrt(M.D))
    elif noise in ("normal", "tangent") and nu > 0:
        for s in range(0, n, chunk):
            E = tangent_frame(M.J(th[s:s + chunk]))[0]                    # (m,D,k)
            m = len(E)
            if noise == "tangent":
                g = rng.normal(size=(m, M.k)); g /= np.linalg.norm(g, axis=1, keepdims=True)
                r = nu * rng.uniform(size=(m, 1)) ** (1 / M.k)
                X[s:s + m] += np.einsum("nDk,nk->nD", E, r * g)
            else:
                g = rng.normal(size=(m, M.D)); g -= np.einsum("nDk,nk->nD", E, np.einsum("nDk,nD->nk", E, g))
                g /= np.linalg.norm(g, axis=1, keepdims=True)
                r = nu * rng.uniform(size=(m, 1)) ** (1 / (M.D - M.k))
                X[s:s + m] += r * g
    return dict(theta=th, X_clean=Xc, X=X, **info)
