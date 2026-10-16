"""Test manifolds with analytically known tangent spaces, embedded isometrically into R^N."""
import numpy as np
from sklearn.datasets import make_swiss_roll, make_s_curve


def _orth(J):
    q, _ = np.linalg.qr(J)
    return q


def circle(n, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    Y = np.c_[np.cos(t), np.sin(t)]
    T = np.stack([-np.sin(t), np.cos(t)], 1)[:, :, None]
    return Y, T


def trefoil(n, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    Y = np.c_[np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)]
    dY = np.c_[np.cos(t) + 4 * np.cos(2 * t), -np.sin(t) + 4 * np.sin(2 * t), -3 * np.cos(3 * t)]
    return Y, (dY / np.linalg.norm(dY, axis=1, keepdims=True))[:, :, None]


def _sphere(n, dim, rng):
    Y = rng.normal(size=(n, dim + 1))
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    P = np.eye(dim + 1) - Y[:, :, None] * Y[:, None, :]       # tangent projector
    # orthonormal tangent basis from the projector (dim leading eigenvectors)
    w, V = np.linalg.eigh(P)
    return Y, V[:, :, -dim:]


def sphere2(n, rng):
    return _sphere(n, 2, rng)


def torus3(n, rng, R=2.0, r=1.0):
    """Standard torus in R^3 with area-uniform sampling by rejection."""
    out = []
    while sum(len(o) for o in out) < n:
        u = rng.uniform(0, 2 * np.pi, 2 * n)
        v = rng.uniform(0, 2 * np.pi, 2 * n)
        keep = rng.uniform(0, R + r, 2 * n) < (R + r * np.cos(v))
        out.append(np.c_[u[keep], v[keep]])
    uv = np.concatenate(out)[:n]
    u, v = uv[:, 0], uv[:, 1]
    Y = np.c_[(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)]
    tu = np.c_[-np.sin(u), np.cos(u), np.zeros(n)]
    tv = np.c_[-np.sin(v) * np.cos(u), -np.sin(v) * np.sin(u), np.cos(v)]
    return Y, np.stack([tu, tv], 2)                           # already orthonormal


def clifford_torus(n, rng):
    """Flat torus S^1 x S^1 in R^4."""
    u, v = rng.uniform(0, 2 * np.pi, (2, n))
    Y = np.c_[np.cos(u), np.sin(u), np.cos(v), np.sin(v)] / np.sqrt(2)
    z = np.zeros(n)
    T = np.stack([np.c_[-np.sin(u), np.cos(u), z, z], np.c_[z, z, -np.sin(v), np.cos(v)]], 2)
    return Y, T


def swiss_roll(n, rng):
    """sklearn.datasets.make_swiss_roll; the generator returns the angle t, from which the tangent follows."""
    Y, t = make_swiss_roll(n, random_state=int(rng.integers(2**31)))
    dt = np.c_[np.cos(t) - t * np.sin(t), np.zeros(n), np.sin(t) + t * np.cos(t)]
    dh = np.c_[np.zeros(n), np.ones(n), np.zeros(n)]
    return Y, np.stack([dt / np.linalg.norm(dt, axis=1, keepdims=True), dh], 2)


def s_curve(n, rng):
    """sklearn.datasets.make_s_curve (x=sin t, y=h, z=sign(t)(cos t-1))."""
    Y, t = make_s_curve(n, random_state=int(rng.integers(2**31)))
    dt = np.c_[np.cos(t), np.zeros(n), -np.sign(t) * np.sin(t)]
    dh = np.c_[np.zeros(n), np.ones(n), np.zeros(n)]
    return Y, np.stack([dt / np.linalg.norm(dt, axis=1, keepdims=True), dh], 2)


def paraboloid_saddle(n, rng):
    """Monge patch z = x^2 - y^2 (negative curvature, boundary)."""
    x, y = rng.uniform(-1, 1, (2, n))
    Y = np.c_[x, y, x ** 2 - y ** 2]
    J = np.stack([np.c_[np.ones(n), np.zeros(n), 2 * x], np.c_[np.zeros(n), np.ones(n), -2 * y]], 2)
    return Y, np.stack([_orth(j) for j in J])


# name -> (sampler, intrinsic dim)
MANIFOLDS = {
    "circle":        (circle, 1),
    "trefoil":       (trefoil, 1),
    "sphere S2":     (sphere2, 2),
    "torus R3":      (torus3, 2),
    "clifford T2":   (clifford_torus, 2),
    "swiss roll":    (swiss_roll, 2),
    "s-curve":       (s_curve, 2),
    "saddle":        (paraboloid_saddle, 2),
}


def sample_embedded(name, n, N, sigma=0.0, noise="ambient", seed=0):
    """Sample n points from manifold `name`, isometrically embed into R^N by a random orthonormal frame,
    add Gaussian noise of per-coordinate std `sigma` and return (X_noisy, Q_true (n,N,d), d).

    noise='ambient'   : isotropic noise in all N dims (realistic).
    noise='normal'    : noise only orthogonal to the tangent space (the LEGO tubular model).
    The true tangent space is that of the *clean* point.
    """
    rng = np.random.default_rng(seed)
    sampler, d = MANIFOLDS[name]
    Y, T = sampler(n, rng)
    Y = Y - Y.mean(0)
    Y = Y / np.linalg.norm(Y, axis=1).max()      # normalise: all manifolds have max radius 1
    D = Y.shape[1]
    assert N >= D
    # fixed random isometry R^D -> R^N
    E = np.linalg.qr(rng.normal(size=(N, D)))[0]
    X = Y @ E.T
    Q = np.einsum("Nd,nde->nNe", E, T)
    eps = rng.normal(scale=sigma, size=(n, N))
    if noise == "normal":
        eps = eps - np.einsum("nNd,nd->nN", Q, np.einsum("nNd,nN->nd", Q, eps))
    return X + eps, Q, d
