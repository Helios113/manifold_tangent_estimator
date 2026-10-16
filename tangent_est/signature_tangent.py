"""Tangent spaces from the signature function of Guidotti (arXiv:2404.00427), using `signature.SignatureFunction`.

The level set {u = 1} is a hypersurface that contains the cloud. For a manifold of codimension > 1 it is a
thin tube around the manifold, so (Sec. 6.8 of the paper) its normal field turns quickly in the directions
normal to the manifold and hardly at all along tangent directions. Two ways to read the tangent space off it:

 * 'perturb'  (the paper's procedure): evaluate the implied normal at P random perturbations of x_i (translation
   by a uniform random distance in [0, rho*bandwidth] along a random direction). The normals span the normal
   space of the manifold (the numerical rank r of the p x P matrix of normals = codimension); the tangent space
   is the orthogonal complement, i.e. the d left singular vectors of smallest singular value. Needs P >= p-d.
 * 'shape'    (infinitesimal limit of the same idea, deterministic): eigendecompose the shape operator
   D nu(x_i) = (I-nu nu^T) D^2u (I-nu nu^T)/|grad u|. Its eigenvalue on nu is 0; normal directions carry the large
   tube curvatures, tangent directions the small ones. The tangent space is spanned by the d eigenvectors (other
   than nu) with smallest |eigenvalue|. Cost O(m p^2) per point.

Both also give a local dimension estimate (`signature_dimension`) from the singular/eigen spectrum.
"""
import numpy as np
from .signature import SignatureFunction


def signature_tangent(X, d, method="shape", sig=None, n_perturb=None, rho=0.3, seed=0, return_info=False, **sig_kw):
    """Tangent bases (n, p, d). `sig_kw` are forwarded to SignatureFunction (bandwidth/k_bw/bw_mult/alpha/kernel/r)."""
    n, p = X.shape
    sig = sig or SignatureFunction(X, **sig_kw)
    Q = np.empty((n, p, d))
    spec = None
    if method == "shape":
        spec = np.empty((n, p))
        for i in range(n):
            S, nu = sig.shape_operator(X[i])
            w, V = np.linalg.eigh(S)
            order = np.argsort(np.abs(w))            # one of the near-zero modes is nu itself: drop it
            ov = np.abs(V.T @ nu)
            keep = [j for j in order]
            drop = int(np.argmax(ov))
            keep = [j for j in keep if j != drop]
            Q[i] = V[:, keep[:d]]
            spec[i] = np.abs(w[keep + [drop]])[: p]
    elif method == "perturb":
        rng = np.random.default_rng(seed)
        P = n_perturb or max(2 * (p - d), 4 * d + 10)
        spec = np.empty((n, min(p, P + 1)))
        for i in range(n):
            dirs = rng.normal(size=(P, p)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            dist = rng.uniform(0, rho * sig.bw, size=(P, 1))
            Y = np.vstack([X[i], X[i] + dist * dirs])
            M = sig.normal(Y).T                      # (p, P+1) implied normals as columns
            U, S, _ = np.linalg.svd(M, full_matrices=True)
            Q[i] = U[:, p - d:] if P + 1 >= p - d else U[:, -d:]   # d directions least spanned by normals
            spec[i, : len(S)] = S[: spec.shape[1]]
    else:
        raise ValueError(method)
    return (Q, dict(spectrum=spec)) if return_info else Q
