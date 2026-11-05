"""Analytic k-sphere in R^(k+1) (any intrinsic dimension k, default the 2-sphere in R^3) with exact tangent spaces, plus a sampler of points on it (experiment 4 of PLAN.md).

Use from other scripts:  Sphere(k).sample(n, generator, noise=0.0) -> (n, k+1) points (optionally with Gaussian noise);  tangent_basis(p) / tangent_projector(p) give the tangent space at any p.
Running this file only checks the geometry and plots it; no points are saved."""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "sphere"


class QuadraticEmbedding:
    """Embedding of R^in_dim into R^D:  f(x) = Q [x ; x_1^2 ; 0 ; ... ; 0],  Q a random D x D orthogonal matrix (D >= in_dim + 1).

    The first in_dim coordinates are x itself, so f is injective and an immersion with the analytic Jacobian J(x) = Q [I ; 2 x_1 e_1^T ; 0]."""

    def __init__(self, ambient_dim, generator=None, dtype=torch.float32, device=None, in_dim=3):
        assert ambient_dim >= in_dim + 1, "need room for x and x_1^2"
        self.in_dim, self.ambient_dim = in_dim, ambient_dim
        self.Q = torch.linalg.qr(torch.randn(ambient_dim, ambient_dim, generator=generator, dtype=dtype, device=device))[0]

    def __call__(self, x):
        """(n, in_dim) -> (n, D)."""
        pad = x.new_zeros(len(x), self.ambient_dim - self.in_dim - 1)
        return torch.cat([x, x[:, :1] ** 2, pad], dim=1) @ self.Q.T

    def jacobian(self, x):
        """(n, in_dim) -> (n, D, in_dim)."""
        n, d = len(x), self.in_dim
        J = x.new_zeros(n, self.ambient_dim, d)
        J[:, :d] = torch.eye(d, dtype=x.dtype, device=x.device)
        J[:, d, 0] = 2 * x[:, 0]
        return self.Q @ J


class Torus:
    """Unit 2-torus (doughnut) in R^4: {(cos θ, sin θ, cos φ, sin φ) : θ, φ ∈ [0, 2π)}.
    
    This is a 2-dimensional manifold embedded in 4-dimensional space.
    The tangent space at any point is spanned by the partial derivatives w.r.t. θ and φ.
    Points have shape (n, 4)."""

    def __init__(self, k=2):
        assert k == 2, "Only 2D torus is currently implemented"
        self.k, self.D = k, 4  # 2D manifold in R^4

    def sample(self, n, generator=None, dtype=torch.float32, device=None, noise=0.0):
        """n points uniform on the torus, shape (n, 4).
        noise > 0 adds N(0, noise^2 I) to every point afterwards (no renormalisation)."""
        θ = 2 * np.pi * torch.rand(n, 1, generator=generator, dtype=dtype, device=device)
        φ = 2 * np.pi * torch.rand(n, 1, generator=generator, dtype=dtype, device=device)
        p = torch.cat([torch.cos(θ), torch.sin(θ), torch.cos(φ), torch.sin(φ)], dim=1)
        if noise > 0:
            p = p + noise * torch.randn(n, self.D, generator=generator, dtype=dtype, device=device)
        return p

    def tangent_projector(self, p):
        """Orthogonal projector onto the tangent space at p: shape (n, D, D)."""
        # For torus, we need to project onto the tangent space
        # The tangent space is orthogonal to both normal vectors
        # Normal 1: (cos θ, sin θ, 0, 0) - radial in first circle
        # Normal 2: (0, 0, cos φ, sin φ) - radial in second circle
        # So the projector is I - n1 n1^T - n2 n2^T
        n1 = p[:, :2]  # First circle normal
        n2 = p[:, 2:]  # Second circle normal
        P1 = torch.eye(self.D, dtype=p.dtype, device=p.device) - n1[:, :, None] * n1[:, None, :]
        P2 = torch.eye(self.D, dtype=p.dtype, device=p.device) - n2[:, :, None] * n2[:, None, :]
        # The tangent projector is the intersection: project onto both
        return P1 @ P2

    def tangent_basis(self, p):
        """Orthonormal basis of the tangent space at p, shape (n, D, k).
        For the 2D torus in R^4, we use:
        e1 = (-sin θ, cos θ, 0, 0) - tangent to first circle
        e2 = (0, 0, -sin φ, cos φ) - tangent to second circle
        These are already orthonormal."""
        e1 = torch.cat([-p[:, 1:2], p[:, 0:1], torch.zeros_like(p[:, :1]), torch.zeros_like(p[:, :1])], dim=1)
        e2 = torch.cat([torch.zeros_like(p[:, :1]), torch.zeros_like(p[:, :1]), -p[:, 3:4], p[:, 2:3]], dim=1)
        return torch.stack([e1, e2], dim=2)

    def embed_points(self, p, emb):
        """Points of the embedded torus: p (n, D) -> (n, ambient_dim)."""
        return emb(p)

    def embed_tangent_basis(self, p, B, emb):
        """Orthonormal tangent basis of the embedded torus at emb(p): (n, ambient_dim, k)."""
        return torch.linalg.qr(emb.jacobian(p) @ B)[0]


EMBEDDINGS = {"quadratic": QuadraticEmbedding, "torus": Torus}


class Sphere:
    """Unit k-sphere {p in R^D : |p| = 1} with D = k + 1: k = 2 (default) the sphere in R^3, Sphere(100) the 100-sphere used by series 8.
    The tangent space at p is p^perp, so it is known in closed form at any point. Points have shape (n, D)."""

    def __init__(self, k=2):
        self.k, self.D = k, k + 1

    def sample(self, n, generator=None, dtype=torch.float32, device=None, noise=0.0):
        """n points uniform on the sphere (normalised Gaussians), shape (n, D). noise > 0 adds N(0, noise^2 I) to every point afterwards (no renormalisation), so the points leave the sphere;
        noise = 0 draws nothing extra from the generator."""
        g = torch.randn(n, self.D, generator=generator, dtype=dtype, device=device)
        p = g / g.norm(dim=1, keepdim=True)
        if noise > 0: p = p + noise * torch.randn(n, self.D, generator=generator, dtype=dtype, device=device)
        return p

    def tangent_projector(self, p):
        """Orthogonal projector onto the tangent space at p: I - p p^T, shape (n, D, D)."""
        return torch.eye(self.D, dtype=p.dtype, device=p.device) - p[:, :, None] * p[:, None, :]

    def tangent_basis(self, p):
        """Orthonormal basis of the tangent space at p, shape (n, D, k). For D = 3: e1 = (a x p)/|a x p| with a the axis least aligned with p, e2 = p x e1.
        For other D: the last k columns of the complete QR factor of p, which are orthonormal and orthogonal to p."""
        if self.D != 3:
            return torch.linalg.qr(p[:, :, None], mode="complete")[0][:, :, 1:]
        a = torch.eye(self.D, dtype=p.dtype, device=p.device)[p.abs().argmin(dim=1)]
        e1 = torch.linalg.cross(a, p, dim=1)
        e1 = e1 / e1.norm(dim=1, keepdim=True)
        e2 = torch.linalg.cross(p, e1, dim=1)
        return torch.stack([e1, e2], dim=2)

    def embed_points(self, p, emb):
        """Points of the embedded sphere: p (n, D) -> (n, ambient_dim)."""
        return emb(p)

    def embed_tangent_basis(self, p, B, emb):
        """Orthonormal tangent basis of the embedded sphere at emb(p): (n, ambient_dim, k). Pushes B forward with the Jacobian, then orthonormalises."""
        return torch.linalg.qr(emb.jacobian(p) @ B)[0]


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    S = Sphere()
    X = S.sample(2000, gen, dtype=torch.float64)
    B = S.tangent_basis(X)
    P = S.tangent_projector(X)

    # Checks, recorded in the output: on the sphere, tangent frame orthonormal, tangent vectors orthogonal to p, B B^T = projector.
    chk = dict(radius_err=(X.norm(dim=1) - 1).abs().max().item(),
               orthonormal_err=(B.transpose(1, 2) @ B - torch.eye(S.k, dtype=B.dtype)).abs().max().item(),
               normal_err=torch.einsum("ni,nij->nj", X, B).abs().max().item(),
               projector_err=(B @ B.transpose(1, 2) - P).abs().max().item())
    print(chk)
    (OUT / "checks.txt").write_text("\n".join(f"{k}: {v:.3e}" for k, v in chk.items()) + "\n")

    Xn, Bn = X.numpy(), B.numpy()
    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(projection="3d")
    ax.scatter(*Xn.T, s=2, alpha=.4)
    for i in torch.randperm(len(X), generator=gen)[:6].tolist():
        for j, c in enumerate(("r", "g")):
            ax.quiver(*Xn[i], *Bn[i, :, j], length=.4, color=c)
    ax.set_box_aspect((1, 1, 1)); ax.set_title("sphere samples with tangent frames")
    plt.savefig(OUT / "sphere.png", dpi=150); plt.close()
