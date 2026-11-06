"""Analytic k-sphere and k-torus with exact tangent spaces, plus a sampler of points on them, and the embeddings used in the experiments (experiment 4 of PLAN.md).

Use from other scripts:  Sphere(k).sample(n, generator, noise=0.0) -> (n, k+1) points; Torus(k).sample(...) -> (n, D) points;  tangent_basis(p) / tangent_projector(p) give the tangent space at any p.
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


class FourierEmbedding:
    """Random Fourier feature embedding: f(x) = Q [cos(Wx + b); sin(Wx + b)], Q random D x 2J orthogonal, W random J x in_dim.
    This is the embedding used in family B of the tangent_est/generators.py, ported to torch."""

    def __init__(self, ambient_dim, generator=None, dtype=torch.float32, device=None, in_dim=3, J=None, w=3):
        rng = torch.Generator() if generator is None else generator
        self.in_dim, self.ambient_dim = in_dim, ambient_dim
        Jn = J or min(ambient_dim // 2, 10 * in_dim + 10)
        self.Jn = Jn
        # Random frequency matrix W: Jn x in_dim
        self.W = torch.randint(-w, w + 1, (Jn, in_dim), generator=rng, dtype=torch.float32, device=device)
        self.b = torch.rand(Jn, generator=rng, dtype=dtype, device=device) * 2 * np.pi
        # Random orthogonal matrix Q: D x 2*Jn
        self.Q = torch.linalg.qr(torch.randn(ambient_dim, 2 * Jn, generator=rng, dtype=dtype, device=device))[0]
        # Scale factors
        c = torch.full((Jn,), 1.0 / np.sqrt(Jn), dtype=dtype, device=device)
        self.scales = torch.cat([c, c])

    def __call__(self, x):
        """(n, in_dim) -> (n, D)."""
        # Compute Wx + b: (n, Jn)
        Wx = x @ self.W.T + self.b
        cos_Wx = torch.cos(Wx)
        sin_Wx = torch.sin(Wx)
        # Stack and scale: (n, 2*Jn)
        features = torch.cat([cos_Wx, sin_Wx], dim=1) * self.scales
        # Apply orthogonal matrix: (n, D)
        return features @ self.Q.T

    def jacobian(self, x):
        """(n, in_dim) -> (n, D, in_dim)."""
        n = len(x)
        Wx = x @ self.W.T + self.b
        cos_Wx = torch.cos(Wx)
        sin_Wx = torch.sin(Wx)
        # Derivative of features w.r.t. x: (n, 2*Jn, in_dim)
        # For cos(Wx + b): derivative is -sin(Wx + b) * W
        # For sin(Wx + b): derivative is cos(Wx + b) * W
        dfeatures = torch.zeros(n, 2 * self.Jn, self.in_dim, dtype=x.dtype, device=x.device)
        dfeatures[:, :self.Jn] = -sin_Wx.unsqueeze(2) * self.W.unsqueeze(0) * self.scales[:self.Jn].unsqueeze(0).unsqueeze(2)
        dfeatures[:, self.Jn:] = cos_Wx.unsqueeze(2) * self.W.unsqueeze(0) * self.scales[self.Jn:].unsqueeze(0).unsqueeze(2)
        # Apply Q: (n, 2*Jn) @ (2*Jn, D).T -> (n, D), but we need (n, D, in_dim)
        # So we do: (n, 2*Jn, in_dim) @ (2*Jn, D).T -> (n, D, in_dim)
        return torch.einsum('nji,Dj->nDi', dfeatures, self.Q)


EMBEDDINGS = {"quadratic": QuadraticEmbedding, "fourier": FourierEmbedding}


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


class Torus:
    """2-torus (doughnut) in R^3: the standard ring torus with major radius R and minor radius r,
    parameterized by
        x = (R + r cos φ) cos θ
        y = (R + r cos φ) sin θ
        z = r sin φ
    This is a 2-dimensional manifold embedded in 3-dimensional space. Points have shape (n, 3).

    For k != 2: a flat k-torus (product of k circles) in R^(2k), where each pair of coordinates
    (x_{2i}, x_{2i+1}) lies on the unit circle. Points have shape (n, 2k)."""

    def __init__(self, k=2, R=1.0, r=0.5):
        self.k, self.D = k, 3 if k == 2 else 2 * k
        self.R, self.r = R, r  # major / minor radius (only used for k=2)

    def sample(self, n, generator=None, dtype=torch.float32, device=None, noise=0.0):
        """n points on the torus, shape (n, D). noise > 0 adds N(0, noise^2 I) afterwards (no renormalisation)."""
        if self.k == 2:
            θ = 2 * np.pi * torch.rand(n, 1, generator=generator, dtype=dtype, device=device)
            φ = 2 * np.pi * torch.rand(n, 1, generator=generator, dtype=dtype, device=device)
            R, r = self.R, self.r
            p = torch.cat([
                (R + r * torch.cos(φ)) * torch.cos(θ),
                (R + r * torch.cos(φ)) * torch.sin(θ),
                r * torch.sin(φ),
            ], dim=1)
        else:
            angles = 2 * np.pi * torch.rand(n, self.k, generator=generator, dtype=dtype, device=device)
            p = torch.zeros(n, self.D, dtype=dtype, device=device)
            for i in range(self.k):
                p[:, 2*i] = torch.cos(angles[:, i])
                p[:, 2*i + 1] = torch.sin(angles[:, i])
        if noise > 0:
            p = p + noise * torch.randn(n, self.D, generator=generator, dtype=dtype, device=device)
        return p

    def tangent_projector(self, p):
        """Orthogonal projector onto the tangent space at p: shape (n, D, D)."""
        B = self.tangent_basis(p)
        return B @ B.transpose(1, 2)

    def tangent_basis(self, p):
        """Orthonormal basis of the tangent space at p, shape (n, D, k)."""
        if self.k == 2:
            R, r = self.R, self.r
            ρ = (p[:, 0]**2 + p[:, 1]**2).sqrt()
            cos_φ = (ρ - R) / r
            sin_φ = p[:, 2] / r
            θ = torch.atan2(p[:, 1], p[:, 0])

            # ∂/∂θ = (-ρ sin θ, ρ cos θ, 0)
            e1 = torch.zeros_like(p)
            e1[:, 0] = -ρ * torch.sin(θ)
            e1[:, 1] = ρ * torch.cos(θ)

            # ∂/∂φ = (-r sin φ cos θ, -r sin φ sin θ, r cos φ)
            e2 = torch.zeros_like(p)
            e2[:, 0] = -r * sin_φ * torch.cos(θ)
            e2[:, 1] = -r * sin_φ * torch.sin(θ)
            e2[:, 2] = r * cos_φ

            e1 = e1 / e1.norm(dim=1, keepdim=True)
            e2 = e2 / e2.norm(dim=1, keepdim=True)
            return torch.stack([e1, e2], dim=2)
        else:
            basis = torch.zeros(p.shape[0], self.D, self.k, dtype=p.dtype, device=p.device)
            for i in range(self.k):
                basis[:, 2*i, i] = -p[:, 2*i + 1]
                basis[:, 2*i + 1, i] = p[:, 2*i]
            return basis

    def embed_points(self, p, emb):
        """Points of the embedded torus: p (n, D) -> (n, ambient_dim)."""
        return emb(p)

    def embed_tangent_basis(self, p, B, emb):
        """Orthonormal tangent basis of the embedded torus at emb(p): (n, ambient_dim, k). Pushes B forward with the Jacobian, then orthonormalises."""
        return torch.linalg.qr(emb.jacobian(p) @ B)[0]


MANIFOLDS = {"sphere": Sphere, "torus": Torus}


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
