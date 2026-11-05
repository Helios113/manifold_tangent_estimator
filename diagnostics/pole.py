"""The pole axis of an encoder and the great circles through an axis (shared by the fold diagnostics)."""

import numpy as np
import torch
from common import jacobian, map_chunks


def pole_axis(net, S, emb, dev, chunk=25, n=1000, seed=7) -> tuple[torch.Tensor, float]:
    # Returns the directions of the smallest eigen value,
    # plus the corresponding eigen value
    # if the eigen value is near 1, it means that the direction V is shared by many points
    # if the eigen value is near 0, it means that no directions is consistently squased -- not a fold

    p = S.sample(n, torch.Generator().manual_seed(seed))
    B = S.tangent_basis(p)
    Q, R = torch.linalg.qr(emb.jacobian(p) @ B)
    x = emb(p)

    def squashed(sl):
        M = (
            (jacobian(net, x[sl].to(dev)) @ Q[sl].to(dev)).detach().cpu()
        )  # the decompositions run on the CPU (cuSOLVER gave an illegal memory access on the 2080 Ti)
        Vh = torch.linalg.svd(M, full_matrices=False).Vh
        c = torch.linalg.solve(R[sl], Vh[:, -1, :, None])[..., 0]
        w = (B[sl] @ c[..., None])[..., 0]
        return w / w.norm(dim=1, keepdim=True)

    W = map_chunks(squashed, len(p), chunk)
    ev, V = torch.linalg.eigh(W.T @ W / len(W))
    return V[:, -1], float(ev[-1])


def great_circle(N, g, phi):
    """A great circle through the axis N.

    N:  a unit vector (D,) — a fixed axis of the sphere (e.g. the pole axis).
    g:  a torch.Generator — randomness is reproducible.
    phi: angles (n_phi,) where the circle is evaluated; the circle itself is
         p(t) = cos(t) N + sin(t) w0 for t in [0, 2π).

    Returns:
      p  (n_phi, D):      points on the sphere.
      T  (n_phi, D, k):   tangent basis along the circle, [dp/dphi, E] where
                          E is a fixed (k-1)-frame perpendicular to the plane
                          of the circle. E is constant in phi — that's what
                          makes the frame continuous around the full circle
                          (any frame you re-derive locally would flip sign
                          somewhere, which breaks determinant-continuity).
    """
    D = len(N)

    # --- a random direction w0 in the plane perpendicular to N ---
    w0 = torch.randn(D, generator=g)
    w0 = w0 - (w0 @ N) * N  # project out the N component (Gram–Schmidt step 1)
    w0 = w0 / w0.norm()  # normalize                          (step 2)

    # --- a fixed orthonormal complement E of the plane spanned by (N, w0) ---
    # QR of the (D, D) matrix [N | w0 | random D-2 columns]: Qf's first two
    # columns re-span the circle's plane, the remaining D-2 columns are
    # orthonormal and ⊥ to both N and w0. QR also fixes their sign
    # deterministically, which is why E is "the same" at every phi.
    Qf = torch.linalg.qr(
        torch.cat([torch.stack([N, w0], 1), torch.randn(D, D - 2, generator=g)], 1)
    )[0]
    E = Qf[:, 2:]  # (D, D-2) — spans the directions "off" the circle

    # --- the circle itself ---
    p = torch.cos(phi)[:, None] * N + torch.sin(phi)[:, None] * w0  # (n_phi, D)

    # --- its velocity, and the tangent frame ---
    dp = -torch.sin(phi)[:, None] * N + torch.cos(phi)[:, None] * w0  # dp/dphi
    # Tangent space of the sphere at p = everything ⊥ to p:
    #   dp/dphi is one direction (along the circle),
    #   E stays ⊥ to N and w0, hence ⊥ to p for every phi.
    # The caller keeps only the first k of these D-1 columns (k = intrinsic dim).
    return p, torch.cat([dp[:, :, None], E[None].expand(len(phi), -1, -1)], 2)
