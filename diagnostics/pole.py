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
    """A great circle through the axis N with a random direction w0 perpendicular to N (generator g): the points p(phi) = cos(phi) N + sin(phi) w0 (n_phi, D) and the basis (n_phi, D, k) of the sphere's tangent space along it,
    [dp/dphi, E] with E the k - 1 fixed vectors perpendicular to N and w0, continuous in phi (a sign of a determinant needs that)."""
    D = len(N)
    w0 = torch.randn(D, generator=g)
    w0 = w0 - (w0 @ N) * N
    w0 = w0 / w0.norm()
    Qf = torch.linalg.qr(
        torch.cat([torch.stack([N, w0], 1), torch.randn(D, D - 2, generator=g)], 1)
    )[0]
    E = Qf[:, 2:]
    p = torch.cos(phi)[:, None] * N + torch.sin(phi)[:, None] * w0
    dp = -torch.sin(phi)[:, None] * N + torch.cos(phi)[:, None] * w0
    return p, torch.cat([dp[:, :, None], E[None].expand(len(phi), -1, -1)], 2)
