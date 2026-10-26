"""Tangent-space experiments on a trained run.

usage: python auto_encoder/tangent.py <run folder> [function ...]      (no function: all of them, in this order)

    pushforward        tangent.npz           the true input, true latent and true reconstructed tangents   (best.pt)
    sample_latent_tangent   tangent_sampled.npz   the estimated latent tangent (local PCA of the nearest encoded points) and the estimated reconstructed tangent   (best.pt, tangent.npz)
    compare            tangent_compare.{npz,json,png}   principal angles between the tangent spaces, and the singular values of the pushforwards                  (tangent.npz, tangent_sampled.npz)

The five tangent spaces at a point x = f(p) of the embedded manifold, with enc the encoder and dec . enc the autoencoder:
    true input              T(x)                       (D x k, orthonormal columns)  the analytic tangent space of the embedded sphere in the ambient space
    true latent             J_enc(x) T(x)              (latent_dim x k)              its pushforward through the encoder, tangent vectors at z = enc(x)
    true reconstructed      J_{dec . enc}(x) T(x)      (D x k)                       its pushforward through the autoencoder, tangent vectors at x_hat = dec(enc(x))
    estimated latent        L(z)                       (latent_dim x k)              the top-k principal directions of the N_NEIGHBOURS encoded points nearest to z, among the encoded points of the same sample
    estimated reconstructed J_dec(z) L(z)              (D x k)                       the estimated latent tangent pushed through the decoder
compare measures how far apart the k-dimensional spaces are by their principal angles (largest angle in degrees; 0 means the same space)."""
import json, sys
import numpy as np, torch
from torch.func import jacrev, vmap
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plots import load, load_net, latent_inputs
from sphere import Sphere

N_NEIGHBOURS = 30                                                                       # encoded points used for the estimated latent tangent


def pushforward(run):
    run, cfg = load(run)
    emb, p, x, rgb = latent_inputs(cfg)
    net = load_net(run, cfg)

    S = Sphere(cfg["intrinsic_dim"])
    T = S.embed_tangent_basis(p, S.tangent_basis(p), emb)                               # (n, D, k): the true input tangent
    T_latent = (vmap(jacrev(net.encode))(x) @ T).detach()                               # (n, latent_dim, k)
    T_recon = (vmap(jacrev(net))(x) @ T).detach()                                       # (n, D, k)
    with torch.no_grad(): z, x_hat = net.encode(x), net(x)

    np.savez(run / "tangent.npz", sphere_points=p.numpy(), x=x.numpy(), z=z.numpy(), x_hat=x_hat.numpy(), rgb=rgb,
             T=T.numpy(), T_latent=T_latent.numpy(), T_recon=T_recon.numpy(), Q=emb.Q.numpy())
    print(f"saved tangent.npz: T {tuple(T.shape)}, T_latent {tuple(T_latent.shape)}, T_recon {tuple(T_recon.shape)}", flush=True)


def latent_tangent(z, k):
    """(n, latent_dim, k): at every encoded point, the top-k principal directions of its N_NEIGHBOURS nearest encoded points."""
    idx = torch.cdist(z, z).topk(N_NEIGHBOURS, largest=False).indices                   # (n, m), every point is its own nearest neighbour
    nbrs = z[idx]
    nbrs = nbrs - nbrs.mean(dim=1, keepdim=True)
    return torch.linalg.svd(nbrs, full_matrices=False)[2][:, :k].transpose(1, 2)


def sample_latent_tangent(run):
    run, cfg = load(run)
    d = np.load(run / "tangent.npz")
    net = load_net(run, cfg)
    z = torch.tensor(d["z"])
    L = latent_tangent(z, cfg["intrinsic_dim"])                                         # (n, latent_dim, k)
    L_recon = (vmap(jacrev(net.decode))(z) @ L).detach()                                # (n, D, k)
    np.savez(run / "tangent_sampled.npz", L=L.numpy(), L_recon=L_recon.numpy(), n_neighbours=N_NEIGHBOURS)
    print(f"saved tangent_sampled.npz: L {tuple(L.shape)}, L_recon {tuple(L_recon.shape)}", flush=True)


def principal_angles(A, B):
    """(n, k) principal angles in degrees, ascending, between the column spaces of A and B (n, D, k); the spaces are orthonormalised first, so rank-deficient columns are fine."""
    k = A.shape[-1]
    orth = lambda M: np.linalg.svd(M, full_matrices=False)[0][..., :k]
    cos = np.linalg.svd(orth(A).transpose(0, 2, 1) @ orth(B), compute_uv=False)
    return np.degrees(np.arccos(np.clip(cos, 0, 1)))[:, ::-1]


def compare(run):
    """Principal angles between the tangent spaces of the same space, per point: the input, reconstructed and latent spaces."""
    run, _ = load(run)
    d, s = np.load(run / "tangent.npz"), np.load(run / "tangent_sampled.npz")
    pairs = {"true input vs true reconstructed":                (d["T"], d["T_recon"]),
             "true input vs estimated reconstructed":           (d["T"], s["L_recon"]),
             "true reconstructed vs estimated reconstructed":   (d["T_recon"], s["L_recon"]),
             "true latent vs estimated latent":                 (d["T_latent"], s["L"])}
    angles = {name: principal_angles(A, B) for name, (A, B) in pairs.items()}
    pushforwards = {"true latent": d["T_latent"], "true reconstructed": d["T_recon"], "estimated reconstructed": s["L_recon"]}
    sv = {name: np.linalg.svd(M, compute_uv=False) for name, M in pushforwards.items()}                        # (n, k) per point, descending: how the pushforward stretches the tangent directions
    np.savez(run / "tangent_compare.npz", **{f"angles_{i}": a for i, a in enumerate(angles.values())}, names=list(angles),
             **{f"singular_values_{i}": v for i, v in enumerate(sv.values())}, singular_value_names=list(sv))

    summary = {name: dict(largest_angle_mean=a[:, -1].mean(), largest_angle_median=np.median(a[:, -1]), largest_angle_p90=np.percentile(a[:, -1], 90),
                          smallest_angle_mean=a[:, 0].mean()) for name, a in angles.items()}
    summary = {n: {k: float(v) for k, v in m.items()} for n, m in summary.items()}
    summary["singular_values"] = {name: dict(mean=v.mean(0).tolist(), median=np.median(v, axis=0).tolist(), mean_smallest_over_largest=float((v[:, -1] / v[:, 0]).mean())) for name, v in sv.items()}
    json.dump(summary, open(run / "tangent_compare.json", "w"), indent=1)
    for name, m in summary["singular_values"].items(): print(f"singular values of the {name} tangent: mean {np.round(m['mean'], 4)}, mean smallest/largest {m['mean_smallest_over_largest']:.3f}", flush=True)
    for name, m in summary.items():
        if name != "singular_values": print(f"{name}: largest angle mean {m['largest_angle_mean']:.1f} deg, median {m['largest_angle_median']:.1f}, 90th percentile {m['largest_angle_p90']:.1f}", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, a) in zip(axes.flat, angles.items()):                                # every panel has its own x range: the comparisons differ by orders of magnitude
        ax.hist(a[:, -1], bins=40, color="C0")
        ax.axvline(np.median(a[:, -1]), color="k", ls="--", lw=1, label=f"median {np.median(a[:, -1]):.3g} deg")
        ax.set_title(name, fontsize=9); ax.set_xlabel("largest principal angle (degrees)"); ax.set_ylabel("number of points"); ax.legend(fontsize=8)
    fig.suptitle(run.name); fig.tight_layout()
    fig.savefig(run / "tangent_compare.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    for name in sys.argv[2:] or ["pushforward", "sample_latent_tangent", "compare"]: globals()[name](sys.argv[1])
