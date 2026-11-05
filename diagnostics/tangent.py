import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.func import jacrev, vmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auto_encoder"))
from common import BLUE, ORANGE, WIDTH, angle_summary, neurips, save, titled
from eval_points import eval_points
from plots import load, load_net

WEIGHTS = "best.pt"  # set from --weights
OUT = None  # a folder to write all tangent files into (set by run_diagnostics.py); None = the run folder


def out_dir(run):
    """Where the tangent files of this run are written: OUT when set, else the run folder for best.pt, otherwise tangent_checkpoints/<weights name>/."""
    if OUT is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        return OUT
    if WEIGHTS == "best.pt":
        return run
    d = run / "tangent_checkpoints" / Path(WEIGHTS).stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def pushforward(run):
    run, cfg = load(run)
    out = out_dir(run)
    pts = eval_points(cfg)
    n_eval = pts["n_eval"]

    p, x, T = (
        pts["p"][:n_eval],
        pts["x"][:n_eval],
        pts["T"],
    )  # the first n_eval of the points shared with lpca_lego.py (the training distribution); T (n, D, k): the true input tangent

    net = load_net(run, cfg, WEIGHTS)

    T_latent = (vmap(jacrev(net.encode))(x) @ T).detach()  # (n, latent_dim, k)

    T_recon = (vmap(jacrev(net))(x) @ T).detach()  # (n, D, k)

    with torch.no_grad():
        z, x_hat = net.encode(x), net(x)

    np.savez(
        out / "tangent.npz",
        sphere_points=p.numpy(),
        x=x.numpy(),
        z=z.numpy(),
        x_hat=x_hat.numpy(),
        T=T.numpy(),
        T_latent=T_latent.numpy(),
        T_recon=T_recon.numpy(),
        Q=pts["emb"].Q.numpy(),
    )
    print(
        f"saved tangent.npz: T {tuple(T.shape)}, T_latent {tuple(T_latent.shape)}, T_recon {tuple(T_recon.shape)}",
        flush=True,
    )


def latent_tangent(z, k, n_eval):
    """(n_eval, latent_dim, k): at each of the first n_eval encoded points, the top-k principal directions of its latent_dim + 1 nearest points among all the encoded points z (fewer points could not span k directions)."""
    n_neighbours = z.shape[1] + 1
    assert n_neighbours >= k, f"{n_neighbours} neighbours cannot give {k} directions"
    idx = (
        torch.cdist(z[:n_eval], z).topk(n_neighbours, largest=False).indices
    )  # (n_eval, m), every point is its own nearest neighbour
    nbrs = z[idx]
    nbrs = nbrs - nbrs.mean(dim=1, keepdim=True)
    return torch.linalg.svd(nbrs, full_matrices=False)[2][:, :k].transpose(1, 2)


def sample_latent_tangent(run):
    run, cfg = load(run)
    out = out_dir(run)
    pts = eval_points(cfg)
    n_eval = pts["n_eval"]
    net = load_net(run, cfg, WEIGHTS)
    with torch.no_grad():
        z = net.encode(
            pts["x"]
        )  # the codes of the whole cloud: the neighbours are searched among all of them
    L = latent_tangent(z, cfg["intrinsic_dim"], n_eval)  # (n_eval, latent_dim, k)
    L_recon = (vmap(jacrev(net.decode))(z[:n_eval]) @ L).detach()  # (n_eval, D, k)
    np.savez(
        out / "tangent_sampled.npz",
        L=L.numpy(),
        L_recon=L_recon.numpy(),
        n_neighbours=z.shape[1] + 1,
    )
    print(
        f"saved tangent_sampled.npz: L {tuple(L.shape)}, L_recon {tuple(L_recon.shape)}",
        flush=True,
    )


def decoder_tangent(run):
    run, cfg = load(run)
    out = out_dir(run)
    d = np.load(out / "tangent.npz")
    net = load_net(run, cfg, WEIGHTS)
    J = vmap(jacrev(net.decode))(torch.tensor(d["z"])).detach()  # (n, D, latent_dim)
    T_dec = torch.linalg.svd(J, full_matrices=False)[0][
        :, :, : cfg["intrinsic_dim"]
    ]  # (n, D, k)
    np.savez(
        out / "tangent_decoder.npz",
        T_dec=T_dec.numpy(),
        singular_values=torch.linalg.svdvals(J).numpy(),
    )
    print(f"saved tangent_decoder.npz: T_dec {tuple(T_dec.shape)}", flush=True)


def cos_between_tangent_spaces(A, B):
    # Assumes that the points at which we have attained A and B are identical
    # Any non-zero reconstruction loss invalidates this assumption
    k = A.shape[-1]
    orth = lambda M: np.linalg.svd(M, full_matrices=False)[0][..., :k]
    cos = np.linalg.svd(orth(A).transpose(0, 2, 1) @ orth(B), compute_uv=False)
    return cos


def principal_angles(A, B):
    """(n, k) principal angles between the tangent spaces A, B (n, D, k) at the same points, ascending, in degrees."""
    return np.sort(np.degrees(np.arccos(np.clip(cos_between_tangent_spaces(A, B), -1, 1))), axis=1)


def compare(run):
    run, _ = load(run)
    out = out_dir(run)
    d, s, dec = (
        np.load(out / "tangent.npz"),
        np.load(out / "tangent_sampled.npz"),
        np.load(out / "tangent_decoder.npz"),
    )
    pairs = {
        "true input vs true reconstructed": (d["T"], d["T_recon"]),
        "true input vs estimated reconstructed": (d["T"], s["L_recon"]),
        "true reconstructed vs estimated reconstructed": (d["T_recon"], s["L_recon"]),
        "true latent vs estimated latent": (d["T_latent"], s["L"]),
        "true input vs decoder Jacobian": (d["T"], dec["T_dec"]),
        "true reconstructed vs decoder Jacobian": (d["T_recon"], dec["T_dec"]),
    }
    singular = {  # the three tangent maps of the comparison, (n, D, k)
        "true latent": d["T_latent"],
        "true reconstructed": d["T_recon"],
        "estimated reconstructed": s["L_recon"],
    }
    angles = {name: principal_angles(A, B) for name, (A, B) in pairs.items()}
    sv = {name: np.linalg.svd(M, compute_uv=False) for name, M in singular.items()}
    np.savez(
        out / "tangent_compare.npz",
        names=np.array(list(angles)),
        singular_value_names=np.array(list(sv)),
        **{f"angles_{i}": a for i, a in enumerate(angles.values())},
        **{f"singular_values_{i}": v for i, v in enumerate(sv.values())},
    )
    summary = {name: angle_summary(a) for name, a in angles.items()}
    json.dump(summary, open(out / "tangent_compare.json", "w"), indent=1)
    plot_compare(out, f"{run.name} {Path(WEIGHTS).stem}", angles)
    print("plotted plot_compare")


@neurips
def plot_compare(out, title, angles):
    """One histogram per comparison: the (n,) largest principal angle a[:, -1] of
    every point, in degrees. The bins are built explicitly: a data range too
    small to hold them (float precision when the spaces agree almost exactly, or
    an exactly constant largest angle) would crash np.histogram with 'cannot
    create N finite-sized bins', so the degenerate case gets a visible window."""
    fig, axes = plt.subplots(
        3, 2, figsize=(WIDTH, 5.4), gridspec_kw=dict(hspace=0.8, wspace=0.3)
    )
    for ax, (letter, (name, a)) in zip(axes.flat, zip("abcdef", angles.items())):
        v = a[:, -1]
        lo, hi = float(v.min()), float(v.max())
        if not hi > lo:
            lo, hi = lo - 0.5, hi + 0.5
        ax.hist(v, bins=np.linspace(lo, hi, 41), color=BLUE)
        ax.axvline(
            np.median(v),
            color=ORANGE,
            ls="--",
            lw=0.9,
            label=f"median {np.median(v):.3g}$^\\circ$",
        )
        titled(ax, letter, name)
        ax.set_xlabel("Largest principal angle (degrees)")
        ax.set_ylabel("Points")
        ax.legend(loc="upper right")
    fig.suptitle(title, y=0.95, fontsize=7.5)
    save(fig, out, "tangent_compare")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--weights" in args:
        i = args.index("--weights")
        WEIGHTS = args[i + 1]
        del args[i : i + 2]
    for name in args[1:] or [
        "pushforward",
        "sample_latent_tangent",
        "decoder_tangent",
        "compare",
    ]:
        globals()[name](args[0])
