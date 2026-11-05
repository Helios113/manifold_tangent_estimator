"""The fold determinant per stage and per step of the encoder, along great circles
through the pole axis. Two axes are analysed: the pole estimate and a random
control axis. Writes, per axis: fold_location.{png,pdf} (is there a fold, where),
fold_origin.{png,pdf} (which stage / step creates it) and loops.{png,pdf} (does a
loop around the circle hit a cut). The fold alignment is in fold_alignment.py, the
loop and path plots in fold_plots.py, the pole axis and the circles in pole.py.

usage: python diagnostics/fold_determinant_stages.py <run name or folder> [<out folder>]
"""

import sys

import numpy as np
import torch
from common import L10, jacobian, load_net, manifold, map_chunks, open_run, output_dir
from fold_alignment import fold_arrays, plot_fold_location, plot_fold_origin
from fold_plots import path_marks, plot_loops
from pole import great_circle, pole_axis
from tqdm import tqdm


def point_quantities(steps, x, JfB):
    """Per-point diagnostics of the encoder chain, composed onto the embedded tangent.

    steps: [(name, fn), ...] the layers applied in order; each invertible, so the
           chain x -> z has Jacobian A = S_L @ ... @ S_1 @ JfB.
    x:     (n, latent) the starting inputs h_0 (the embedded circle points).
    JfB:   (n, latent, k) the embedding Jacobian times the manifold tangent basis.

    Returns (per point): sv (accumulated singular values, stage 0..L),
    sv_steps (the singular values of each step against the incoming orthonormal
    frame), logvol (log volume per stage), steps (log volume each step adds),
    sign and logabs (slogdet of A), z (the code).
    """
    h, J = x, JfB.double().cpu()

    sv = [torch.linalg.svdvals(J)]  # (n, k) per stage, grows per stage
    stepsv = []

    for j, (_, fn) in enumerate(steps, 1):
        Q = torch.linalg.qr(J)[0]  # orthonormal frame of the incoming tangent image
        S_s = jacobian(fn, h).double()
        # What step j alone does to that frame, with the accumulated anisotropy removed.
        stepsv.append(torch.linalg.svdvals((S_s @ Q).cpu()))
        J = S_s @ J  # chain rule
        h = fn(h)  # advance the point
        sv.append(torch.linalg.svdvals(J))

    sg, la = torch.linalg.slogdet(J)
    sv, stepsv = torch.stack(sv, 1), torch.stack(stepsv, 1)  # (n, stages (+1), k)

    out = dict(
        sv=sv,
        sv_steps=stepsv,
        logvol=sv.log().sum(-1),
        steps=stepsv.log().sum(-1),
        sign=sg,
        logabs=la,
        z=h.double().cpu(),
    )
    return {key: v.detach() for key, v in out.items()}
    # detach: without it each point retains its autograd graph (the weights
    # require grad) and the memory fills up over the batch.


def circle_loops(net, S, emb, dev, N, seed, n_circles, step_deg, n_deg, pq_chunk=1):
    """n_circles great circles through the axis N (seed), each walked for n_deg
    degrees in steps of step_deg: point_quantities at every sample, pooled into
    (circles, phi, ...) arrays."""
    steps = net.encoder_steps()
    g = torch.Generator().manual_seed(seed)
    k = S.k
    n_phi = n_deg // step_deg + 1
    phi = torch.linspace(0, np.radians(n_deg), n_phi)
    res = {}
    for ci in tqdm(range(n_circles), desc="circles", mininterval=30):
        p, B = great_circle(N, g, phi)
        JfB = emb.jacobian(p) @ B
        x = emb(p)
        torch.randn(
            n_phi, k, generator=g
        )  # keeps the random stream aligned with the earlier fold scripts
        q = map_chunks(
            lambda sl: point_quantities(steps, x[sl].to(dev), JfB[sl].to(dev)),
            n_phi,
            pq_chunk,
            desc=f"circle {ci}",
        )
        for key, v in q.items():
            res.setdefault(key, []).append(v.detach().numpy())
    return phi.numpy(), {key: np.stack(v) for key, v in res.items()}


def plot_arrays(Q):
    """The arrays the plots need, from the per-point quantities Q (circles, phi, ...):

    lv   (circles, phi, stages + 1): accumulated log10 tangent volumes.
    ls   (circles, phi, stages):     per-step log10 volume contributions.
    rel  (circles, phi, stages + 1): orientation of det A relative to the start of the path (+1 / -1).
    rest (circles, phi, stages + 1): log10 |det(L_s Q_s)|, the remainder after the tangent volume.
    z    (circles, phi, k):          the codes.
    """
    lv, ls = Q["logvol"] / L10, Q["steps"] / L10
    rest = Q["logabs"][:, :, None] / L10 - lv
    # Guard against invalid values in rest
    rest = np.where(np.isfinite(rest), rest, 0.0)
    return {
        "lv": lv,
        "ls": ls,
        "rest": rest,
        "rel": Q["sign"] * Q["sign"][:, :1],
        "z": Q["z"],
    }


def fold_determinant_stages(
    run,
    out=None,
    n_circles=20,
    seed=11,
    step_deg=2,
    phi_start=90,
    n_loops=2,
    threshold=0.1,
    pole_chunk=25,
    pq_chunk=1,
):
    """run: run name (results/autoencoder/<run>) or run folder; out: folder to write
    the figures into (default results/diagnostics/fold_determinant_stages/<run>),
    the random control axis into <out>/random_control."""
    p, cfg = open_run(run)
    out = output_dir("fold_determinant_stages", p, out)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S, emb = manifold(cfg)
    assert cfg["embedded_dim"] == S.k, (
        "the pushforward is square only when the latent dimension equals k"
    )
    net = load_net(p, cfg).to(dev)
    steps = net.encoder_steps()
    labels = ["input"] + [s[0] for s in steps]
    step_labels = [
        f"step {j + 1}: stage {j} -> stage {j + 1} ({labels[j + 1]})"
        for j in range(len(steps))
    ]

    N_hat, confidence = pole_axis(net, S, emb, dev, chunk=pole_chunk)
    N_rand = torch.nn.functional.normalize(
        torch.randn(S.D, generator=torch.Generator().manual_seed(9)), dim=0
    )  # control axis: random, with its own circles (seed 12)

    i0 = int(round(phi_start / step_deg))
    per = int(round(360 / step_deg))
    loop_ends = [i0 + l * per for l in range(1, n_loops + 1)]

    for (
        axis,
        N,
        conf,
        circle_seed,
        folder,
    ) in (
        (
            "pole axis",
            N_hat,
            confidence,
            seed,
            out,
        ),
        ("random control axis", N_rand, None, 12, out / "random_control"),
    ):
        folder.mkdir(parents=True, exist_ok=True)
        phi, Q = circle_loops(
            net,
            S,
            emb,
            dev,
            N,
            circle_seed,
            n_circles,
            step_deg,
            phi_start + 360 * n_loops,
            pq_chunk,
        )
        A = plot_arrays(Q)
        F = fold_arrays(phi, Q, A["rel"], A["rest"], A["ls"], threshold)
        marks = path_marks(phi, i0, loop_ends)
        name = f"{p.name} ({axis})"
        plot_fold_location(folder, name, F, marks, confidence)
        plot_fold_origin(folder, name, F, labels, step_labels)
        plot_loops(folder, name, phi, A, i0, loop_ends, marks)


if __name__ == "__main__":
    fold_determinant_stages(*sys.argv[1:3])
