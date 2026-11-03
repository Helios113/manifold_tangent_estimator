"""The fold determinant per stage and per step of the encoder, and loops around every great circle (is there a cut?). (merges the former diagnostics/fold_stages.py; see auto_encoder/MATH.md; the plots are in fold_plots.py, the fold alignment in fold_alignment.py, the pole axis and the circles in pole.py)
usage: python diagnostics/fold_determinant_stages.py <run name or folder> [<out folder>]      (default out: results/diagnostics/fold_determinant_stages/<run>)
Writes <out>/{results.json, data.npz, fold_location.{png,pdf}, fold_origin.{png,pdf}, loops.{png,pdf}} for the pole estimate and the same files in <out>/random_control for a random control axis. Call fold_determinant_stages(run, ...) from Python for the other parameters.

The encoder is a chain x = h_0 -> h_1 -> ... -> z given by net.encoder_steps() (stage s = after step s; the steps are the layers, the pair builder of arctan2_project, the layer norm, the Arctan2, the last matrix).
J_s = d h_s / d x . J_f B is (width of h_s) x k and S_s = d h_s / d h_(s-1) is the step from stage s-1 to s.
Only the last J_s (= A, the pushforward) is square, so only A has a signed determinant. For the other stages the determinant is the volume sqrt(det(J_s^T J_s)) = the product of the singular values of J_s, and for a step
the volume of S_s Q_(s-1) (Q_(s-1) an orthonormal basis of the column space of J_(s-1)), which is vol(J_s) / vol(J_(s-1)). A fold makes a volume dip. The sign of an intermediate stage is not defined: the frame Q_s of its column space
can only be fixed by J_s itself (QR with positive diagonal), and then the determinant of every step in that frame is positive.
The signed quantity that exists at every stage is the remainder: the rest of the network applied to the tangent image of stage s, L_s Q_s (L_s = d z / d h_s, Q_s the orthonormal basis of the column space of J_s from the QR with positive diagonal).
Since A = L_s J_s = L_s Q_s R_s and det R_s > 0, det(L_s Q_s) = det(A) / vol(J_s): its sign is the sign of det A at EVERY stage (the sign is identical for all s, so it is reported once per loop), and its size is |det A| / vol(J_s).
The stage where the fold arises is the first one at which vol(J_s) collapses; before it, the remainder |det A| / vol(J_s) is what goes to zero at the sign change of det A.
Plan
  1. Axis N: the pole estimate (pole.pole_axis: the common axis of the directions that the encoder squashes most); control: a random axis (seed 9, circles seed 12), analysed identically. n_circles great circles through N (seed): p(phi) = cos(phi) N + sin(phi) w0, basis [dp/dphi, E] continuous in phi.
  2. The path is n_loops loops around the circle, started at phi_start (default 90 degrees, a point away from the pole), every step_deg degrees (defaults: two loops, phi = 0 .. 810 degrees, 406 steps).
     The pole is crossed at phi = 0, 180, 360, ...
  3. At every point: log10 volume of every J_s (stage) and of every step, slogdet(A) (sign relative to phi = 0 and log|det|), the Arctan2 output theta (m angles) and the latent code z.
  3b. Per stage the remainder log10 |det(L_s Q_s)| = log10 |det A| - log10 vol(J_s), and the sign of det A relative to phi = 0 per circle; the number of sign changes in every loop (and which circles).
  4. Per stage and per step: dip = median over phi of log10 volume minus the minimum over phi, median over circles: how deep the volume falls somewhere on the circle, and the phi (modulo 180) of the minimum.
  4b. The folds (the former fold_stages.py): a fold is a sign change of det A along a circle (anywhere on the path, all loops), located at the sample of the pair with the smaller smallest singular value of A; a circle without a sign change has no fold.
      Per stage and per step the median over all folds of the smallest singular value s_min at the fold, against the typical s_min (median over the path and the circles), their ratio, the condition number at the fold,
      and the first stage / step whose ratio is below threshold (default 0.1) = where the fold first occurs / is created. Plots fold_location.png (the sign of det A and s_min of A along every circle, the folds marked) and fold_origin.png (every stage and step around the folds: s_min, the remainder / step volume, the ratio).
  5. Loops. z(p) is a function of p, so z at the end of a loop equals z at the start (sanity check, distance ~ 0). The Arctan2 angles are lifted: the m angles along the path are unwrapped (np.unwrap), and the lifted displacement
     theta(end) - theta(start) is a multiple of 2 pi per angle (the winding) when the path closes. Per loop report the winding of every angle on every circle and the lifted distance |theta_lifted(end) - theta_lifted(start)|:
     near 0 when the loop does not wind around a cut, a multiple of 2 pi when it does. Also the largest jump of z between two consecutive points relative to the median jump (a cut shows as a jump).
     np.unwrap is only reliable where theta moves less than pi between samples, so near the pole (small Arctan2 radius) the lifted values are unreliable; the minimum Arctan2 radius along each circle is reported.
     Without an Arctan2 in the run there are no angles: only the volumes, the determinant and z are reported."""

import json
import sys

import numpy as np
import torch
from common import L10, jacobian, load_net, manifold, map_chunks, open_run, output_dir
from fold_alignment import fold_alignment, plot_fold_location, plot_fold_origin
from fold_plots import path_marks, plot_loops
from pole import great_circle, pole_axis
from tqdm import tqdm


def point_quantities(steps, i_arctan, x, JfB):
    """x (n, D), JfB (n, D, k): singular values of J_s (n, stages + 1, k) and of the steps S_s Q_(s-1) (n, stages, k), their log volumes (n, stages + 1), (n, stages), slogdet of A (sign, log|det|), z (n, k), and with an Arctan2 its input pairs' radii (n, m) and output theta (n, m)."""
    h, J = x, JfB.double()
    sv = [torch.linalg.svdvals(J.cpu())]
    stepsv = []
    extra = {}
    for j, (_, fn) in enumerate(steps, 1):
        Q = torch.linalg.qr(J.cpu())[0].to(J.device)
        S_s = jacobian(fn, h).double()
        stepsv.append(torch.linalg.svdvals((S_s @ Q).cpu()))
        if j == i_arctan:
            a = h.double().chunk(2, -1)
            extra["radius"] = torch.sqrt(a[0] ** 2 + a[1] ** 2).cpu()
        J = S_s @ J
        h = fn(h)
        if j == i_arctan:
            extra["theta"] = h.double().cpu()
        sv.append(torch.linalg.svdvals(J.cpu()))
    sg, la = torch.linalg.slogdet(J.cpu())
    sv, stepsv = torch.stack(sv, 1), torch.stack(stepsv, 1)
    out = dict(
        sv=sv,
        sv_steps=stepsv,
        logvol=sv.log().sum(-1),
        steps=stepsv.log().sum(-1),
        sign=sg,
        logabs=la,
        z=h.double().cpu(),
        **extra,
    )
    return {
        key: v.detach() for key, v in out.items()
    }  # without the detach every point keeps its autograd graph (the network weights require grad) and the memory fills up


def circle_loops(net, S, emb, dev, N, seed, n_circles, step_deg, n_deg, pq_chunk=1):
    """Quantities of point_quantities along n_circles great circles through N, phi = 0 .. n_deg degrees. pq_chunk: points per call (the step Jacobian is width x width per point)."""
    steps = net.encoder_steps()
    i_arctan = 1 + [s[0] for s in steps].index("Arctan2") if net.arctan2_after else None
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
        )  # keeps the random stream of the earlier fold scripts
        q = map_chunks(
            lambda sl: point_quantities(
                steps, i_arctan, x[sl].to(dev), JfB[sl].to(dev)
            ),
            n_phi,
            pq_chunk,
            desc=f"circle {ci}",
        )
        for key, v in q.items():
            res.setdefault(key, []).append(v.detach().numpy())
    return np.degrees(phi.numpy()), {k_: np.stack(v) for k_, v in res.items()}


def summarise(phi, Q, labels, step_labels, i0, loop_ends):
    """Everything reported: the dict for results.json and the arrays the plots need (lv, ls, rest, rel, thu, z)."""
    lv, ls = (
        Q["logvol"] / L10,
        Q["steps"] / L10,
    )  # (circles, phi, stages + 1), (circles, phi, stages): log10 volumes
    rel = Q["sign"] * Q["sign"][:, :1]
    rest = (
        Q["logabs"][:, :, None] / L10 - lv
    )  # (circles, phi, stages + 1): log10 |det(L_s Q_s)|, the remainder after stage s
    ch = lambda a, b: (np.diff(rel[:, a : b + 1], axis=1) != 0).sum(
        1
    )  # sign changes of det A per circle between two path indices
    bounds = list(zip([i0] + loop_ends[:-1], loop_ends))
    sign_loops = {
        f"loop {l}": dict(
            sign_changes_per_circle=[int(v) for v in ch(a, b)],
            circles_with_a_sign_change=int((ch(a, b) > 0).sum()),
            phi_of_sign_changes=[
                [float(phi[a + j]) for j in np.nonzero(np.diff(rel[c, a : b + 1]))[0]]
                for c in range(len(rel))
            ],
        )
        for l, (a, b) in enumerate(bounds, 1)
    }
    summ = lambda v, lab: [
        dict(
            index=j,
            label=lab[j],
            median_log10_volume=float(np.median(v[..., j])),
            min_log10_volume_median=float(np.median(v[..., j].min(1))),
            dip_decades_median=float(
                np.median(np.median(v[..., j], 1) - v[..., j].min(1))
            ),
            phi_of_min_mod_180_median=float(np.median(phi[v[..., j].argmin(1)] % 180)),
        )
        for j in range(v.shape[-1])
    ]
    stage_rows = summ(lv, labels)
    for j, row in enumerate(stage_rows):
        row.update(
            remainder_median_log10=float(np.median(rest[..., j])),
            remainder_min_log10_median=float(np.median(rest[..., j].min(1))),
            remainder_dip_decades_median=float(
                np.median(np.median(rest[..., j], 1) - rest[..., j].min(1))
            ),
            sign_of_remainder="same as sign(det A) at every stage",
            **{
                f"sign_changes_loop{l}": sign_loops[f"loop {l}"][
                    "circles_with_a_sign_change"
                ]
                for l in range(1, len(bounds) + 1)
            },
        )
    z = Q["z"]  # (circles, phi, k)
    jumps = np.linalg.norm(np.diff(z, axis=1), axis=-1)
    jump_ratio = jumps.max(1) / np.median(jumps, 1)
    loops, thu = {}, None
    if "theta" in Q:
        thu = np.unwrap(Q["theta"], axis=1)  # (circles, phi, m)
        wind = lambda i: np.round(
            (thu[:, i] - thu[:, i0]) / (2 * np.pi)
        )  # (circles, m) integer winding of the lifted angles from the start to index i
        lift = lambda i: np.linalg.norm(thu[:, i] - thu[:, i0], axis=-1)
    for l, i in enumerate(loop_ends, 1):
        loops[f"loop {l}"] = dict(
            end_phi=float(phi[i]),
            z_distance_start_end_max=float(
                np.linalg.norm(z[:, i] - z[:, i0], axis=-1).max()
            ),
        )
        if thu is not None:
            w = wind(i)
            loops[f"loop {l}"].update(
                lifted_theta_distance_median=float(np.median(lift(i))),
                lifted_theta_distance_max=float(lift(i).max()),
                circles_with_nonzero_winding=int((np.abs(w).max(1) > 0).sum()),
                winding_counts={int(v): int((w == v).sum()) for v in np.unique(w)},
            )
    step_deg = phi[1] - phi[0]
    res = dict(
        phi_start=float(phi[i0]),
        pole_crossings_phi=[
            float(v) for v in phi[np.arange(0, len(phi), int(round(180 / step_deg)))]
        ],
        stages=stage_rows,
        sign_of_det_A_per_loop=sign_loops,
        steps=summ(ls, step_labels),
        A=dict(
            sign_changes_per_circle_mean_over_loops=float(
                (np.diff(rel, axis=1) != 0).sum(1).mean()
            ),
            circles_with_a_sign_change=int(
                ((np.diff(rel, axis=1) != 0).sum(1) > 0).sum()
            ),
            log10_abs_det_min_median=float(np.median(Q["logabs"].min(1)) / L10),
        ),
        loops=loops,
        z_jump_max_over_median_jump_median=float(np.median(jump_ratio)),
    )
    if "radius" in Q:
        res["min_arctan2_radius_median"] = float(np.median(Q["radius"].min((1, 2))))
    return res, dict(
        lv=lv,
        ls=ls,
        rest=rest,
        rel=rel,
        thu=thu,
        z=z,
        wind=wind if thu is not None else None,
    )


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
    """run: run name (results/autoencoder/<run>) or run folder; out: folder to write into (default results/diagnostics/fold_determinant_stages/<run>). Returns the results dict of the pole estimate."""
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

    N_hat, share = pole_axis(net, S, emb, dev, chunk=pole_chunk)

    N_rand = torch.nn.functional.normalize(
        torch.randn(S.D, generator=torch.Generator().manual_seed(9)), dim=0
    )  # control axis: random, with its own circles (seed 12)

    i0 = int(round(phi_start / step_deg))
    per = int(round(360 / step_deg))
    loop_ends = [i0 + l * per for l in range(1, n_loops + 1)]
    res = None
    for axis, N, circle_seed, folder in (
        ("Direction of smallest eigen value", N_hat, seed, out),
        ("Random direction", N_rand, 12, out / "random_control"),
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
        r, A = summarise(phi, Q, labels, step_labels, i0, loop_ends)
        fold, F = fold_alignment(
            phi, Q, A["rel"], A["rest"], A["ls"], labels, step_labels, threshold
        )
        r = dict(
            axis=axis,
            top_eigenvalue_share=share if N is N_hat else None,
            fold_alignment=fold,
            **r,
        )
        print(json.dumps(r, indent=1), flush=True)
        json.dump(r, open(folder / "results.json", "w"), indent=1)
        np.savez(
            folder / "data.npz",
            phi=phi,
            N=N.numpy(),
            start_index=i0,
            loop_end_indices=np.array(loop_ends),
            stage_labels=np.array(labels),
            step_labels=np.array(step_labels),
            remainder_log10=A["rest"],
            rel_sign=A["rel"],
            fold_circle=np.array([c for c, _ in F["folds"]], dtype=int),
            fold_index=np.array([i for _, i in F["folds"]], dtype=int),
            **Q,
            **({"theta_lifted": A["thu"]} if A["thu"] is not None else {}),
        )

        marks = path_marks(phi, i0, loop_ends)
        name = f"{p.name} ({axis})"
        plot_fold_location(folder, name, F, marks)
        plot_fold_origin(folder, name, F, labels, step_labels)
        plot_loops(folder, name, phi, A, i0, loop_ends, marks)
        res = res or r
    return res


if __name__ == "__main__":
    fold_determinant_stages(*sys.argv[1:3])
