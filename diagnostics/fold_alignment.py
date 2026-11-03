"""The folds of fold_determinant_stages.py: where they are along every circle, and what every stage and step does to the tangent directions there.
A fold is a point of the path where det A (the determinant of the encoder's pushforward A) changes sign: the smallest singular value of A goes through ~0 between two samples of different sign. Every sign change on every circle (the whole path, all loops) is one fold,
located at the one of the two samples with the smaller smallest singular value; a circle without a sign change has no fold. Per stage and step: the smallest singular value at the folds against its typical value (median over the path).
Two figures: fold_location.png (is there a fold, and where) and fold_origin.png (which stage / step creates it)."""

import matplotlib.pyplot as plt
import numpy as np
from common import (
    BLUE,
    FAINT,
    INK,
    MUTED,
    ORANGE,
    WIDTH,
    neurips,
    save,
    stage_colors,
    titled,
)
from fold_plots import draw_marks, finish, legend_right
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def fold_alignment(
    phi, Q, rel, rest, step_vol, labels, step_labels, threshold, window=30
):
    """rel: (circles, phi) sign of det A relative to the start of the path; rest: (circles, phi, stages + 1) log10 |det(L_s Q_s)|; step_vol: (circles, phi, stages) log10 volume of every step;
    threshold: a stage / step whose smallest singular value at the folds is below threshold x its typical value counts as folded; window: samples around a fold that are kept.
    Returns the dict of results.json and the arrays the plots need."""
    smin, smax, smin_s = (
        Q["sv"][..., -1],
        Q["sv"][..., 0],
        Q["sv_steps"][..., -1],
    )  # (circles, phi, stages + 1), (circles, phi, stages)
    smin_A = smin[:, :, -1]
    folds = [
        (c, j if smin_A[c, j] <= smin_A[c, j + 1] else j + 1)
        for c in range(len(rel))
        for j in np.nonzero(np.diff(rel[c]))[0]
    ]
    typical, typical_s = np.median(smin, axis=(0, 1)), np.median(smin_s, axis=(0, 1))
    res = dict(
        n_folds=len(folds),
        circles=len(rel),
        circles_with_a_fold=len({c for c, _ in folds}),
        fold_phi_per_circle=[
            [float(phi[i]) for c2, i in folds if c2 == c] for c in range(len(rel))
        ],
    )
    F = dict(
        threshold=threshold,
        phi=phi,
        smin=smin,
        rel=rel,
        folds=folds,
        window=window,
        ratio=None,
        ratio_s=None,
        aligned={},
    )
    if folds:
        at, at_s = (
            np.median(np.stack([smin[c, i] for c, i in folds]), 0),
            np.median(np.stack([smin_s[c, i] for c, i in folds]), 0),
        )
        F.update(ratio=at / typical, ratio_s=at_s / typical_s)
        kept = [(c, i) for c, i in folds if window <= i < len(phi) - window]
        if kept:
            F["aligned"] = {
                key: np.stack([a[c, i - window : i + window + 1] for c, i in kept])
                for key, a in dict(
                    smin=smin, rest=rest, smin_s=smin_s, step_vol=step_vol
                ).items()
            }
        first = next((j for j, r in enumerate(F["ratio"]) if r < threshold), None)
        first_s = next((j for j, r in enumerate(F["ratio_s"]) if r < threshold), None)
        res["stages"] = [
            {
                "index": j,
                "label": labels[j],
                "smin_at_fold": float(at[j]),
                "smin_typical": float(typical[j]),
                "ratio_fold_over_typical": float(F["ratio"][j]),
                "cond_at_fold": float(
                    np.median([smax[c, i, j] / smin[c, i, j] for c, i in folds])
                ),
            }
            for j in range(len(labels))
        ]
        res[f"first_stage_with_ratio_below_{threshold}"] = (
            None if first is None else {"index": first, "label": labels[first]}
        )
        res["steps"] = [
            {
                "index": j + 1,
                "label": step_labels[j],
                "smin_at_fold": float(at_s[j]),
                "smin_typical": float(typical_s[j]),
                "ratio_fold_over_typical": float(F["ratio_s"][j]),
            }
            for j in range(len(step_labels))
        ]
        res[f"first_step_with_ratio_below_{threshold}"] = (
            None
            if first_s is None
            else {"index": first_s + 1, "label": step_labels[first_s]}
        )
    return res, F


@neurips
def plot_fold_location(out, name, F, marks):
    """Is there a fold, and where: (a) the orientation of the encoder along every circle (a colour change is a fold), (b) the smallest singular value of A along every circle, with the folds marked."""
    phi, rel, smin_A = F["phi"], F["rel"], F["smin"][:, :, -1]
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(WIDTH, 4.8),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.1], hspace=0.35),
    )
    a1.imshow(
        rel,
        aspect="auto",
        cmap=ListedColormap([ORANGE, BLUE]),
        vmin=-1,
        vmax=1,
        extent=[phi[0], phi[-1], len(rel) - 0.5, -0.5],
        interpolation="nearest",
    )
    a1.spines[["left", "bottom"]].set_visible(True)
    a1.set_ylabel("great circle")
    a1.yaxis.get_major_locator().set_params(integer=True)
    legend_right(
        a1,
        handles=[
            Patch(color=BLUE, label="same orientation\nas at the start"),
            Patch(color=ORANGE, label="flipped"),
        ],
    )
    titled(
        a1,
        "a",
        "Orientation of the encoder along each great circle: every colour change is a fold",
    )
    for c in range(len(smin_A)):
        a2.plot(phi, np.log10(smin_A[c]), color=BLUE, lw=0.5, alpha=0.35)
    a2.plot(phi, np.median(np.log10(smin_A), 0), color=INK, lw=1.1)
    if F["folds"]:
        a2.plot(
            [phi[i] for _, i in F["folds"]],
            [np.log10(smin_A[c, i]) for c, i in F["folds"]],
            "o",
            ms=2.5,
            color=ORANGE,
        )
    a2.set_ylabel("log$_{10}$ smallest singular value of $A$")
    titled(
        a2,
        "b",
        "Smallest singular value of the pushforward: it drops towards 0 at every fold",
    )
    legend_right(
        a2,
        handles=[
            Line2D([], [], color=BLUE, lw=0.6, alpha=0.6, label="one circle"),
            Line2D([], [], color=INK, label="median"),
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="o",
                ls="",
                ms=3,
                label=f"fold ({len(F['folds'])})",
            ),
        ],
    )
    finish(fig, [a1, a2], phi, marks)
    fig.suptitle(f"{name}: {len(rel)} great circles", y=0.95, fontsize=7.5)
    save(fig, out, "fold_location")


def around_panel(ax, F, key, names, colors, ylabel):
    """A quantity around the folds (median over all folds), one curve per stage / step, against the angle from the fold."""
    al = F["aligned"].get(key)
    if al is None:
        ax.text(
            0.5,
            0.5,
            "no fold with a full window",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
        )
        return
    dphi = np.arange(-F["window"], F["window"] + 1) * (F["phi"][1] - F["phi"][0])
    for j, n in enumerate(names):
        ax.plot(dphi, np.median(al[..., j], 0), color=colors[j], label=n)
    ax.axvline(0, color=FAINT, lw=0.6, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Angle from the fold (degrees)")


def ratio_panel(ax, F, ratio, names):
    """The smallest singular value at the folds over its typical value, per stage / step: below the threshold the stage / step has folded."""
    if ratio is None:
        ax.text(
            0.5,
            0.5,
            "no fold",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
        )
        return
    x = np.arange(len(names))
    ax.bar(
        x,
        ratio,
        color=[ORANGE if r < F["threshold"] else BLUE for r in ratio],
        width=0.6,
    )
    ax.set_yscale("log")
    ax.axhline(1, color=MUTED, lw=0.5, ls=":")
    ax.axhline(F["threshold"], color=ORANGE, lw=0.7, ls="--")
    ax.text(
        len(names) - 0.5,
        F["threshold"],
        f"threshold {F['threshold']:g} ",
        ha="right",
        va="bottom",
        fontsize=6,
        color=ORANGE,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("smallest singular value\nat the folds / typical")


@neurips
def plot_fold_origin(out, name, F, labels, step_labels):
    """Which stage or step creates the fold. Left: the stages (cumulative: the encoder up to the stage); right: the steps (what each step does on its own). Rows: the smallest singular value around the folds, the volume (left: what the rest of the network does
    to the tangent image after the stage; right: what the step adds), and the ratio at the folds over the typical value."""
    step_names, colors = (
        labels[1:],
        stage_colors(len(labels)),
    )  # step s ends in stage s: it has the name and the colour of that stage
    fig, ax = plt.subplots(
        3,
        2,
        figsize=(WIDTH, 6.6),
        gridspec_kw=dict(hspace=0.55, wspace=0.45, height_ratios=[1.2, 1.2, 1]),
    )
    around_panel(
        ax[0, 0], F, "smin", labels, colors, "log$_{10}$ smallest singular value"
    )
    around_panel(
        ax[0, 1],
        F,
        "smin_s",
        step_names,
        colors[1:],
        "log$_{10}$ smallest singular value",
    )
    around_panel(ax[1, 0], F, "rest", labels, colors, "log$_{10}$ |det($L_s Q_s$)|")
    around_panel(
        ax[1, 1],
        F,
        "step_vol",
        step_names,
        colors[1:],
        "log$_{10}$ volume of $S_s Q_{s-1}$",
    )
    ratio_panel(ax[2, 0], F, F["ratio"], labels)
    ratio_panel(ax[2, 1], F, F["ratio_s"], step_names)
    for a in ax[0:2].ravel():
        a.margins(x=0)
    handles, names = ax[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            names,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=len(names),
            handlelength=1.2,
            columnspacing=1.2,
            title="stage (left) and the step that ends in it (right)",
            title_fontsize=6,
        )
    titled(ax[0, 0], "a", "Stages: smallest singular value")
    titled(ax[0, 1], "b", "Steps: smallest singular value")
    titled(ax[1, 0], "c", "Stages: what the rest does after it")
    titled(ax[1, 1], "d", "Steps: what each step adds")
    titled(ax[2, 0], "e", "Stages: folded?")
    titled(ax[2, 1], "f", "Steps: folded?")
    fig.suptitle(
        f"{name}: around the {len(F['folds'])} folds (median over folds; a ratio below the threshold means folded)",
        y=1.0,
        fontsize=7.5,
    )
    save(fig, out, "fold_origin")
