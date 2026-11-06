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


def fold_arrays(phi, Q, rel, rest, step_vol, threshold, window=30):
    """Everything the two figures need.

    Q:        per-point quantities from point_quantities; sv / sv_steps indexed
              (circles, phi, stages (+1 for sv)).
    rel:      (circles, phi) orientation of det A relative to the start.
    rest:     (circles, phi, stages + 1) log10 |det(L_s Q_s)|.
    step_vol: (circles, phi, stages) log10 volume of every step.
    threshold: a stage / step whose smallest singular value at the folds is below
              threshold x its typical value counts as folded.
    window:   samples kept around each fold (folds closer to the ends are dropped).

    Returns F with:
      phi, rel, folds      — path, orientation, fold locations (circle, index);
      smin, smin_s         — smallest singular value per stage / per step;
      ratio, ratio_s       — the above at the folds over its median over the path
                             (None when there is no fold);
      aligned              — for smin, rest, smin_s and step_vol, the
                             (folds, 2*window+1, stages(+1)) windows around the folds
                             (empty when no fold has a full window).
    """
    smin, smin_s = Q["sv"][..., -1], Q["sv_steps"][..., -1]
    smin_A = smin[:, :, -1]  # the smallest singular value of A itself

    # One fold per sign change of rel, at the sample with the smaller smin_A.
    folds = [
        (c, j if smin_A[c, j] <= smin_A[c, j + 1] else j + 1)
        for c in range(len(rel))
        for j in np.nonzero(np.diff(rel[c]))[0]
    ]
    F = dict(
        phi=phi,
        rel=rel,
        smin=smin,
        smin_s=smin_s,
        folds=folds,
        threshold=threshold,
        window=window,
        ratio=None,
        ratio_s=None,
        aligned={},
    )
    if not folds:
        return F

    # Typical value: the median of each stage's / step's smallest singular value
    # over the whole path; the ratio at the folds against it says who folded.
    typical, typical_s = np.median(smin, axis=(0, 1)), np.median(smin_s, axis=(0, 1))
    at = np.median(np.stack([smin[c, i] for c, i in folds]), 0)
    at_s = np.median(np.stack([smin_s[c, i] for c, i in folds]), 0)
    F["ratio"], F["ratio_s"] = at / typical, at_s / typical_s

    # The windows around the folds that are not too close to the ends of the path.
    kept = [(c, i) for c, i in folds if window <= i < len(phi) - window]
    if kept:
        F["aligned"] = {
            key: np.stack([a[c, i - window : i + window + 1] for c, i in kept])
            for key, a in dict(
                smin=smin, rest=rest, smin_s=smin_s, step_vol=step_vol
            ).items()
        }
    return F


@neurips
def plot_fold_location(out, name, F, marks, confidence):
    """Is there a fold, and where. (a) The orientation of the encoder along every
    circle: a colour change is a fold. (b) The smallest singular value of A along
    every circle, with the folds marked."""
    phi, rel, smin_A = F["phi"], F["rel"], F["smin"][:, :, -1]
    fig, (a1, a2) = plt.subplots(
        2,
        1,
        figsize=(WIDTH, 4.8),
        sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.1], hspace=0.35),
    )

    # (a) orientation per circle as a colour raster
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
    a1.set_ylabel("path along circle")
    a1.yaxis.get_major_locator().set_params(integer=True)
    legend_right(
        a1,
        handles=[
            Patch(color=BLUE, label="same orientation\nas at the start"),
            Patch(color=ORANGE, label="flipped"),
        ],
    )
    titled(a1, "a", "Orientation of the encoder along each circular path")

    # (b) smallest singular value of A per circle, median, and the folds
    # Guard against log10 of zero or negative values
    smin_A_safe = np.maximum(smin_A, 1e-300)  # avoid log10(0) or log10(negative)
    for c in range(len(smin_A_safe)):
        a2.plot(phi, np.log10(smin_A_safe[c]), color=BLUE, lw=0.5, alpha=0.35)
    a2.plot(phi, np.median(np.log10(smin_A_safe), 0), color=INK, lw=1.1)
    if F["folds"]:
        a2.plot(
            [phi[i] for _, i in F["folds"]],
            [np.log10(smin_A_safe[c, i]) for c, i in F["folds"]],
            "o",
            ms=2.5,
            color=ORANGE,
        )
    a2.set_ylabel("log$_{10}$ smallest singular value of pushforward")
    titled(a2, "b", "Smallest singular value of the pushforward")
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
    suptitle = f"{name}: {len(rel)} circles"
    if confidence is not None:
        suptitle = f"{name}: {len(rel)} circles, confidence for axis is {confidence}"

    fig.suptitle(suptitle, y=0.95, fontsize=7.5)
    save(fig, out, "fold_check")


def around_panel(ax, F, key, names, colors, ylabel):
    """One curve per stage / step: the median of a quantity over all folds, against
    the angle from the fold. Text instead of curves when no fold has a full window."""
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
    ax.set_xlabel("Angle from the fold")


def ratio_panel(ax, F, ratio, names):
    """Per stage / step: the smallest singular value at the folds over its typical
    value, a bar chart. Below the threshold line the stage / step has folded."""
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
    """Which stage or step creates the fold. Columns: stages (cumulative: the
    encoder up to that stage) vs steps (what each step does on its own). Rows:
    the smallest singular value around the folds, the volume (left: what the rest
    of the network does after the stage; right: what the step adds), and the
    at-the-folds / typical ratio of the smallest singular value."""
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
        f"{name}: around the {len(F['folds'])} folds"
        " (median over folds; a ratio below the threshold means folded)",
        y=1.0,
        fontsize=7.5,
    )
    save(fig, out, "fold_origin")
