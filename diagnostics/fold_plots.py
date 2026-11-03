"""The marks along the path and the loops plot of fold_determinant_stages.py (the fold plots are in fold_alignment.py)."""
import numpy as np
import matplotlib.pyplot as plt

from common import WIDTH, BLUE, ORANGE, AQUA, INK, MUTED, FAINT, band, neurips, save, titled

XLABEL = r"Angle $\varphi$ along the path (degrees)"
NOTE = "dotted: the axis is crossed; green dashed: start; black dashed: end of a loop"


def path_marks(phi, i0, loop_ends):
    """Where the path crosses the axis (every 180 degrees), where it starts and where each loop ends."""
    return dict(poles=np.arange(0, phi[-1] + 1, 180), start=phi[i0], ends=[phi[i] for i in loop_ends])


def draw_marks(ax, marks):
    for v in marks["poles"]: ax.axvline(v, color=FAINT, ls=":", lw=.7, zorder=0)
    ax.axvline(marks["start"], color=AQUA, ls="--", lw=.8, zorder=0)
    for v in marks["ends"]: ax.axvline(v, color=INK, ls="--", lw=.8, zorder=0)


def finish(fig, axes, phi, marks):
    """The marks on every panel, one x label under the last one."""
    for ax in axes: draw_marks(ax, marks); ax.set_xlim(phi[0], phi[-1])
    axes[-1].set_xlabel(XLABEL + "\n" + NOTE)


def legend_right(ax, **kw):
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), handlelength=1.2, **kw)


@neurips
def plot_loops(out, name, phi, A, i0, loop_ends, marks):
    """(a) the latent code and (b) the lifted Arctan2 angles against their value at the start: they return to 0 when a loop closes without a cut; (c) how many times each angle winds around a cut."""
    z, thu, wind = A["z"], A["thu"], A["wind"]
    paths = [(np.linalg.norm(z - z[:, i0:i0 + 1], axis=-1), "Latent code $z$: distance from its value at the start", "$|z(\\varphi) - z(\\mathrm{start})|$")]
    if thu is not None: paths.append((np.linalg.norm(thu - thu[:, i0:i0 + 1], axis=-1), "Arctan2 angles, unwrapped along the path: distance from the start", "$|\\theta(\\varphi) - \\theta(\\mathrm{start})|$ (rad)"))
    n = len(paths) + (thu is not None)
    fig = plt.figure(figsize=(WIDTH, 1.7 * n + .4)); gs = fig.add_gridspec(n, 1, hspace=.65)
    axes = [fig.add_subplot(gs[i]) for i in range(len(paths))]
    for i, (ax, (d, t, y)) in enumerate(zip(axes, paths)):
        band(ax, phi, d, BLUE, "median (10th to 90th percentile shaded)"); ax.plot(phi, d.max(0), color=ORANGE, lw=.8, label="largest over circles")
        titled(ax, "abc"[i], t); ax.set_ylabel(y); ax.set_ylim(bottom=0)
    if thu is not None: axes[1].axhline(2 * np.pi, color=MUTED, lw=.5, ls="-"); axes[1].text(phi[-1], 2 * np.pi, "$2\\pi$ ", ha="right", va="bottom", fontsize=6, color=MUTED)
    axes[0].legend(loc="lower center", fontsize=5.5)
    finish(fig, axes, phi, marks)
    if thu is not None:
        ax = fig.add_subplot(gs[n - 1]); ks = np.unique(np.concatenate([wind(i).ravel() for i in loop_ends])); w = .8 / len(loop_ends); cols = [BLUE, ORANGE, AQUA]
        for l, i in enumerate(loop_ends): ax.bar(ks + (l - (len(loop_ends) - 1) / 2) * w, [(wind(i) == v).sum() for v in ks], w, color=cols[l % 3], label=f"after loop {l + 1}")
        ax.set_xticks(ks); ax.set_xlabel("winding of a lifted angle (multiples of $2\\pi$; 0 = the loop closes without crossing a cut)"); ax.set_ylabel("(circle, angle) pairs"); ax.legend(loc="upper right")
        titled(ax, "c", "Winding of the Arctan2 angles after each loop")
    fig.suptitle(f"{name}:\ndistance from the start point ($\\varphi$ = {phi[i0]:.0f}$^\\circ$) along the loops", y=.97, fontsize=7.5)
    save(fig, out, "loops")
