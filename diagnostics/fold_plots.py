"""The marks along the path and the loops plot of fold_determinant_stages.py (the fold plots are in fold_alignment.py)."""
import numpy as np
import matplotlib.pyplot as plt

from common import WIDTH, BLUE, ORANGE, AQUA, INK, FAINT, band, neurips, save, titled

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
    """The latent code along the loops: its distance from the start returns to 0 when a loop closes without a cut."""
    z = A["z"]
    d = np.linalg.norm(z - z[:, i0:i0 + 1], axis=-1)
    fig = plt.figure(figsize=(WIDTH, 2.4)); gs = fig.add_gridspec(1, 1, hspace=.65)
    ax = fig.add_subplot(gs[0])
    band(ax, phi, d, BLUE, "median (10th to 90th percentile shaded)"); ax.plot(phi, d.max(0), color=ORANGE, lw=.8, label="largest over circles")
    titled(ax, "a", "Latent code $z$: distance from its value at the start"); ax.set_ylabel("$|z(\\varphi) - z(\\mathrm{start})|$"); ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", fontsize=5.5)
    finish(fig, [ax], phi, marks)
    fig.suptitle(f"{name}:\ndistance from the start point ($\\varphi$ = {phi[i0]:.0f}$^\\circ$) along the loops", y=.97, fontsize=7.5)
    save(fig, out, "loops")
