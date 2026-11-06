"""Helpers shared by the diagnostics scripts: finding a run and its output folder, chunked evaluation, Jacobians, the plot idioms that repeat."""

import functools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.func import jacrev, vmap
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auto_encoder"))
from plots import (
    latent_inputs,
    load,
    load_net,
    manifold,
)  # re-exported: the scripts import all run loading from here

L10 = np.log(10)


def open_run(run):
    """(run folder, config) of a run name (results/autoencoder/<run>) or a run folder."""
    p = Path(run)
    return load(p if p.exists() else ROOT / "results" / "autoencoder" / run)


def output_dir(script, run, out=None):
    """The folder a script writes into: out, or results/diagnostics/<script>/<run folder name> (without the run folder when run is None, for a script that compares several runs); created."""
    out = (
        Path(out)
        if out
        else ROOT / "results" / "diagnostics" / script / (Path(run).name if run else "")
    )
    out.mkdir(parents=True, exist_ok=True)
    return out


def map_chunks(fn, n, chunk, desc=None):
    """fn(slice) for consecutive slices of range(n) with at most chunk entries, the results (tensors, or dicts of tensors) concatenated: the memory of a Jacobian batch stays bounded. desc: label of a progress bar."""
    starts = range(0, n, chunk)
    starts = tqdm(starts, desc=desc, leave=False, mininterval=30) if desc else starts
    parts = [fn(slice(s, s + chunk)) for s in starts]
    return (
        {k: torch.cat([q[k] for q in parts]) for k in parts[0]}
        if isinstance(parts[0], dict)
        else torch.cat(parts)
    )


def jacobian(f, x):
    """(n, out, in) Jacobian of f at every row of x (n, in)."""
    return vmap(jacrev(f))(x)


def band(ax, x, y, color, label, lo=10, hi=90):
    """Median over the first axis of y (circles, len(x)) with the lo-th to hi-th percentile band."""
    ax.plot(x, np.median(y, 0), color=color, label=label)
    ax.fill_between(
        x, np.percentile(y, lo, 0), np.percentile(y, hi, 0), color=color, alpha=0.2
    )


def angle_summary(a):
    """Summary of principal angles a (n, k) in degrees (ascending per point):
    mean / median / p90 of the largest angle and mean of the smallest and of all angles."""
    largest = a[:, -1]
    return dict(
        largest_angle_mean=float(largest.mean()),
        largest_angle_median=float(np.median(largest)),
        largest_angle_p90=float(np.percentile(largest, 90)),
        mean_angle_mean=float(a.mean()),
        smallest_angle_mean=float(a[:, 0].mean()),
    )


def stage_colors(n):
    """n colours along viridis, one per stage (the yellow end is left out: it does not show on white)."""
    cm = plt.get_cmap("viridis")
    return [cm(0.05 + 0.85 * j / max(n - 1, 1)) for j in range(n)]


# ---- the look of every figure: a NeurIPS text width (5.5 in), serif text, no Type 3 fonts, the categorical slots of the reference palette
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
WIDTH = 5.5
STYLE = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": MUTED,
    "figure.dpi": 100,
}


def neurips(fn):
    """Run a plotting function with the figure style STYLE."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)

    return wrapped


def save(fig, folder, name):
    """folder/name.png (300 dpi) and folder/name.pdf (vector), then close the figure."""
    fig.savefig(folder / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def titled(ax, letter, title):
    """Panel title '(a)  title' at the left edge of the axes."""
    ax.set_title(f"({letter})  {title}", loc="left", pad=4, color=INK)
