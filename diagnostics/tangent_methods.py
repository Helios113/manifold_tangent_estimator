"""Tangent estimation compared across methods: the angle to the true tangent space of the autoencoder (its exact pushforward, its estimate from the encoded points, its decoder Jacobian) against LPCA and LEGO on the data, on one axis.
usage: python diagnostics/tangent_methods.py <tangent folder> <lpca_lego folder> [<out folder>]      (the folders of diagnostics/tangent.py and diagnostics/lpca_lego.py; default out: the parent of the tangent folder / tangent_methods)
Writes <out>/{tangent_methods.png, tangent_methods.pdf, summary.json}.

Plan
  1. Angles: tangent_compare.npz of tangent.py (principal angles per point between the true input tangent and the three autoencoder tangents) and lpca_lego.npz of lpca_lego.py (LPCA, LEGO against the same true tangent). Both use the same points
     (eval_points.py: the training distribution of the run), so the curves compare the methods on the same data.
  2. Plot: the cumulative distribution over the points of (a) the largest principal angle and (b) the mean principal angle, one curve per method; the further left a curve, the better the tangent."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import WIDTH, BLUE, AQUA, VIOLET, ORANGE, MAGENTA, FAINT, neurips, save, titled

AUTOENCODER = {"true input vs true reconstructed": ("Autoencoder: exact pushforward", BLUE), "true input vs estimated reconstructed": ("Autoencoder: from encoded points", AQUA), "true input vs decoder Jacobian": ("Autoencoder: decoder Jacobian", VIOLET)}
BASELINES = {"LPCA": ("LPCA on the data", ORANGE), "LEGO": ("LEGO on the data", MAGENTA)}


def load_angles(tangent_dir, lpca_dir):
    """{label: ((n, k) principal angles in degrees, colour)} for every method."""
    c, l = np.load(Path(tangent_dir) / "tangent_compare.npz"), np.load(Path(lpca_dir) / "lpca_lego.npz")
    names = [str(n) for n in c["names"]]; angles = {}
    for key, (label, colour) in AUTOENCODER.items(): angles[label] = (c[f"angles_{names.index(key)}"], colour)
    for key, (label, colour) in BASELINES.items(): angles[label] = (l[f"angles_{key}"], colour)
    return angles


@neurips
def plot_methods(out, name, angles):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    for ax, (letter, title, f) in zip(axes, (("a", "Worst direction of the tangent space", lambda a: a[:, -1]), ("b", "Average over the directions", lambda a: a.mean(1)))):
        for label, (a, colour) in angles.items():
            v = np.sort(f(a)); ax.plot(np.maximum(v, 1e-2), np.arange(1, len(v) + 1) / len(v), color=colour, label=label)
        ax.set_xscale("log"); ax.set_xlim(1e-1, 90); ax.set_xlabel("Angle to the true tangent space (degrees)"); titled(ax, letter, title); ax.axhline(.5, color=FAINT, lw=.5, ls=":")
    axes[0].set_ylabel("Fraction of points at or below the angle"); axes[0].legend(loc="upper left", fontsize=5.5)
    fig.suptitle(f"{name}: the further left, the better the tangent", y=1.03, fontsize=7.5)
    save(fig, out, "tangent_methods")


def tangent_methods(tangent_dir, lpca_dir, out=None, name=None):
    """tangent_dir, lpca_dir: the output folders of tangent.py and lpca_lego.py; out: folder to write into (default: tangent_dir/../tangent_methods). Returns the summary dict."""
    tangent_dir = Path(tangent_dir); out = Path(out) if out else tangent_dir.parent / "tangent_methods"; out.mkdir(parents=True, exist_ok=True)
    angles = load_angles(tangent_dir, lpca_dir)
    summary = {label: dict(n=len(a), largest_angle_median=float(np.median(a[:, -1])), largest_angle_p90=float(np.percentile(a[:, -1], 90)), mean_angle_median=float(np.median(a.mean(1)))) for label, (a, _) in angles.items()}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    plot_methods(out, name or tangent_dir.parent.parent.name, angles)
    return summary


if __name__ == "__main__":
    print(json.dumps(tangent_methods(*sys.argv[1:4]), indent=1))
