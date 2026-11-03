"""How close is the reconstruction to the input, and how are the principal angles between the true and the reconstructed tangent spaces distributed (mean and variance of every direction)? Needs tangent.npz and tangent_compare.npz of the run folder (from diagnostics/tangent.py).
usage: python diagnostics/reconstruction_error.py <run folder> [<run folder> ...]      (writes into each run folder; call reconstruction_error(run, out=None) from Python for another folder)
Writes <out>/reconstruction_error.toml (the numbers below, and one array each for index, angle_mean_deg, angle_variance_deg2) and <out>/reconstruction_error.{png,pdf}.

Plan (per run, on the points of tangent.npz; x = input, x_hat = dec(enc(x)), T = true input tangent (D x k), T_recon = J_{dec.enc}(x) T)
  1. Point error:  e = |x - x_hat|, relative to |x| and to the typical distance between two input points (the scale of the data).
  2. Where the error points: e_tan = |T^T (x_hat - x)| (inside the tangent space of the input manifold), e_normal = sqrt(e^2 - e_tan^2) (off the manifold).
  3. Principal angles between T and T_recon (tangent_compare.npz, first comparison), all k of them, not only the largest: mean and variance of the angle of every direction (index 1 = smallest angle, k = largest),
     how many exceed each threshold per point (default 10 and 45 degrees), the median largest and second largest angle.
  4. Singular values of the pushforward T_recon (tangent_compare.npz): smallest / largest, a collapsed direction shows as a small ratio."""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import WIDTH, BLUE, ORANGE, INK, neurips, save, titled


def toml_value(v):
    if isinstance(v, str): return f'"{v}"'
    if isinstance(v, (list, np.ndarray)): return "[" + ", ".join(toml_value(t) for t in v) + "]"
    if isinstance(v, (int, np.integer)): return str(int(v))
    return f"{float(v):.6g}"


@neurips
def plot_reconstruction_error(out, name, e, xn, e_tan, e_nor, ang, sv, thresholds, i, mean, var):
    """Two groups: how well the points are reconstructed (a, b), and how well the tangent spaces are (c to f)."""
    fig = plt.figure(figsize=(WIDTH, 3.9)); top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1], hspace=.02)
    top.suptitle("Reconstruction of the points", x=.01, ha="left", fontsize=7.5, fontweight="bold", color=INK); bottom.suptitle("Reconstruction of the tangent spaces (principal angles to the true tangent)", x=.01, ha="left", fontsize=7.5, fontweight="bold", color=INK)
    a, b = top.subplots(1, 2, gridspec_kw=dict(wspace=.35, left=.08, right=.98, top=.7, bottom=.26)); c, d, f, g = bottom.subplots(1, 4, gridspec_kw=dict(wspace=.75, left=.08, right=.98, top=.7, bottom=.26))
    a.hist(e / xn, bins=30, color=BLUE); a.set_xlabel("$|x - \\hat{x}| \\,/\\, |x|$"); a.set_ylabel("Points"); titled(a, "a", "How far is the reconstruction?")
    b.hist(e_tan, bins=30, alpha=.6, color=BLUE, label="along the manifold"); b.hist(e_nor, bins=30, alpha=.6, color=ORANGE, label="off the manifold"); b.set_xlabel("Error component"); b.set_ylim(0, 1.4 * b.get_ylim()[1]); b.legend(loc="upper left"); titled(b, "b", "Where does the error point?")
    c.plot(i, mean, "o-", ms=3, color=BLUE); c.set_ylim(0, 90); c.set_ylabel("Degrees"); titled(c, "c", "Mean angle")
    d.plot(i, var, "o-", ms=3, color=ORANGE); d.set_ylabel("Degrees$^2$"); titled(d, "d", "Variance")
    for x in (c, d): x.set_xlabel("Direction"); x.set_xticks(i) if len(i) <= 10 else None
    f.hist((ang > thresholds[0]).sum(1), bins=np.arange(0, ang.shape[1] + 2) - .5, color=BLUE, rwidth=.8); f.set_xlabel(f"Angles above {thresholds[0]}$^\\circ$"); f.set_ylabel("Points"); titled(f, "e", "Tilted directions")
    g.hist(sv[:, -1] / sv[:, 0], bins=30, color=BLUE); g.set_xlabel("Smallest / largest"); g.set_ylabel("Points"); titled(g, "f", "Singular values")
    fig.suptitle(f"{name}: {len(ang)} points", y=1.04, fontsize=7.5)
    save(fig, out, "reconstruction_error")


def reconstruction_error(run, out=None, thresholds=(10, 45), n_scale=200, data=None):
    """run: run folder with tangent.npz and tangent_compare.npz (or data: the folder that holds them); out: folder to write into (default: run); thresholds: angles (degrees) counted per point; n_scale: points on each side of the
    pair-distance sample that sets the data scale. Returns the dict of numbers."""
    run = Path(run); out = Path(out) if out else run; out.mkdir(parents=True, exist_ok=True)
    data = Path(data) if data else run
    d, c = np.load(data / "tangent.npz"), np.load(data / "tangent_compare.npz")
    x, xh, T = d["x"].astype(np.float64), d["x_hat"].astype(np.float64), d["T"].astype(np.float64)
    ang = c["angles_0"]                                                                  # (n, k) ascending: true input vs true reconstructed
    sv = c["singular_values_1"]                                                          # (n, k) descending: pushforward T_recon
    err = x - xh
    e = np.linalg.norm(err, axis=1); xn = np.linalg.norm(x, axis=1)
    e_tan = np.linalg.norm(np.einsum("ndk,nd->nk", T, err), axis=1)
    e_nor = np.sqrt(np.maximum(e ** 2 - e_tan ** 2, 0))
    scale = np.median(np.linalg.norm(x[:n_scale, None] - x[None, n_scale:2 * n_scale], axis=2))   # typical distance between two input points
    r = dict(run=run.name, n=len(x), x_norm_median=np.median(xn), typical_pair_distance=scale,
             err_median=np.median(e), err_rel_norm_median=np.median(e / xn), err_rel_scale_median=np.median(e) / scale,
             err_tangential_median=np.median(e_tan), err_normal_median=np.median(e_nor),
             **{f"angles_over_{t}_mean": (ang > t).sum(1).mean() for t in thresholds}, largest_angle_median=np.median(ang[:, -1]),
             smallest_sv_over_largest_mean=(sv[:, -1] / sv[:, 0]).mean())
    if ang.shape[1] > 1: r["second_largest_angle_median"] = np.median(ang[:, -2])
    i = np.arange(1, ang.shape[1] + 1); mean, var = ang.mean(0), ang.var(0)
    (out / "reconstruction_error.toml").write_text("".join(f"{k} = {toml_value(v)}\n" for k, v in {**r, "index": i, "angle_mean_deg": mean, "angle_variance_deg2": var}.items()))

    plot_reconstruction_error(out, run.name, e, xn, e_tan, e_nor, ang, sv, thresholds, i, mean, var)
    return r


if __name__ == "__main__":
    for run in sys.argv[1:]: print(reconstruction_error(run), flush=True)
