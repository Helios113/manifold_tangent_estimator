"""Does the orthogonality penalty get stuck on near-zero singular values of the weights? Over the checkpoints of a run (checkpoints/step*.pt, the weights saved every ckpt_every steps).
usage: python diagnostics/singular_values_over_training.py <run name or folder> [<run name or folder> ...]      (default out: results/diagnostics/singular_values_over_training/<run>)
Writes <out>/{summary.csv, jacobian_summary.csv, singular_values.{png,pdf}, jacobians.{png,pdf}}. Call singular_values_over_training(run, ...) from Python for the other parameters.

Plan (per checkpoint, per weight matrix W_i)
  1. All singular values of W_i (svdvals).
  2. Record: the smallest singular value, how many are below each threshold (default 0.5 and 0.1), the largest, and sum (sigma^2 - 1)^2 (the orthogonality penalty in the Frobenius form).
  3. The Jacobian of the layer of every W_i at n_points data points (clean points of the manifold, the first ones of plots.latent_inputs): the steps of net.encoder_steps() that belong to the layer (net.encoder_layers())
     (bias and activation, and after the layer arctan2_after the pair builder, layer norm and Arctan2), the last layer without activation. Its smallest singular value per point: the median and the minimum over the points. -> jacobian_summary.csv, jacobians.png
  4. Plot against training step: the smallest singular value of every W_i (log y), the number of singular values below the first threshold, and the orthogonality penalty of the run (history.csv, left to the run's own definition),
     so a plateau of the penalty can be matched with a singular value stuck near zero."""
import sys
from pathlib import Path
import pandas as pd, torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import WIDTH, INK, MUTED, BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED, open_run, output_dir, load_net, latent_inputs, jacobian, neurips, save, titled

CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]


def checkpoints(run):
    """[(step, path)] of run/checkpoints/step*.pt in order."""
    return [(int(ck.stem[4:]), ck) for ck in sorted((Path(run) / "checkpoints").glob("step*.pt"))]


def weight_singular_values(run, thresholds=(0.5, 0.1)):
    """One row per checkpoint and weight matrix: smallest and largest singular value, the number below each threshold and the penalty sum (sigma^2 - 1)^2."""
    rows = []
    for step, ck in checkpoints(run):
        for key, W in torch.load(ck).items():
            if not (key.startswith("W") and key[1:].isdigit()): continue                    # the weight matrices W1..Wn (not P of arctan2_project)
            s = torch.linalg.svdvals(W.double())
            rows.append({"step": step, "matrix": key, "shape": str(tuple(W.shape)), "smin": float(s.min()), "smax": float(s.max()),
                         **{f"n_below_{t}": int((s < t).sum()) for t in thresholds}, "penalty_sum": float(((s ** 2 - 1) ** 2).sum())})
    return pd.DataFrame(rows)


def layer_jacobians(run, cfg, n_points, dev):
    """One row per checkpoint and layer: the smallest singular value of the layer Jacobian at n_points points (median and minimum over the points) and the median largest."""
    rows = []
    x = latent_inputs(cfg)[2][:n_points].to(dev)
    for step, ck in checkpoints(run):
        net = load_net(run, cfg, f"checkpoints/{ck.name}").to(dev)
        h = x
        for i, f in enumerate(net.encoder_layers(), 1):
            with torch.no_grad(): J = jacobian(f, h)                                      # (n_points, out, in)
            sv = torch.linalg.svdvals(J)                                                  # (n_points, min(out, in))
            smin = sv[:, -1]
            rows.append({"step": step, "matrix": f"W{i}", "jac_shape": str(tuple(J.shape[1:])), "jac_smin_median": float(smin.median()), "jac_smin_min": float(smin.min()), "jac_smax_median": float(sv[:, 0].median())})
            with torch.no_grad(): h = f(h)
    return pd.DataFrame(rows)


@neurips
def plot_singular_values(out, name, df, history, first_threshold, window):
    """(a) what we observe: the orthogonality penalty of the run; then the explanation: (b) the smallest singular value of every weight matrix and (c) how many singular values are below the threshold."""
    pen = next((c for c in ("train_orth_penalty", "train_orth") if c in history), None)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(WIDTH, 5.6), gridspec_kw=dict(hspace=.4))
    if pen: ax[0].semilogy(history.step, history[pen].rolling(window, min_periods=1).mean(), color=INK); ax[0].set_ylabel("Orthogonality penalty"); titled(ax[0], "a", f"Orthogonality penalty of the run ({window}-step mean)")
    for i, (m, g) in enumerate(df.groupby("matrix")):
        c = CAT[i % len(CAT)]
        ax[1].semilogy(g.step, g.smin, "o-", ms=3, color=c); ax[1].text(g.step.iloc[-1], g.smin.iloc[-1], f"  {m} {g['shape'].iloc[0]}", color=c, va="center", fontsize=6)
        ax[2].plot(g.step, g[f"n_below_{first_threshold}"], "o-", ms=3, color=c); ax[2].text(g.step.iloc[-1], g[f"n_below_{first_threshold}"].iloc[-1], f"  {m}", color=c, va="center", fontsize=6)
    ax[1].set_ylabel("Smallest singular value"); titled(ax[1], "b", "Smallest singular value of each weight matrix (1 = orthogonal)")
    ax[2].set_ylabel(f"Number below {first_threshold}"); ax[2].yaxis.get_major_locator().set_params(integer=True); ax[2].set_ylim(bottom=-.2); titled(ax[2], "c", f"Singular values below {first_threshold} (a stuck direction)")
    ax[2].set_xlabel("Training step"); ax[0].margins(x=.1); fig.suptitle(name, y=.94, fontsize=7.5)
    save(fig, out, "singular_values")


@neurips
def plot_jacobians(out, name, jd):
    """The smallest singular value of each layer's Jacobian over training: median over the points (solid) and the worst point (dashed)."""
    fig, ax = plt.subplots(figsize=(WIDTH * .75, 2.8))
    for i, (m, g) in enumerate(jd.groupby("matrix")):
        c = CAT[i % len(CAT)]
        ax.semilogy(g.step, g.jac_smin_median, "o-", ms=3, color=c); ax.semilogy(g.step, g.jac_smin_min, "x--", ms=3.5, color=c, alpha=.7, lw=.8)
        ax.text(g.step.iloc[-1], g.jac_smin_median.iloc[-1], f"  {m} {g['jac_shape'].iloc[0]}", color=c, va="center", fontsize=6)
    ax.set_xlabel("Training step"); ax.set_ylabel("Smallest singular value of the layer Jacobian"); ax.margins(x=.15)
    ax.legend(handles=[Line2D([], [], color=MUTED, marker="o", ms=3, label="median over points"), Line2D([], [], color=MUTED, marker="x", ms=3.5, ls="--", lw=.8, label="worst point")], loc="best")
    ax.set_title(f"{name}: how much each layer squashes the data tangent", loc="left", pad=4); save(fig, out, "jacobians")


def singular_values_over_training(run, out=None, n_points=50, thresholds=(0.5, 0.1), window=200):
    """run: run name (results/autoencoder/<run>) or run folder; out: folder to write into (default results/diagnostics/singular_values_over_training/<run>); n_points: data points of the layer Jacobians;
    thresholds: singular values counted below (the first one is plotted); window: steps of the rolling mean of the penalty. Returns (weight table, Jacobian table)."""
    p, cfg = open_run(run); out = output_dir("singular_values_over_training", p, out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = weight_singular_values(p, thresholds); df.to_csv(out / "summary.csv", index=False)
    plot_singular_values(out, p.name, df, pd.read_csv(p / "history.csv"), thresholds[0], window)
    jd = layer_jacobians(p, cfg, n_points, dev); jd.to_csv(out / "jacobian_summary.csv", index=False)
    plot_jacobians(out, p.name, jd)
    return df, jd


if __name__ == "__main__":
    for r in sys.argv[1:]:
        df, jd = singular_values_over_training(r)
        print(r, df[df.matrix == "W1"].to_string(index=False), jd.to_string(index=False), sep="\n", flush=True)
