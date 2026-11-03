"""LPCA and LEGO on the points of an autoencoder run's generator, compared with the true tangent space (same style as tangent_compare of diagnostics/tangent.py).
usage: python diagnostics/lpca_lego.py <run name or folder> [<out folder>]      Writes results/diagnostics/lpca_lego/<run>/. Call lpca_lego(run, out=None) from Python for another folder.

Plan
  1. Points: the training distribution of the run, the same points as tangent.py (eval_points.py): sphere points with the run's Gaussian noise added before the embedding, as in its training batches. N points, seed (diagnostics/diagnostics.toml).
  2. True tangent at x_i = f(p_i): the analytic tangent of the embedded sphere at the noiseless p_i (Sphere.embed_tangent_basis), (D x k).
  3. Estimators on the noisy cloud, k = the intrinsic dimension each, LPCA with ambient dim + 1 neighbours; LEGO with the setting of diagnostics/diagnostics.toml (series 8 tuned: 1616 neighbours, m = 200, m0 = 400, beta = 0.5).
     Tangents are taken at the first n_eval points.
  4. Per point, the principal angles between the estimated and the true tangent space.
  5. Saves lpca_lego.{npz,json} and the figures into results/diagnostics/lpca_lego/<run>/:
       tangent_compare.png   histograms of the largest principal angle (LPCA, LEGO) and of the mean principal angle (LPCA, LEGO)
       angles_all.png        all k sorted principal angles, mean over points, one curve per method"""
import json, sys
import numpy as np, torch
import matplotlib.pyplot as plt

from common import ROOT, WIDTH, BLUE, ORANGE, open_run, output_dir, neurips, save, titled
from eval_points import eval_points, settings
sys.path.insert(0, str(ROOT))
from tangent import principal_angles
from tangent_torch import LPCA, LEGO, DEV


COLORS = {"LPCA": BLUE, "LEGO": ORANGE}


@neurips
def plot_lpca_lego(out, run_name, angles, k, cfg, n):
    """tangent_compare: the angle to the true tangent space per point, LPCA against LEGO on the same axes; angles_all: the mean angle of each direction."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5), sharey=True)
    bins = np.linspace(0, 90, 46)
    for ax, (letter, title, f) in zip(axes, (("a", "Worst direction of the tangent space", lambda a: a[:, -1]), ("b", "Average over the directions", lambda a: a.mean(1)))):
        for name, a in angles.items():
            v = f(a); w = np.ones(len(v)) / len(v)
            ax.hist(v, bins=bins, weights=w, histtype="stepfilled", color=COLORS[name], alpha=.3, lw=0); ax.hist(v, bins=bins, weights=w, histtype="step", color=COLORS[name], lw=1, label=f"{name} (median {np.median(v):.2g}$^\\circ$)")
        titled(ax, letter, title); ax.set_xlabel("Angle to the true tangent space (degrees; 0 = exact)"); ax.set_xlim(0, 90); ax.legend(loc="upper right")
    axes[0].set_ylabel("Fraction of points")
    fig.suptitle(f"{run_name}: $S^{{{k}}}$ in $\\mathbb{{R}}^{{{cfg['ambient_dim']}}}$, noise {cfg['noise']}, {n} points", y=1.02, fontsize=7.5)
    save(fig, out, "tangent_compare")

    fig, ax = plt.subplots(figsize=(WIDTH * .6, 2.4))
    for name, a in angles.items(): ax.plot(np.arange(1, k + 1), a.mean(0), "o-", ms=3, color=COLORS[name], label=name)
    ax.set_xlabel("Direction (principal angles sorted from smallest to largest)"); ax.set_ylabel("Mean angle over points (degrees)"); ax.set_ylim(bottom=0); ax.legend()
    if k <= 10: ax.set_xticks(np.arange(1, k + 1))
    ax.set_title("Angle to the true tangent space, per direction", loc="left", pad=4)
    save(fig, out, "angles_all")


def lpca_lego(run, out=None):
    """run: run name (results/autoencoder/<run>) or run folder; out: folder to write into (default results/diagnostics/lpca_lego/<run>). Returns the summary dict."""
    run, cfg = open_run(run); out = output_dir("lpca_lego", run, out)
    pts = eval_points(cfg); N, N_EVAL, SEED, lego = pts["n"], pts["n_eval"], settings()["points"]["seed"], settings()["lego"]
    k = cfg["intrinsic_dim"]; k_lpca = cfg["ambient_dim"] + 1                                # LPCA: ambient dim + 1 neighbours
    X, T = pts["x"], pts["T"].numpy()                                                        # the noisy cloud and the true tangents at its first N_EVAL points
    print(f"points {tuple(X.shape)}, true tangent {T.shape}, device {DEV}", flush=True)

    Xg = X.to(DEV)
    ids = torch.arange(N_EVAL, device=DEV)
    est = {"LPCA": lambda: LPCA(k, k=k_lpca).fit(Xg).tangents_at(ids),
           "LEGO": lambda: LEGO(k, k=lego["k"], m=lego["m"], m0=lego["m0"], beta=lego["beta"]).fit(Xg).tangents_at(ids)}

    angles, tangents = {}, {}
    for name, fn in est.items():
        E = fn().cpu().numpy().astype(np.float64)
        tangents[name] = E
        angles[name] = principal_angles(E, T.astype(np.float64))
        a = angles[name]
        print(f"{name}: largest angle mean {a[:, -1].mean():.1f} deg, median {np.median(a[:, -1]):.1f}, p90 {np.percentile(a[:, -1], 90):.1f}; mean angle {a.mean():.1f}", flush=True)
        torch.cuda.empty_cache()

    np.savez(out / "lpca_lego.npz", **{f"angles_{n}": a for n, a in angles.items()}, **{f"tangent_{n}": t.astype(np.float32) for n, t in tangents.items()}, T_true=T)
    summary = {n: dict(largest_angle_mean=float(a[:, -1].mean()), largest_angle_median=float(np.median(a[:, -1])), largest_angle_p90=float(np.percentile(a[:, -1], 90)),
                       mean_angle_mean=float(a.mean()), smallest_angle_mean=float(a[:, 0].mean())) for n, a in angles.items()}
    summary["setup"] = dict(n=N, n_eval=N_EVAL, seed=SEED, noise=cfg["noise"], k=k, D=cfg["ambient_dim"], lpca_k=k_lpca, lego=lego)
    json.dump(summary, open(out / "lpca_lego.json", "w"), indent=1)

    plot_lpca_lego(out, run.name, angles, k, cfg, N)
    return summary


if __name__ == "__main__":
    lpca_lego(*sys.argv[1:3])
