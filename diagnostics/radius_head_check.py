"""Is the radius head effectively a constant? Hypothesis: in high dimension the pair radii that the Arctan2 drops are nearly the same for all points, so the decoder can use constant radii and a fold costs nothing.
usage: python diagnostics/radius_head_check.py <run> [<run> ...]      Writes results/diagnostics/radius_head_check/{summary.csv, <run>.png}. Call radius_head_check(runs, out=None) from Python for another folder.

Plan (per run with a radius head, 2000 fresh clean sphere points)
  1. True radii: the radii sqrt(a^2 + b^2) of the pairs (a, b) = the two halves of the activated output of weight matrix arctan2_after, the quantity Arctan2 drops. Per pair, the coefficient of variation
     (std / mean over the points); median over the pairs.
  2. Predicted radii: radius_head applied to the unit pairs of the decoded angles (the decoder path from z = enc(x)). Same coefficient of variation; per-pair correlation with the true radii; median relative error.
  3. Reconstruction MSE of dec(enc(x)) with four choices of the radii: the radius head (the model), the per-pair mean of the predicted radii (constant), the per-pair mean of the true radii (constant),
     and the true radii of every point (oracle: information the latent code does not carry).
  If the constant radii give the same MSE as the radius head, the radius head carries no information about the point."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt

from common import WIDTH, BLUE, ORANGE, MUTED, open_run, output_dir, load_net, latent_inputs, neurips, save, titled


@neurips
def plot_run(out, name, r_true, r_pred, cv_true, cv_pred, mse):
    """(a) how much each pair's radius varies over the points, true against the head; (b) head against truth for the pair that varies most; (c) the reconstruction error with four choices of the radii."""
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, 2.3), gridspec_kw=dict(wspace=.75, width_ratios=[1, 1, 1.25]))
    order = torch.argsort(cv_true).numpy(); x = np.arange(len(order))
    ax[0].plot(x, cv_true.numpy()[order], "o-", ms=3, color=BLUE, label="true radii"); ax[0].plot(x, cv_pred.numpy()[order], "s-", ms=3, color=ORANGE, label="radius head")
    ax[0].set_xlabel("Pair (sorted by variation)"); ax[0].set_ylabel("Std / mean over the points"); ax[0].set_ylim(bottom=0); ax[0].legend(); titled(ax[0], "a", "Variation of the radii")
    if len(x) <= 10: ax[0].set_xticks(x)
    j = int(torch.argsort(cv_true)[-1]); t, p = r_true[:, j].numpy(), r_pred[:, j].numpy(); lim = [min(t.min(), p.min()), max(t.max(), p.max())]
    ax[1].plot(t, p, ".", ms=2, color=BLUE, alpha=.6); ax[1].plot(lim, lim, "--", color=MUTED, lw=.6); ax[1].text(.05, .95, f"$r$ = {np.corrcoef(t, p)[0, 1]:.2f}", transform=ax[1].transAxes, va="top", fontsize=6.5)
    ax[1].set_xlabel("True radius"); ax[1].set_ylabel("Radius head"); titled(ax[1], "b", f"Most varying pair (pair {j})")
    names = ["radius head (the model)", "constant: mean predicted", "constant: mean true", "oracle: true radius per point"]; vals = list(mse.values()); y = np.arange(4)[::-1]
    ax[2].barh(y, vals, color=[ORANGE, BLUE, BLUE, BLUE], height=.6); ax[2].set_yticks(y); ax[2].set_yticklabels(names); ax[2].set_xscale("log"); ax[2].set_xlabel("Reconstruction MSE")
    for yy, v in zip(y, vals): ax[2].text(v, yy, f" {v:.2g}", va="center", fontsize=6)
    ax[2].margins(x=.35); titled(ax[2], "c", "Reconstruction with other radii")
    fig.suptitle(Path(name).name, y=1.03, fontsize=7.5)
    save(fig, out, Path(name).name)


def run_row(name, out):
    """The numbers of one run (a dict, one row of summary.csv) and its figure <out>/<run>.png."""
    run, cfg = open_run(name)
    net = load_net(run, cfg); x = latent_inputs(cfg)[2]
    with torch.no_grad():
        z = net.encode(x); r_true = net.true_radii(x); r_pred = net.predicted_radii(z)
        const_pred = r_pred.mean(0, keepdim=True).expand_as(r_pred); const_true = r_true.mean(0, keepdim=True).expand_as(r_true)
        mse = lambda xh: ((xh - x) ** 2).mean().item()
        m = dict(model=mse(net.decode(z)), const_predicted_mean=mse(net.decode(z, const_pred)), const_true_mean=mse(net.decode(z, const_true)), oracle_true_radii=mse(net.decode(z, r_true)))
    cv = lambda r: (r.std(0) / r.mean(0))
    corr = torch.stack([torch.corrcoef(torch.stack([r_pred[:, j], r_true[:, j]]))[0, 1] for j in range(r_true.shape[1])])
    rel = ((r_pred - r_true).abs() / r_true).median(0).values
    row = dict(run=name, pairs=r_true.shape[1], cv_true_median=cv(r_true).median().item(), cv_pred_median=cv(r_pred).median().item(), corr_pred_true_median=corr.nan_to_num().median().item(),
               rel_error_pred_median=rel.median().item(), data_var=x.var(0).mean().item(), **{f"mse_{k}": v for k, v in m.items()})
    plot_run(out, name, r_true, r_pred, cv(r_true), cv(r_pred), m)
    return row


def radius_head_check(runs, out=None):
    """runs: run names or folders; out: folder to write into (default results/diagnostics/radius_head_check). Returns the table (one row per run)."""
    out = output_dir("radius_head_check", None, out); rows = []
    for name in runs:
        rows.append(run_row(name, out)); print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "summary.csv", index=False)
    return df


if __name__ == "__main__":
    radius_head_check(sys.argv[1:])
