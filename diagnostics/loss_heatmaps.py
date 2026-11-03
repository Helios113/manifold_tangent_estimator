"""
usage: python diagnostics/loss_heatmaps.py <sweep folder>      Writes <sweep>/loss_heatmaps/{best_losses.csv, <loss>.png}.
Plan: for every run of the sweep take the losses at its best step (best.pt: the step of the lowest validation reconstruction error; metrics.json best_step), nothing else.
  losses: relative_mse (val MSE / data variance), val_dist (distance loss), orth_euclidian_composed and orth_spectral_composed (after the final orthogonalisation), enc_dec_latent_err
One figure per loss. Rows = every on/off combination of (layers, embedded_extra, noise); columns = radius head off | on | on / off (ratio, < 1 means the head helps).
Each panel is a heat map, intrinsic_dim (x) by ambient_dim (y), log colour scale shared by the off and on columns of that row; empty cells are grid points that do not exist or did not finish."""
import json, sys, tomllib
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

LOSSES = ("relative_mse", "val_dist", "orth_euclidian_composed", "orth_spectral_composed", "enc_dec_latent_err")


def best_losses(sweep):
    rows = []
    for run in sorted((sweep / "runs").glob("*")):
        if not (run / "metrics.json").exists(): continue
        cfg, m = tomllib.load(open(run / "config.toml", "rb")), json.load(open(run / "metrics.json"))
        h = pd.read_csv(run / "history.csv", usecols=["step", "val_dist"]).dropna()
        rows.append(dict(run=run.name, intrinsic_dim=cfg["intrinsic_dim"], ambient_dim=cfg["ambient_dim"], layers=len(cfg["hidden_dims"]) + 1, embedded_extra=cfg["embedded_dim"] - cfg["intrinsic_dim"],
                         noise=cfg["noise"], radius_head=int(cfg["radius_head"] > 0), best_step=m["best_step"], relative_mse=m["relative_mse"], val_dist=h.loc[h.step == m["best_step"], "val_dist"].iloc[0],
                         orth_euclidian_composed=m["orth_after"]["orth_euclidian_composed"], orth_spectral_composed=m["orth_after"]["orth_spectral_composed"], enc_dec_latent_err=m["enc_dec_latent_err"]))
    return pd.DataFrame(rows)


def grid(df, loss, ks, ds, head):
    g = df[df.radius_head == head].pivot(index="ambient_dim", columns="intrinsic_dim", values=loss)
    return g.reindex(index=ds, columns=ks)


def plot(df, loss, out):
    ks, ds = sorted(df.intrinsic_dim.unique()), sorted(df.ambient_dim.unique())
    combos = sorted(df[["layers", "embedded_extra", "noise"]].drop_duplicates().itertuples(index=False))
    fig, axes = plt.subplots(len(combos), 3, figsize=(15, 3.6 * len(combos)), squeeze=False)
    for r, (layers, extra, noise) in enumerate(combos):
        sub = df[(df.layers == layers) & (df.embedded_extra == extra) & (df.noise == noise)]
        off, on = grid(sub, loss, ks, ds, 0), grid(sub, loss, ks, ds, 1)
        vals = np.concatenate([off.values.ravel(), on.values.ravel()]); vals = vals[np.isfinite(vals) & (vals > 0)]
        norm = LogNorm(vals.min(), vals.max()) if len(vals) else None
        ratio = (on / off).replace([np.inf, -np.inf], np.nan); rv = ratio.values[np.isfinite(ratio.values) & (ratio.values > 0)]
        rnorm = TwoSlopeNorm(1, min(rv.min(), 0.99), max(rv.max(), 1.01)) if len(rv) else None
        for c, (g, title, nm, cmap) in enumerate(((off, "head off", norm, "viridis"), (on, "head on", norm, "viridis"), (ratio, "on / off", rnorm, "RdBu_r"))):
            ax = axes[r, c]; im = ax.imshow(np.ma.masked_invalid(g.values), origin="lower", aspect="auto", cmap=cmap, norm=nm)
            ax.set_xticks(range(len(ks)), ks, fontsize=7); ax.set_yticks(range(len(ds)), ds, fontsize=7)
            ax.set_title(f"{title}: layers {layers}, extra {extra}, noise {noise}", fontsize=9); fig.colorbar(im, ax=ax)
            if r == len(combos) - 1: ax.set_xlabel("intrinsic_dim")
            if c == 0: ax.set_ylabel("ambient_dim")
    fig.suptitle(f"{loss} at the best step", fontsize=14, y=1.0)
    fig.savefig(out / f"{loss}.png", dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    sweep = Path(sys.argv[1]).resolve(); out = sweep / "loss_heatmaps"; out.mkdir(exist_ok=True)
    df = best_losses(sweep); df.to_csv(out / "best_losses.csv", index=False); print(f"{len(df)} runs", flush=True)
    for loss in LOSSES: plot(df, loss, out); print(loss, flush=True)
