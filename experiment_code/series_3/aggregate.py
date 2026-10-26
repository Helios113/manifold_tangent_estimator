"""Series 3 (large k): aggregate results/series_3/cells/*.csv. Outputs in results/series_3/: summary_table.csv, failure_log.csv, failure_summary.csv, feasibility_table.csv, figures/*.png"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est import aggregate as AG, plots as PL
OUT = ROOT / "results" / "series_3"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True, parents=True)


def main():
    df = AG.load(OUT / "cells")
    S = AG.summary_table(df); S.to_csv(OUT / "summary_table.csv", index=False)
    bad = AG.failure_log(df); bad.to_csv(OUT / "failure_log.csv", index=False); AG.skipped_summary(bad).to_csv(OUT / "failure_summary.csv", index=False)
    lab = df[df.draw == "test"].groupby(["id", "family", "k", "D", "n", "noise", "level", "diam_over_tau_target"], dropna=False).agg(
        tau=("tau", "mean"), diam_over_tau=("diam_over_tau", "mean"), delta=("delta", "mean"), noise_over_nn=("noise_over_nn", "mean")).reset_index()
    lab.to_csv(OUT / "feasibility_table.csv", index=False)
    # error vs diam/tau, one panel per k (family D), A as separate markers
    for nm, lv in (("none", 0.0), ("iso", 0.1)):
        for yc in ("med_e", "max_ang"):
            fig, axs = plt.subplots(1, 3, figsize=(11.5, 2.8), constrained_layout=True, sharey=True)
            for ax, k in zip(axs, (50, 100, 300)):
                g = S[(S.k == k) & (S.noise == nm) & (S.level == lv)]
                for m in ("lpca", "msvd", "lego", "guidotti", "random", "pca", "oracle_lpca"):
                    h = g[(g.method == m) & (g.family == "D")].sort_values("diam_over_tau_target")
                    if len(h) > 1 or (len(h) == 1 and m in ("random",)):
                        ax.errorbar(h.diam_over_tau, h[yc], yerr=h[yc + "_sd"], marker="o" if m not in AG.NAMES or m not in PL.STYLE else None, ms=3, lw=1, color=AG.COLORS[m], label=AG.NAMES[m], **PL.STYLE.get(m, {}))
                    ha = g[(g.method == m) & (g.family == "A")]
                    if len(ha) and m not in PL.STYLE: ax.scatter(ha.diam_over_tau, ha[yc], marker="*", s=40, color=AG.COLORS[m], zorder=5)
                ax.set_xscale("log"); ax.set_yscale("log"); PL._style(ax, "diam / $\\tau$ (stars: family A)", yc.replace("_", " "), f"k={k}, D=1000, n=10$^4$, {nm} {lv:g}")
            axs[0].legend(frameon=False, fontsize=6); fig.savefig(FIG / f"dtau_{nm}{lv:g}_{yc}.png"); plt.close(fig)
    for cid in df[df.family == "D"].id.unique()[:1].tolist() + ["D_k100_D1000_n10000_uniform_none0_dt0.1", "A_k100_D1000_n10000_uniform_none0"]:
        PL.error_vs_scale(df, cid, "med_e", FIG / f"scale_{cid}.png")
    print("summary rows", len(S), "failures", len(bad))


if __name__ == "__main__":
    main()
