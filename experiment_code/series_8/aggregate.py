"""Series 8 (S^100 in R^1000, analytic tangents): tuned errors of every method. Outputs in results/series_8/: summary_table.csv, failure_log.csv, tables.md, figures/{bars_med_e,bars_max_ang,scale_<id>}.png"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est import aggregate as AG, plots as PL
OUT = ROOT / "results" / "series_8"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True, parents=True)
ORDER = ["lpca", "msvd", "al2", "al3", "al4", "al", "ldle", "lego", "guidotti", "pca", "oracle_lpca", "random"]
CID = "S_k100_D1000_n10000_uniform_none0"


def main():
    df = AG.load(OUT / "cells")
    S = AG.summary_table(df); S.to_csv(OUT / "summary_table.csv", index=False)
    AG.failure_log(df).to_csv(OUT / "failure_log.csv", index=False)
    S = S.set_index("method").reindex([m for m in ORDER if m in set(S.method)])
    cols = ["med_e", "p90_e", "max_e", "med_ang", "max_ang"]
    tab = pd.DataFrame({c: [f"{S.loc[m, c]:.3g} ± {S.loc[m, c + '_sd']:.2g}" for m in S.index] for c in cols}, index=[AG.NAMES[m] for m in S.index])
    (OUT / "tables.md").write_text("S^100 in R^1000, n = 1e4, no noise; e = mean sin^2 of the principal angles, angles in degrees (mean ± s.d. over 5 test draws, grid point tuned on a validation draw)\n\n" + tab.to_markdown() + "\n")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, (yc, title) in zip(axs, (("med_e", "median over points of e = mean sin^2(theta)"), ("max_ang", "largest principal angle, max over points (deg)"))):
        ax.bar(range(len(S)), S[yc], yerr=S[yc + "_sd"], color=[AG.COLORS[m] for m in S.index], error_kw=dict(lw=1),
               hatch=None); ax.set_xticks(range(len(S))); ax.set_xticklabels([AG.NAMES[m] for m in S.index], rotation=45, ha="right", fontsize=8)
        PL._style(ax, "", title, "S^100 in R^1000, n=1e4, analytic tangent ground truth")
    fig.savefig(FIG / "bars.png"); plt.close(fig)
    PL.error_vs_scale(df, CID, "med_e", FIG / f"scale_{CID}.png")
    print(tab)


if __name__ == "__main__":
    main()
