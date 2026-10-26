"""Series 6 (series 5 over k = 3..300, D = 1000): tuned errors vs k. Outputs in results/series_6/: summary_table.csv, failure_log.csv, tables.md, figures/err_vs_k_*.png"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est import aggregate as AG, plots as PL
OUT = ROOT / "results" / "series_6"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True, parents=True)
ORDER = ["lpca", "msvd", "al2", "al3", "al4", "al", "lego", "guidotti", "pca", "oracle_lpca", "random"]
CONDS = [("none", 0.0), ("iso", 0.01), ("iso", 0.1), ("normal", 0.1)]


def main():
    df = AG.load(OUT / "cells")
    S = AG.summary_table(df); S.to_csv(OUT / "summary_table.csv", index=False)
    AG.failure_log(df).to_csv(OUT / "failure_log.csv", index=False)
    parts = []
    for yc, title in (("med_e", "median over points of e = mean sin^2(theta)"), ("max_ang", "largest principal angle, max over points (deg)")):
        for n in sorted(S.n.unique()):
            for fam in ("A", "B", "D"):
                for noise, level in CONDS:
                    g = S[(S.n == n) & (S.family == fam) & (S.noise == noise) & (S.level == level)]
                    if not len(g): continue
                    t = g.pivot_table(index="method", columns="k", values=yc).reindex([m for m in ORDER if m in g.method.unique()])
                    sd = g.pivot_table(index="method", columns="k", values=yc + "_sd").reindex(t.index)
                    tab = t.copy().astype(object)
                    for r in t.index:
                        for c in t.columns:
                            tab.loc[r, c] = f"{t.loc[r, c]:.3g} ± {sd.loc[r, c]:.2g}" if pd.notna(t.loc[r, c]) else "skipped"
                    parts.append(f"### {title}; family {fam}, n = {n}, noise {noise} {level}, columns = k\n\n" + tab.to_markdown() + "\n")
    (OUT / "tables.md").write_text("\n".join(parts))
    for yc in ("med_e", "max_ang"):
        for noise, level in CONDS:
            fig, axs = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
            for i, n in enumerate(sorted(S.n.unique())):
                for j, fam in enumerate(("A", "B", "D")):
                    ax = axs[i, j]; g = S[(S.n == n) & (S.family == fam) & (S.noise == noise) & (S.level == level)]
                    for m in [m for m in ORDER if m in g.method.unique()]:
                        h = g[g.method == m].sort_values("k")
                        ax.errorbar(h.k, h[yc], yerr=h[yc + "_sd"], color=AG.COLORS[m], label=AG.NAMES[m], marker="o", ms=3, lw=1, **AG.STYLE.get(m, {}))
                    ax.set_xscale("log"); ax.set_yscale("log")
                    PL._style(ax, "intrinsic dimension k", yc.replace("_", " "), f"family {fam}, n={n}, D=1000, {noise} {level}")
            axs[0, 0].legend(frameon=False, fontsize=6, ncol=2); fig.savefig(FIG / f"err_vs_k_{yc}_{noise}{level:g}.png"); plt.close(fig)
    print(len(S), "summary rows")


if __name__ == "__main__":
    main()
