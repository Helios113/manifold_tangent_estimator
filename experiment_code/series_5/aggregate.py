"""Series 5 (k = 3, D = 100 pre-study): tuned errors of every method. Outputs in results/series_5/: summary_table.csv, failure_log.csv, tables.md, figures/*.png"""
import sys, pathlib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est import aggregate as AG, plots as PL
OUT = ROOT / "results" / "series_5"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True, parents=True)
ORDER = ["lpca", "msvd", "al2", "al3", "al4", "al", "lego", "guidotti", "pca", "oracle_lpca", "random"]


def main():
    df = AG.load(OUT / "cells")
    S = AG.summary_table(df); S.to_csv(OUT / "summary_table.csv", index=False)
    AG.failure_log(df).to_csv(OUT / "failure_log.csv", index=False)
    S["cond"] = S.noise.where(S.noise == "none", S.noise + " " + S.level.astype(str))
    parts = []
    for yc, title in (("med_e", "median over points of e = mean sin^2(theta)"), ("max_ang", "largest principal angle, max over points (deg)"), ("med_ang", "largest principal angle, median over points (deg)")):
        for n in sorted(S.n.unique()):
            for fam in ("A", "B", "D"):
                g = S[(S.n == n) & (S.family == fam)]
                t = g.pivot_table(index="method", columns="cond", values=yc).reindex([m for m in ORDER if m in g.method.unique()])
                sd = g.pivot_table(index="method", columns="cond", values=yc + "_sd").reindex(t.index)
                tab = t.copy().astype(object)
                for r in t.index:
                    for c in t.columns:
                        tab.loc[r, c] = f"{t.loc[r, c]:.3g} ± {sd.loc[r, c]:.2g}" if pd.notna(t.loc[r, c]) else "-"
                parts.append(f"### {title}; family {fam}, n = {n}\n\n" + tab.to_markdown() + "\n")
    (OUT / "tables.md").write_text("\n".join(parts))
    # figure: tuned median e per method, grouped by condition, one panel per family and n
    for yc in ("med_e", "max_ang"):
        fig, axs = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
        for i, n in enumerate(sorted(S.n.unique())):
            for j, fam in enumerate(("A", "B", "D")):
                ax = axs[i, j]; g = S[(S.n == n) & (S.family == fam)]; conds = [c for c in ["none", "iso 0.01", "iso 0.1", "normal 0.1"] if c in g.cond.unique()]
                ms = [m for m in ORDER if m in g.method.unique() and m != "random"]
                w = 0.8 / len(ms)
                for q, m in enumerate(ms):
                    v = [g[(g.method == m) & (g.cond == c)][yc].mean() for c in conds]; e = [g[(g.method == m) & (g.cond == c)][yc + "_sd"].mean() for c in conds]
                    ax.bar(np.arange(len(conds)) + q * w, v, w, yerr=e, color=AG.COLORS[m], label=AG.NAMES[m], error_kw=dict(lw=0.5), hatch="//" if m in ("pca", "oracle_lpca") else None)
                ax.set_xticks(np.arange(len(conds)) + 0.4); ax.set_xticklabels(conds, fontsize=7); ax.set_yscale("log")
                PL._style(ax, "noise", yc.replace("_", " "), f"family {fam}, n={n}, k=3, D=100")
        axs[0, 0].legend(frameon=False, fontsize=6, ncol=2); fig.savefig(FIG / f"bars_{yc}.png"); plt.close(fig)
    for cid in ("A_k3_D100_n1000_uniform_iso0.1", "D_k3_D100_n10000_uniform_iso0.1_dt1", "B_k3_D100_n1000_uniform_none0"):
        PL.error_vs_scale(df, cid, "med_e", FIG / f"scale_{cid}.png")
    print(len(S), "summary rows")


if __name__ == "__main__":
    main()
