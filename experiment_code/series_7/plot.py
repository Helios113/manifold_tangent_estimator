"""Series 7 figure: Guidotti time, peak GPU memory and error against the ambient dimension D.  python plot.py -> results/series_7/{summary.csv, guidotti_vs_D.png, .pdf}"""
import pathlib, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = pathlib.Path(__file__).resolve().parent.parent.parent / "results" / "series_7"
df = pd.concat([pd.read_csv(f) for f in sorted((R / "cells").glob("D*.csv"))])
g = df.groupby("D").agg(["mean", "min", "max"]); g.columns = ["_".join(c) for c in g.columns]; g.to_csv(R / "summary.csv")
fig, ax = plt.subplots(1, 4, figsize=(19, 4))
for a, (col, lab) in zip(ax, (("time_fit", "fit time (s)"), ("time_per_point", "tangent estimate, s per point"), ("peak_tangent", "peak extra GPU memory, tangent estimate (bytes)"), ("ang_med", "median largest principal angle (deg)"))):
    a.errorbar(g.index, g[col + "_mean"], yerr=[g[col + "_mean"] - g[col + "_min"], g[col + "_max"] - g[col + "_mean"]], marker="o", capsize=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("ambient dimension D"); a.set_ylabel(lab); a.grid(alpha=.3)
fig.suptitle("Guidotti signature function, product of circles k=2, n=2000, mean and min-max over 3 seeds")
fig.tight_layout(); fig.savefig(R / "guidotti_vs_D.png", dpi=150); fig.savefig(R / "guidotti_vs_D.pdf")
