"""Series 2 (small k): aggregate results/series_2/cells/*.csv (+ rate_check.csv) into tables and figures (plan.md step 9).
Outputs in results/series_2/: summary_table.csv, failure_log.csv, failure_summary.csv, feasibility_table.csv, rate_check_slopes.csv, figures/*.png, README.md"""
import sys, pathlib, json, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est import aggregate as AG, plots as PL, rate as RT
OUT = ROOT / "results" / "series_2"; FIG = OUT / "figures"; FIG.mkdir(exist_ok=True, parents=True)


def md(df, floatfmt=".3g"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def main():
    df = AG.load(OUT / "cells")
    S = AG.summary_table(df)
    S.to_csv(OUT / "summary_table.csv", index=False)
    bad = AG.failure_log(df); bad.to_csv(OUT / "failure_log.csv", index=False)
    fs = AG.skipped_summary(bad); fs.to_csv(OUT / "failure_summary.csv", index=False)
    # feasibility gate: per cell labels (delta, diam/tau, bottleneck/h_max) and which AL orders were feasible
    lab = df[df.draw == "test"].groupby(["id", "family", "base", "k", "D", "n", "sampling", "noise", "level"], dropna=False).agg(
        tau=("tau", "mean"), diam_over_tau=("diam_over_tau", "mean"), delta=("delta", "mean"), noise_over_nn=("noise_over_nn", "mean"), bottleneck_over_h_max=("bottleneck_over_h_max", "mean")).reset_index()
    feas = df[(df.draw == "val") & df.method.isin(["al3", "al4"])].groupby(["id", "method"]).status.agg(lambda s: "feasible" if (s == "ok").any() else "skipped").unstack().reset_index()
    lab = lab.merge(feas, on="id", how="left"); lab.to_csv(OUT / "feasibility_table.csv", index=False)
    # figures
    for k in (1, 2, 5, 10):
        for nm in ("iso", "normal", "tangent"):
            for yc in ("med_e", "max_ang"):
                PL.error_vs_noise(S, k, 1000, 1000, "uniform", nm, yc, FIG / f"noise_k{k}_{nm}_{yc}.png", "tuned,")
        for (nm, lv) in (("none", 0.0), ("iso", 0.1), ("normal", 0.1)):
            for yc in ("med_e", "max_ang"):
                PL.error_vs_n(S, k, 1000, nm, lv, yc, FIG / f"n_k{k}_{nm}{lv:g}_{yc}.png")
        for (nm, lv) in (("none", 0.0), ("iso", 0.1)):
            PL.error_vs_dtau(S, k, 1000, 1000, nm, lv, "med_e", FIG / f"dtau_k{k}_{nm}{lv:g}.png")
    for cid in ["A_k2_D1000_n1000_uniform_iso0.1", "B_k5_D1000_n1000_uniform_normal0.1", "D_k5_D1000_n1000_uniform_none0_dt1", "A_k5_D1000_n10000_uniform_iso0.1"]:
        PL.error_vs_scale(df, cid, "med_e", FIG / f"scale_{cid}.png")
    rc = OUT / "rate_check.csv"
    if rc.exists():
        R = pd.read_csv(rc); T, g = RT.rate_table(R); T.to_csv(OUT / "rate_check_slopes.csv", index=False); RT.rate_figure(g, FIG / "rate_check.png")
    print("summary rows", len(S), "failures", len(bad))


if __name__ == "__main__":
    main()
