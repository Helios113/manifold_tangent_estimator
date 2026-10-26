"""Aggregation of the cell files written by tangent_est.bench: tuning on the validation draw, mean +- s.d. over the test seeds, tables, failure log and figures
(protocol step 9).  Used by experiment_code/series_2/aggregate.py and series_3/aggregate.py."""
import glob, json, pathlib
import numpy as np, pandas as pd

METHODS = ["lpca", "msvd", "al2", "al3", "al4", "ldle", "lego", "guidotti"]
BASELINES = ["random", "pca", "oracle_lpca"]
NAMES = dict(lpca="LPCA", msvd="MS-SVD", al2="AL m=2", al3="AL m=3", al4="AL m=4", al="AL (tuned m)", ldle="LDLE", lego="LEGO", guidotti="Guidotti", random="random subspace",
             pca="global PCA", oracle_lpca="oracle LPCA")
COLORS = dict(lpca="#0072B2", msvd="#56B4E9", al2="#999999", al3="#E69F00", al4="#D55E00", al="#E69F00", ldle="#F0E442", lego="#009E73", guidotti="#CC79A7", random="#888888", pca="#000000", oracle_lpca="#0072B2")
STYLE = dict(random=dict(ls="--"), pca=dict(ls=":"), oracle_lpca=dict(ls="-."))
METRICS = ["med_e", "p90_e", "max_e", "med_ang", "p90_ang", "max_ang", "med_sin", "p90_sin", "max_sin"]


def _stable(cfg):
    """Grid-point identity without the data-dependent radius h (it differs between the validation and test draws; the target ball count identifies the grid point)."""
    c = json.loads(cfg) if isinstance(cfg, str) else {}
    c.pop("h", None); c.pop("cap", None)
    return json.dumps(c, sort_keys=True)


def load(cells_dir):
    fs = sorted(glob.glob(str(pathlib.Path(cells_dir) / "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    if len(df): df["cfg_raw"], df["cfg"] = df["cfg"], df["cfg"].map(_stable)
    return df


def tune(df):
    """For every (cell, method): pick the grid point with the lowest validation median e; return one row per (cell, draw=test, seed, method) at that grid point,
    with columns tuned_cfg and n_val_ok. Also builds 'al' = the order m (and h) with the lowest validation error among al2, al3, al4."""
    d = df[df.method.isin(METHODS)].copy()
    ok = d[d.status == "ok"]
    val = ok[ok.draw == "val"].sort_values("med_e")
    best = val.groupby(["id", "method"]).first()[["cfg", "med_e"]].rename(columns=dict(cfg="tuned_cfg", med_e="val_med_e")).reset_index()
    test = ok[ok.draw == "test"].merge(best, on=["id", "method"])
    test = test[test.cfg == test.tuned_cfg]
    al = best[best.method.isin(["al2", "al3", "al4"])].sort_values("val_med_e").groupby("id").first().reset_index()
    al_test = ok[ok.draw == "test"].merge(al[["id", "method", "tuned_cfg", "val_med_e"]], on=["id", "method"])
    al_test = al_test[al_test.cfg == al_test.tuned_cfg].copy(); al_test["method"] = "al"
    return pd.concat([test, al_test], ignore_index=True)


def baselines(df):
    ok = df[(df.status == "ok") & (df.draw == "test")]
    out = ok[ok.method.isin(["random", "pca"])].copy()
    o = ok[ok.method == "oracle_lpca"]
    if len(o):                                                                    # oracle: the grid point with the lowest mean test error
        g = o.groupby(["id", "cfg"]).med_e.mean().reset_index().sort_values("med_e").groupby("id").first().reset_index()[["id", "cfg"]]
        out = pd.concat([out, o.merge(g, on=["id", "cfg"])], ignore_index=True)
    return out


KEYS = ["id", "family", "base", "k", "D", "n", "sampling", "noise", "level", "diam_over_tau_target"]


def summary_table(df):
    """One row per (cell, method): mean and s.d. over the test seeds of the metrics, plus the labels required by the protocol."""
    t = pd.concat([tune(df), baselines(df)], ignore_index=True)
    cols = [m for m in METRICS + ["int_med_e", "bnd_med_e", "int_max_ang", "bnd_max_ang", "curv_med", "curv_max", "curv_norm", "seconds"] if m in t.columns]
    g = t.groupby(KEYS + ["method"], dropna=False)
    mean, sd = g[cols].mean(), g[cols].std()
    out = mean.join(sd, rsuffix="_sd").reset_index()
    lab = t.groupby(KEYS + ["method"], dropna=False).agg(n_seeds=("seed", "nunique"), tuned_cfg=("tuned_cfg", "first") if "tuned_cfg" in t else ("cfg", "first"),
                                                             tau=("tau", "mean"), diam_over_tau=("diam_over_tau", "mean"), delta=("delta", "mean"), nu=("nu", "mean"),
                                                             noise_over_nn=("noise_over_nn", "mean")).reset_index()
    out = out.merge(lab, on=KEYS + ["method"])
    for c in ("sigma_le_h_over_4", "high_noise_regime", "h", "ball_count_median"):
        if c in t.columns:
            out = out.merge(t.groupby(KEYS + ["method"], dropna=False)[c].agg("mean" if c in ("sigma_le_h_over_4", "high_noise_regime") else "median").reset_index(), on=KEYS + ["method"], how="left")
    return out


def failure_log(df):
    bad = df[df.status != "ok"].copy()
    keep = [c for c in ("id", "family", "base", "k", "D", "n", "noise", "level", "draw", "seed", "method", "cfg", "status", "reason", "trace") if c in bad.columns]
    return bad[keep]


def skipped_summary(bad):
    g = bad.groupby(["method", "status", "reason"]).agg(n_rows=("id", "size"), n_cells=("id", "nunique")).reset_index().sort_values("n_rows", ascending=False)
    return g
