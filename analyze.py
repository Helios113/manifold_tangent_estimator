"""Summarise results.csv -> tables (markdown) and figures in results/."""
import sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.concat([pd.read_csv("results/results_ABCD.csv"), pd.read_csv("results/results_E.csv")], ignore_index=True)
METHODS = ["lpca", "ldle", "lego"]; COL = {"lpca": "#1f77b4", "ldle": "#ff7f0e", "lego": "#2ca02c"}
METRIC = "mean_max"   # mean over points of the largest principal angle (deg)
keys = ["exp", "manifold", "sigma", "n", "N", "seed"]

# per dataset: default config, and oracle best-over-grid config (chosen per dataset with ground truth!)
dflt = df[df.is_default].copy()
best = df.loc[df.groupby(keys + ["method"])[METRIC].idxmin()].copy()
# "tuned" = one config per (method, experiment, manifold, sigma...) chosen by mean over seeds (less optimistic)
grp = [c for c in keys if c != "seed"]
avg = df.groupby(grp + ["method", "params"])[METRIC].mean().reset_index()
tuned = avg.loc[avg.groupby(grp + ["method"])[METRIC].idxmin()]

def agg(d, extra=()):
    return d.groupby(list(extra) + ["method"])[METRIC].mean().unstack("method")[METHODS]

out = []
def md(title, t):
    out.append(f"### {title}\n\n" + t.round(2).to_markdown() + "\n")

A = lambda d: d[d.exp == "A_noise_ambient"]
for nm, d, lab in [("default hyper-parameters", dflt, "default"), ("tuned per (manifold, noise) on ground truth, averaged over seeds", tuned, "tuned")]:
    md(f"Exp A (ambient noise), mean max principal angle in degrees, {nm}: by noise level (avg over manifolds)", agg(A(d), ["sigma"]))
    md(f"Exp A, {nm}: by manifold (avg over noise levels)", agg(A(d), ["manifold"]))
    for s in sorted(A(d).sigma.unique()):
        if s in (0.0, 0.02, 0.05):
            md(f"Exp A, {nm}: sigma={s}", agg(A(d)[A(d).sigma == s], ["manifold"]))
    md(f"Exp B (tubular noise), {nm}", agg(d[d.exp == "B_noise_normal"], ["manifold", "sigma"]))
    md(f"Exp C (sample size, sigma=0.02), {nm}", agg(d[d.exp == "C_n_sweep"], ["manifold", "n"]))
    md(f"Exp D (ambient dimension, fixed noise energy), {nm}", agg(d[d.exp == "D_ambient_dim"], ["manifold", "N"]))

# win rate under tuned configs (A)
t = tuned[tuned.exp == "A_noise_ambient"].pivot_table(index=grp, columns="method", values=METRIC)[METHODS]
t = t[t.index.get_level_values("sigma") > 0]
wins = t.idxmin(axis=1).value_counts()
out.append("### Win counts (tuned, Exp A, sigma>0)\n\n" + wins.to_markdown() + "\n")
tm = df[df.is_default].groupby("method")["time"].median()
out.append("### Median wall-clock per run (default config, n varies, 1 thread)\n\n" + tm.round(2).to_markdown() + "\n")
md_ = lambda d, e: d[d.exp == e]
for nm, d in [("default", dflt), ("tuned", tuned)]:
    md(f"Exp E (ambient N=1000, noise energy sigma*sqrt(N)), {nm}", agg(md_(d, "E_N1000").assign(energy=lambda x: (x.sigma * np.sqrt(1000)).round(2)), ["manifold", "energy"]))

# ---- intrinsic dimension
dm = pd.read_csv("results/dim.csv")
for N_, lab in [(20, "N=20, all noise levels"), (1000, "N=1000")]:
    sub = dm[dm.N == N_]
    out.append(f"### Intrinsic-dimension accuracy (fraction of points with d_hat = d), {lab}, by rule\n\n" +
               sub.groupby(["rule", "method"]).acc.mean().unstack("method")[METHODS].round(3).to_markdown() + "\n")
sub = dm[(dm.N == 20) & (dm.rule == "fvar=0.9")]
out.append("### Dimension accuracy, fvar=0.9, N=20, by noise\n\n" + sub.groupby(["sigma", "method"]).acc.mean().unstack("method")[METHODS].round(3).to_markdown() + "\n")
out.append("### Dimension accuracy, fvar=0.9, N=20, by manifold\n\n" + sub.groupby(["manifold", "method"]).acc.mean().unstack("method")[METHODS].round(3).to_markdown() + "\n")
# ---- speed
sp = pd.read_csv("results/speed.csv")
out.append("### Speed (s, single thread, defaults, excl. kNN search) vs n (N=20)\n\n" + sp[sp.N == 20].groupby(["n", "method"]).time.mean().unstack("method")[METHODS].round(2).to_markdown() + "\n")
out.append("### Speed vs ambient N (n=2000)\n\n" + sp[sp.n == 2000].groupby(["N", "method"]).time.mean().unstack("method")[METHODS].round(2).to_markdown() + "\n")
out.append("### Accuracy (mean max angle, deg) in the speed runs vs n (N=20, sigma=0.02)\n\n" + sp[sp.N == 20].groupby(["n", "method"]).mean_max.mean().unstack("method")[METHODS].round(2).to_markdown() + "\n")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for m in METHODS:
    g = sp[sp.N == 20].groupby("n")[["time", "mean_max"]].mean()
    g = sp[(sp.N == 20) & (sp.method == m)].groupby("n")[["time", "mean_max"]].mean()
    axes[0].loglog(g.index, g.time, "o-", color=COL[m], label=m.upper())
    axes[1].plot(g.time, g.mean_max, "o-", color=COL[m], label=m.upper())
    g2 = sp[(sp.n == 2000) & (sp.method == m)].groupby("N").time.mean()
    axes[2].loglog(g2.index, g2.values, "o-", color=COL[m], label=m.upper())
axes[0].set_xlabel("n"); axes[0].set_ylabel("time (s)"); axes[1].set_xscale("log"); axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("mean max angle (deg)")
axes[2].set_xlabel("ambient N"); axes[2].set_ylabel("time (s)"); axes[0].legend()
for a in axes: a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("results/speed.png", dpi=130); plt.close(fig)
open("results/summary.md", "w").write("\n".join(out)); print("\n".join(out))

# ---- figures
def curve(ax, d, x, **kw):
    for m in METHODS:
        g = d[d.method == m].groupby(x)[METRIC].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], g["std"], label=m.upper(), color=COL[m], marker="o", capsize=2, **kw)
    ax.set_ylabel("mean max principal angle (deg)"); ax.grid(alpha=.3)

mans = list(df[df.exp == "A_noise_ambient"].manifold.unique())
for tag, src_df in [("default", dflt), ("tuned", tuned)]:
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True)
    for ax, m in zip(axes.ravel(), mans):
        d = src_df[(src_df.exp == "A_noise_ambient") & (src_df.manifold == m)]
        # per-seed std only meaningful for 'default'; for tuned there is 1 row per sigma
        curve(ax, d, "sigma"); ax.set_title(m); 
    for ax in axes[-1]: ax.set_xlabel("noise sigma (per ambient coord)")
    axes[0, 0].legend(); fig.suptitle(f"Exp A: tangent error vs noise ({tag} hyper-parameters, N=20, n=2000)")
    fig.tight_layout(); fig.savefig(f"results/expA_{tag}.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (e, x, t_) in zip(axes, [("C_n_sweep", "n", "sample size n (sigma=0.02)"), ("D_ambient_dim", "N", "ambient dim N (fixed noise energy)"),
                                  ("B_noise_normal", "sigma", "tubular noise sigma")]):
    curve(ax, tuned[tuned.exp == e].groupby(["method", x])[METRIC].mean().reset_index(), x); ax.set_xlabel(t_)
    if e != "B_noise_normal": ax.set_xscale("log")
axes[0].legend(); fig.suptitle("Tuned hyper-parameters, averaged over manifolds"); fig.tight_layout(); fig.savefig("results/expBCD.png", dpi=130)
