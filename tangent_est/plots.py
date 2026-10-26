"""Figures of the protocol (step 9): error vs noise, error vs n, error vs diam/tau (each per k), error vs scale parameter, rate check."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .aggregate import COLORS, NAMES, STYLE

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 200, "figure.dpi": 130})
SHOW = ["lpca", "msvd", "al", "lego", "guidotti", "random", "pca", "oracle_lpca"]
FAMS = [("A", ""), ("C", "A"), ("B", ""), ("C", "B"), ("D", "")]
FNAME = {("A", ""): "A: circles", ("C", "A"): "C: A with boundary", ("B", ""): "B: Fourier", ("C", "B"): "C: B with boundary", ("D", ""): "D: near-flat"}


def _style(ax, xl, yl, title=None):
    ax.grid(True, color="#dddddd", lw=0.4); ax.set_axisbelow(True); ax.set_xlabel(xl); ax.set_ylabel(yl)
    if title: ax.set_title(title, loc="left", pad=3)


def _line(ax, g, xcol, ycol, m, **kw):
    g = g.sort_values(xcol)
    st = dict(color=COLORS[m], marker="o" if m not in STYLE else None, ms=3, lw=1.1, mec="white", mew=0.3, **STYLE.get(m, {}))
    st.update(kw)
    ax.errorbar(g[xcol], g[ycol], yerr=g.get(ycol + "_sd"), capsize=0, elinewidth=0.5, label=NAMES[m], **st)


def error_vs_noise(T, k, D, n, sampling, noise_model, ycol, out, title=""):
    """Rows: families; x: sigma sqrt(D)/tau (0 plotted at the left); y: ycol."""
    fig, axs = plt.subplots(1, 5, figsize=(13.5, 2.7), constrained_layout=True, sharey=True)
    for ax, (fam, base) in zip(axs, FAMS):
        g = T[(T.family == fam) & (T.base.fillna("") == base) & (T.k == k) & (T.D == D) & (T.n == n) & (T.sampling == sampling)]
        for m in SHOW:
            h = g[(g.method == m) & ((g.noise == noise_model) | (g.noise == "none"))].copy()
            if len(h) < 2: continue
            h["x"] = h.level.replace(0, np.nan).fillna(h.level[h.level > 0].min() / 3 if (h.level > 0).any() else 0.003)
            _line(ax, h, "x", ycol, m)
        ax.set_xscale("log"); ax.set_yscale("log"); _style(ax, "$\\sigma\\sqrt{D}/\\tau_{eff}$ (0 = leftmost)", ycol.replace("_", " "), FNAME[(fam, base)])
    axs[0].legend(frameon=False, ncol=1, loc="lower right")
    fig.suptitle(f"{title} k={k}, D={D}, n={n}, {sampling}, {noise_model} noise", fontsize=8)
    fig.savefig(out); plt.close(fig)


def error_vs_n(T, k, D, noise_model, level, ycol, out):
    fig, axs = plt.subplots(1, 5, figsize=(13.5, 2.7), constrained_layout=True, sharey=True)
    for ax, (fam, base) in zip(axs, FAMS):
        g = T[(T.family == fam) & (T.base.fillna("") == base) & (T.k == k) & (T.D == D) & (T.sampling == "uniform") & (T.noise == noise_model) & (T.level == level)]
        for m in SHOW:
            h = g[g.method == m]
            if h.n.nunique() < 2: continue
            _line(ax, h, "n", ycol, m)
        ax.set_xscale("log"); ax.set_yscale("log"); _style(ax, "n", ycol.replace("_", " "), FNAME[(fam, base)])
    axs[0].legend(frameon=False, loc="lower left")
    fig.suptitle(f"k={k}, D={D}, {noise_model} {level:g}", fontsize=8)
    fig.savefig(out); plt.close(fig)


def error_vs_dtau(T, k, D, n, noise_model, level, ycol, out, dcol="diam_over_tau"):
    """Scatter/lines of the error against the measured diam/tau over the cells (families colour-coded by marker)."""
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    g = T[(T.k == k) & (T.D == D) & (T.n == n) & (T.sampling == "uniform") & (T.noise == noise_model) & (T.level == level)]
    mk = {("A", ""): "o", ("C", "A"): "s", ("B", ""): "^", ("C", "B"): "v", ("D", ""): "D"}
    for m in SHOW[:5]:
        for (fam, base), mm in mk.items():
            h = g[(g.method == m) & (g.family == fam) & (g.base.fillna("") == base)]
            if len(h): ax.scatter(h[dcol], h[ycol], s=14, marker=mm, color=COLORS[m], alpha=0.85, label=NAMES[m] if (fam, base) == ("A", "") else None, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log"); _style(ax, "diam / $\\tau$ (measured)", ycol.replace("_", " "), f"k={k}, D={D}, n={n}, {noise_model} {level:g}")
    ax.legend(frameon=False); fig.savefig(out); plt.close(fig)


def error_vs_scale(df, cid, ycol, out):
    """Test-draw error against the scale parameter for every method in one cell."""
    d = df[(df.id == cid) & (df.draw == "test") & (df.status == "ok")].copy()
    if d.empty: return
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6), constrained_layout=True)
    def xval(r):
        c = __import__("json").loads(r)
        return c.get("count", c.get("k", c.get("bw_mult", c.get("gap", np.nan))))
    d["x"] = d.cfg.map(xval)
    panels = [("neighbour count (LPCA, AL ball count)", ["lpca", "al2", "al3", "al4"]), ("MS-SVD gap threshold", ["msvd"]), ("LEGO graph neighbours", ["lego"]), ("Guidotti bandwidth multiplier", ["guidotti"])]
    for ax, (xl, ms) in zip(axs, panels):
        for m in ms:
            h = d[d.method == m]
            if m == "guidotti":
                for a, hh in h.assign(alpha=h.cfg.map(lambda s: __import__("json").loads(s)["alpha"])).groupby("alpha"):
                    g = hh.groupby("x")[ycol].agg(["mean", "std"]).reset_index(); ax.errorbar(g.x, g["mean"], yerr=g["std"], marker="o", ms=3, lw=1, label=f"Guidotti α={a:g}")
            else:
                g = h.groupby("x")[ycol].agg(["mean", "std"]).reset_index()
                if len(g): ax.errorbar(g.x, g["mean"], yerr=g["std"], marker="o", ms=3, lw=1, color=COLORS[m], label=NAMES[m])
        ax.set_yscale("log"); ax.set_xscale("log" if ms != ["msvd"] else "linear"); _style(ax, xl, ycol.replace("_", " ")); ax.legend(frameon=False)
    fig.suptitle(cid, fontsize=8); fig.savefig(out); plt.close(fig)
