"""Tables and figures for the GPU benchmark (results/torch_noise_*.csv, results/torch_scale.csv)."""
import glob, textwrap, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["lpca", "ldle", "lego", "sig"]
NAME = dict(lpca="LPCA", ldle="LDLE", lego="LEGO", sig="Signature")
COL = dict(lpca="#0072B2", ldle="#D55E00", lego="#009E73", sig="#CC79A7")     # Okabe-Ito (colour-blind safe)
MK = dict(lpca="o", ldle="s", lego="^", sig="D")
LS = dict(lpca="-", ldle="--", lego="-.", sig=":")
# natural ambient dimension D of each manifold (MANIFOLDS.md); intrinsic dimension d
DIMS = {"circle": (1, 2), "trefoil": (1, 3), "sphere S2": (2, 3), "torus R3": (2, 3), "clifford T2": (2, 4),
        "swiss roll": (2, 3), "s-curve": (2, 3), "saddle": (2, 3), "sphere S3": (3, 4)}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "savefig.dpi": 300, "figure.dpi": 150})
W = 5.5                       # ICLR text width in inches
out = []

def style(ax, ylab=None, xlab=None, title=None):
    ax.grid(True, color="#dddddd", lw=0.4); ax.set_axisbelow(True)
    if ylab: ax.set_ylabel(ylab)
    if xlab: ax.set_xlabel(xlab)
    if title: ax.set_title(title, loc="left", pad=3)

def line(ax, x, y, m):
    ax.plot(x, y, color=COL[m], marker=MK[m], ms=3.6, lw=1.2, ls=LS[m], label=NAME[m], mec="white", mew=0.4)

def legend(fig, ncol=4, y=1.0):
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y), handlelength=2.4, columnspacing=1.6)

def save(fig, name, caption, top=0.93, bottom_pad=0.0):
    """Setup note below the plot; saves PNG and PDF."""
    note = textwrap.fill(caption, width=int(fig.get_figwidth() * 21))
    nl = note.count("\n") + 1
    h = fig.get_figheight(); foot = (nl * 0.125 + 0.1) / h
    fig.tight_layout(rect=[0, foot, 1, top])
    fig.text(0.01, 0.008, note, ha="left", va="bottom", fontsize=6.3, color="#444444", linespacing=1.25)
    for ext in ("png", "pdf"): fig.savefig(f"results/{name}.{ext}", bbox_inches="tight")
    plt.close(fig)

def md(title, df, nd=2):
    out.append(f"### {title}\n\n" + df.round(nd).to_markdown() + "\n")

# ------------------------------------------------------------------ accuracy vs noise
N = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/torch_noise_*.csv"))], ignore_index=True)
N = N[N.err.isna()] if "err" in N else N
dflt = N[N.is_default]
g = N.groupby(["manifold", "sigma", "method", "params"])[["in_mean", "oos_mean"]].mean().reset_index()
tuned = g.loc[g.groupby(["manifold", "sigma", "method"]).in_mean.idxmin()]          # chosen on in-sample error only
def agg(d, col): return d.groupby(["sigma", "method"])[col].mean().unstack("method")[ORDER]
sets = {"default hyper-parameters": (dflt, agg), "best hyper-parameters per manifold and noise": (tuned, agg)}

SETUP_NOISE = ("Setup: 9 manifolds (intrinsic dim. d = 1–3), each rescaled to max radius 1 and isometrically embedded from its natural "
               "dimension D = 2–4 into $\\mathbb{R}^{20}$ by a random orthonormal frame; i.i.d. Gaussian noise of std σ on all 20 coordinates. "
               "n = 2000 training points + 200 new points, 3 seeds per manifold and σ; curves are means over manifolds and seeds. "
               "Error = mean over points of the largest principal angle to the analytic tangent space. "
               "Defaults: LPCA k=20; LDLE k=25, k_lv=20; LEGO k=15, m=20, m0=60; Signature bandwidth 1.5×, α=1e-3. "
               "Best = setting with lowest in-sample error per manifold and σ (chosen with ground truth).")
fig, axes = plt.subplots(2, 2, figsize=(W, 4.6), sharex=True, sharey=True)
for j, (lab, (d, _)) in enumerate(sets.items()):
    for i, (col, what) in enumerate([("in_mean", "points in the cloud"), ("oos_mean", "new points")]):
        t = agg(d, col); ax = axes[i, j]
        for m in ORDER: line(ax, t.index, t[m], m)
        style(ax, "largest principal angle (deg)" if j == 0 else None, "noise level σ" if i == 1 else None,
              f"{'Default' if j == 0 else 'Best'} settings, {what}")
        md(f"Error (deg) vs noise, {lab}, {what}", t, 2)
legend(fig); save(fig, "fig_noise", SETUP_NOISE, top=0.93)

fig, axes = plt.subplots(3, 3, figsize=(W, 5.4), sharex=True)
for ax, man in zip(axes.ravel(), dflt.manifold.unique()):
    t = dflt[dflt.manifold == man].groupby(["sigma", "method"]).in_mean.mean().unstack("method")[ORDER]
    for m in ORDER: line(ax, t.index, t[m], m)
    d_, D_ = DIMS[man]
    style(ax, None, None, f"{man} ($d$={d_}, $D$={D_})")
for ax in axes[:, 0]: ax.set_ylabel("error (deg)")
for ax in axes[-1]: ax.set_xlabel("noise level σ")
legend(fig)
save(fig, "fig_noise_per_manifold",
     "Per-manifold error at default settings, for points in the cloud. $d$ = intrinsic dimension, $D$ = natural ambient dimension of the "
     "manifold; each is rescaled to max radius 1 and isometrically embedded into $\\mathbb{R}^{20}$ by a random orthonormal frame, with i.i.d. "
     "Gaussian noise of std σ on all 20 coordinates. n = 2000 points, mean over 3 seeds. Error = mean over points of the largest principal "
     "angle to the analytic tangent space. Default settings as in the noise figure.", top=0.93)
md("Default hyper-parameters, in the cloud, by manifold and noise (deg)", dflt.groupby(["manifold", "sigma", "method"]).in_mean.mean().unstack("method")[ORDER].unstack("sigma").round(1))
nfail = pd.concat([pd.read_csv(f) for f in glob.glob("results/torch_noise_*.csv")])
out.append(f"Failed runs (blank rows): {int(nfail.err.notna().sum())} of {len(nfail)}, all Signature with α = 1e-6 (fp32 Cholesky not positive definite).\n")

# ------------------------------------------------------------------ time and memory vs n
S = pd.read_csv("results/torch_scale.csv"); S = S[S.err.isna()] if "err" in S else S
A = S.groupby(["n", "method"]).mean(numeric_only=True).reset_index()
def piv(col, scale=1.0): return A.pivot(index="n", columns="method", values=col)[ORDER] * scale
SETUP_SCALE = ("Setup: sphere S² and Swiss roll (d = 2, D = 3) embedded isometrically into $\\mathbb{R}^{20}$ by a random orthonormal frame, "
               "Gaussian noise σ = 0.02 on all coordinates; default settings for all methods (LPCA k=20; LDLE k=25, k_lv=20; "
               "LEGO k=15, m=20, m0=60; Signature bandwidth 1.5×, α=1e-3); one run per point, averaged over the two manifolds. "
               "NVIDIA A40 (48 GB), PyTorch, fp32.")

t = piv("t_total")
md("Time (s) for all n tangent spaces, incl. kNN", t, 3)
md("  of which kNN search (s)", piv("t_knn"), 3); md("  of which fit (s)", piv("t_fit"), 3); md("  of which extracting tangents (s)", piv("t_tangent"), 3)

pk, st = piv("peak_extra_bytes", 1e-6), piv("state_bytes", 1e-6)
tp, tp1, mem = piv("oos100_time", 1000 / 100), piv("oos1_time", 1000), piv("oos100_peak_extra_bytes", 1e-6)
fig, axes = plt.subplots(2, 3, figsize=(W + 1.6, 4.0), sharex=True)
for m in ORDER:
    line(axes[0, 0], t.index, t[m], m); line(axes[0, 1], pk.index, pk[m], m); line(axes[0, 2], st.index, st[m].clip(lower=1e-3), m)
    line(axes[1, 0], tp.index, tp[m], m); line(axes[1, 1], mem.index, mem[m], m)
for ax in axes.ravel()[:5]: ax.set_xscale("log"); ax.set_yscale("log")
style(axes[0, 0], "wall-clock time (s)", None, "(a) Time, all $n$ points")
style(axes[0, 1], "peak extra GPU memory (MB)", None, "(b) Memory to compute")
style(axes[0, 2], "kept after fitting (MB)", None, "(c) Memory kept after fit")
style(axes[1, 0], "time per new point (ms)", "$n$ points in the cloud", "(d) Time, new points")
style(axes[1, 1], "extra peak GPU memory (MB)", "$n$ points in the cloud", "(e) Memory, new points")
axes[0, 2].set_xlabel("")
axes[1, 2].axis("off")
h, l = axes[0, 0].get_legend_handles_labels()
axes[1, 2].legend(h, l, loc="center", frameon=False, handlelength=2.4)
for ax in axes[0]: ax.tick_params(labelbottom=True)
save(fig, "time_memory_scaling",
     SETUP_SCALE + " (a) Total time for tangent spaces at all $n$ points, kNN included. (b,c) Memory excludes the data and output; "
     "LPCA keeps nothing (drawn at the 10$^{-3}$ MB floor). (d,e) A batch of 100 new points not in the cloud: LPCA and Signature evaluate "
     "directly; LDLE and LEGO, as published, append the points to the cloud and refit.", top=0.97)
md("Peak extra GPU memory while computing (MB)", pk, 0); md("State kept after fitting (MB), excluding the data and the output", st, 3)
md("Data (MB) and output tangents (MB)", pd.DataFrame({"data": A[A.method == "lpca"].set_index("n").data_bytes / 1e6,
                                                         "output": A[A.method == "lpca"].set_index("n").output_bytes / 1e6}), 2)
md("Extra time per new point (ms), batch of 100 new points", tp, 3); md("Extra time (ms), a single new point", tp1, 3)
md("Extra peak GPU memory for 100 new points (MB)", mem, 1)
md("Error on new points (deg), batch of 100, mean over manifolds", piv("oos100_mean"), 2)
md("Error in the cloud (deg) at default settings, mean over the two manifolds, σ = 0.02", piv("in_mean"), 2)
open("results/torch_summary.md", "w").write("\n".join(out)); print("\n".join(out))
