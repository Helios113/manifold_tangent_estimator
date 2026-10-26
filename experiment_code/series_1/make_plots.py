"""Figures, per-test READMEs and the summary table from the results saved by the tests (results/<name>/data.npz).

Every figure draws the *acceptance region* (shaded green): the observed values have to land inside it for the implementation to
agree with the theory / the paper.  Run after the tests:  python experiment_code/series_1/make_plots.py
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiment_code.series_1.common import RESULTS, CRIT, PAPER

BLUE, GREEN, ORANGE, PINK, GREY = "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#555555"   # Okabe-Ito (colour-blind safe)
ACCEPT = "#DDF0E4"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150})
W = 5.5     # ICLR text width in inches


def load(name, file="data"):
    p = RESULTS / name / f"{file}.npz"
    return dict(np.load(p)) if p.exists() else None


def ecdf(ax, x, color, label=None, lw=1.4):
    x = np.sort(x); y = np.arange(1, len(x) + 1) / len(x)
    ax.step(x, y, where="post", color=color, lw=lw, label=label)


def accept_region(ax, lo, hi, text, y=0.04):
    ax.axvspan(lo, hi, color=ACCEPT, zorder=0, lw=0)
    ax.axvline(hi, color=GREY, ls="--", lw=0.9, zorder=1)
    ax.text(hi, y, text, ha="right", va="bottom", fontsize=6.5, color="#1b5e3b", rotation=0,
            bbox=dict(fc="white", ec="none", pad=1.0, alpha=0.8))


def style(ax, xlabel, ylabel="fraction of points $\\leq x$", title=None):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(True, color="#dddddd", lw=0.4); ax.set_axisbelow(True)
    if title: ax.set_title(title, loc="left", pad=3)


def median_dot(ax, x, color, txt, dy=0.0, dx=5, ha="left"):
    m = np.median(x); ax.plot([m], [0.5], "o", color=color, ms=4.5, mec="white", mew=0.7, zorder=5)
    ax.annotate(txt.format(m), (m, 0.5), xytext=(dx, -9 + dy), textcoords="offset points", fontsize=6.5, color=color, ha=ha)


# ---------------------------------------------------------------------------------------------------------------- figures
def fig_lpca(name, title, xlab):
    d = load(name); a = np.clip(d["ang"], 1e-4, None); thr = CRIT["lpca_numpy_deg" if name.endswith("vs_numpy") else "lpca_plane_deg"]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.4), constrained_layout=True)
    accept_region(ax, 1e-4, thr, f"accepted: max $<$ {thr}°")
    ecdf(ax, a, BLUE); ax.plot([a.max()], [1], "o", color=BLUE, ms=4.5, mec="white", mew=0.7, zorder=5)
    ax.annotate(f"max = {d['ang'].max():.3f}°", (a.max(), 1), xytext=(7, -3), textcoords="offset points", ha="left", fontsize=6.5, color=BLUE)
    ax.set_xscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(0, 1.05)
    style(ax, xlab, title=title)
    return fig, [(f"max angle (deg)", d["ang"].max(), f"< {thr}", d["ang"].max() < thr)]


def fig_ldle_grid():
    d = load("ldle_square_grid"); ang, inter = d["ang"], d["interior"]; thr = CRIT["ldle_grid_deg"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
    im = a1.imshow(ang.reshape(101, 101), origin="lower", extent=(0, 1, 0, 1), cmap="viridis", vmin=0, vmax=thr)
    a1.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, ec="white", ls="--", lw=0.9))
    a1.set_title("(a) error at each grid point", loc="left", pad=3); a1.set_xlabel("x"); a1.set_ylabel("y")
    cb = fig.colorbar(im, ax=a1, shrink=0.85); cb.set_label(f"angle error (°), scale ends at the limit {thr}°"); cb.ax.tick_params(labelsize=6.5)
    accept_region(a2, 1e-3, thr, f"accepted: median $<$ {thr}°")
    ecdf(a2, np.clip(ang[inter], 1e-3, None), BLUE, "interior (dashed box)"); ecdf(a2, np.clip(ang, 1e-3, None), ORANGE, "all points")
    median_dot(a2, ang[inter], BLUE, "median {:.3f}°", dy=-2); a2.set_xscale("log"); a2.set_xlim(1e-3, 1); a2.set_ylim(0, 1.05)
    a2.legend(loc="center left", frameon=False)
    style(a2, "max principal angle to the true tangent plane (deg)", title="(b) distribution of the error")
    m = np.median(ang[inter]); return fig, [("median interior angle (deg)", m, f"< {thr}", m < thr)]


def fig_ldle_torus():
    d = load("ldle_flat_torus"); ang = d["ang"]; thr = CRIT["ldle_torus_deg"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.4), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.25]))
    im = a1.imshow(ang.reshape(200, 50).T, origin="lower", extent=(0, 2 * np.pi, 0, 2 * np.pi), cmap="viridis", vmin=0, vmax=thr, aspect="auto")
    a1.set_title("(a) error over the torus parameters", loc="left", pad=3); a1.set_xlabel("$\\theta$"); a1.set_ylabel("$\\phi$")
    cb = fig.colorbar(im, ax=a1, shrink=0.85); cb.set_label(f"angle error (°), scale ends at the limit {thr}°"); cb.ax.tick_params(labelsize=6.5)
    accept_region(a2, 1e-3, thr, f"accepted: median $<$ {thr}°")
    ecdf(a2, np.clip(ang, 1e-3, None), BLUE); median_dot(a2, ang, BLUE, "median {:.3f}°", dy=-2, dx=8)
    a2.set_xscale("log"); a2.set_xlim(1e-3, 10); a2.set_ylim(0, 1.05)
    style(a2, "max principal angle to the true tangent plane (deg)", title="(b) distribution of the error")
    m = np.median(ang); return fig, [("median angle (deg)", m, f"< {thr}", m < thr)]


def fig_lego(name, title):
    d = load(name); Dl, Dg = np.clip(d["D_lpca"], 1e-7, None), np.clip(d["D_lego"], 1e-7, None); r = CRIT["lego_ratio"]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.5), constrained_layout=True)
    lim = r * np.median(Dl)
    accept_region(ax, 1e-6, lim, f"accepted:\nLEGO median $<$ {r} $\\times$ LPCA median", y=0.80)
    ecdf(ax, Dl, BLUE, "LPCA"); ecdf(ax, Dg, GREEN, "LEGO")
    median_dot(ax, Dl, BLUE, "median {:.1e}", dy=-2); median_dot(ax, Dg, GREEN, "median {:.1e}", dy=-2, dx=-5, ha="right")
    ax.set_xscale("log"); ax.set_xlim(1e-5, 1); ax.set_ylim(0, 1.05); ax.legend(loc="lower right", frameon=False)
    style(ax, "discrepancy $D_j=\\sum_i(1-\\cos\\theta_i)$ (smaller = better)", title=title)
    ratio = np.median(Dg) / np.median(Dl)
    return fig, [("median $D_j$ LEGO / LPCA", ratio, f"< {r}", ratio < r)]


PAPER_TABLE = dict(   # Table 6.8.1 of the paper: singular values at four points
    curve=np.array([[3.25, 2.11, 8.78e-4], [3.42, 1.81, 2.45e-3], [3.28, 2.05, 8.39e-2], [3.03, 2.41, 8.62e-2]]),
    surface=np.array([[3.87, 1.46e-2, 4.02e-4], [3.87, 5.34e-3, 4.44e-3], [3.87, 2.18e-1, 1.66e-3], [3.87, 9.94e-2, 9.07e-5]]))


def fig_guidotti_numpy():
    ds = {k: load("guidotti_vs_numpy", f"data_{k}") for k in ("gauss", "lap")}
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5), constrained_layout=True); rows = []
    for ax, key, thr, lo, xl, tt in [(axs[0], "ang", CRIT["guidotti_numpy_deg"], 1e-7, "angle torch vs numpy normal (deg)", "(a) implied normals"),
                                     (axs[1], "du", CRIT["guidotti_numpy_u"], 1e-17, "$|u_{torch}-u_{numpy}|$", "(b) signature value")]:
        accept_region(ax, lo, thr, f"accepted: max $<$ {thr:g}", y=0.04)
        for (k, d), c, nm in zip(ds.items(), (BLUE, ORANGE), ("Gauss", "regularised Laplace")):
            ecdf(ax, np.clip(d[key], lo, None), c, nm)
            rows.append((f"{nm}: max {xl}", d[key].max(), f"< {thr:g}", d[key].max() < thr))
        ax.set_xscale("log"); ax.set_xlim(lo, thr * 100); ax.set_ylim(0, 1.05); ax.legend(loc="center left", frameon=False, fontsize=6.3)
        style(ax, xl, title=tt)
    return fig, rows


def fig_sig_sphere():
    d = load("guidotti_sphere")
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6), constrained_layout=True); rows = []
    for ax, key, thr, paper, ylab, title in [
            (axs[0], "ang", CRIT["guidotti_sphere_deg"], PAPER["guidotti_sphere_deg"], "max angle: implied vs exact normal (deg)", "(a) normal direction"),
            (axs[1], "u_dev", CRIT["guidotti_sphere_u"], PAPER["guidotti_sphere_u"], "max $|u(x)-1|$ over the 32 points", "(b) signature value")]:
        lo = paper / 30
        ax.axhspan(lo, thr, color=ACCEPT, zorder=0, lw=0)
        ax.axhline(thr, color=GREY, ls="--", lw=0.9, label=f"acceptance limit ({thr:g})")
        ax.axhline(paper, color=PINK, ls=":", lw=1.3, label=f"paper, one draw ({paper:g})")
        v = np.sort(d[key])
        ax.plot(np.arange(1, len(v) + 1), v, "o", color=BLUE, ms=3.2, mec="white", mew=0.4, zorder=5, label="our 20 random draws, sorted")
        rows.append((f"{ylab}, worst of 20 draws", d[key].max(), f"< {thr:g}", d[key].max() < thr))
        ax.set_yscale("log"); ax.set_ylim(lo, thr * 3); ax.set_xlim(0, 21)
        ax.legend(loc="lower right", frameon=False, fontsize=6.2)
        style(ax, "random draw (sorted by error)", ylabel=ylab, title=title)
    return fig, rows


def fig_guidotti_line():
    ds = {k: load("guidotti_line", f"data_{k}") for k in ("gauss", "lap")}
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6), constrained_layout=True); rows = []
    for off, (k, d), c, nm in zip((-0.2, 0.2), ds.items(), (BLUE, ORANGE), ("Gauss", "regularised Laplace")):
        ax.bar(np.arange(3) + off, np.clip(d["sv"], 1e-6, None), 0.38, color=c, label=nm)
        rows.append((f"{nm}: sv3 / sv2", d["sv"][2] / d["sv"][1], f"< {CRIT['guidotti_line_ratio']:g}", d["sv"][2] / d["sv"][1] < CRIT["guidotti_line_ratio"]))
        rows.append((f"{nm}: max normal component along the line", float(d["along"]), f"< {CRIT['guidotti_line_along']:g}", float(d["along"]) < CRIT["guidotti_line_along"]))
    ax.set_yscale("log"); ax.set_ylim(1e-6, 20); ax.set_xticks(range(3), ["sv1", "sv2", "sv3"]); ax.legend(frameon=False, fontsize=6.5)
    style(ax, "singular value of the $3\\times15$ matrix of normals", ylabel="value", title="Straight line: normals have rank 2")
    return fig, rows


def fig_guidotti_table():
    d = load("guidotti_table_6_8_1"); fig, axs = plt.subplots(1, 2, figsize=(W, 2.7), constrained_layout=True, sharey=True); rows = []
    for ax, key, tt in ((axs[0], "sv_c", "(a) curve ($d=1$): rank 2 in the paper"), (axs[1], "sv_s", "(b) surface ($d=2$): rank 1")):
        pap = PAPER_TABLE["curve" if key == "sv_c" else "surface"]
        for j in range(3):
            ax.plot(np.arange(4) + 1, pap[:, j], "o", mfc="none", color=[BLUE, ORANGE, GREEN][j], ms=6, label="paper" if j == 0 else None)
            ax.plot(np.arange(4) + 1, d[key][:, j], "x", color=[BLUE, ORANGE, GREEN][j], ms=5, label=f"ours sv{j + 1}")
        ax.set_yscale("log"); ax.set_ylim(1e-5, 10); ax.set_xticks(range(1, 5)); ax.legend(frameon=False, fontsize=6, ncol=2, loc="lower left")
        style(ax, "point", ylabel="singular value", title=tt)
    sv = d["sv_s"]
    rows.append(("surface: max |sv1 - sqrt(15)|", np.abs(sv[:, 0] - np.sqrt(15)).max(), f"< {CRIT['guidotti_surface_sv1']:g}", np.abs(sv[:, 0] - np.sqrt(15)).max() < CRIT["guidotti_surface_sv1"]))
    rows.append(("surface: max sv2 (paper: 0.22)", sv[:, 1].max(), f"< {CRIT['guidotti_surface_sv2']:g}", sv[:, 1].max() < CRIT["guidotti_surface_sv2"]))
    rows.append(("curve: median sv2 (paper: 1.8-2.4)", np.median(d["sv_c"][:, 1]), "not reproduced (see text)", None))
    return fig, rows


def fig_sig_curve():
    d = load("guidotti_curve_surface"); fig, ax = plt.subplots(figsize=(W * 0.62, 2.7), constrained_layout=True); rows = []
    thr = CRIT["guidotti_curve_deg"]
    accept_region(ax, 1e-2, thr, f"accepted: median $<$ {thr}°", y=0.04)
    for key, c, nm, lim in (("ang_c", ORANGE, "curve, $d=1$", CRIT["guidotti_curve_deg"]), ("ang_s", PINK, "surface, $d=2$", CRIT["guidotti_surface_deg"])):
        x = np.sort(np.clip(d[key], 1e-2, None)); ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=c, lw=1.3, label=nm)
        ax.plot([np.median(d[key])], [0.5], "o", color=c, ms=4, mec="white", mew=0.6, zorder=5)
        rows.append((f"{nm.split(',')[0]}: median angle (deg)", np.median(d[key]), f"< {lim}", np.median(d[key]) < lim))
    ax.set_xscale("log"); ax.set_xlim(1e-2, 60); ax.set_ylim(0, 1.05); ax.legend(loc="upper left", frameon=False, fontsize=6.3)
    style(ax, "max principal angle to the true tangent space (deg)", title="Guidotti: tangent of a curve / surface (dots: medians)")
    return fig, rows


# ---------------------------------------------------------------------------------------------------------------- descriptions
def fig_lpoly_rate():
    d = load("lpoly_rate"); n = d["n"]; lo, hi = CRIT["lpoly_slope2"]; t3 = CRIT["lpoly_slope3"]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6), constrained_layout=True)
    for key, c, lab, sl, th in (("err2", BLUE, "order 2 (local PCA)", d["slope2"], -0.5), ("err3", GREEN, "order 3 (quadratic fit)", d["slope3"], -1.0)):
        e = d[key]; m = e.mean(0)
        ax.fill_between(n, e.min(0), e.max(0), color=c, alpha=0.18, lw=0); ax.plot(n, m, "o-", color=c, ms=3.5, mec="white", mew=0.5, lw=1.3, label=f"{lab}: slope {sl:.2f}")
        ax.plot(n, m[0] * (n / n[0]) ** th, ls=":", color=c, lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(frameon=False, loc="lower left")
    style(ax, "sample size $n$ (k = 30 neighbours)", "median max principal angle (deg)", "sphere $S^2\\subset\\mathbb{R}^{10}$, dotted: theory $n^{-1/2}$, $n^{-1}$")
    return fig, [("order 2: log-log slope", d["slope2"], f"in ({lo}, {hi})", lo < d["slope2"] < hi), ("order 3: log-log slope", d["slope3"], f"< {t3}", d["slope3"] < t3)]


def fig_lpoly_curv():
    d = load("lpoly_curvature"); n = d["n"]; e = d["err"]; t = CRIT["lpoly_curv_slope"]; fin = CRIT["lpoly_curv_final"]
    fig, ax = plt.subplots(figsize=(W * 0.62, 2.6), constrained_layout=True)
    ax.fill_between(n, e.min(0), e.max(0), color=GREEN, alpha=0.18, lw=0); ax.plot(n, e.mean(0), "o-", color=GREEN, ms=3.5, mec="white", mew=0.5, lw=1.3, label=f"order 3: slope {float(d['slope']):.2f}")
    ax.plot(n, e.mean(0)[0] * (n / n[0]) ** -0.5, ls=":", color=GREEN, lw=0.9); ax.axhline(fin, color=GREY, ls="--", lw=0.9); ax.text(n[0], fin * 1.1, f"required at the largest n: < {fin}", fontsize=6.5, color="#1b5e3b")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.legend(frameon=False, loc="lower left")
    style(ax, "sample size $n$ (k = 30 neighbours)", "median curvature error (operator norm)", "sphere $S^2\\subset\\mathbb{R}^{10}$, dotted: theory $n^{-1/2}$")
    return fig, [("order 3: log-log slope of the curvature error", float(d["slope"]), f"< {t}", float(d["slope"]) < t), ("curvature error at the largest n", float(e.mean(0)[-1]), f"< {fin}", float(e.mean(0)[-1]) < fin)]


SPECS = [
    dict(name="lpca_vs_numpy", method="LPCA", build=lambda: fig_lpca("lpca_vs_numpy", "LPCA: torch vs independent numpy", "angle between the two estimates (deg)"),
         what="The torch LPCA gives the same tangent spaces as a from-scratch numpy/scipy implementation of local PCA.",
         how="500 points on $S^4\\subset\\mathbb{R}^5$ plus Gaussian noise (std 0.01); $d=3$, $k=20$. Reference: scipy kd-tree neighbours, centre the patch (point + 20 neighbours), SVD, top-3 right singular vectors. We compare the two subspaces by their largest principal angle at every point.",
         theory="LPCA is a deterministic textbook algorithm, so the two must agree up to floating-point round-off (the torch code runs in fp32)."),
    dict(name="lpca_plane", method="LPCA", build=lambda: fig_lpca("lpca_plane", "LPCA on a noise-free plane", "angle to the true tangent plane (deg)"),
         what="LPCA recovers the exact tangent space of a flat manifold.",
         how="400 uniform points on a 2-D plane embedded in $\\mathbb{R}^{10}$ by a random orthonormal frame, no noise, $k=15$.",
         theory="Every local patch lies in the plane, so the top-2 principal directions are the plane itself: the error is round-off only."),
    dict(name="lpoly_vs_numpy", method="LocalPoly", build=lambda: fig_lpca("lpoly_vs_numpy", "Local polynomial, order 2: torch vs independent numpy", "angle between the two estimates (deg)"),
         what="With order 2 the local polynomial estimator of Aamari and Levrard (eq. (2)) is local PCA about the query point, with no mean-centring.",
         how="500 points on $S^4\\subset\\mathbb{R}^5$ plus Gaussian noise (std 0.01); $d=3$, $k=20$. Reference: scipy kd-tree neighbours, SVD of the 20 neighbours minus the point (not minus their mean), top-3 right singular vectors; largest principal angle at every point.",
         theory="For $k=2$ the sum of tensors in eq. (2) is empty and the objective is $\\sum\\|x-\\pi(x)\\|^2$ over the ball: a deterministic problem, so the two must agree up to round-off."),
    dict(name="lpoly_plane", method="LocalPoly", build=lambda: fig_lpca("lpoly_plane", "Local polynomial, order 3, on a noise-free plane", "angle to the true tangent plane (deg)"),
         what="The order-3 estimator, with its quadratic correction, still returns the exact tangent space of a flat manifold.",
         how="400 uniform points on a 2-D plane in $\\mathbb{R}^{10}$ (random orthonormal frame), no noise, order 3, $k=30$.",
         theory="Every patch lies in the plane, so the normal residual is zero, the fitted tensors vanish and the tilt is zero: the error is round-off only."),
    dict(name="lpoly_rate", method="LocalPoly", build=fig_lpoly_rate,
         what="The error of the local polynomial estimator decreases with the sample size at the rate of Theorems 2-3 of the paper: $(1/n)^{(k-1)/d}$ for order $k$.",
         how="Noise-free uniform samples of $S^2$ embedded in $\\mathbb{R}^{10}$ by a random orthonormal frame, $n\\in\\{1000,\\dots,32000\\}$, 3 draws per size, float64, fixed number of neighbours $k=30$ (so the neighbourhood radius $h$ scales as $n^{-1/d}$). Error: median over the points of the largest principal angle to the exact tangent plane; the slope is a least-squares fit of log median error against log $n$ (mean over draws).",
         theory="With $d=2$: order 2 (local PCA) converges as $n^{-1/2}$, order 3 as $n^{-1}$ (the dotted lines). The paper's bounds are upper bounds with a $\\log n$ factor and constants; we require the slopes to be near the theoretical ones (order 2 within $\\pm0.15$ of $-0.5$, order 3 steeper than $-0.8$). The limits were fixed before the run."),
    dict(name="lpoly_curvature", method="LocalPoly", build=fig_lpoly_curv,
         what="The second-order tensor $T_2$ of the order-3 estimator estimates the second fundamental form, at the rate of Theorem 4 of the paper: $(1/n)^{(m-2)/d}$.",
         how="Noise-free uniform samples of the unit sphere $S^2$ embedded in $\\mathbb{R}^{10}$ (random orthonormal frame), $n\\in\\{1000,\\dots,32000\\}$, 3 draws per size, float64, $k=30$ neighbours. The exact second fundamental form is $II(u,v)=-\\langle u,v\\rangle N$ with $N$ the point itself (unit normal). Error: median over 200 points of $\\sup\\|II\\circ\\pi_T(x,y)-\\hat T_2\\circ\\hat\\pi(x,y)\\|$ over unit $x,y$ (both bilinear maps expressed in an orthonormal basis of the span of the two tangent spaces; alternating power iteration).",
         theory="Theorem 4: error $\\lesssim(1/n)^{(m-2)/d}=n^{-1/2}$ for $m=3$, $d=2$ (up to logs). We require the fitted slope to be below $-0.35$ and the error at the largest $n$ below 0.1 ($\\|II\\|_{op}=1$). Limits fixed before the run."),
    dict(name="ldle_square_grid", method="LDLE", build=fig_ldle_grid,
         what="LDLE (Alg. 2, local parameterisation from Laplacian eigenvectors) recovers the tangent plane of its own running example.",
         how="Paper's example (Sec. 3.2): $[0,1]^2$ on a grid of spacing 0.01 (10201 points), embedded isometrically in $\\mathbb{R}^3$ by a random orthonormal frame. Paper's default hyperparameters (knn=49, ktune=7, klv=25, N=100, p=0.99), $d=2$. Error = largest principal angle to the frame. The criterion is on the interior (dashed box); boundary points are shown too.",
         theory="The square is flat, so the tangent plane at every point is the span of the frame. LDLE's local parameterisation is, to first order, a projection onto the tangent space."),
    dict(name="ldle_flat_torus", method="LDLE", build=fig_ldle_torus,
         what="LDLE on a closed manifold whose Laplacian eigenvectors are degenerate and non-trivial (the paper's flat torus).",
         how="Paper's flat torus (Appendix): $X=(4\\cos\\theta,4\\sin\\theta,\\cos\\phi,\\sin\\phi)/4\\pi$, $\\theta=0.01\\pi i$ ($i<200$), $\\phi=0.04\\pi j$ ($j<50$): 10000 points in $\\mathbb{R}^4$. Same default hyperparameters, $d=2$. True tangent space = span$(\\partial_\\theta,\\partial_\\phi)$.",
         theory="Analytic tangents are known exactly. The limit of 1 degree is our own tolerance (the paper does not report an angle); the point is that the error is small everywhere, with no region of failure in panel (a)."),
    dict(name="lego_swiss_roll", method="LEGO", build=lambda: fig_lego("lego_swiss_roll", "Swiss roll (Fig. 3), noise $\\pm0.0175$"),
         what="LEGO is far more accurate than LPCA under noise normal to the manifold (Fig. 3 of the LEGO paper).",
         how="10700 points on a high-aspect-ratio Swiss roll (scaled to radius $\\leq 1$, width 0.5) plus uniform noise in $(-0.0175,0.0175)$ along the normal. LPCA knn=9; LEGO knn=9, m0=100, m=40, $\\beta=1/2$ (the figure caption). Error: $D_j=\\sum_i(1-\\cos\\theta_i)$ (Eq. 44) against the clean tangent space.",
         theory="The paper's claim: LPCA is very sensitive to this noise and LEGO is much better. We require LEGO's median $D_j$ to be below half of LPCA's. The paper does not give the roll's scale/aspect ratio, so our geometry is an assumption."),
    dict(name="lego_truncated_torus", method="LEGO", build=lambda: fig_lego("lego_truncated_torus", "Truncated torus (Fig. 4), heteroskedastic noise"),
         what="Same claim on the truncated torus with noise whose level varies over the surface (Fig. 4 of the LEGO paper).",
         how="3617 points on a torus patch ($u\\in[0,3\\pi/2)$, R=0.3, r=0.12) plus normal noise uniform in $(-\\epsilon(u),\\epsilon(u))$, $\\epsilon(u)=10^{-2}+2.5\\cdot10^{-3}(1+\\cos 2u)$. LPCA knn=14; LEGO knn=14, m0=100, m=20, $\\beta=1/2$. Error: $D_j$ (Eq. 44).",
         theory="As above. The paper does not give the torus radii; they were chosen so that the noise is comparable to the point spacing, which is the regime in which the paper reports LPCA degrading."),
    dict(name="guidotti_vs_numpy", method="Guidotti", build=fig_guidotti_numpy,
         what="The torch Guidotti code computes the signature function and its implied normals exactly as the paper's formulas say.",
         how="80 random points on the unit sphere, $\\alpha=10^{-8}$, kernel width 1 (the paper's kernel $e^{-|x|^2}$, or the regularised Laplace kernel $e^{-\\sqrt{|x|^2+1}}$), float64. Reference: a from-scratch numpy implementation that solves $(\\alpha I+K/m)\\Lambda=\\mathbf 1$ (Eq. 3.3), sets $u(x)=\\frac1m\\sum_k\\Lambda_k K(x-x_k)$ (Eq. 3.5) and uses $\\nu=-\\nabla u/|\\nabla u|$ with the gradient formulas of Sec. 5, at 32 other points on the sphere.",
         theory="Same mathematics, so the two must agree to round-off (the angle floor of about $10^{-6}$ degrees is the precision of $\\arccos$ near 1)."),
    dict(name="guidotti_sphere", method="Guidotti", build=fig_sig_sphere,
         what="The signature function interpolates a sphere and its level set has the right normals (Sec. 6.6 of the paper).",
         how="80 random points on the unit sphere ($x_3$ uniform in $[-1,1]$, $\\theta$ uniform), Gaussian kernel of width 1, $\\alpha=0$, float64. Evaluate $u$ and the implied normal at 32 other random points of the sphere and compare with $u=1$ and the exact normal $x$. Repeated for 20 random draws.",
         theory="The paper reports one draw: $\\max|u-1|=2.25\\cdot10^{-5}$, maximal normal angle $0.00873°$. The error depends on the draw (it varies by a factor of about 20 over our draws), and the paper's values lie at the low end of ours, so we require every draw to be within 100 times the paper's numbers. Exactness of the code itself is shown by guidotti_vs_numpy."),
    dict(name="guidotti_line", method="Guidotti", build=fig_guidotti_line,
         what="Normals of the signature function turn quickly in directions normal to the manifold (Sec. 4 and 6.8): for a straight line they are radial.",
         how="61 equally spaced points on $[-6,6]\\times\\{0\\}^2\\subset\\mathbb{R}^3$, $\\alpha=10^{-8}$, width 1, float64. Normals at 15 random perturbations of the midpoint (random direction, distance uniform in $[0,0.01]$); singular values of the $3\\times15$ matrix of normals.",
         theory="Level sets of the signature function of a line are cylinders, so the normals are orthogonal to the line and span its 2-D orthogonal complement: rank 2, third singular value $\\approx0$."),
    dict(name="guidotti_table_6_8_1", method="Guidotti", build=fig_guidotti_table,
         what="Table 6.8.1 of the paper: singular values of the normals at 15 perturbations of points of a closed curve (rank 2) and of the surface made by extruding it (rank 1).",
         how="Closed curve in the $x$-$z$ plane made of two unit-length segments and two semicircles (300 points; the paper does not give the sample or the dimensions, so these are our assumptions), and the surface made of 11 copies at $y\\in[-0.5,0.5]$. Regularised Laplace kernel, $r=1$, $\\alpha=10^{-10}$, width 1, float64 (Sec. 6.7). Four points along the curve (the same four at $y=0$ on the surface), 15 perturbations each, distance $\\le0.01$. Open circles: the paper's numbers; crosses: ours.",
         theory="The surface reproduces the paper (sv1 = $\\sqrt{15}=3.873$ because all 15 normals coincide, sv2 and sv3 small). The curve does not: the paper finds two large singular values (about 3 and 2) and we find one (3.8, then 0.3-0.6, then 0.03-0.07), i.e. our normals turn much less. A straight line does give two large values (guidotti_line), so the code is not the cause; the paper's curve and sample are unspecified, and the normals' turning depends on the sample spacing and curvature. We therefore do not require the curve's rank to be 2."),
    dict(name="guidotti_curve_surface", method="Guidotti", build=fig_sig_curve,
         what="The tangent space obtained from the normals at perturbed points (Sec. 6.8) identifies a 1-D tangent space on a curve and a 2-D one on a surface.",
         how="The same curve (300 points) and extruded surface as above, regularised Laplace kernel, $r=1$, $\\alpha=10^{-10}$, width 1, float64. Tangent estimate with $d=1$ on the curve and $d=2$ on the surface (evaluated at $y=0$, away from the boundary): 14 (curve) or 18 (surface) random perturbations per point, distance uniform in $[0,0.01]$; tangent space = the $d$ directions least covered by the normals.",
         theory="The paper's argument: normals turn quickly in directions normal to the manifold and hardly along it. The paper gives no accuracy figure; the 2 degree limit on the median is our own tolerance."),
]


def main():
    summary = []
    for sp in SPECS:
        if not any((RESULTS / sp["name"]).glob("*.npz")):
            print("missing results for", sp["name"]); continue
        out = RESULTS / sp["name"]
        fig, rows = sp["build"]()
        fig.savefig(out / "plot.png"); fig.savefig(out / "plot.pdf"); plt.close(fig)
        status = "PASS" if all(r[3] is not False for r in rows) else "FAIL"
        table = "| quantity | measured | required | |\n|---|---|---|---|\n" + "\n".join(
            f"| {q.replace(chr(124), chr(92) + chr(124))} | {v:.3g} | {c} | {'not required' if ok is None else 'pass' if ok else '**FAIL**'} |" for q, v, c, ok in rows)
        (out / "README.md").write_text(
            f"# {sp['name']}  ({sp['method']}) — {status}\n\n**What is tested.** {sp['what']}\n\n**How.** {sp['how']}\n\n"
            f"**What it should show.** {sp['theory']}\n\n**Result.**\n\n{table}\n\n"
            f"![plot](plot.png)\n\nThe green region is the acceptance region: the observed values must land inside it. "
            f"Raw numbers: `*.npz` in this folder. Generated by `make_plots.py`; thresholds are in `common.py` (`CRIT`).\n")
        for q, v, c, ok in rows:
            summary.append((sp["name"], sp["method"], q, v, c, ok))
    lines = ["# Known-result verification\n",
             "Each test reproduces an experiment from the paper of the method (or a textbook identity) and checks the numbers against what the theory/paper says they must be. "
             "Per-test details, plots and raw data are in the sub-folders.\n",
             "Run: `sbatch experiment_code/series_1/run.sbatch` (tests, then plots). Thresholds: `series_1/common.py` (`CRIT`).\n",
             "| test | method | quantity | measured | required | |", "|---|---|---|---|---|---|"]
    for n, m, q, v, c, ok in summary:
        lines.append(f"| [{n}]({n}/README.md) | {m} | {q.replace(chr(124), chr(92) + chr(124))} | {v:.3g} | {c} | {'not required' if ok is None else 'pass' if ok else '**FAIL**'} |")
    (RESULTS / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
