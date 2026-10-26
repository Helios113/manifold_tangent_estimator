"""Series 4: Guidotti bandwidth ramp on a 100-dimensional curved patch (patch of a sphere in R^101, chart z ~ N(0, I_100), radius 100 sqrt(100/300)) embedded in R^1000,
n = 4000, no noise, d given, float64. Bandwidth multiplier (x mean 10-NN distance) x alpha; 200 evaluation points.
  python bandwidth_ramp.py compute   -> results/series_4/ramp.csv, spectra.npz
  python bandwidth_ramp.py plot      -> figures and README in results/series_4/"""
import sys, pathlib, time, gc, numpy as np, pandas as pd, torch
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_torch import Guidotti, LPCA, to_t
from tangent_torch.common import sqdist, DEV
OUT = ROOT / "results" / "series_4"
D, P_AMB, N, NSUB, F64 = 100, 1000, 4000, 200, torch.float64
MULTS = (0.5, 1, 1.5, 2, 3, 5, 10, 20, 50, 100, 300, 1000)
ALPHAS = (1e-3, 1e-6, 1e-10)


def make(d=D, n=N, p=P_AMB, seed=0):
    rng = np.random.default_rng(seed)
    R = 100.0 * np.sqrt(d / 300)
    z = rng.normal(size=(n, d))
    Y = np.concatenate([z, np.sqrt(R ** 2 - (z ** 2).sum(1, keepdims=True))], 1)
    E = np.linalg.qr(rng.normal(size=(p, d + 1)))[0]
    return Y @ E.T, Y / R, E


def angles(T, Yt, Et):
    """Principal angles (deg, ascending) between span(T) (m,p,d), orthonormal, and the true tangent space at the points with unit chart normals Yt (m,d+1)."""
    A = Et.T @ T; A = A - Yt[:, :, None] * (Yt[:, None, :] @ A)
    Rm = T - Et @ A
    return torch.rad2deg(torch.asin(torch.linalg.eigvalsh(Rm.transpose(1, 2) @ Rm).clamp(0, 1).sqrt()))


def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    X, Y, E = make(); Xt, Yt, Et = to_t(X, F64), to_t(Y[:NSUB], F64), to_t(E, F64)
    dd = sqdist(Xt[:500], Xt).sqrt(); knn_ratio = float(dd.kthvalue(11, 1).values.mean() / dd.mean()); mean_pair = float(dd.mean())
    rnd = angles(torch.linalg.qr(torch.randn(NSUB, P_AMB, D, dtype=F64, device=DEV)).Q, Yt, Et)
    ang = angles(LPCA(D, k=400).fit(to_t(X)).tangents_train()[:NSUB].to(F64), Yt, Et)
    lpca = dict(lpca_mean_angle=float(ang.mean()), lpca_largest_angle=float(ang[:, -1].mean()))
    print("knn10/pairwise", knn_ratio, "mean pairwise", mean_pair, "random", float(rnd.mean()), lpca, flush=True)
    rows, spectra = [], {}
    for a in ALPHAS:
        for m in MULTS:
            r = dict(d=D, p=P_AMB, n=N, alpha=a, bw_mult=m, knn10_over_pairwise=knn_ratio, random_mean_angle=float(rnd.mean()), random_largest_angle=float(rnd[:, -1].mean()), **lpca)
            t0 = time.perf_counter()
            try:
                g = Guidotti(D, bw_mult=m, alpha=a, dtype=F64).fit(Xt)
                ang = angles(g.tangents_query(Xt[:NSUB]), Yt, Et)
                gen = torch.Generator(device=DEV).manual_seed(0)
                w = g._eig(Xt[:3], gen)[0]                                 # (3, p) ascending normal-Gram spectra
                u = torch.exp(-sqdist(g.Xs, g.Xs)) @ g.lam / g.n
                r.update(status="ok", bw=g.bw, bw_over_pairwise=g.bw / mean_pair, mean_angle=float(ang.mean()), largest_angle=float(ang[:, -1].mean()),
                         gap_ratio=float((w[:, D] / w[:, D - 1]).mean()), top_share=float((w[:, -1] / w.sum(1)).mean()), u_min=float(u.min()), u_max=float(u.max()))
                spectra[f"a{a:g}_m{m:g}"] = w.cpu().numpy()
                del g
            except RuntimeError as e:
                r["status"] = f"error: {str(e)[:80]}"
            r["seconds"] = time.perf_counter() - t0; rows.append(r); print(r, flush=True)
            pd.DataFrame(rows).to_csv(OUT / "ramp.csv", index=False); np.savez(OUT / "spectra.npz", **spectra)
            gc.collect(); torch.cuda.empty_cache()


def plot():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                         "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 150})
    C = {1e-3: "#CC79A7", 1e-6: "#E69F00", 1e-10: "#009E73"}
    df = pd.read_csv(OUT / "ramp.csv"); ok = df[df.status == "ok"]; sp = np.load(OUT / "spectra.npz")

    def style(ax, xl, yl, t):
        ax.grid(True, color="#dddddd", lw=0.4); ax.set_axisbelow(True); ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(t, loc="left", pad=3); ax.set_xscale("log")

    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.0), constrained_layout=True)
    for a, g in ok.groupby("alpha"):
        axs[0].plot(g.bw_mult, g.mean_angle, marker="o", ms=3.5, lw=1.2, color=C[a], mec="white", mew=0.4, label=f"$\\alpha={a:g}$")
        axs[1].plot(g.bw_mult, g.largest_angle, marker="o", ms=3.5, lw=1.2, color=C[a], mec="white", mew=0.4, label=f"$\\alpha={a:g}$")
        axs[2].plot(g.bw_mult, g.top_share, marker="o", ms=3.5, lw=1.2, color=C[a], mec="white", mew=0.4, label=f"$\\alpha={a:g}$")
    for ax, k, rk in ((axs[0], "mean_angle", "random_mean_angle"), (axs[1], "largest_angle", "random_largest_angle")):
        ax.axhline(df[rk].iloc[0], color="#888888", ls="--", lw=0.9, label="random basis")
        ax.axhline(df["lpca_" + k].iloc[0], color="#0072B2", ls="-.", lw=0.9, label="LPCA, $k=400$"); ax.set_ylim(0, 92); ax.axvline(1.5, color="#777777", ls=":", lw=0.8)
    axs[0].legend(frameon=False, loc="center left")
    style(axs[0], "bandwidth multiplier (× mean 10-NN distance)", "mean principal angle (deg)", "mean angle, $d=100$ in $\\mathbb{R}^{1000}$")
    style(axs[1], "bandwidth multiplier (× mean 10-NN distance)", "largest principal angle (deg)", "largest angle")
    style(axs[2], "bandwidth multiplier (× mean 10-NN distance)", "top eigenvalue / trace", "share of the top eigenvalue of the normal Gram")
    axs[2].set_yscale("log"); fig.savefig(OUT / "fig_ramp.png"); fig.savefig(OUT / "fig_ramp.pdf"); plt.close(fig)

    keys = [m for m in (1.5, 10, 100, 1000)]
    fig, axs = plt.subplots(1, len(ALPHAS), figsize=(10.5, 3.0), constrained_layout=True, sharey=True)
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(keys)))
    for ax, a in zip(axs, ALPHAS):
        for c, m in zip(cm, keys):
            k = f"a{a:g}_m{m:g}"
            if k in sp: ax.plot(np.arange(1, P_AMB + 1), sp[k][0], color=c, lw=1.0, label=f"bw ×{m:g}")
        ax.axvline(P_AMB - D, color="#777777", ls=":", lw=0.8); ax.set_yscale("log"); ax.set_xlabel("eigenvalue index (ascending)"); ax.grid(True, color="#dddddd", lw=0.4)
        ax.set_title(f"normal-Gram spectrum, $\\alpha={a:g}$", loc="left", pad=3)
    axs[0].set_ylabel("eigenvalue"); axs[0].legend(frameon=False); fig.savefig(OUT / "fig_spectra.png"); fig.savefig(OUT / "fig_spectra.pdf"); plt.close(fig)

    def tab(k): return ok.pivot(index="bw_mult", columns="alpha", values=k).round(3).to_markdown()
    r0 = df.iloc[0]
    txt = f"""# Series 4 — Guidotti bandwidth ramp, $d=100$ curved patch in $\\mathbb{{R}}^{{1000}}$

Patch of a sphere in $\\mathbb{{R}}^{{101}}$ (chart $z\\sim N(0,I_{{100}})$, radius 57.7, angular extent about 10°) embedded in $\\mathbb{{R}}^{{1000}}$ by a random orthonormal frame, n = {N}, no noise, d given, float64, 200 evaluation points (`bandwidth_ramp.py`). The bandwidth is the multiplier times the mean 10-NN distance; the ratio of the 10-NN distance to the mean pairwise distance is {r0.knn10_over_pairwise:.2f}. Error = principal angles between the estimated and the true tangent space (deg). Random basis (chance): mean {r0.random_mean_angle:.1f}°, largest {r0.random_largest_angle:.1f}°. LPCA (k = 400): mean {r0.lpca_mean_angle:.2f}°, largest {r0.lpca_largest_angle:.2f}°.

![ramp](fig_ramp.png)

![spectra](fig_spectra.png)

Mean principal angle (deg) by bandwidth multiplier (rows) and alpha (columns):

{tab('mean_angle')}

Largest principal angle (deg):

{tab('largest_angle')}

Share of the top eigenvalue of the perturbed-normal Gram (about 1/(p-d) = {1/(P_AMB-D):.4f} if the normals were isotropic in the normal space, 1 if they all point the same way):

{tab('top_share')}

Gap ratio (eigenvalue d / eigenvalue d-1, ascending; 1 = no gap):

{tab('gap_ratio')}

Failed configurations: {', '.join(f'alpha={r.alpha:g}, mult={r.bw_mult:g} ({r.status})' for r in df[df.status != 'ok'].itertuples()) or 'none'}.
"""
    (OUT / "README.md").write_text(txt)


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
