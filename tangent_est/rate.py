"""Rate check of the Aamari-Levrard estimator (plan.md step 6): fitted slope of log(error) vs log(n) next to the prediction -(m-1)/k (tangent) and -(m-2)/k (curvature)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .plots import _style


def rate_table(R):
    """R: rate_check.csv rows. For every (family, k, m, n): mean over test seeds of the error at the validation-chosen h; returns the per-(family,k,m) slopes."""
    ok = R[(R.status == "ok") & (R.chosen == True)]
    g = ok.groupby(["family", "k", "m", "n"]).agg(max_ang=("max_ang", "mean"), med_ang=("med_ang", "mean"), curv_max=("curv_max", "mean"), curv_med=("curv_med", "mean")).reset_index()
    rows = []
    for (fam, k, m), h in g.groupby(["family", "k", "m"]):
        h = h.sort_values("n")
        if len(h) < 3: continue
        ln = np.log(h.n.values)
        r = dict(family=fam, k=k, m=m, n_points=len(h), n_min=int(h.n.min()), n_max=int(h.n.max()),
                 slope_max_ang=np.polyfit(ln, np.log(h.max_ang.values), 1)[0], predicted=-(m - 1) / k,
                 slope_med_ang=np.polyfit(ln, np.log(h.med_ang.values), 1)[0])
        if m >= 3 and h.curv_max.notna().all():
            r.update(slope_curv_max=np.polyfit(ln, np.log(h.curv_max.values), 1)[0], predicted_curv=-(m - 2) / k)
        rows.append(r)
    return pd.DataFrame(rows), g


def rate_figure(g, out):
    fams = sorted(g.family.unique()); ks = sorted(g.k.unique())
    fig, axs = plt.subplots(len(fams), len(ks), figsize=(3.2 * len(ks), 2.5 * len(fams)), constrained_layout=True, squeeze=False)
    col = {2: "#999999", 3: "#E69F00", 4: "#D55E00"}
    for i, fam in enumerate(fams):
        for j, k in enumerate(ks):
            ax = axs[i, j]
            for m in (2, 3, 4):
                h = g[(g.family == fam) & (g.k == k) & (g.m == m)].sort_values("n")
                if len(h) < 2: continue
                ax.plot(h.n, h.max_ang, "o-", ms=3, lw=1, color=col[m], label=f"m={m}")
                ax.plot(h.n, h.max_ang.iloc[0] * (h.n / h.n.iloc[0]) ** (-(m - 1) / k), ":", color=col[m], lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log"); _style(ax, "n", "max angle (deg)", f"family {fam}, k={k}")
            if i == 0 and j == 0: ax.legend(frameon=False, title="dotted: $n^{-(m-1)/k}$")
    fig.savefig(out); plt.close(fig)
