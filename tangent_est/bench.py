"""Benchmark runner for the protocol in plan.md: one *cell* = one condition (family, k, D, n, sampling, noise model, noise level, curvature) evaluated on a validation
draw (used to tune the hyper-parameters of every method) and on N_TEST test draws (new manifold instance, sample and noise each).  Every grid point of every method
is evaluated on every draw, so the same file gives the tuned result (the grid point with the lowest validation median error) and the error-vs-scale curve.

Rows: one per (draw, method, grid point); methods lpca, msvd, al2/al3/al4 (Aamari-Levrard local polynomial of order m), lego, guidotti and the baselines random,
pca, oracle_lpca.  Skipped / failed configurations are rows with status != 'ok' and a reason (the failure log)."""
import gc, json, math, sys, time, pathlib, traceback
import numpy as np, pandas as pd, torch
from tqdm import tqdm
from .generators import make_manifold, make_dataset, eps_for_diam_over_tau, tangent_frame
from tangent_torch import LPCA, LDLE, LEGO, Guidotti, LocalPoly, MultiscaleSVD
from tangent_torch.lpoly import n_monomials
from tangent_torch.common import DEV, knn_train, knn_query, chunks

NEV, N_TEST, VAL_SEED = 200, 5, 1000
F64, F32 = torch.float64, torch.float32


# ------------------------------------------------------------------------------------------------------------------------------ metrics
def principal(T, E):
    """T, E (m,D,k) orthonormal -> principal angles (deg), ascending, (m,k)."""
    c = np.linalg.svd(np.einsum("nDk,nDj->nkj", T, E), compute_uv=False).clip(0, 1)
    return np.degrees(np.arccos(c))


def agg(ang, name, mask=None):
    """Median / 90th percentile / max over the points of e = mean sin^2(theta) (= ||P-P'||_F^2/(2k)), the largest angle (deg) and sin(largest angle)."""
    k = ang.shape[1]
    e = (np.sin(np.radians(ang)) ** 2).mean(1); mx = ang.max(1); sm = np.sin(np.radians(mx))
    out = {}
    for nm, v in (("e", e), ("ang", mx), ("sin", sm)):
        vv = v if mask is None else v[mask]
        out.update({f"{name}med_{nm}": float(np.median(vv)), f"{name}p90_{nm}": float(np.percentile(vv, 90)), f"{name}max_{nm}": float(vv.max())})
    return out


def curvature_error(E, II, Bm, T2):
    """||II o pi_T - T2 o pi_hat||_op per point. E (m,D,k) true frames, II (m,D,k,k) true second form in that frame, Bm (m,D,k) estimated basis,
    T2 (m,D,k,k) estimated tensor in that basis. Both are bilinear maps R^D x R^D -> R^D; expressed in an orthonormal basis of span([E, Bm]) and the operator norm
    sup ||Delta(x,y)|| over unit x,y approximated by alternating power iteration (8 random starts)."""
    dev = DEV
    E, II, Bm, T2 = (torch.as_tensor(a, dtype=F64, device=dev) for a in (E, II, Bm, T2))
    m, D, k = E.shape
    Wb = torch.linalg.qr(torch.cat([E, Bm], 2))[0]                                # (m,D,2k)
    A1, A2 = Wb.transpose(1, 2) @ E, Wb.transpose(1, 2) @ Bm
    T1 = torch.einsum("nDab,nca,ndb->nDcd", II, A1, A1)
    T2m = torch.einsum("nDab,nca,ndb->nDcd", T2, A2, A2)
    Dl = T1 - T2m                                                               # (m,D,r,r)
    r = Dl.shape[2]; S = 8
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(m, r, S, dtype=F64, device=dev, generator=g); y = torch.randn(m, r, S, dtype=F64, device=dev, generator=g)
    x, y = x / x.norm(dim=1, keepdim=True), y / y.norm(dim=1, keepdim=True)
    for _ in range(40):
        z = torch.einsum("nDcd,ncs,nds->nDs", Dl, x, y); z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-300)
        x = torch.einsum("nDcd,nDs,nds->ncs", Dl, z, y); x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-300)
        z = torch.einsum("nDcd,ncs,nds->nDs", Dl, x, y); z = z / z.norm(dim=1, keepdim=True).clamp_min(1e-300)
        y = torch.einsum("nDcd,nDs,ncs->nds", Dl, z, x); y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-300)
    val = torch.einsum("nDcd,ncs,nds->nDs", Dl, x, y).norm(dim=1).amax(1)
    return val.cpu().numpy()


# ------------------------------------------------------------------------------------------------------------------------------ grids
def count_grid(k, n):
    """Neighbour counts for LPCA / MS-SVD: (2k+2) * 2^j, j = 0..5, at most n/3 (LPCA needs more than 2k points)."""
    b = 2 * k + 2
    g = [b * 2 ** j for j in range(6) if b * 2 ** j <= n // 3]
    return g


def al_counts(k, m, n):
    """Target median ball counts for AL of order m: {2,4,8,16} p(k,m), at least 2k+2 (stage (a) needs more than 2k points), at most n/3.
    Returns (counts, reason-if-infeasible)."""
    if m == 2:
        return count_grid(k, n), None
    p = n_monomials(k, m)
    lo = max(2 * p, 2 * k + 2)
    cs = [int(lo * f) for f in (1, 2, 4, 8) if lo * f <= n // 3]
    if not cs:
        return [], f"median ball count must be >= 2 p(k,m) = {2 * p} (p(k,m) = {p} monomials per output coordinate) but n/3 = {n // 3}"
    return cs, None


def lego_grid(k, n, D):
    out = []
    pairs = [(20, 100), (40, 200)] if k <= 10 else [(2 * k, 4 * k)]
    for c in (2, 4, 8):
        kn = c * (2 * k + 2)
        if kn > n // 3: continue
        for m, m0 in pairs:
            out.append(dict(k=kn, m=m, m0=min(m0, n - 2)))
    return out


def lego_memory_ok(cfg, n, D):
    return D * cfg["m"] * n * 4 <= 3.0e10, f"gradient tensor D*m*n*4 B = {D * cfg['m'] * n * 4 / 1e9:.0f} GB"


def ldle_grid(k, n):
    """LDLE: graph neighbours and local-view size both (2k+2) * {1, 2, 4} (at most n/3); N = 2k eigenvectors (the selection of k of them needs N > k)."""
    return [dict(k=c * (2 * k + 2), klv=c * (2 * k + 2), N=2 * k) for c in (1, 2, 4) if c * (2 * k + 2) <= n // 3]


GUIDOTTI_GRID = [dict(bw_mult=b, alpha=a) for b in (1.5, 5.0, 20.0) for a in (1e-3, 1e-6)]


def gc_cuda():
    gc.collect(); torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------------------------ one draw
def estimate_tau(M, rng, pilot=1000):
    th, _ = M.sample_theta(pilot, rng)
    Xp = M.F(th)
    diam = M.diameter(th, Xp)
    kappa = M.max_curvature(th[:200], rng, n_dirs=100)
    bott = M.bottleneck(th, Xp)
    tau = M.reach_exact if getattr(M, "reach_exact", None) else min(M.reach_estimate(th[:500], Xp[:500]), 1 / max(kappa, 1e-12))
    return dict(tau=float(tau), kappa_max=float(kappa), bottleneck=float(bott), diam=float(diam))


def make_draw(cell, seed):
    """Manifold + dataset for one draw. Returns (M, ds, meta)."""
    rng = np.random.default_rng([seed, 7, cell["k"], cell["D"], cell["n"]])
    kw = dict(cell.get("gen", {}))
    M = make_manifold(cell["family"], cell["k"], cell["D"], seed, **kw)
    if cell["family"] == "D" and cell.get("diam_over_tau") is not None:
        M.set_eps(eps_for_diam_over_tau(M, cell["diam_over_tau"], rng))
    t = estimate_tau(M, rng)
    tau_eff = min(t["tau"], t["diam"])
    nu = cell["level"] * tau_eff if cell["noise"] != "none" else 0.0
    ds = make_dataset(M, cell["n"], rng, uniform=cell["sampling"] == "uniform", noise=cell["noise"], nu=nu)
    meta = dict(**t, tau_eff=float(tau_eff), nu=float(nu), diam_over_tau=t["diam"] / t["tau"], acceptance=ds["acceptance"], eps=getattr(M, "eps", None))
    return M, ds, meta


def run_draw(cell, seed, role, rows, base_row):
    t0 = time.time()
    k, D, n = cell["k"], cell["D"], cell["n"]
    M, ds, meta = make_draw(cell, seed)
    X = torch.as_tensor(ds["X"], dtype=F64, device=DEV)
    th = ds["theta"]
    ev = np.random.default_rng(seed).permutation(n)[:NEV]
    ev_t = torch.as_tensor(ev, device=DEV)
    J = M.J(th[ev]); sv = np.linalg.svd(J, compute_uv=False)
    meta["min_sv_J"] = float(sv.min())
    if sv.min() < 1e-8:
        rows.append(dict(base_row, draw=role, seed=seed, method="generator", status="rejected: J not full rank (min singular value %.2e)" % sv.min(), **meta)); return None
    Etrue = M.tangent(th[ev])
    bmask = M.boundary_mask(th[ev]) if M.has_boundary() else None
    # --- nearest-neighbour structure and the feasibility gate (spacing ratio)
    counts_lpca = count_grid(k, n)
    al_plan = {m: al_counts(k, m, n) for m in (2, 3, 4)}
    kmax = int(min(n - 1, max([2 * c for m in (3, 4) for c in al_plan[m][0]] + counts_lpca + [2 * k + 2])))
    dist, idx = knn_train(X, kmax)
    meta["delta"] = float(dist[:, 0].median().item() / meta["diam"])
    meta["nn_over_diam_q10"] = float(dist[:, 0].quantile(0.1).item() / meta["diam"])
    meta["noise_over_nn"] = meta["nu"] / float(dist[:, 0].median().item())
    bott_ratio = {c: meta["bottleneck"] / float(dist[:, c - 1].median().item()) for c in counts_lpca}
    meta["bottleneck_over_h_max"] = bott_ratio[max(counts_lpca)] if counts_lpca else None
    bk = dict(base_row, draw=role, seed=seed, **{k_: meta[k_] for k_ in ("tau", "tau_eff", "kappa_max", "bottleneck", "diam", "diam_over_tau", "delta", "nu", "noise_over_nn", "min_sv_J", "bottleneck_over_h_max", "eps")})
    bk["sampling_acceptance"] = meta["acceptance"]

    def record(method, cfg, T, extra=None, seconds=None, status="ok", reason=""):
        r = dict(bk, method=method, cfg=json.dumps(cfg, sort_keys=True), status=status, reason=reason, seconds=seconds)
        if T is not None:
            ang = principal(T, Etrue)
            r.update(agg(ang, ""))
            if bmask is not None:
                r.update(agg(ang, "int_", ~bmask)); r.update(agg(ang, "bnd_", bmask))
            r["ang_profile"] = json.dumps([round(float(v), 4) for v in np.median(ang, 0)])
        if extra: r.update(extra)
        rows.append(r)

    def to_np(T): return T.detach().cpu().numpy().astype(np.float64)

    def safe(method, cfg, fn, extra_fn=None):
        t1 = time.time()
        print(f"  [{role} {seed}] {method} {json.dumps(cfg, sort_keys=True)}", flush=True)
        try:
            T, extra = fn()
            if not np.isfinite(T).all():
                record(method, cfg, None, seconds=time.time() - t1, status="failed", reason="non-finite output")
            else:
                record(method, cfg, T, extra, seconds=time.time() - t1)
        except Exception as e:
            record(method, cfg, None, seconds=time.time() - t1, status="failed", reason=f"{type(e).__name__}: {str(e)[:150]}")
        gc_cuda()

    # --- baselines
    g = np.random.default_rng(seed + 99)
    Tr = np.linalg.qr(g.normal(size=(NEV, D, k)))[0]
    record("random", {}, Tr, dict(expected_e=1 - k / D))
    Xc = X - X.mean(0)
    w, V = torch.linalg.eigh(Xc.T @ Xc)
    Tp = np.repeat(to_np(V[:, -k:])[None], NEV, 0)
    record("pca", {}, Tp)
    if role == "test":                                                          # oracle LPCA: clean, dense sample (query at the clean evaluation points)
        n_or = int(min(max(20 * n, 20000), 100000))
        th_or, _ = M.sample_theta(n_or, np.random.default_rng([seed, 11]))
        Xo = torch.as_tensor(M.F(th_or), dtype=F64, device=DEV); Xq = torch.as_tensor(M.F(th[ev]), dtype=F64, device=DEV)
        best = None
        for c in count_grid(k, n_or):
            def fn():
                return to_np(LPCA(k, k=c).fit(Xo, nn=(None, None)).tangents_query(Xq)), dict(n_oracle=n_or)
            safe("oracle_lpca", dict(k=c), fn)
        del Xo, Xq
        gc_cuda()
    # --- LPCA
    for c in counts_lpca:
        safe("lpca", dict(k=c), lambda c=c: (to_np(LPCA(k, k=c).fit(X, nn=(dist, idx)).tangents_at(ev_t)), dict(h_median=float(dist[:, c - 1].median()))))
    if not counts_lpca:
        record("lpca", {}, None, status="skipped", reason=f"n/3 = {n // 3} < 2k+2 = {2 * k + 2}")
    # --- multiscale SVD
    if counts_lpca:
        for gap in (2.0, 5.0, 20.0):
            def fn(gap=gap):
                est = MultiscaleSVD(k, scales=counts_lpca, gap=gap).fit(X, nn=(dist, idx))
                T = to_np(est.tangents_at(ev_t))
                sc = est.last_scale.cpu().numpy()
                return T, dict(scale_median=float(np.median([counts_lpca[i] for i in sc])), h_median=float(dist[:, int(np.median([counts_lpca[i] for i in sc])) - 1].median()))
            safe("msvd", dict(gap=gap, scales=counts_lpca), fn)
    # --- Aamari-Levrard local polynomial (m = 2, 3, 4)
    cache = {}
    for m in (2, 3, 4):
        cs, why = al_plan[m]
        if why:
            record(f"al{m}", {}, None, status="skipped", reason=why + "; fall back to m = 2 (= LPCA)"); continue
        for c in cs:
            h = float(dist[:, c - 1].median().item()); cap = int(min(kmax, 2 * c, n - 1))
            def fn(m=m, c=c, h=h, cap=cap):
                est = LocalPoly(k, k=cap, order=m, radius=h).fit(X, nn=(dist, idx))
                cnt = est.ball_counts(ev_t).double().median().item()
                if m == 2 or k > 10:
                    return to_np(est.tangents_at(ev_t)), dict(ball_count_median=cnt, h=h)
                T, T2 = est.tangents_at(ev_t, want_T2=True)
                if "II" not in cache: cache["II"] = M.second_form(th[ev])
                E_, II = cache["II"]
                ce = curvature_error(E_, II, to_np(T), to_np(T2))
                return to_np(T), dict(ball_count_median=cnt, h=h, curv_med=float(np.median(ce)), curv_p90=float(np.percentile(ce, 90)), curv_max=float(ce.max()),
                                      curv_norm=float(np.median(np.linalg.norm(II.reshape(NEV, -1), axis=1))))
            safe(f"al{m}", dict(m=m, count=c, h=h, cap=cap), fn)
            if m >= 2 and meta["nu"] > 0:
                rows[-1]["sigma_le_h_over_4"] = bool(meta["nu"] <= h / 4)
            rows[-1]["high_noise_regime"] = bool(meta["nu"] > 10 * n ** (-m / k)) if meta["nu"] > 0 else False
    # --- LDLE (float32); tangents at the evaluation points from its local-view step
    Xf = X.to(F32)
    for cfg in ldle_grid(k, n):
        def fn(cfg=cfg):
            est = LDLE(k, k=cfg["k"], klv=cfg["klv"], N=cfg["N"]).fit(Xf)
            dl, ix = est.nn
            ids = [ev_t[s] for s in chunks(len(ev_t), est._chunk())]
            T = torch.cat([est._tangent(est.X[i], est.Phi[i], dl[i], ix[i]) for i in ids]); del est
            return to_np(T), {}
        safe("ldle", cfg, fn)
    if not ldle_grid(k, n):
        record("ldle", {}, None, status="skipped", reason=f"graph neighbour count 2k+2 = {2 * k + 2} > n/3")
    for cfg in lego_grid(k, n, D):
        ok, why = lego_memory_ok(cfg, n, D)
        if not ok:
            record("lego", cfg, None, status="skipped", reason=why + " exceeds the 30 GB budget"); continue
        def fn(cfg=cfg):
            est = LEGO(k, k=cfg["k"], m=cfg["m"], m0=cfg["m0"], beta=0.5).fit(Xf)
            T = to_np(est.tangents_at(ev_t)); del est
            return T, {}
        safe("lego", cfg, fn)
    if not lego_grid(k, n, D):
        record("lego", {}, None, status="skipped", reason=f"graph neighbour count 2(2k+2) = {2 * (2 * k + 2)} > n/3")
    del Xf
    # --- Guidotti signature function (float64: alpha down to 1e-6)
    for cfg in GUIDOTTI_GRID:
        def fn(cfg=cfg):
            est = Guidotti(k, bw_mult=cfg["bw_mult"], alpha=cfg["alpha"], dtype=F64).fit(X)
            T = to_np(est.tangents_at(ev_t))
            extra = dict(bw=est.bw)
            gen = torch.Generator(device=DEV).manual_seed(0)                    # failure diagnostic: how often is grad u ~ 0 at the perturbed points (normal undefined)?
            Y = est.X[ev_t[:5]]; dirs = torch.randn(5, 64, D, dtype=F64, device=DEV, generator=gen); dirs /= dirs.norm(dim=2, keepdim=True)
            Ys = (Y[:, None, :] + dirs * est.rho * est.bw).reshape(-1, D)
            Ysc = (Ys - est.mu0) / est.bw
            _, K1, _ = est._derivs(((Ysc * Ysc).sum(1)[:, None] + (est.Xs * est.Xs).sum(1)[None, :] - 2 * Ysc @ est.Xs.T).clamp_min(0))
            W1 = K1 * est.lam; gr = 2 * (Ysc * W1.sum(1, keepdim=True) - W1 @ est.Xs) / est.n
            extra["grad_u_norm_median"] = float(gr.norm(dim=1).median()); extra["grad_u_zero_frac"] = float((gr.norm(dim=1) < 1e-10).double().mean())
            del est
            return T, extra
        safe("guidotti", cfg, fn)
    del X, dist, idx
    gc_cuda()
    meta["seconds"] = time.time() - t0
    return meta


def run_cell(cell, out_dir):
    """Run one cell (validation draw + N_TEST test draws); write <out_dir>/<id>.csv and <id>.meta.json. Returns the dataframe."""
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    base = {kk: cell[kk] for kk in ("id", "family", "k", "D", "n", "sampling", "noise", "level")}
    base["base"] = cell.get("gen", {}).get("base", "")
    base["diam_over_tau_target"] = cell.get("diam_over_tau")
    rows, metas = [], {}
    bar = tqdm([("val", VAL_SEED)] + [("test", s) for s in range(N_TEST)], desc=cell["id"], unit="draw", file=sys.stdout, mininterval=30)
    for role, seed in bar:
        bar.set_postfix_str(f"{role} {seed}")
        try:
            metas[f"{role}{seed}"] = run_draw(cell, seed, role, rows, base)
        except Exception as e:
            rows.append(dict(base, draw=role, seed=seed, method="draw", status="failed", reason=f"{type(e).__name__}: {str(e)[:200]}", trace=traceback.format_exc()[-800:]))
        gc_cuda()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{cell['id']}.csv", index=False)
    (out_dir / f"{cell['id']}.meta.json").write_text(json.dumps(dict(cell=cell, metas=metas), default=str, indent=1))
    return df


def worker(cells, out_dir, shard=0, nshards=1, only=None):
    """Run the cells of this shard (cell i belongs to shard i % nshards), skipping finished ones (restartable)."""
    out_dir = pathlib.Path(out_dir)
    print("device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu", flush=True)
    for i, c in enumerate(cells):
        if i % nshards != shard or (only and only not in c["id"]) or (out_dir / f"{c['id']}.csv").exists():
            continue
        t0 = time.time(); df = run_cell(c, out_dir)
        bad = df[df.status != "ok"]
        print(f"[{i}/{len(cells)}] {c['id']}: {time.time() - t0:.0f} s, {len(df)} rows, {len(bad)} not ok", flush=True)
