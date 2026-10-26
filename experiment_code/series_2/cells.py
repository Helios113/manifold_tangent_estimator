"""Series 2 (small k): list of cells (conditions) -- see plan.md. Not a full cross product (that would be ~6400 cells x 6 draws x ~40 grid points): a core grid at
n = 1e3, D = 1000, uniform sampling, plus one-factor extensions (n = 1e4, D = 300, non-uniform sampling).
  core      family in {A, B, C(A), C(B), D} x k in {1,2,5,10} x noise {none, iso/normal/tangent x sigma sqrt(D)/tau in (0.01, 0.1, 0.5)}     n = 1e3, D = 1000, uniform
  n = 1e4   same families and k, noise {none, iso 0.1, normal 0.1}
  D = 300   same families and k, noise {none, iso 0.1}
  nonuniform  same, n = 1e3, D = 1000, noise {none, iso 0.1}
  D = 3     sanity configuration: A (k = 1), D-family (k = 2)"""
DTAU = 10.0          # family D: diam / tau (curvature) -- strongly curved, so that global PCA fails (non-triviality, step 2.4); the near-flat regime is series 3


def fams():
    return [("A", {}), ("C", dict(base="A")), ("B", {}), ("C", dict(base="B")), ("D", {})]


def cid(fam, gen, k, D, n, samp, noise, level):
    f = fam + gen.get("base", "")
    return f"{f}_k{k}_D{D}_n{n}_{samp}_{noise}{level:g}"


def cell(fam, gen, k, D, n, samp, noise, level, dtau=None):
    c = dict(family=fam, gen=gen, k=k, D=D, n=n, sampling=samp, noise=noise, level=level)
    if fam == "D": c["diam_over_tau"] = dtau if dtau is not None else DTAU
    c["id"] = cid(fam, gen, k, D, n, samp, noise, level) + (f"_dt{c['diam_over_tau']:g}" if fam == "D" else "")
    return c


def all_cells():
    out = []
    core_noise = [("none", 0.0)] + [(m, l) for m in ("iso", "normal", "tangent") for l in (0.01, 0.1, 0.5)]
    for k in (1, 2, 5, 10):
        for fam, gen in fams():
            for noise, level in core_noise:
                out.append(cell(fam, gen, k, 1000, 1000, "uniform", noise, level))
                if fam == "D": out.append(cell(fam, gen, k, 1000, 1000, "uniform", noise, level, dtau=1.0))      # also the moderately curved case, diam/tau = 1
    for k in (1, 2, 5, 10):
        for fam, gen in fams():
            for noise, level in (("none", 0.0), ("iso", 0.1), ("normal", 0.1)):
                out.append(cell(fam, gen, k, 1000, 10000, "uniform", noise, level))
    for k in (1, 2, 5, 10):
        for fam, gen in fams():
            for noise, level in (("none", 0.0), ("iso", 0.1)):
                out.append(cell(fam, gen, k, 300, 1000, "uniform", noise, level))
                out.append(cell(fam, gen, k, 1000, 1000, "nonuniform", noise, level))
    for noise, level in (("none", 0.0), ("iso", 0.1)):
        out.append(cell("A", {}, 1, 3, 1000, "uniform", noise, level))
        out.append(cell("D", {}, 2, 3, 1000, "uniform", noise, level))
    seen, uniq = set(), []
    for c in out:
        if c["id"] not in seen: seen.add(c["id"]); uniq.append(c)
    return uniq
