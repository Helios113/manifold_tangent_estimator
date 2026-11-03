"""The points the tangent diagnostics (tangent.py, lpca_lego.py) share, so that the autoencoder, LPCA and LEGO are compared on the same points: the training distribution of the run.
A sphere point p is drawn as in a training batch (uniform on the sphere), the run's noise is added to it (no renormalisation), and the noisy point is embedded: x = f(p + noise). The true tangent space of a point is the analytic tangent of the embedded sphere at the
noiseless p (the parent of the noisy point). n points make the cloud, the first n_eval are the ones whose tangent is computed and compared. n, n_eval and the seed are in diagnostics/diagnostics.toml."""

import tomllib
import torch
from common import ROOT, manifold


def settings():
    """The settings of diagnostics/diagnostics.toml: {"points": {n, n_eval, seed}, "lego": {...}}."""
    return tomllib.load(open(ROOT / "diagnostics" / "diagnostics.toml", "rb"))


def eval_points(cfg):
    """cfg: the config of a run. Returns a dict: S, emb; p (n, D) the noiseless sphere points; x (n, ambient_dim) the embeddings of the noisy points (the cloud); T (n_eval, ambient_dim, k) the true tangent bases at f(p[:n_eval]); n, n_eval."""
    st = settings()["points"]
    n, n_eval, seed = st["n"], st["n_eval"], st["seed"]
    S, emb = manifold(cfg)
    p = S.sample(n, torch.Generator().manual_seed(seed))
    p_noisy = S.sample(
        n, torch.Generator().manual_seed(seed), noise=cfg.get("noise", 0.0)
    )  # the same draw plus the noise, as in a training batch
    T = S.embed_tangent_basis(p[:n_eval], S.tangent_basis(p[:n_eval]), emb)
    return dict(S=S, emb=emb, p=p, x=emb(p_noisy), T=T, n=n, n_eval=n_eval)
