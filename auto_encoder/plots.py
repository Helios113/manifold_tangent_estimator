"""All plots of a finished training run, one function each; every function takes the run folder and writes its figure(s) into it.

usage: python auto_encoder/plots.py <run folder> [function ...]      (no function: plot_all)

    plot_loss            loss.png                  train / validation MSE, and the gradient norm of every W_i   (history.csv)
    plot_orth            orth_{euclidian,spectral}.png   orthonormality error of every W_i and of their product   (history.csv)
    plot_latent          latent.png, latent.npz, latent_checkpoints/step*.png   original points next to the latent code   (best.pt, checkpoints/)
    plot_latent_3d       latent_3d.html            the same for the best weights, interactive in 3-D (open in a browser)   (best.pt)
    plot_gram            gram.png                  Gram matrix of every W_i                           (best.pt)
    plot_gram_composed   gram_composed.png         Gram matrix of the product W1 ... Wn               (best.pt)"""
import json, sys, tomllib
from pathlib import Path
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import make_model
from sphere import Sphere, EMBEDDINGS

N_LATENT = 2000


def load(run):
    run = Path(run)
    return run, tomllib.load(open(run / "config.toml", "rb"))


def load_net(run, cfg, weights="best.pt"):
    net = make_model(cfg)
    net.load_state_dict(torch.load(run / weights))
    return net


def history(run):
    """The rows of history.csv where the validation set was evaluated, and the sample count of the best step."""
    run, cfg = load(run)
    h = pd.read_csv(run / "history.csv")
    best_samples = json.load(open(run / "metrics.json"))["best_step"] * cfg["batch"]
    return h, h.dropna(subset=["val_mse"]), best_samples


def save(fig, ax, path, ylabel, best_samples, run):
    if best_samples is not None: ax.axvline(best_samples, ls=":", color="k", lw=.8, label="best step")
    ax.set_xlabel("samples seen"); ax.set_ylabel(ylabel); ax.set_title(Path(run).name); ax.legend()
    fig.savefig(path, dpi=150); plt.close(fig)


# ---- training curves ------------------------------------------------------------------------------------------------------------------------------------
def plot_loss(run):
    """Top: train / validation MSE. Bottom: gradient norm of every W_i."""
    h, v, best = history(run)
    fig, (ax, ax_grad) = plt.subplots(2, 1, sharex=True, figsize=(7, 8))
    ax.semilogy(h.samples, h.train_loss, alpha=.25, color="C0", label="train batch")
    ax.semilogy(v.samples, v.val_mse, color="C0", label="val")
    for i, c in enumerate(c for c in h.columns if c.startswith("grad_W")):
        ax_grad.semilogy(h.samples, h[c], color=f"C{i}", alpha=.15)
        ax_grad.semilogy(h.samples, h[c].rolling(200, min_periods=1).mean(), color=f"C{i}", label=c[-2:])
    ax_grad.set_xlabel("samples seen"); ax_grad.set_ylabel("||grad||_F (faint: per step, line: 200-step mean)")
    ax.set_ylabel("MSE"); ax.set_title(Path(run).name)
    for a in (ax, ax_grad): a.axvline(best, ls=":", color="k", lw=.8, label="best step"); a.legend()
    fig.savefig(Path(run) / "loss.png", dpi=150); plt.close(fig)


def plot_orth(run):
    """One figure per norm: every W_i (solid) and the product of all W_i (dashed black)."""
    h, v, best = history(run)
    for kind, ylabel in (("euclidian", "||G - I||_F"), ("spectral", "||G - I||_2")):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, c in enumerate(c for c in h.columns if c.startswith(f"orth_{kind}_")):
            composed = c.endswith("composed")
            ax.semilogy(v.samples, v[c], color="k" if composed else f"C{i}", ls="--" if composed else "-", lw=2 if composed else 1.5, label=c.split("_", 2)[2])
        save(fig, ax, Path(run) / f"orth_{kind}.png", ylabel, best, run)


# ---- latent space ---------------------------------------------------------------------------------------------------------------------------------------
def latent_inputs(cfg):
    """Fresh points of the manifold (not the validation set): the embedding, the points p, their embedding x and a colour per point."""
    gen = torch.Generator().manual_seed(cfg["data_seed"])
    S = Sphere(cfg["intrinsic_dim"])
    emb = EMBEDDINGS[cfg["embedding"]](cfg["ambient_dim"], gen, in_dim=S.D)                 # train.py draws the embedding first from this seed
    p = S.sample(N_LATENT, torch.Generator().manual_seed(123))
    x = S.embed_points(p, emb)
    rgb = ((p + 1) / 2).clamp(0, 1).numpy() if S.D == 3 else plt.cm.hsv((torch.atan2(p[:, 1], p[:, 0]) / (2 * np.pi) % 1).numpy())[:, :3]
    return emb, p, x, rgb


def pca3(M):
    """M (n, ambient_dim) projected on its own top three principal components, and the share of the variance each of them holds."""
    centred = M - M.mean(dim=0)
    _, S, Vh = torch.linalg.svd(centred, full_matrices=False)
    return (centred @ Vh[:3].T).numpy(), (S[:3] ** 2 / (S ** 2).sum()).numpy()


def pca_title(what, evr, sep=": "):
    return f"{what}{sep}top 3 PCA (variance " + " / ".join(f"{v:.2f}" for v in evr) + ")"


def encode_decode(net, x):
    """Latent code z, the reconstruction x_hat, and the reconstruction MSE."""
    with torch.no_grad():
        z = net.encode(x); x_hat = net.decode(z)
    return z.numpy(), x_hat, ((x_hat - x) ** 2).mean().item()


def scatter_latent(net, p, x, rgb, path, title):
    """Static figure: original points, PCA of the embedded input points x, latent code, and PCA of the reconstruction x_hat (the PCAs are in the ambient space)."""
    z, x_hat, mse = encode_decode(net, x)
    pn = p.numpy()[:, :3]
    (inp, inp_evr), (rec, rec_evr) = pca3(x), pca3(x_hat)
    z2 = z if z.shape[1] > 1 else np.hstack([z, np.zeros_like(z)])
    three = z.shape[1] == 3
    rows, cols = (2, 4) if three else (1, 4)
    fig = plt.figure(figsize=(20, 10 if three else 4.5))                  # panels in order: original points, embedded input PCA, latent, reconstruction PCA
    ax = fig.add_subplot(rows, cols, 1, projection="3d"); ax.scatter(*pn.T, c=rgb, s=4); ax.set_title("original sphere points"); ax.set_box_aspect((1, 1, 1))
    if three:
        ax = fig.add_subplot(rows, cols, 3, projection="3d"); ax.scatter(*z.T, c=rgb, s=4); ax.set_title("latent space (R^3)"); ax.set_box_aspect(np.ptp(z, axis=0))
        for k, (i, j) in enumerate([(0, 1), (0, 2)]):
            ax = fig.add_subplot(rows, cols, 5 + k); ax.scatter(z[:, i], z[:, j], c=rgb, s=4); ax.set_xlabel(f"z{i}"); ax.set_ylabel(f"z{j}"); ax.set_title(f"latent z{i} vs z{j}")
    else:
        ax = fig.add_subplot(rows, cols, 3); ax.scatter(z2[:, 0], z2[:, 1], c=rgb, s=4); ax.set_xlabel("z0"); ax.set_ylabel("z1"); ax.set_title(f"latent space (R^{z.shape[1]}): z0 vs z1"); ax.set_aspect("equal")
    ax = fig.add_subplot(rows, cols, 4, projection="3d"); ax.scatter(*rec.T, c=rgb, s=4); ax.set_title(pca_title("reconstructed points x_hat", rec_evr), fontsize=9); ax.set_box_aspect(np.ptp(rec, axis=0))
    ax = fig.add_subplot(rows, cols, 2, projection="3d"); ax.scatter(*inp.T, c=rgb, s=4); ax.set_title(pca_title("embedded input points x", inp_evr), fontsize=9); ax.set_box_aspect(np.ptp(inp, axis=0))
    fig.suptitle(f"{title}: colour = position on the sphere; reconstruction MSE {mse:.2g}")
    fig.savefig(path, dpi=150); plt.close(fig)
    return z, mse


def plot_latent(run):
    """Static figures of fresh points through the best weights and through every checkpoint."""
    run, cfg = load(run)
    emb, p, x, rgb = latent_inputs(cfg)

    net = load_net(run, cfg)
    z, mse = scatter_latent(net, p, x, rgb, run / "latent.png", f"{run.name} best weights")
    np.savez(run / "latent.npz", sphere_points=p.numpy(), latent=z)
    print(f"best: reconstruction MSE on {N_LATENT} fresh points {mse:.4g}", flush=True)

    (run / "latent_checkpoints").mkdir(exist_ok=True)
    for ck in sorted((run / "checkpoints").glob("step*.pt")):
        step = int(ck.stem[4:])
        net.load_state_dict(torch.load(ck))
        scatter_latent(net, p, x, rgb, run / "latent_checkpoints" / f"step{step:07d}.png", f"{run.name} step {step} ({step * cfg['batch']} samples)")


def plot_latent_3d(run):
    """Interactive version of the best weights' latent figure: four 3-D scenes (original points, embedded input PCA, latent code, reconstruction PCA) that rotate and zoom in the browser."""
    run, cfg = load(run)
    emb, p, x, rgb = latent_inputs(cfg)
    z, x_hat, mse = encode_decode(load_net(run, cfg), x)
    (inp, inp_evr), (rec, rec_evr) = pca3(x), pca3(x_hat)
    z = np.hstack([z, np.zeros((len(z), 3 - z.shape[1]))])[:, :3]                           # latent dims beyond three are not shown, fewer are padded with zeros
    colors = [f"rgb({r:.0f},{g:.0f},{b:.0f})" for r, g, b in rgb * 255]

    titles = ["original sphere points", pca_title("embedded input points x", inp_evr, "<br>"), f"latent space (first {min(cfg['embedded_dim'], 3)} of {cfg['embedded_dim']} dims)", pca_title("reconstructed points x_hat", rec_evr, "<br>")]
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}] * 2] * 2, subplot_titles=titles, vertical_spacing=.12)
    for col, pts in enumerate((p.numpy()[:, :3], inp, z, rec), 1):
        fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", marker=dict(size=2, color=colors)), row=(col - 1) // 2 + 1, col=(col - 1) % 2 + 1)
    fig.update_scenes(aspectmode="data")
    fig.update_annotations(font_size=13)                                                    # the subplot titles
    fig.update_layout(title=f"{run.name} best weights: colour = position on the sphere; reconstruction MSE {mse:.2g}", showlegend=False, height=1000, margin=dict(t=120))
    fig.write_html(run / "latent_3d.html", include_plotlyjs=True)


# ---- Gram matrices --------------------------------------------------------------------------------------------------------------------------------------
def gram(W):
    """W^T W for a tall W, W W^T for a wide W; the identity would mean orthonormal columns (tall) or rows (wide)."""
    return W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T


def heatmap(fig, ax, G, title):
    vmax = G.abs().max().item()
    im = ax.imshow(G.numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(G)):
        for j in range(len(G)): ax.text(j, i, f"{G[i, j].item():.6g}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(G))); ax.set_yticks(range(len(G)))
    ax.set_title(f"{title}\n||. - I||_F = {torch.linalg.matrix_norm(G - torch.eye(len(G))).item():.6g}")
    fig.colorbar(im, ax=ax, fraction=.046)


def plot_gram(run):
    run, cfg = load(run)
    Ws = [W.detach() for W in load_net(run, cfg).weight_matrices()]
    fig, axes = plt.subplots(1, len(Ws), figsize=(5.5 * len(Ws), 5), squeeze=False)
    for i, (ax, W) in enumerate(zip(axes[0], Ws), 1): heatmap(fig, ax, gram(W), f"Gram of W{i} {tuple(W.shape)}")
    fig.suptitle(f"{run.name} (best weights)")
    fig.savefig(run / "gram.png", dpi=150); plt.close(fig)


def plot_gram_composed(run):
    run, cfg = load(run)
    Ws = [W.detach() for W in load_net(run, cfg).weight_matrices()]
    P = Ws[0]
    for W in Ws[1:]: P = P @ W
    fig, ax = plt.subplots(figsize=(5.5, 5))
    heatmap(fig, ax, gram(P), f"{run.name} (best weights)\nGram of P = W1 ... W{len(Ws)} {tuple(P.shape)}")
    fig.savefig(run / "gram_composed.png", dpi=150); plt.close(fig)


def plot_all(run):
    for f in (plot_loss, plot_orth, plot_latent, plot_latent_3d, plot_gram, plot_gram_composed): f(run)


if __name__ == "__main__":
    for name in sys.argv[2:] or ["plot_all"]: globals()[name](sys.argv[1])
