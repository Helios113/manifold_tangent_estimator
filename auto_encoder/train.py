"""Train the tied-weight AutoEncoder on a manifold embedded in R^ambient_dim.

usage: train.py <run folder>

The run folder holds config.toml (see DEFAULTS for the keys; any key left out takes its default) and receives every result:
    history.csv, metrics.json, loss.png (also rewritten at every checkpoint), orth_euclidian.png, orth_spectral.png, best.pt, checkpoints/step*.pt, and the Gram plots (best weights and every checkpoint) and, for intrinsic_dim <= 3, the latent plots.

Streaming: every step draws a fresh batch from the manifold, until total_samples have been seen.
The best weights are chosen on a fixed validation set."""
import json, sys, tomllib
from pathlib import Path
import pandas as pd, torch
from model import make_model
from sphere import Sphere, EMBEDDINGS
from plots import plot_all, plot_loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"                                   # the network and the data of a run live here; saved weights are always CPU tensors

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}              # SGD is plain: no momentum, no weight decay

DEFAULTS = {
    "intrinsic_dim": 2,                 # k of the k-sphere, which lives in R^(k+1)
    "embedding": "quadratic",           # key of sphere.EMBEDDINGS
    "ambient_dim": 10,
    "embedded_dim": 3,                  # latent dimension
    "hidden_dims": [6],                 # widths between ambient_dim and embedded_dim: one weight matrix more than entries
    "activation": "smooth_leaky_relu",  # key of model.ACTIVATIONS: "smooth_leaky_relu", "leaky_relu" or "jump"
    "arctan2_after": 0,                 # k >= 1: apply model.Arctan2 to the activated output of weight matrix W_k (its width must be even; halves the input of W_(k+1)); 0 = off
    "radius_head": 0,                   # width of the hidden layer of the untied MLP that predicts the Arctan2 pair radii from the decoded angles in decode; 0 = unit radii
    "prior_lambda": 0.0,                # weight of the MMD between the batch of latent codes and a same-size sample of N(0, I), so that z ~ N(0, I) can be sampled later; 0 skips it entirely
    "init": "uniform",                  # key of model.INITS: "uniform" (nn.Linear's default draw) or "orthogonal" (all singular values of every weight matrix start at 1)
    "radius_lambda": 0.0,               # weight of the radius loss: the mean squared relative error between the radius head's prediction (from the latent code) and the true radii of the Arctan2 pairs (before the layer norm), the true radii being a constant target; needs radius_head > 0; 0 skips it entirely
    "arctan2_layernorm": False,         # layer norm (no learned parameters) over the features just before the Arctan2 of the encoder; needs arctan2_after >= 1
    "arctan2_project": False,           # instead of doubling W_k (arctan2_after = k), keep it at its width m and build each Arctan2 pair as (h, h P) with a separate untied square matrix P; the width of W_k is then the number of angles (any size); needs arctan2_after >= 1
    "arctan2_x_hidden": [],             # with arctan2_project: widths of the hidden layers of an untied MLP that builds the x entry of every pair from h (m -> widths -> m, no activation after the last matrix) instead of the matrix P; empty = use P
    "arctan2_x_activation": "",         # key of model.ACTIVATIONS for that MLP; empty = the model's activation
    "bias": False,                      # add a bias b_i (zero at the start) to every weight matrix; the decoder subtracts them (the biases are not part of the orthogonality penalty)
    "optimizer": "adam",                # key of OPTIMIZERS
    "lr": 2e-4,
    "recon_lambda": 1.0,                # weight of the reconstruction MSE in the loss; 0 leaves it out of the loss (it is still logged as train_recon / val_recon)
    "orth_lambda": 0.0,                 # weight of the orthogonality penalty sum_i ||G_i - I||_F (the Euclidian norm, not squared) on the weight matrices: nothing is divided by the size; 0 skips it entirely
    "dist_lambda": 0.0,                 # weight of the mean squared difference between the pairwise distances of the batch inputs and of their reconstructions (all pairs i < j of the batch); 0 skips it entirely
    "topk_lambda": 0.0,                 # weight of the top-k loss: the sum, over all vectors of the batch, of the squared reconstruction errors of the topk_k entries of that vector with the largest error; 0 skips it entirely
    "topk_k": 1,                        # number of entries per vector the top-k loss sums over
    "plane_lambda": 0.0,                # weight of the mean squared distance of the latent codes from the plane plane_normal . z = plane_offset; 0 skips it entirely
    "plane_normal": [],                 # normal of the plane in the latent space (embedded_dim numbers, any length); empty = the last latent axis
    "plane_offset": 0.0,                # signed distance of the plane from the origin along its unit normal
    "noise": 0.0,                       # std of the Gaussian noise added to the sphere points of every training batch (before the embedding); the validation set stays clean; 0 = none
    "batch": 32,
    "total_samples": 5_000_000,
    "n_val": 1000,
    "eval_every": 1000,                 # steps
    "ckpt_every": 10_000,               # steps; 0 = save no checkpoints
    "data_seed": 0,                     # embedding and validation set
    "init_seed": 0,                     # network initialisation
    "stream_seed": 1,                   # stream of training batches
}


def toml_value(v):
    """TOML text of a str, bool, int, float or list of those."""
    if isinstance(v, bool): return "true" if v else "false"
    if isinstance(v, str): return json.dumps(v)
    if isinstance(v, list): return "[" + ", ".join(map(toml_value, v)) + "]"
    return repr(v)


def load_config(run):
    cfg = {**DEFAULTS, **tomllib.load(open(run / "config.toml", "rb"))}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown: raise KeyError(f"unknown config keys: {sorted(unknown)}")
    if isinstance(cfg["hidden_dims"], int): cfg["hidden_dims"] = [cfg["hidden_dims"]]
    if cfg["radius_lambda"] > 0 and not (cfg["arctan2_after"] and cfg["radius_head"]): raise ValueError("radius_lambda needs arctan2_after >= 1 and radius_head > 0")
    if cfg["topk_lambda"] > 0 and not 1 <= cfg["topk_k"] <= cfg["ambient_dim"]: raise ValueError("topk_k must be between 1 and ambient_dim")
    if cfg["plane_normal"] and len(cfg["plane_normal"]) != cfg["embedded_dim"]: raise ValueError("plane_normal needs embedded_dim numbers")
    (run / "config.toml").write_text("".join(f"{k} = {toml_value(v)}\n" for k, v in cfg.items()))      # the config with every default filled in
    return cfg


HEADER_EVERY = 20                                                                       # the column names are repeated every this many log rows


def grad_names(net):
    return [f"grad_W{i + 1}" for i in range(net.n)]


def log_columns(net, cfg):
    """(header, row key) of the log table: the total train loss, the reconstruction loss, the other losses, then the Euclidian and the spectral orthonormality error of every W_i and of their product, the gradient norms."""
    cols = [("step", "step"), ("samples", "samples"), ("train_total", "train_total"), ("train_recon", "train_recon"), ("val_recon", "val_recon")]
    if cfg["orth_lambda"] > 0: cols += [("train_orth", "train_orth"), ("train_orth_scaled", "train_orth_scaled")]
    for term in ("dist", "topk", "radius", "plane", "prior"):
        if cfg[f"{term}_lambda"] > 0: cols += [(k, k) for k in (f"train_{term}", f"train_{term}_scaled", f"val_{term}", f"val_{term}_scaled")]
    for short, kind in (("eucl", "euclidian"), ("spec", "spectral")):
        cols += [(f"{short}_W{i + 1}", f"orth_{kind}_W{i + 1}") for i in range(net.n)] + [(f"{short}_comp", f"orth_{kind}_composed")]
    return cols + [(f"grad_W{i + 1}", f"grad_W{i + 1}") for i in range(net.n)]


def format_row(cols, row):
    return " ".join(f"{row[k]:>{max(10, len(h))}d}" if k in ("step", "samples") else f"{row[k]:>{max(10, len(h))}.3e}" for h, k in cols)


def format_header(cols):
    return " ".join(f"{h:>{max(10, len(h))}}" for h, _ in cols)


# ---- training -------------------------------------------------------------------------------------------------------------------------------------------
def plane_distance(z, normal, offset):
    """Mean squared distance of the latent codes z (n, d) from the plane normal . z = offset (the normal is normalised); an empty normal means the last axis."""
    n = torch.eye(z.shape[1], device=z.device)[-1] if not normal else torch.tensor(normal, dtype=z.dtype, device=z.device)
    return ((z @ (n / n.norm()) - offset) ** 2).mean()


def distance_loss(x, x_hat):
    """Mean squared difference between the pairwise distances of the inputs x (n, D) and of their reconstructions x_hat (all pairs i < j): the reconstruction must keep the points as far apart as they were."""
    return ((torch.pdist(x_hat) - torch.pdist(x)) ** 2).mean()


def topk_loss(x, x_hat, k):
    """Sum over the vectors (n, D) of the batch of the squared errors of the k entries of each vector with the largest error."""
    return ((x_hat - x) ** 2).topk(k, dim=1).values.sum()


def prior_mmd(z, scales=(0.5, 1.0, 2.0, 4.0)):
    """Biased squared MMD between the codes z (n, d) and a fresh sample of N(0, I) of the same size, with Gaussian kernels exp(-|a - b|^2 / (2 s d)) summed over the scales s."""
    y = torch.randn_like(z)
    k = lambda a, b: sum(torch.exp(-torch.cdist(a, b) ** 2 / (2 * s * z.shape[1])) for s in scales).mean()
    return k(z, z) + k(y, y) - 2 * k(z, y)


def cpu_state(net):
    return {k: v.cpu() for k, v in net.state_dict().items()}


def train_step(net, opt, x, cfg):
    """One optimiser step on batch x. Returns the logged values (gradient norms are taken before the step)."""
    z = net.encode(x)
    x_hat = net.decode(z)
    mse = ((x_hat - x) ** 2).mean()
    loss = cfg["recon_lambda"] * mse
    if cfg["orth_lambda"] > 0:
        penalty = net.orth_penalty()
        loss = loss + cfg["orth_lambda"] * penalty
    if cfg["dist_lambda"] > 0:
        dist = distance_loss(x, x_hat)
        loss = loss + cfg["dist_lambda"] * dist
    if cfg["topk_lambda"] > 0:
        topk = topk_loss(x, x_hat, cfg["topk_k"])
        loss = loss + cfg["topk_lambda"] * topk
    if cfg["radius_lambda"] > 0:
        radius = net.radius_loss(x, z)
        loss = loss + cfg["radius_lambda"] * radius
    if cfg["plane_lambda"] > 0:
        plane = plane_distance(z, cfg["plane_normal"], cfg["plane_offset"])
        loss = loss + cfg["plane_lambda"] * plane
    if cfg["prior_lambda"] > 0:
        prior = prior_mmd(z)
        loss = loss + cfg["prior_lambda"] * prior
    opt.zero_grad(); loss.backward()
    grads = {name: W.grad.norm().item() for name, W in zip(grad_names(net), net.weight_matrices())}      # Frobenius norm of the batch gradient
    opt.step()
    row = dict(train_recon=mse.item(), train_total=loss.item(), **grads)      # train_recon is the reconstruction MSE, train_total = recon_lambda x train_recon + the sum of the *_scaled terms
    for term, value in (("orth", penalty if cfg["orth_lambda"] > 0 else None), ("dist", dist if cfg["dist_lambda"] > 0 else None), ("topk", topk if cfg["topk_lambda"] > 0 else None), ("radius", radius if cfg["radius_lambda"] > 0 else None),
                        ("plane", plane if cfg["plane_lambda"] > 0 else None), ("prior", prior if cfg["prior_lambda"] > 0 else None)):
        if value is not None: row[f"train_{term}"], row[f"train_{term}_scaled"] = value.item(), cfg[f"{term}_lambda"] * value.item()      # raw term, and lambda x raw term
    return row


def evaluate(net, X_val, cfg):
    with torch.no_grad():
        z = net.encode(X_val)
        x_hat = net.decode(z)
        row = dict(val_recon=((x_hat - X_val) ** 2).mean().item(), **net.orth_errors())
        val = {}
        if cfg["dist_lambda"] > 0: val["dist"] = distance_loss(X_val, x_hat).item()
        if cfg["topk_lambda"] > 0: val["topk"] = topk_loss(X_val, x_hat, cfg["topk_k"]).item()
        if cfg["radius_lambda"] > 0: val["radius"] = net.radius_loss(X_val, z).item()
        if cfg["plane_lambda"] > 0: val["plane"] = plane_distance(z, cfg["plane_normal"], cfg["plane_offset"]).item()
        if cfg["prior_lambda"] > 0: val["prior"] = prior_mmd(z).item()
        for term, value in val.items(): row[f"val_{term}"], row[f"val_{term}_scaled"] = value, cfg[f"{term}_lambda"] * value
    return row


def train(cfg, run, S, emb, X_val):
    """Train on a stream of fresh batches. Returns the history rows and the metrics of the best weights."""
    torch.manual_seed(cfg["init_seed"])
    stream = torch.Generator().manual_seed(cfg["stream_seed"])
    net = make_model(cfg).to(DEVICE)
    orth_before = net.orth_errors()
    opt = OPTIMIZERS[cfg["optimizer"]](net.parameters(), cfg["lr"])

    cols, n_logged = log_columns(net, cfg), 0
    rows = []
    best_val, best_step, step, seen = float("inf"), -1, 0, 0
    while seen < cfg["total_samples"]:
        x = S.embed_points(S.sample(cfg["batch"], stream, noise=cfg["noise"]), emb).to(DEVICE)      # drawn on the CPU: the stream generator is a CPU one
        row = train_step(net, opt, x, cfg)
        step += 1; seen += cfg["batch"]
        row.update(step=step, samples=seen)

        if step % cfg["eval_every"] == 0:
            row.update(evaluate(net, X_val, cfg))
            if n_logged % HEADER_EVERY == 0: print(format_header(cols), flush=True)
            print(format_row(cols, row), flush=True)
            n_logged += 1
            if row["val_recon"] < best_val:
                best_val, best_step = row["val_recon"], step
                torch.save(cpu_state(net), run / "best.pt")
        rows.append(row)
        if cfg["ckpt_every"] and step % cfg["ckpt_every"] == 0:
            torch.save(cpu_state(net), run / "checkpoints" / f"step{step:06d}.pt")
            pd.DataFrame(rows).to_csv(run / "history.csv", index=False); plot_loss(run)            # the loss plot follows the training

    net.load_state_dict(torch.load(run / "best.pt"))                                    # evaluate the best weights
    with torch.no_grad():
        z = net.encode(X_val)
        data_var = X_val.var(0).mean().item()
        metrics = dict(best_step=best_step, steps=step, samples=seen, val_mse=best_val, data_var=data_var, relative_mse=best_val / data_var,
                       enc_dec_latent_err=(net.encode(net.decode(z)) - z).norm(dim=1).mean().item(),
                       orth_before=orth_before, orth_after=net.orth_errors())
    print(metrics, flush=True)
    return rows, metrics


# ---- main -----------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    run = Path(sys.argv[1]).resolve()
    cfg = load_config(run)
    if (run / "history.csv").exists(): raise FileExistsError(f"{run} already holds results")
    (run / "checkpoints").mkdir(exist_ok=True)
    print("run folder:", run, "device:", DEVICE, flush=True)
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()                                         # the whole GPU, including other users' jobs on it
        print(f"gpu: {torch.cuda.get_device_name()}, free VRAM {free / 2**30:.1f} of {total / 2**30:.1f} GiB", flush=True)

    gen = torch.Generator().manual_seed(cfg["data_seed"])
    S = Sphere(cfg["intrinsic_dim"])
    emb = EMBEDDINGS[cfg["embedding"]](cfg["ambient_dim"], gen, in_dim=S.D)      # one fixed embedding
    X_val = S.embed_points(S.sample(cfg["n_val"], gen), emb).to(DEVICE)

    rows, metrics = train(cfg, run, S, emb, X_val)
    history = pd.DataFrame(rows)
    history.to_csv(run / "history.csv", index=False)
    json.dump(metrics, open(run / "metrics.json", "w"), indent=1)

    plot_all(run)
