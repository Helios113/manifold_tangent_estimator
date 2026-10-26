"""Train the tied-weight AutoEncoder on a manifold embedded in R^ambient_dim.

usage: train.py <run folder>

The run folder holds config.toml (see DEFAULTS for the keys; any key left out takes its default) and receives every result:
    history.csv, metrics.json, loss.png, orth_euclidian.png, orth_spectral.png, best.pt, checkpoints/step*.pt, and the latent / Gram plots.

Streaming: every step draws a fresh batch from the manifold, until total_samples have been seen.
The best weights are chosen on a fixed validation set."""
import json, sys, tomllib
from pathlib import Path
import pandas as pd, torch
from model import make_model
from sphere import Sphere, EMBEDDINGS
from plots import plot_all

OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}              # SGD is plain: no momentum, no weight decay

DEFAULTS = {
    "intrinsic_dim": 2,                 # k of the k-sphere, which lives in R^(k+1)
    "embedding": "quadratic",           # key of sphere.EMBEDDINGS
    "ambient_dim": 10,
    "embedded_dim": 3,                  # latent dimension
    "hidden_dims": [6],                 # widths between ambient_dim and embedded_dim: one weight matrix more than entries
    "optimizer": "adam",                # key of OPTIMIZERS
    "lr": 2e-4,
    "orth_lambda": 0.0,                 # weight of the orthogonality penalty sum_i ||G_i - I||_F^2 on the weight matrices ; 0 skips it entirely
    "plane_lambda": 0.0,                # weight of the mean squared distance of the latent codes from the plane plane_normal . z = plane_offset; 0 skips it entirely
    "plane_normal": [],                 # normal of the plane in the latent space (embedded_dim numbers, any length); empty = the last latent axis
    "plane_offset": 0.0,                # signed distance of the plane from the origin along its unit normal
    "batch": 32,
    "total_samples": 5_000_000,
    "n_val": 1000,
    "eval_every": 1000,                 # steps
    "ckpt_every": 10_000,               # steps
    "data_seed": 0,                     # embedding and validation set
    "init_seed": 0,                     # network initialisation
    "stream_seed": 1,                   # stream of training batches
}


def toml_value(v):
    """TOML text of a str, int, float or list of those."""
    if isinstance(v, str): return json.dumps(v)
    if isinstance(v, list): return "[" + ", ".join(map(toml_value, v)) + "]"
    return repr(v)


def load_config(run):
    cfg = {**DEFAULTS, **tomllib.load(open(run / "config.toml", "rb"))}
    unknown = set(cfg) - set(DEFAULTS)
    if unknown: raise KeyError(f"unknown config keys: {sorted(unknown)}")
    if isinstance(cfg["hidden_dims"], int): cfg["hidden_dims"] = [cfg["hidden_dims"]]
    if cfg["plane_normal"] and len(cfg["plane_normal"]) != cfg["embedded_dim"]: raise ValueError("plane_normal needs embedded_dim numbers")
    (run / "config.toml").write_text("".join(f"{k} = {toml_value(v)}\n" for k, v in cfg.items()))      # the config with every default filled in
    return cfg


HEADER_EVERY = 20                                                                       # the column names are repeated every this many log rows


def grad_names(net):
    return [f"grad_W{i + 1}" for i in range(net.n)]


def log_columns(net, cfg):
    """(header, row key) of the log table: losses, then the Euclidian and the spectral orthonormality error of every W_i and of their product, the gradient norms."""
    cols = [("step", "step"), ("samples", "samples"), ("train_mse", "train_loss"), ("val_mse", "val_mse")]
    if cfg["orth_lambda"] > 0: cols.append(("train_pen", "train_orth_penalty"))
    if cfg["plane_lambda"] > 0: cols += [("train_plane", "train_plane"), ("val_plane", "val_plane")]
    for short, kind in (("eucl", "euclidian"), ("spec", "spectral")):
        cols += [(f"{short}_W{i + 1}", f"orth_{kind}_W{i + 1}") for i in range(net.n)] + [(f"{short}_comp", f"orth_{kind}_composed")]
    return cols + [(f"grad_W{i + 1}", f"grad_W{i + 1}") for i in range(net.n)]


def format_row(cols, row):
    return " ".join(f"{row[k]:>10d}" if k in ("step", "samples") else f"{row[k]:>10.3e}" for _, k in cols)


def format_header(cols):
    return " ".join(f"{h:>10}" for h, _ in cols)


# ---- training -------------------------------------------------------------------------------------------------------------------------------------------
def plane_distance(z, normal, offset):
    """Mean squared distance of the latent codes z (n, d) from the plane normal . z = offset (the normal is normalised); an empty normal means the last axis."""
    n = torch.eye(z.shape[1])[-1] if not normal else torch.tensor(normal, dtype=z.dtype)
    return ((z @ (n / n.norm()) - offset) ** 2).mean()


def train_step(net, opt, x, cfg):
    """One optimiser step on batch x. Returns the logged values (gradient norms are taken before the step)."""
    z = net.encode(x)
    mse = ((net.decode(z) - x) ** 2).mean()
    loss = mse
    if cfg["orth_lambda"] > 0:
        penalty = net.orth_penalty()
        loss = loss + cfg["orth_lambda"] * penalty
    if cfg["plane_lambda"] > 0:
        plane = plane_distance(z, cfg["plane_normal"], cfg["plane_offset"])
        loss = loss + cfg["plane_lambda"] * plane
    opt.zero_grad(); loss.backward()
    grads = {name: W.grad.norm().item() for name, W in zip(grad_names(net), net.weight_matrices())}      # Frobenius norm of the batch gradient
    opt.step()
    row = dict(train_loss=mse.item(), **grads)
    if cfg["orth_lambda"] > 0: row["train_orth_penalty"] = penalty.item()
    if cfg["plane_lambda"] > 0: row["train_plane"] = plane.item()
    return row


def evaluate(net, X_val, cfg):
    with torch.no_grad():
        z = net.encode(X_val)
        row = dict(val_mse=((net.decode(z) - X_val) ** 2).mean().item(), **net.orth_errors())
        if cfg["plane_lambda"] > 0: row["val_plane"] = plane_distance(z, cfg["plane_normal"], cfg["plane_offset"]).item()
    return row


def train(cfg, run, S, emb, X_val):
    """Train on a stream of fresh batches. Returns the history rows and the metrics of the best weights."""
    torch.manual_seed(cfg["init_seed"])
    stream = torch.Generator().manual_seed(cfg["stream_seed"])
    net = make_model(cfg)
    orth_before = net.orth_errors()
    opt = OPTIMIZERS[cfg["optimizer"]](net.parameters(), cfg["lr"])

    cols, n_logged = log_columns(net, cfg), 0
    rows = []
    best_val, best_step, step, seen = float("inf"), -1, 0, 0
    while seen < cfg["total_samples"]:
        x = S.embed_points(S.sample(cfg["batch"], stream), emb)
        row = train_step(net, opt, x, cfg)
        step += 1; seen += cfg["batch"]
        row.update(step=step, samples=seen)

        if step % cfg["eval_every"] == 0:
            row.update(evaluate(net, X_val, cfg))
            if n_logged % HEADER_EVERY == 0: print(format_header(cols), flush=True)
            print(format_row(cols, row), flush=True)
            n_logged += 1
            if row["val_mse"] < best_val:
                best_val, best_step = row["val_mse"], step
                torch.save(net.state_dict(), run / "best.pt")
        if step % cfg["ckpt_every"] == 0:
            torch.save(net.state_dict(), run / "checkpoints" / f"step{step:06d}.pt")
        rows.append(row)

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
    print("run folder:", run, flush=True)

    gen = torch.Generator().manual_seed(cfg["data_seed"])
    S = Sphere(cfg["intrinsic_dim"])
    emb = EMBEDDINGS[cfg["embedding"]](cfg["ambient_dim"], gen, in_dim=S.D)      # one fixed embedding
    X_val = S.embed_points(S.sample(cfg["n_val"], gen), emb)

    rows, metrics = train(cfg, run, S, emb, X_val)
    history = pd.DataFrame(rows)
    history.to_csv(run / "history.csv", index=False)
    json.dump(metrics, open(run / "metrics.json", "w"), indent=1)

    plot_all(run)
