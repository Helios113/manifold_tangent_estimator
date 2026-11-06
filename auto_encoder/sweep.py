"""Hyper-parameter sweep of train.py: one run per point of a grid, each submitted as its own slurm job.

usage: python auto_encoder/sweep.py <sweep folder> [submit | summarise]      (default: submit)

The sweep folder holds sweep.toml, e.g.
    sbatch_args = ["--time=04:00:00"]                       # optional: extra sbatch arguments for every job
    [base]                                                  # config.toml keys shared by every run (see train.DEFAULTS)
    total_samples = 5000000
    [grid]                                                  # one run per combination of these values
    embedded_dim = [2, 3, 4]
    embedded_extra = [0, 1]                                 # not a train.py key: sets embedded_dim = intrinsic_dim + embedded_extra (do not also give embedded_dim)
    layers = [1, 2, 3, 4]                                   # not a train.py key: number of weight matrices; sets hidden_dims by hidden_widths (do not also give hidden_dims)
    expansion = [false, true]                               # not a train.py key: the second width is 2 * ambient_dim (needs layers >= 3)
    lr = [1e-3, 2e-4, 5e-5]
    orth_lambda = [0.0, 0.1, 1.0]

    max_runs = 1000                                         # optional, top level: submit at most this many unfinished runs (the first ones of the grid)
    max_concurrent = 10                                     # optional, top level: at most this many array tasks active at once (default 10)
    runs_per_job = 4                                        # optional, top level: runs trained at the same time by one array task, on its one GPU (default 1; give the job --cpus-per-task >= runs_per_job)

    submit      creates <sweep folder>/runs/<name>/config.toml for every grid point and submits the runs as job arrays (sweep.sbatch), at most max_concurrent running at a time; jobs.csv lists the job ids.
                A run that already has a history.csv or is queued / running is skipped, so a half-finished sweep can be submitted again.
    summarise   summary.csv (config and metrics of every finished run) and summary.png (relative validation MSE against orth_lambda, one panel per embedded_dim, one line per optimizer, lr and hidden_dims)."""
import itertools, json, os, subprocess, sys, tomllib
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import DEFAULTS, toml_value

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "auto_encoder" / "sweep.sbatch"
MAX_ARRAY = 1000                                                                        # slurm allows arrays of 1001 tasks
PSEUDO_KEYS = {"embedded_extra", "layers", "expansion"}
METRICS = ("best_step", "val_mse", "relative_mse", "enc_dec_latent_err")


def load_sweep(sweep):
    spec = tomllib.load(open(sweep / "sweep.toml", "rb"))
    unknown = (set(spec["base"]) | set(spec["grid"])) - set(DEFAULTS) - PSEUDO_KEYS
    if unknown: raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return spec


def fmt(v):
    """A config value as text for run names and tables: [8, 6] -> 8-6."""
    return "-".join(map(str, v)) if isinstance(v, list) else str(v)


def even(x):
    return 2 * round(x / 2)


def hidden_widths(ambient, latent, layers, expansion):
    """Hidden widths of a network with `layers` weight matrices (layers - 1 widths): the first is ambient_dim; with expansion the second is 2 * ambient_dim.
    The rest step linearly down to the latent size, so the widths contract after the first (or the second, with expansion). All rounded to even."""
    widths = ([ambient] + ([2 * ambient] if expansion else []))[:layers - 1]
    n_rest = layers - 1 - len(widths)
    start = widths[-1] if widths else ambient
    return widths + [even(start + (latent - start) * (i + 1) / (n_rest + 1)) for i in range(n_rest)]


def valid(cfg):
    """False for the grid points that cannot exist or duplicate another one."""
    layers, expansion = cfg.get("layers", 2), cfg.get("expansion", False)
    if expansion and layers < 3: return False                       # needs both the ambient_dim and the 2 * ambient_dim widths
    if cfg.get("ambient_dim", DEFAULTS["ambient_dim"]) < cfg.get("intrinsic_dim", DEFAULTS["intrinsic_dim"]) + 2: return False       # the quadratic embedding needs ambient_dim >= intrinsic_dim + 2
    return True


def grid_points(spec):
    """(run name, config) for every valid combination of the grid values."""
    keys = list(spec["grid"])
    for values in itertools.product(*spec["grid"].values()):
        cfg = {**spec["base"], **dict(zip(keys, values))}
        if not valid(cfg): continue
        if "embedded_extra" in cfg: cfg["embedded_dim"] = cfg.get("intrinsic_dim", DEFAULTS["intrinsic_dim"]) + cfg.pop("embedded_extra")
        if "layers" in cfg:
            cfg["hidden_dims"] = hidden_widths(cfg.get("ambient_dim", DEFAULTS["ambient_dim"]), cfg.get("embedded_dim", DEFAULTS["embedded_dim"]), cfg.pop("layers"), cfg.pop("expansion", False))
        yield "_".join(f"{k}{fmt(v)}" for k, v in zip(keys, values)), cfg


def queued_run_names():
    """Names of this user's jobs that are queued or running (arrays are listed as the array name)."""
    out = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%j"], capture_output=True, text=True, check=True)
    return set(out.stdout.split())


def submit(sweep):
    """Writes every run folder, then submits the unfinished runs as job arrays of equal size, at most MAX_ARRAY tasks each, each array throttled to max_concurrent running tasks (sweep.toml, default 10)
    and starting only after the previous array has ended, so no more than max_concurrent runs are ever active."""
    spec = load_sweep(sweep)
    queued, pending = queued_run_names(), []
    for name, cfg in grid_points(spec):
        run = sweep / "runs" / name
        if (run / "history.csv").exists(): print(f"skip {name}: already finished", flush=True); continue
        if name in queued: print(f"skip {name}: already queued", flush=True); continue
        if len(pending) >= spec.get("max_runs", float("inf")): break
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.toml").write_text("".join(f"{k} = {toml_value(v)}\n" for k, v in cfg.items()))
        pending.append(run)
    rows, previous = [], None
    (sweep / "queue").mkdir(exist_ok=True)
    k = spec.get("runs_per_job", 1)                                                        # runs trained at the same time by one array task (they share its GPU)
    tasks = [pending[i:i + k] for i in range(0, len(pending), k)]
    size = -(-len(tasks) // max(-(-len(tasks) // MAX_ARRAY), 1))                             # the arrays are of equal size (+-1): as many as needed for MAX_ARRAY, then the tasks divided evenly
    stamp = subprocess.run(["date", "+%Y%m%d_%H%M%S"], capture_output=True, text=True).stdout.strip()
    for c, i in enumerate(range(0, len(tasks), size)):
        chunk = tasks[i:i + size]
        listing = sweep / "queue" / f"chunk{c:03d}_{stamp}.txt"
        listing.write_text("".join(" ".join(map(str, task)) + "\n" for task in chunk))
        dep = [f"--dependency=afterany:{previous}"] if previous else []
        out = subprocess.run(["sbatch", "--parsable", f"--job-name=sweep_{sweep.name}", f"--array=0-{len(chunk) - 1}%{spec.get('max_concurrent', 10)}", *dep, *spec.get("sbatch_args", []), str(SBATCH), str(listing)],
                             capture_output=True, text=True, check=True)
        previous = out.stdout.strip().split(";")[0]
        rows += [dict(job_id=f"{previous}_{j}", run=run.name) for j, task in enumerate(chunk) for run in task]
        print(f"Job {previous}: array of {len(chunk)} tasks, {sum(map(len, chunk))} runs ({chunk[0][0].name} ... {chunk[-1][-1].name})", flush=True)
    if rows: pd.DataFrame(rows).to_csv(sweep / "jobs.csv", index=False)


def summarise(sweep):
    rows = []
    for run in sorted((sweep / "runs").glob("*")):
        if not (run / "metrics.json").exists(): continue
        cfg, m = tomllib.load(open(run / "config.toml", "rb")), json.load(open(run / "metrics.json"))
        hidden = cfg["hidden_dims"] if isinstance(cfg["hidden_dims"], list) else [cfg["hidden_dims"]]
        rows.append(dict(run=run.name, **{k: cfg[k] for k in {*load_sweep(sweep)["grid"], "embedded_dim", "optimizer", "lr", "orth_lambda"} - {"hidden_dims"} - PSEUDO_KEYS},
                         hidden_dims=fmt(hidden), n_weight_matrices=len(hidden) + 1, **{k: m[k] for k in METRICS},
                         orth_euclidian_composed=m["orth_after"]["orth_euclidian_composed"], orth_spectral_composed=m["orth_after"]["orth_spectral_composed"]))
    df = pd.DataFrame(rows)
    df.to_csv(sweep / "summary.csv", index=False)
    print(df.to_string(index=False), flush=True)

    dims = sorted(df.embedded_dim.unique())
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=True, squeeze=False)
    for ax, dim in zip(axes[0], dims):
        for (optimizer, lr, hidden), g in df[df.embedded_dim == dim].groupby(["optimizer", "lr", "hidden_dims"]):
            g = g.sort_values("orth_lambda"); ax.plot(g.orth_lambda, g.relative_mse, "o-", label=f"{optimizer} lr {lr:g}, hidden {hidden}")
        ax.set_xscale("symlog", linthresh=1e-2); ax.set_yscale("log")                  # symlog so that orth_lambda = 0 has a place
        ax.set_title(f"embedded_dim {dim}"); ax.set_xlabel("orth_lambda"); ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("relative validation MSE (best weights)")
    fig.savefig(sweep / "summary.png", dpi=150, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    sweep = Path(sys.argv[1]).resolve()
    {"submit": submit, "summarise": summarise}[sys.argv[2] if len(sys.argv) > 2 else "submit"](sweep)
