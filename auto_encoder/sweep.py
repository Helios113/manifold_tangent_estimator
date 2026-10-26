"""Hyper-parameter sweep of train.py: one run per point of a grid, each submitted as its own slurm job.

usage: python auto_encoder/sweep.py <sweep folder> [submit | summarise]      (default: submit)

The sweep folder holds sweep.toml, e.g.
    sbatch_args = ["-w", "mauao", "--time=04:00:00"]       # optional: extra sbatch arguments for every job
    [base]                                                  # config.toml keys shared by every run (see train.DEFAULTS)
    total_samples = 5000000
    [grid]                                                  # one run per combination of these values
    embedded_dim = [2, 3, 4]
    hidden_dims = [[6], [8, 6], [10, 8, 6]]                 # network depth: one list of hidden widths per entry, so 2, 3 and 4 weight matrices
    lr = [1e-3, 2e-4, 5e-5]
    orth_lambda = [0.0, 0.1, 1.0]

    submit      creates <sweep folder>/runs/<name>/config.toml for every grid point and submits train.sbatch for it; jobs.csv lists the job ids.
                A run that already has a history.csv is skipped, so a half-finished sweep can be submitted again.
    summarise   summary.csv (config and metrics of every finished run) and summary.png (relative validation MSE against orth_lambda, one panel per embedded_dim, one line per optimizer, lr and hidden_dims)."""
import itertools, json, subprocess, sys, tomllib
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from train import DEFAULTS, toml_value

ROOT = Path(__file__).resolve().parents[1]
SBATCH = ROOT / "auto_encoder" / "train.sbatch"
METRICS = ("best_step", "val_mse", "relative_mse", "enc_dec_latent_err")


def load_sweep(sweep):
    spec = tomllib.load(open(sweep / "sweep.toml", "rb"))
    unknown = (set(spec["base"]) | set(spec["grid"])) - set(DEFAULTS)
    if unknown: raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return spec


def fmt(v):
    """A config value as text for run names and tables: [8, 6] -> 8-6."""
    return "-".join(map(str, v)) if isinstance(v, list) else str(v)


def grid_points(spec):
    """(run name, config) for every combination of the grid values."""
    keys = list(spec["grid"])
    for values in itertools.product(*spec["grid"].values()):
        yield "_".join(f"{k}{fmt(v)}" for k, v in zip(keys, values)), {**spec["base"], **dict(zip(keys, values))}


def submit(sweep):
    spec = load_sweep(sweep)
    jobs = []
    for name, cfg in grid_points(spec):
        run = sweep / "runs" / name
        if (run / "history.csv").exists(): print(f"skip {name}: already finished", flush=True); continue
        run.mkdir(parents=True, exist_ok=True)
        (run / "config.toml").write_text("".join(f"{k} = {toml_value(v)}\n" for k, v in cfg.items()))
        out = subprocess.run(["sbatch", "--parsable", f"--job-name={name}", *spec.get("sbatch_args", []), str(SBATCH), str(run)], capture_output=True, text=True, check=True)
        jobs.append(dict(job_id=out.stdout.strip().split(";")[0], run=name))
        print(f"Job {jobs[-1]['job_id']}: {name}", flush=True)
    pd.DataFrame(jobs).to_csv(sweep / "jobs.csv", index=False)


def summarise(sweep):
    rows = []
    for run in sorted((sweep / "runs").glob("*")):
        if not (run / "metrics.json").exists(): continue
        cfg, m = tomllib.load(open(run / "config.toml", "rb")), json.load(open(run / "metrics.json"))
        hidden = cfg["hidden_dims"] if isinstance(cfg["hidden_dims"], list) else [cfg["hidden_dims"]]
        rows.append(dict(run=run.name, **{k: cfg[k] for k in {*load_sweep(sweep)["grid"], "embedded_dim", "optimizer", "lr", "orth_lambda"} - {"hidden_dims"}},
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
