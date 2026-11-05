"""All diagnostics of diagnostics/ on a finished run, written into the run folder: <run folder>/eval/<diagnostic>/ (not into results/diagnostics/).
usage: python diagnostics/run_diagnostics.py [--delete-tangent-npz] [--only a,b,...] <run name or folder> [<run name or folder> ...]
--delete-tangent-npz: once tangent, reconstruction_error and tangent_methods are done (their plots written), delete eval/tangent/*.npz to save space
--only a,b: run just these diagnostics and keep the other entries of an existing eval/status.json

Plan (per run; a diagnostic that fails is logged and the others still run; <run folder>/eval/status.json says for each: done, skipped (why) or failed (the error), and the seconds)
  1. tangent                    eval/tangent/                   the true / estimated / decoder tangent spaces and their principal angles (pushforward, sample_latent_tangent, decoder_tangent, compare of tangent.py)
  2. reconstruction_error       eval/reconstruction_error/      the reconstruction error and the angle statistics per direction (reads eval/tangent/)
  3. lpca_lego                  eval/lpca_lego/                 LPCA and LEGO against the true tangent space on the run's generator
  4. tangent_methods            eval/tangent_methods/           the angle to the true tangent of the autoencoder, LPCA and LEGO on one axis (reads eval/tangent/ and eval/lpca_lego/)
  5. activation_distributions   eval/activation_distributions/  the element-wise distribution of the values through the encoder and the decoder
  6. singular_values_over_training  eval/singular_values_over_training/   the singular values of the weights and the layer Jacobians over the checkpoints (needs checkpoints)
  7. fold_determinant_stages    eval/fold_determinant_stages/   the folds (where, and which stage creates them) and the loops along great circles, pole estimate and random control (needs embedded_dim = intrinsic_dim)"""

import json
import sys
import time
import traceback

import tangent
from activation_distributions import activation_distributions
from common import open_run
from fold_determinant_stages import fold_determinant_stages
from lpca_lego import lpca_lego
from reconstruction_error import reconstruction_error
from singular_values_over_training import singular_values_over_training
from tangent_methods import tangent_methods


def run_tangent(run, out):
    tangent.OUT = out
    for name in ("pushforward", "sample_latent_tangent", "decoder_tangent", "compare"):
        getattr(tangent, name)(run)


def diagnostics(run, cfg, eval_dir):
    """[(name, function(out folder), reason to skip or None)] in the order they run."""
    return [
        ("tangent", lambda out: run_tangent(run, out), None),
        (
            "reconstruction_error",
            lambda out: reconstruction_error(run, out, data=eval_dir / "tangent"),
            None,
        ),
        ("lpca_lego", lambda out: lpca_lego(run, out), None),
        (
            "tangent_methods",
            lambda out: tangent_methods(
                eval_dir / "tangent", eval_dir / "lpca_lego", out, run.name
            ),
            None,
        ),
        (
            "activation_distributions",
            lambda out: activation_distributions(run, out),
            None,
        ),
        (
            "singular_values_over_training",
            lambda out: singular_values_over_training(run, out),
            None
            if list((run / "checkpoints").glob("step*.pt"))
            else "no checkpoints in the run",
        ),
        (
            "fold_determinant_stages",
            lambda out: fold_determinant_stages(run, out),
            None
            if cfg["embedded_dim"] == cfg["intrinsic_dim"]
            else "the pushforward is square only when embedded_dim = intrinsic_dim",
        ),
    ]


def run_diagnostics(run, delete_tangent_npz=False, only=None):
    """run: run name (results/autoencoder/<run>) or run folder. Returns the status dict that is also written to <run folder>/eval/status.json."""
    run, cfg = open_run(run)
    eval_dir = run / "eval"
    eval_dir.mkdir(exist_ok=True)
    status = (
        json.load(open(eval_dir / "status.json"))
        if only and (eval_dir / "status.json").exists()
        else {}
    )
    for name, fn, skip in diagnostics(run, cfg, eval_dir):
        if only and name not in only:
            continue
        print(
            f"=== {run.name}: {name}" + (f" skipped: {skip}" if skip else ""),
            flush=True,
        )
        if skip:
            status[name] = dict(status="skipped", reason=skip)
            continue
        t0 = time.time()
        try:
            fn(eval_dir / name)
            status[name] = dict(status="done", seconds=round(time.time() - t0))
        except Exception as e:
            traceback.print_exc()
            status[name] = dict(
                status="failed",
                error=f"{type(e).__name__}: {e}",
                seconds=round(time.time() - t0),
            )
        json.dump(status, open(eval_dir / "status.json", "w"), indent=1)
    json.dump(status, open(eval_dir / "status.json", "w"), indent=1)
    if delete_tangent_npz and all(
        status[n]["status"] == "done"
        for n in ("tangent", "reconstruction_error", "tangent_methods")
    ):
        for f in (eval_dir / "tangent").glob("*.npz"):
            f.unlink()
    return status


if __name__ == "__main__":
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only")
        only = args[i + 1].split(",")
        del args[i : i + 2]
    flag = "--delete-tangent-npz" in args
    for r in [a for a in args if a != "--delete-tangent-npz"]:
        print(r, json.dumps(run_diagnostics(r, flag, only), indent=1), flush=True)
