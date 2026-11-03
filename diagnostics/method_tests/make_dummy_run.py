"""A dummy run folder with a randomly initialised (untrained) model, to look at the plots of diagnostics/run_diagnostics.py without a training run.
usage: python diagnostics/method_tests/make_dummy_run.py [<name>]      (default dummy_run; writes results/autoencoder/<name>/{config.toml, best.pt, checkpoints/step*.pt, history.csv})
S^2 in R^20 (quadratic embedding, noise 0.01), dims [20, 16, 8, 2], Arctan2 after layer 2 with layer norm and a radius head, biases, orthogonal init. The checkpoints are the weights plus growing noise, so the
curves over the checkpoints are not flat; the history has a made-up orthogonality penalty."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "auto_encoder"))
from model import make_model

NAME = sys.argv[1] if len(sys.argv) > 1 else "dummy_run"
CFG = dict(intrinsic_dim=2, embedding="quadratic", ambient_dim=20, embedded_dim=2, hidden_dims=[16, 8], activation="smooth_leaky_relu", arctan2_after=2, radius_head=16,
           arctan2_layernorm=True, bias=True, init="orthogonal", noise=0.01, data_seed=0, init_seed=0)
out = ROOT / "results" / "autoencoder" / NAME; (out / "checkpoints").mkdir(parents=True, exist_ok=True)
(out / "config.toml").write_text("".join(f'{k} = {repr(v).replace(chr(39), chr(34)).lower() if isinstance(v, bool) else (chr(34) + v + chr(34) if isinstance(v, str) else v)}\n' for k, v in CFG.items()))
torch.manual_seed(0); net = make_model(CFG)
torch.save(net.state_dict(), out / "best.pt")
for step in (1000, 2000, 3000, 4000):
    sd = {k: v + (step / 4000) * 0.05 * torch.randn_like(v) for k, v in net.state_dict().items()}
    torch.save(sd, out / "checkpoints" / f"step{step:06d}.pt")
steps = np.arange(1, 4001)
pd.DataFrame(dict(step=steps, train_orth=2.0 * np.exp(-steps / 1500) + 0.05 * np.random.default_rng(0).random(len(steps)))).to_csv(out / "history.csv", index=False)
print(out)
