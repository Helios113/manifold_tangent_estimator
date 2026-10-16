import sys; sys.path.insert(0, ".")
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tangent_est import MANIFOLDS
from tangent_est.manifolds import MANIFOLDS as M
rng = np.random.default_rng(0)
fig = plt.figure(figsize=(13, 10))
for i, (name, (f, d)) in enumerate(M.items()):
    Y, T = f(1500, rng)
    ax = fig.add_subplot(3, 3, i + 1, projection="3d" if Y.shape[1] >= 3 else None)
    if Y.shape[1] == 2: ax.scatter(*Y.T, s=2); ax.set_aspect("equal")
    else:
        ax.scatter(*Y[:, :3].T, s=2, c=Y[:, 0], cmap="viridis"); ax.set_box_aspect(np.ptp(Y[:, :3], axis=0))
    ax.set_title(f"{name}  (d={d}, in R^{Y.shape[1]}" + (", first 3 coords shown)" if Y.shape[1] > 3 else ")"), fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); 
    if Y.shape[1] >= 3: ax.set_zticks([])
fig.tight_layout(); fig.savefig("results/manifolds.png", dpi=120)
