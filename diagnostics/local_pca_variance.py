"""Diagnostic: Local PCA variance per direction.

For a trained autoencoder, this script:
1. Samples points on the manifold
2. For each point, performs PCA on its local neighborhood in the EMBEDDED AMBIENT space
3. Plots the variance explained by each principal component
4. Shows how the local tangent space varies across the manifold

This version does NOT assume knowledge of the intrinsic dimension - it performs
PCA on the full ambient space of the EMBEDDED INPUT points (before autoencoder).

Usage: python diagnostics/local_pca_variance.py <run folder> [--n_points 1000] [--radius 0.1]
"""

import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "auto_encoder"))

from sphere import MANIFOLDS, EMBEDDINGS
from common import load_net, open_run, output_dir


def local_pca_variance(run, n_points=1000, radius=0.1, n_neighbors=100, max_pcs=None):
    """Compute and plot local PCA variance per direction.
    
    Performs PCA on the full ambient space without assuming intrinsic dimension.
    
    Args:
        run: Path to the run folder
        n_points: Number of points to sample on the manifold
        radius: Radius for local neighborhood (in latent space)
        n_neighbors: Minimum number of neighbors for PCA
        max_pcs: Maximum number of principal components to plot (None = all)
    """
    run, cfg = open_run(run)
    out = output_dir("local_pca_variance", run)
    
    # Get manifold and embedding from config
    S = MANIFOLDS[cfg["manifold"]](cfg["intrinsic_dim"])
    emb = EMBEDDINGS[cfg["embedding"]](
        cfg["ambient_dim"], 
        torch.Generator().manual_seed(cfg["data_seed"]), 
        in_dim=S.D
    )
    
    # Load the trained network
    net = load_net(run, cfg)
    
    # Sample points on the manifold
    gen = torch.Generator().manual_seed(42)
    p = S.sample(n_points, gen)  # Points on the manifold (n_points, S.D)
    x = emb(p)  # Embedded points (n_points, ambient_dim)
    
    # Get latent codes
    with torch.no_grad():
        z = net.encode(x)  # (n_points, embedded_dim)
    
    # Get ambient dimension
    ambient_dim = cfg["ambient_dim"]
    
    # For each point, find local neighborhood in LATENT space and perform PCA on EMBEDDED AMBIENT space
    # We use latent space for neighborhood finding, but PCA is on the embedded ambient space
    all_variances = []
    
    for i in range(n_points):
        # Find neighbors within radius in latent space
        dists = torch.norm(z - z[i], dim=1)
        neighbors = dists < radius
        
        # If not enough neighbors, take the closest n_neighbors
        if neighbors.sum() < n_neighbors:
            _, indices = torch.topk(dists, k=n_neighbors, largest=False)
            neighbor_indices = indices
        else:
            neighbor_indices = torch.where(neighbors)[0]
        
        # Get the EMBEDDED ambient space points for these neighbors
        neighbor_x = x[neighbor_indices]  # (n_neighbors, ambient_dim)
        
        # Center the neighborhood in ambient space
        neighbor_x_centered = neighbor_x - neighbor_x.mean(dim=0)
        
        # Perform PCA using SVD on the full ambient space
        _, S, Vt = torch.linalg.svd(neighbor_x_centered, full_matrices=False)
        
        # Get variances (squared singular values divided by n-1)
        # S has shape (min(n_neighbors, ambient_dim),) but we want all ambient_dim variances
        # Pad with zeros if needed
        variances = torch.zeros(ambient_dim, dtype=S.dtype, device=S.device)
        n_components = min(len(S), ambient_dim)
        variances[:n_components] = S[:n_components] ** 2 / (len(neighbor_x) - 1)
        
        all_variances.append(variances.cpu().numpy())
    
    # Stack all variances
    all_variances = np.stack(all_variances)  # (n_points, ambient_dim)
    
    # Sort variances in descending order for each point
    sorted_variances = np.sort(all_variances, axis=1)[:, ::-1]  # (n_points, ambient_dim)
    
    # Take only the top max_pcs for plotting (or all if max_pcs is None)
    if max_pcs is not None:
        sorted_variances = sorted_variances[:, :max_pcs]
    else:
        max_pcs = sorted_variances.shape[1]
    
    # Plot variance per direction
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean variance per principal component
    ax = axes[0, 0]
    mean_var = sorted_variances.mean(axis=0)
    std_var = sorted_variances.std(axis=0)
    ax.errorbar(range(len(mean_var)), mean_var, yerr=std_var, fmt='o-', capsize=5)
    ax.set_xlabel('Principal Component (sorted by variance)')
    ax.set_ylabel('Variance')
    ax.set_title(f'Mean Variance per PC (top {max_pcs} of {ambient_dim})')
    ax.grid(True, alpha=0.3)
    
    # 2. Log-scale variance to see the drop-off
    ax = axes[0, 1]
    ax.semilogy(range(len(mean_var)), mean_var, 'o-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance (log scale)')
    ax.set_title('Variance Drop-off (Log Scale)')
    ax.grid(True, alpha=0.3)
    
    # 3. Variance ratio (first PC / sum of all PCs)
    ax = axes[1, 0]
    total_var = sorted_variances.sum(axis=1)
    variance_ratios = sorted_variances[:, 0] / total_var
    ax.hist(variance_ratios, bins=50, alpha=0.7)
    ax.set_xlabel('Variance Ratio (PC1 / Total)')
    ax.set_ylabel('Frequency')
    ax.set_title('Variance Explained by First PC')
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative variance
    ax = axes[1, 1]
    cum_var = np.cumsum(sorted_variances, axis=1) / total_var[:, np.newaxis]
    mean_cum_var = cum_var.mean(axis=0)
    for i in range(0, n_points, max(1, n_points // 20)):
        ax.plot(range(len(mean_var)), cum_var[i], alpha=0.1, color='blue')
    ax.plot(range(len(mean_var)), mean_cum_var, 'r-', linewidth=2, label='Mean')
    
    # Find where cumulative variance hits 99%
    pc_99 = np.argmax(mean_cum_var >= 0.99)
    if mean_cum_var[pc_99] >= 0.99:
        ax.axvline(x=pc_99, color='green', linestyle='--', linewidth=1.5)
        ax.annotate(f'99% at PC{pc_99 + 1}',
                   xy=(pc_99, 0.99),
                   xytext=(pc_99 + 5, 0.95),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   color='green',
                   fontsize=9)
    
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance')
    ax.set_title(f'Cumulative Variance Explained (top {max_pcs})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / "local_pca_variance.png", dpi=150)
    plt.close()
    
    # Save the variance data
    np.savez(out / "local_pca_variance.npz", 
             all_variances=all_variances,
             sorted_variances=sorted_variances,
             mean_variance=mean_var,
             std_variance=std_var,
             variance_ratios=variance_ratios,
             cumulative_variance=cum_var.mean(axis=0),
             ambient_dim=ambient_dim)
    
    # Find where cumulative variance hits 99% for print output
    mean_cum_var_full = cum_var.mean(axis=0)
    pc_99_print = np.argmax(mean_cum_var_full >= 0.99)
    
    print(f"Local PCA variance analysis saved to {out}")
    print(f"Ambient dimension: {ambient_dim}")
    print(f"Mean variance per PC (top 10): {mean_var[:10]}")
    print(f"Mean variance ratio (PC1/total): {variance_ratios.mean():.4f}")
    print(f"Cumulative variance at PC{min(10, len(mean_var))}: {mean_cum_var_full[min(9, len(mean_var)-1)]:.4f}")
    if mean_cum_var_full[pc_99_print] >= 0.99:
        print(f"99% variance reached at PC{pc_99_print + 1}")
    else:
        print(f"99% variance NOT reached (max cumulative: {mean_cum_var_full[-1]:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Local PCA variance analysis on ambient space')
    parser.add_argument('run', type=str, help='Run folder or name')
    parser.add_argument('--n_points', type=int, default=1000, help='Number of points to sample')
    parser.add_argument('--radius', type=float, default=0.1, help='Neighborhood radius in latent space')
    parser.add_argument('--n_neighbors', type=int, default=100, help='Minimum number of neighbors')
    parser.add_argument('--max_pcs', type=int, default=None, help='Maximum number of PCs to plot (None = all components)')
    args = parser.parse_args()
    
    run_path = Path(args.run)
    if not run_path.exists():
        run_path = ROOT / "results" / "autoencoder" / args.run
    
    local_pca_variance(run_path, 
                      n_points=args.n_points,
                      radius=args.radius,
                      n_neighbors=args.n_neighbors,
                      max_pcs=args.max_pcs)
