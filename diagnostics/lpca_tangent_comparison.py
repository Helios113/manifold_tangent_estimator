"""Diagnostic: Compare LPCA on latent space vs true tangent space.

For a trained autoencoder where embedded_dim > intrinsic_dim, this script:
1. Samples points on the manifold
2. Computes the true tangent space at each point (from the analytic manifold)
3. Performs LPCA on the latent space neighborhood
4. Compares the LPCA tangent space with the true tangent space
5. Plots principal angles between the two spaces

This helps evaluate how well the autoencoder's latent space preserves the
true manifold geometry.

Usage: python diagnostics/lpca_tangent_comparison.py <run folder> [--n_points 1000] [--radius 0.1]
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


def principal_angles(A, B):
    """Compute principal angles (in radians) between two subspaces.
    
    Args:
        A: (d, k) matrix with orthonormal columns (basis for subspace A)
        B: (d, k) matrix with orthonormal columns (basis for subspace B)
    
    Returns:
        angles: (k,) array of principal angles in [0, pi/2]
    """
    # Detach to avoid grad issues
    A = A.detach()
    B = B.detach()
    
    # QR decomposition to ensure orthonormality
    Qa, _ = torch.linalg.qr(A)
    Qb, _ = torch.linalg.qr(B)
    
    # Compute SVD of Qa^T Qb
    M = Qa.T @ Qb
    _, S, _ = torch.linalg.svd(M)
    
    # Principal angles are arccos of singular values
    angles = torch.arccos(torch.clamp(S, -1.0, 1.0))
    return angles.cpu().numpy()


def lpca_tangent_comparison(run, n_points=1000, radius=0.1, n_neighbors=100):
    """Compare LPCA on latent space vs true tangent space.
    
    Args:
        run: Path to the run folder
        n_points: Number of points to sample on the manifold
        radius: Radius for local neighborhood (in latent space)
        n_neighbors: Minimum number of neighbors for LPCA
    """
    run, cfg = open_run(run)
    out = output_dir("lpca_tangent_comparison", run)
    
    # Check that embedded_dim > intrinsic_dim
    intrinsic_dim = cfg["intrinsic_dim"]
    embedded_dim = cfg["embedded_dim"]
    ambient_dim = cfg["ambient_dim"]
    
    if embedded_dim <= intrinsic_dim:
        print(f"Warning: embedded_dim ({embedded_dim}) <= intrinsic_dim ({intrinsic_dim}). "
              f"Comparison may not be meaningful.")
    
    # Get manifold and embedding from config
    S = MANIFOLDS[cfg["manifold"]](intrinsic_dim)
    emb = EMBEDDINGS[cfg["embedding"]](
        ambient_dim, 
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
    
    # Get true tangent basis at each point
    true_tangent_basis = S.tangent_basis(p)  # (n_points, S.D, intrinsic_dim)
    
    # Embed the true tangent basis to get it in the ambient space
    # We need to push it through the embedding Jacobian
    true_tangent_ambient = emb.jacobian(p) @ true_tangent_basis  # (n_points, ambient_dim, intrinsic_dim)
    
    # Now we need to compare LPCA in latent space with true tangent in ambient space
    # But they're in different spaces. We have two options:
    # 1. Compare LPCA in latent space with the true tangent mapped to latent space
    # 2. Compare LPCA in ambient space (from reconstructions) with true tangent in ambient space
    
    # Let's do option 1: map true tangent to latent space using the decoder Jacobian
    # The true tangent in ambient space should correspond to some direction in latent space
    
    # Actually, a better approach: perform LPCA in latent space and compare with
    # the true tangent space structure (which we know analytically)
    
    # For LPCA in latent space:
    # 1. For each point, find neighbors in latent space
    # 2. Perform PCA on the latent codes of neighbors
    # 3. Get the top k (intrinsic_dim) principal components
    
    # For true tangent space:
    # We have the analytic tangent basis in the manifold coordinates
    # We can map it to latent space using the autoencoder's encoder Jacobian
    
    # Get the encoder Jacobian (derivative of encoder w.r.t. input)
    # This is tricky because we need the Jacobian of the encoder
    # Let's use torch.func.jacrev
    from torch.func import jacrev, vmap
    
    def encode_fn(x_flat):
        return net.encode(x_flat.unsqueeze(0)).squeeze(0)
    
    # Compute Jacobian of encoder at each point
    # This gives us how small changes in input (ambient space) affect latent codes
    encoder_jacobian = vmap(jacrev(encode_fn))(x)  # (n_points, embedded_dim, ambient_dim)
    
    # Map true tangent basis from ambient space to latent space
    # true_tangent_ambient: (n_points, ambient_dim, intrinsic_dim)
    # encoder_jacobian: (n_points, embedded_dim, ambient_dim)
    # We need: (n_points, embedded_dim, intrinsic_dim)
    # einsum: batch matrix multiplication: (n, D, a) @ (n, a, k) -> (n, D, k)
    true_tangent_latent = torch.einsum('nDa,nak->nDk', encoder_jacobian, true_tangent_ambient)
    
    # Orthonormalize the true tangent in latent space
    true_tangent_latent_ortho = torch.zeros_like(true_tangent_latent)
    for i in range(n_points):
        Q, R = torch.linalg.qr(true_tangent_latent[i])
        true_tangent_latent_ortho[i] = Q
    
    # Perform LPCA in latent space
    lpca_basis = torch.zeros((n_points, embedded_dim, intrinsic_dim))
    lpca_variances = torch.zeros((n_points, intrinsic_dim))
    
    for i in range(n_points):
        # Find neighbors in latent space
        dists = torch.norm(z - z[i], dim=1)
        neighbors = dists < radius
        
        if neighbors.sum() < n_neighbors:
            _, indices = torch.topk(dists, k=n_neighbors, largest=False)
            neighbor_indices = indices
        else:
            neighbor_indices = torch.where(neighbors)[0]
        
        # Get latent codes of neighbors
        neighbor_z = z[neighbor_indices]  # (n_neighbors, embedded_dim)
        
        # Center
        neighbor_z_centered = neighbor_z - neighbor_z.mean(dim=0)
        
        # Perform PCA
        _, S, Vt = torch.linalg.svd(neighbor_z_centered, full_matrices=False)
        
        # Get top intrinsic_dim components
        lpca_basis[i] = Vt[:intrinsic_dim].T  # (embedded_dim, intrinsic_dim)
        lpca_variances[i] = S[:intrinsic_dim] ** 2 / (len(neighbor_z) - 1)
    
    # Transpose lpca_basis to (n_points, embedded_dim, intrinsic_dim)
    lpca_basis = lpca_basis.transpose(1, 2)
    
    # Compute principal angles between LPCA basis and true tangent in latent space
    principal_angles_list = []
    for i in range(n_points):
        # Both are (embedded_dim, intrinsic_dim)
        angles = principal_angles(lpca_basis[i].T, true_tangent_latent_ortho[i].T)
        principal_angles_list.append(angles)
    
    principal_angles_arr = np.stack(principal_angles_list)  # (n_points, intrinsic_dim)
    
    # Convert to degrees
    principal_angles_deg = np.degrees(principal_angles_arr)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean principal angles
    ax = axes[0, 0]
    mean_angles = principal_angles_deg.mean(axis=0)
    std_angles = principal_angles_deg.std(axis=0)
    ax.errorbar(range(intrinsic_dim), mean_angles, yerr=std_angles, fmt='o-', capsize=5)
    ax.set_xlabel('Principal Angle Index')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Mean Principal Angles between LPCA and True Tangent')
    ax.grid(True, alpha=0.3)
    
    # 2. Histogram of maximum principal angle (worst alignment)
    ax = axes[0, 1]
    max_angles = principal_angles_deg.max(axis=1)
    ax.hist(max_angles, bins=50, alpha=0.7)
    ax.set_xlabel('Maximum Principal Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Maximum Principal Angle')
    ax.grid(True, alpha=0.3)
    
    # 3. Mean angle per component
    ax = axes[1, 0]
    for k in range(intrinsic_dim):
        ax.hist(principal_angles_deg[:, k], bins=50, alpha=0.5, label=f'PC{k+1}')
    ax.set_xlabel('Principal Angle (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Principal Angle Distribution per Component')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. LPCA variances
    ax = axes[1, 1]
    lpca_variances_np = lpca_variances.cpu().numpy()
    mean_lpca_var = lpca_variances_np.mean(axis=0)
    std_lpca_var = lpca_variances_np.std(axis=0)
    ax.errorbar(range(intrinsic_dim), mean_lpca_var, yerr=std_lpca_var, fmt='o-', capsize=5)
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance')
    ax.set_title('LPCA Variances in Latent Space')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out / "lpca_tangent_comparison.png", dpi=150)
    plt.close()
    
    # Save data
    np.savez(out / "lpca_tangent_comparison.npz",
             principal_angles_deg=principal_angles_deg,
             principal_angles_rad=principal_angles_arr,
             lpca_variances=lpca_variances_np,
             mean_principal_angles=mean_angles,
             max_principal_angles=max_angles,
             intrinsic_dim=intrinsic_dim,
             embedded_dim=embedded_dim)
    
    print(f"LPCA tangent comparison saved to {out}")
    print(f"Intrinsic dimension: {intrinsic_dim}, Embedded dimension: {embedded_dim}")
    print(f"Mean principal angles (deg): {mean_angles}")
    print(f"Mean max principal angle (deg): {max_angles.mean():.2f}")
    print(f"95th percentile max angle (deg): {np.percentile(max_angles, 95):.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compare LPCA on latent space vs true tangent space')
    parser.add_argument('run', type=str, help='Run folder or name')
    parser.add_argument('--n_points', type=int, default=1000, help='Number of points to sample')
    parser.add_argument('--radius', type=float, default=0.1, help='Neighborhood radius in latent space')
    parser.add_argument('--n_neighbors', type=int, default=100, help='Minimum number of neighbors')
    args = parser.parse_args()
    
    run_path = Path(args.run)
    if not run_path.exists():
        run_path = ROOT / "results" / "autoencoder" / args.run
    
    lpca_tangent_comparison(run_path,
                           n_points=args.n_points,
                           radius=args.radius,
                           n_neighbors=args.n_neighbors)
