#!/usr/bin/env python
"""Run UMAP on latent vectors from runs and save plots in the same directory.

Usage:
    python umap_latent.py [run1 run2 ...]

If umap-learn is not installed, falls back to t-SNE from sklearn.
Saves umap.png in each run's directory.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import UMAP, fall back to t-SNE
try:
    import umap.umap_ as umap
    USE_UMAP = True
except ImportError:
    from sklearn.manifold import TSNE
    USE_UMAP = False
    print("umap-learn not found, using t-SNE instead")


def load_latent(run_path):
    """Load latent vectors from a run's latent.npz file."""
    latent_path = Path(run_path) / "latent.npz"
    if not latent_path.exists():
        raise FileNotFoundError(f"No latent.npz found in {run_path}")
    data = np.load(latent_path)
    return data["latent"], data.get("sphere_points")


def run_umap(latent_vectors, random_state=42):
    """Run UMAP on latent vectors."""
    if USE_UMAP:
        reducer = umap.UMAP(random_state=random_state, n_components=2)
        return reducer.fit_transform(latent_vectors)
    else:
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        return reducer.fit_transform(latent_vectors)


def plot_embedding(embedding, run_name, out_path):
    """Plot 2D embedding for a single run."""
    plt.figure(figsize=(10, 8))
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        s=5,
        alpha=0.6,
        color='blue'
    )
    plt.title(f"{'UMAP' if USE_UMAP else 't-SNE'} of Latent Vectors - {run_name}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.savefig(out_path.with_suffix('.pdf'))
    print(f"Saved plot to {out_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python umap_latent.py run1 [run2 ...]")
        print("Example: python umap_latent.py a_run_22 a_run_23")
        sys.exit(1)
    
    runs = sys.argv[1:]
    base_path = Path("results/autoencoder")
    
    for run_name in runs:
        run_path = base_path / run_name
        try:
            latent, _ = load_latent(run_path)
            print(f"Loaded {len(latent)} latent vectors from {run_name}")
            print(f"Latent dimension: {latent.shape[1]}")
            
            # Run UMAP/t-SNE
            print(f"Running {'UMAP' if USE_UMAP else 't-SNE'}...")
            embedding = run_umap(latent)
            print(f"Embedding shape: {embedding.shape}")
            
            # Save plot in the run directory
            out_path = run_path / "umap.png"
            plot_embedding(embedding, run_name, out_path)
            
        except Exception as e:
            print(f"Error processing {run_name}: {e}")


if __name__ == "__main__":
    main()
