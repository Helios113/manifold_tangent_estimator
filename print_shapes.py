#!/usr/bin/env python
"""Print weight and activation shapes for a given run.

Usage:
    python print_shapes.py <run_name> [--decoder]

Example:
    python print_shapes.py a_run_24f
    python print_shapes.py a_run_24f --decoder
"""

import sys
import torch
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from auto_encoder.model import DeepAutoEncoder as AutoEncoder
from diagnostics.common import open_run


def print_encoder_shapes(model, x_shape=(1, 12)):
    """Print shapes for encoder forward pass."""
    print("\n" + "="*60)
    print("ENCODER")
    print("="*60)
    
    # Get weight matrices and biases
    Ws = model.weight_matrices()
    bs = model.biases()
    
    print(f"\nInput shape: {x_shape}")
    
    h = torch.zeros(x_shape)
    for i, (W, b) in enumerate(zip(Ws, bs), 1):
        print(f"\nLayer {i}:")
        print(f"  W{i}: {tuple(W.shape)}")
        print(f"  b{i}: {tuple(b.shape)}")
        
        h = h @ W + b
        print(f"  h @ W{i} + b{i}: {tuple(h.shape)}")
        
        if i < len(Ws):  # Not the last layer
            h = model.act(h)
            print(f"  after activation: {tuple(h.shape)}")
    
    if model.norm:
        h = model.norm(h)
        print(f"  after RMS norm: {tuple(h.shape)}")
    
    print(f"\nLatent code z: {tuple(h.shape)}")
    return h


def print_decoder_shapes(model, z_shape=(1, 2)):
    """Print shapes for decoder forward pass."""
    print("\n" + "="*60)
    print("DECODER")
    print("="*60)
    
    # Get decoder weight matrices and biases
    Ws = model.decoder_weight_matrices()
    bs = model.decoder_biases()
    
    print(f"\nLatent input shape: {z_shape}")
    
    h = torch.zeros(z_shape)
    
    if model.tied_weights:
        # For tied weights, decoder starts with (z - b_n) @ W_n^T
        print(f"\nTied weights: True")
        print(f"  Using transposed encoder weights")
        
        # First layer is special
        print(f"\nLayer (bottleneck):")
        print(f"  W{len(Ws)}^T: {tuple(Ws[-1].shape)}")
        print(f"  b{len(Ws)}: {tuple(bs[-1].shape)}")
        h = (h - bs[-1]) @ Ws[-1]
        print(f"  (z - b{len(Ws)}) @ W{len(Ws)}^T: {tuple(h.shape)}")
        
        # Remaining layers
        for i in range(len(Ws) - 1, 0, -1):
            print(f"\nLayer {i}:")
            print(f"  W{i}^T: {tuple(Ws[i-1].shape)}")
            print(f"  b{i}: {tuple(bs[i-1].shape)}")
            
            h = model.act_inv(h)
            print(f"  after inverse activation: {tuple(h.shape)}")
            
            h = (h - bs[i-1]) @ Ws[i-1]
            print(f"  (h - b{i}) @ W{i}^T: {tuple(h.shape)}")
    else:
        print(f"\nTied weights: False")
        print(f"  Using separate decoder weights")
        
        for i, (W, b) in enumerate(zip(Ws, bs), 1):
            print(f"\nLayer {i}:")
            print(f"  V{i}: {tuple(W.shape)}")
            print(f"  c{i}: {tuple(b.shape)}")
            
            h = h @ W + b
            print(f"  h @ V{i} + c{i}: {tuple(h.shape)}")
            
            if i < len(Ws):
                h = model.act(h)
                print(f"  after activation: {tuple(h.shape)}")
    
    print(f"\nReconstruction x': {tuple(h.shape)}")
    return h


def main():
    if len(sys.argv) < 2:
        print("Usage: python print_shapes.py <run_name> [--decoder]")
        print("Example: python print_shapes.py a_run_24f")
        sys.exit(1)
    
    run_name = sys.argv[1]
    print_decoder = '--decoder' in sys.argv
    
    # Load run
    run_path, cfg = open_run(run_name)
    
    print(f"\nConfiguration for {run_name}:")
    print(f"  intrinsic_dim: {cfg.get('intrinsic_dim')}")
    print(f"  ambient_dim: {cfg.get('ambient_dim')}")
    print(f"  embedded_dim: {cfg.get('embedded_dim')}")
    print(f"  hidden_dims: {cfg.get('hidden_dims')}")
    print(f"  activation: {cfg.get('activation')}")
    print(f"  tied_weights: {cfg.get('tied_weights')}")
    print(f"  bias: {cfg.get('bias')}")
    print(f"  rms_norm: {cfg.get('rms_norm')}")
    
    # Build model
    dims = [cfg['ambient_dim']] + cfg['hidden_dims'] + [cfg['embedded_dim']]
    model = AutoEncoder(
        dims=dims,
        activation=cfg['activation'],
        init=cfg.get('init', 'uniform'),
        bias=cfg.get('bias', True),
        rms_norm=cfg.get('rms_norm', False),
        rms_gain=cfg.get('rms_gain', True),
        rms_eps=cfg.get('rms_eps', 1e-6),
        tied_weights=cfg.get('tied_weights', True),
        activation_alpha=cfg.get('activation_alpha'),
        activation_threshold=cfg.get('activation_threshold'),
    )
    
    # Load weights from checkpoint
    ckpt_path = Path(f"results/autoencoder/{run_name}/best.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # Check if checkpoint has 'model' key or direct weights
        if 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        print(f"\nLoaded weights from {ckpt_path}")
    else:
        print(f"\nUsing randomly initialized weights (no checkpoint at {ckpt_path})")
    
    # Print shapes
    x_shape = (1, cfg['ambient_dim'])
    z = print_encoder_shapes(model, x_shape)
    
    if print_decoder:
        print_decoder_shapes(model, tuple(z.shape))


if __name__ == "__main__":
    main()
