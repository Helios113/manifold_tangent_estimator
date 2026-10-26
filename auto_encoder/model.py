import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_leaky_relu(x, a=0.5, eps=1.0):
    """Analytic leaky ReLU: a bijection of R with slope in (a, 1)."""
    return 0.5 * ((1 + a) * x + (1 - a) * torch.sqrt(x * x + eps * eps))


def smooth_leaky_relu_inv(y, a=0.5, eps=1.0):
    """Closed-form inverse of smooth_leaky_relu, defined on all of R."""
    return ((1 + a) * y - (1 - a) * torch.sqrt(y * y + a * eps * eps)) / (2 * a)


def linear_init(fan_in, fan_out):
    """(fan_in, fan_out) weights drawn like nn.Linear's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1 / math.sqrt(fan_in)
    return (torch.rand(fan_in, fan_out) * 2 - 1) * bound


def gram_error_euclidian(W):
    """||G - I||_F with G = W^T W for a tall W or W W^T for a wide W."""
    G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    # how far away are we from Identity
    return torch.linalg.matrix_norm(G - torch.eye(len(G), device=W.device)).item()


def gram_error_spectral(W):
    """||G - I||_2 (largest singular value, so never negative) with G = W^T W for a tall W or W W^T for a wide W."""
    G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    return torch.linalg.matrix_norm(G - torch.eye(len(G), device=W.device), ord=2).item()




class DeepAutoEncoder(nn.Module):
    """Same tied-weight scheme with any number of weight matrices W1..Wn (no biases), dims = [ambient, hidden..., embedded].

    encode: x -> act(x W1) -> act(. W2) -> ... -> . Wn;   decode: the exact reverse with the transposes and act_inv."""

    def __init__(self, dims):
        super().__init__()
        self.n = len(dims) - 1
        for i in range(self.n): 
            setattr(self, f"W{i + 1}",
                    nn.Parameter(
                        linear_init(dims[i],
                                    dims[i + 1])
                        )
                    )
        self.act = smooth_leaky_relu
        self.act_inv = smooth_leaky_relu_inv

    def weight_matrices(self):
        return [getattr(self, f"W{i + 1}") for i in range(self.n)]

    def encode(self, x):
        Ws = self.weight_matrices()
        for W in Ws[:-1]: x = self.act(x @ W)
        return x @ Ws[-1]

    def decode(self, z):
        Ws = self.weight_matrices()
        h = z @ Ws[-1].T
        for W in reversed(Ws[:-1]): h = self.act_inv(h) @ W.T
        return h

    def hidden(self, x):
        """Activations feeding the last weight matrix."""
        for W in self.weight_matrices()[:-1]: x = self.act(x @ W)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def orth_penalty(self):
        """Differentiable sum over the weight matrices of ||G - I||_F^2, G = W^T W for a tall W or W W^T for a wide W."""
        total = 0.
        for W in self.weight_matrices():
            G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
            total = total + ((G - torch.eye(len(G), device=W.device)) ** 2).sum()
        return total

    def orth_error_euclidian(self):
        return [gram_error_euclidian(W) for W in self.weight_matrices()]

    def orth_error_composed_euclidian(self):
        """||P^T P - I||_F for P = W1 W2 ... Wn, the product of all weight matrices (activations ignored)."""
        P = self.weight_matrices()[0].detach()
        for W in self.weight_matrices()[1:]: P = P @ W.detach()
        return gram_error_euclidian(P)
    
    def orth_error_spectral(self):
        return [gram_error_spectral(W) for W in self.weight_matrices()]

    def orth_error_composed_spectral(self):
        """||P^T P - I||_2 for P = W1 W2 ... Wn, the product of all weight matrices (activations ignored)."""
        P = self.weight_matrices()[0].detach()
        for W in self.weight_matrices()[1:]: P = P @ W.detach()
        return gram_error_spectral(P)


    def orth_errors(self):
        """Orthonormality errors of every weight matrix W1..Wn and of their product, by Euclidian (Frobenius) and spectral norm."""
        errors = {}
        for kind in ("euclidian", "spectral"):
            for i, e in enumerate(getattr(self, f"orth_error_{kind}")()): errors[f"orth_{kind}_W{i + 1}"] = e
            errors[f"orth_{kind}_composed"] = getattr(self, f"orth_error_composed_{kind}")()
        return errors


def make_model(cfg):
    """DeepAutoEncoder with dims [ambient_dim, *hidden_dims, embedded_dim] from a run config."""
    return DeepAutoEncoder([cfg["ambient_dim"], *cfg["hidden_dims"], cfg["embedded_dim"]])
