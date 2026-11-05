import functools
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_leaky_relu(x, a=0.5, eps=1.0, threshold=None):
    """Analytic leaky ReLU: a bijection of R with slope in (a, 1).
    threshold parameter is ignored for compatibility."""
    return 0.5 * ((1 + a) * x + (1 - a) * torch.sqrt(x * x + eps * eps))


def smooth_leaky_relu_inv(y, a=0.5, eps=1.0, threshold=None):
    """Closed-form inverse of smooth_leaky_relu, defined on all of R.
    threshold parameter is ignored for compatibility."""
    return ((1 + a) * y - (1 - a) * torch.sqrt(y * y + a * eps * eps)) / (2 * a)


def leaky_relu(x, a=0.5, threshold=None):
    """Piecewise-linear leaky ReLU: slope 1 for x > 0, a for x < 0.
    threshold parameter is ignored for compatibility."""
    return F.leaky_relu(x, a)


def leaky_relu_inv(y, a=0.5, threshold=None):
    """Exact inverse of leaky_relu, defined on all of R.
    threshold parameter is ignored for compatibility."""
    return F.leaky_relu(y, 1 / a)


def relu(x):
    """Standard ReLU: max(0, x)."""
    return F.relu(x)


def relu_inv(y):
    """Right inverse of ReLU: returns y for y > 0, 0 otherwise."""
    return F.relu(y)


def fat_relu(x, threshold=0.0):
    """FATReLU: Forced-Activation Threshold ReLU. x when x >= threshold, 0 otherwise."""
    return torch.where(x >= threshold, x, torch.zeros_like(x))


def fat_relu_inv(y, threshold=0.0):
    """Right inverse of FATReLU. Since FATReLU is not bijective for x < threshold, 
    we return y + threshold for y > 0 (approximate inverse)."""
    return torch.where(y > 0, y + threshold, torch.zeros_like(y))


def jump(x):
    """x + 1 for x > 0, x - 1 for x < 0 (x + sign(x)): slope 1, with a jump of 2 at 0 that cuts the input space along every hyperplane where a pre-activation is 0."""
    return x + torch.sign(x)


def jump_inv(y):
    """Right inverse of jump on its range |y| > 1 (y - sign(y)); the gap |y| <= 1 is not in the range and is sent to 0, the pre-image of the jump."""
    return torch.where(y.abs() > 1, y - torch.sign(y), torch.zeros_like(y))


ACTIVATIONS = {
    "smooth_leaky_relu": (smooth_leaky_relu, smooth_leaky_relu_inv),
    "leaky_relu": (leaky_relu, leaky_relu_inv),
    "relu": (relu, relu_inv),
    "fat_relu": (fat_relu, fat_relu_inv),
    "jump": (jump, jump_inv),
}


class BottleneckRMSNorm(nn.Module):
    """RMS norm on the latent code: z / sqrt(mean(z^2) + eps), optionally
    rescaled by a learnable per-dimension gain. Without the gain every latent
    vector lies on the sphere of radius sqrt(d), which imposes the point-2-style
    norm constraint on the bottleneck; the decoder is NOT given the norm back,
    so the scale is a deliberate degree of freedom the model must do without."""

    def __init__(self, dim, gain=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if gain else None

    def forward(self, z):
        z = z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return z * self.weight if self.weight is not None else z


def linear_init(fan_in, fan_out):
    """(fan_in, fan_out) weights drawn like nn.Linear's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1 / math.sqrt(fan_in)
    return (torch.rand(fan_in, fan_out) * 2 - 1) * bound


def orthogonal_init(fan_in, fan_out):
    """(fan_in, fan_out) weights with orthonormal columns (tall) or rows (wide): every singular value is 1, so the matrix starts at the minimum of the orthogonality penalty and is far from singular."""
    return torch.nn.init.orthogonal_(torch.empty(fan_in, fan_out))


def normal_init(fan_in, fan_out):
    """(fan_in, fan_out) weights drawn from normal distribution N(0, 1/sqrt(fan_in))."""
    std = 1 / math.sqrt(fan_in)
    return torch.randn(fan_in, fan_out) * std


INITS = {"uniform": linear_init, "orthogonal": orthogonal_init, "normal": normal_init}


def gram_error_euclidian(W):
    """||G - I||_F with G = W^T W for a tall W or W W^T for a wide W."""
    G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    # how far away are we from Identity
    return torch.linalg.matrix_norm(G - torch.eye(len(G), device=W.device)).item()


def gram_error_spectral(W):
    """||G - I||_2 (largest singular value, so never negative) with G = W^T W for a tall W or W W^T for a wide W."""
    G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    return torch.linalg.matrix_norm(
        G - torch.eye(len(G), device=W.device), ord=2
    ).item()


class DeepAutoEncoder(nn.Module):
    def __init__(
        self,
        dims,
        activation="smooth_leaky_relu",
        init="uniform",
        bias=False,
        rms_norm=False,
        rms_gain=True,
        rms_eps=1e-6,
        activation_alpha=None,
        tied_weights=True,
        activation_threshold=None,
    ):
        super().__init__()
        self.n = len(dims) - 1
        self.bias = bias
        self.tied_weights = tied_weights
        act, act_inv = ACTIVATIONS[activation]
        if activation_alpha is not None:
            # bind the slope a into the (forward, inverse) pair; jump ignores it
            act = functools.partial(act, a=activation_alpha)
            act_inv = functools.partial(act_inv, a=activation_alpha)
        if activation_threshold is not None:
            # bind the threshold into FATReLU
            act = functools.partial(act, threshold=activation_threshold)
            act_inv = functools.partial(act_inv, threshold=activation_threshold)
        self.act, self.act_inv = act, act_inv
        self.rms_norm = rms_norm
        self.norm = (
            BottleneckRMSNorm(dims[-1], gain=rms_gain, eps=rms_eps)
            if rms_norm
            else None
        )
        for i in range(self.n):
            setattr(self, f"W{i + 1}", nn.Parameter(INITS[init](dims[i], dims[i + 1])))
            if bias:
                setattr(self, f"b{i + 1}", nn.Parameter(torch.zeros(dims[i + 1])))
        # Decoder weights (untied if tied_weights=False)
        if not tied_weights:
            for i in range(self.n):
                setattr(self, f"V{i + 1}", nn.Parameter(INITS[init](dims[i + 1], dims[i])))
                if bias:
                    setattr(self, f"c{i + 1}", nn.Parameter(torch.zeros(dims[i])))

    def weight_matrices(self):
        return [getattr(self, f"W{i + 1}") for i in range(self.n)]

    def decoder_weight_matrices(self):
        """Decoder weight matrices: either transposed encoder weights (tied) or separate weights (untied)."""
        if self.tied_weights:
            return [w.T for w in self.weight_matrices()]
        else:
            return [getattr(self, f"V{i + 1}") for i in range(self.n)]

    def biases(self):
        """The bias of every weight matrix, zeros when the model has none."""
        return [getattr(self, f"b{i + 1}") if self.bias else 0.0 for i in range(self.n)]

    def decoder_biases(self):
        """The bias of every decoder weight matrix, zeros when the model has none."""
        if self.tied_weights:
            return self.biases()
        else:
            return [getattr(self, f"c{i + 1}") if self.bias else 0.0 for i in range(self.n)]

    def composed_matrices(self):
        """Weight matrices whose product is the linear chain: all of them."""
        return self.weight_matrices()

    def encode(self, x):
        x = self.hidden(x)
        z = x @ self.weight_matrices()[-1] + self.biases()[-1]
        return self.norm(z) if self.norm is not None else z

    def decode(self, z):
        """The decoder; z is the (RMS-normalised) latent code. When rms_norm is
        on, encode normalises z and decode consumes the normalised code: the
        norm is the bottleneck's business, never decoded back."""
        return self._decode(z)

    def _decode(self, z, record=None):
        """The decoder; record, a list, collects (label, tensor) after every step (see decoder_activations)."""
        Ws, bs = self.decoder_weight_matrices(), self.decoder_biases()
        note = (
            (lambda label, h: record.append((label, h)))
            if record is not None
            else (lambda label, h: None)
        )
        dec_act = self.act_inv if self.tied_weights else self.act
        if self.tied_weights:
            h = (z - bs[-1]) @ Ws[-1]
            note(f"decoder: (z - b{self.n}) W{self.n}^T", h)
            for i in range(self.n - 1, 0, -1):
                h = dec_act(h)
                note(f"decoder: layer {i} inverse activation", h)
                h = (h - bs[i - 1]) @ Ws[i - 1]
                note(f"decoder: layer {i} output" + (" (reconstruction)" if i == 1 else ""), h)
        else:
            h = z @ Ws[-1] + bs[-1]
            note(f"decoder: z V{self.n} + c{self.n}", h)
            for i in range(self.n - 1, 0, -1):
                h = dec_act(h)
                note(f"decoder: layer {i} activation", h)
                h = h @ Ws[i - 1] + bs[i - 1]
                note(f"decoder: layer {i} output" + (" (reconstruction)" if i == 1 else ""), h)
        return h

    def encoder_steps(self):
        """[(label, fn)]: the encoder as a chain of functions h_(s-1) -> h_s (a layer with its bias and activation, the last matrix, and the bottleneck RMS norm); applying them in order is encode."""
        Ws, bs, steps = self.weight_matrices(), self.biases(), []
        for i in range(1, self.n):
            steps.append(
                (f"layer {i}", lambda h, W=Ws[i - 1], b=bs[i - 1]: self.act(h @ W + b))
            )
        steps.append(
            (f"layer {self.n} (output z)", lambda h, W=Ws[-1], b=bs[-1]: h @ W + b)
        )
        if self.norm is not None:
            steps.append(("bottleneck RMS norm", self.norm))
        return steps

    def intermediates(self, x):
        """[x, h_1, ..., z]: the input and the output of every step of encoder_steps."""
        out = [x]
        for _, fn in self.encoder_steps():
            out.append(fn(out[-1]))
        return out

    def encoder_layers(self):
        """One function per weight matrix: the map from the input of the layer to its output, composed of the consecutive steps of encoder_steps that belong to it (the layer with its bias and activation, plus the bottleneck RMS norm on the last one)."""
        groups = []
        for label, fn in self.encoder_steps():
            if re.match(r"layer \d", label):
                groups.append([])
            groups[-1].append(fn)

        def compose(fns):
            def f(h):
                for g in fns:
                    h = g(h)
                return h

            return f

        return [compose(g) for g in groups]

    def encoder_activations(self, x):
        """[(label, tensor)]: the input, then every step of encoder_steps; a layer with an activation gives two entries, its pre-activation x W + b (what the nonlinearity receives) and its activation (what the next layer receives). The last tensor is encode(x)."""
        Ws, bs = self.weight_matrices(), self.biases()
        out = [("input", x)]
        h = x
        for label, fn in self.encoder_steps():
            new = fn(h)
            if re.match(r"layer \d", label) and "output" not in label:
                i = int(label.split()[1])
                out += [
                    (f"layer {i} pre-activation", h @ Ws[i - 1] + bs[i - 1]),
                    (f"layer {i} activation", new),
                ]
            else:
                out.append((label, new))
            h = new
        return out

    def decoder_activations(self, z):
        """[(label, tensor)]: the decoder from z, step by step as in decode: (z - b_n) W_n^T, then for every layer the inverse activation and the layer output. The last tensor is decode(z)."""
        record = []
        self._decode(z, record)
        return record

    def hidden(self, x):
        """Activations feeding the last weight matrix."""
        for W, c in zip(self.weight_matrices()[:-1], self.biases()[:-1]):
            x = self.act(x @ W + c)
        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def orth_penalty(self):
        """Differentiable sum over the weight matrices of the Euclidian norm ||G - I||_F (the square root of the sum of the squared entries of G - I, not squared), G = W^T W for a tall W or W W^T for a wide W."""
        total = 0.0
        for W in self.weight_matrices():
            G = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
            total = total + torch.linalg.matrix_norm(
                G - torch.eye(len(G), device=W.device)
            )
        return total

    def orth_error_euclidian(self):
        return [gram_error_euclidian(W) for W in self.weight_matrices()]

    def orth_error_composed_euclidian(self):
        """||P^T P - I||_F for P = W1 W2 ... Wn, the product of all weight matrices (activations ignored)."""
        P = self.composed_matrices()[0].detach()
        for W in self.composed_matrices()[1:]:
            P = P @ W.detach()
        return gram_error_euclidian(P)

    def orth_error_spectral(self):
        return [gram_error_spectral(W) for W in self.weight_matrices()]

    def orth_error_composed_spectral(self):
        """||P^T P - I||_2 for P = W1 W2 ... Wn, the product of all weight matrices (activations ignored)."""
        P = self.composed_matrices()[0].detach()
        for W in self.composed_matrices()[1:]:
            P = P @ W.detach()
        return gram_error_spectral(P)

    def orth_errors(self):
        """Orthonormality errors of every weight matrix W1..Wn and of their product (see composed_matrices), by Euclidian (Frobenius) and spectral norm."""
        errors = {}
        for kind in ("euclidian", "spectral"):
            for i, e in enumerate(getattr(self, f"orth_error_{kind}")()):
                errors[f"orth_{kind}_W{i + 1}"] = e
            errors[f"orth_{kind}_composed"] = getattr(
                self, f"orth_error_composed_{kind}"
            )()
        return errors


def make_model(cfg):
    """DeepAutoEncoder with dims [ambient_dim, *hidden_dims, embedded_dim] from a run config."""
    return DeepAutoEncoder(
        [cfg["ambient_dim"], *cfg["hidden_dims"], cfg["embedded_dim"]],
        cfg["activation"],
        cfg.get("init", "uniform"),
        cfg.get("bias", False),
        cfg.get("rms_norm", False),
        cfg.get("rms_gain", True),
        cfg.get("rms_eps", 1e-6),
        cfg.get("activation_alpha"),
        cfg.get("tied_weights", True),
        cfg.get("activation_threshold"),
    )
