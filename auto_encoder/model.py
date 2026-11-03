import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_leaky_relu(x, a=0.5, eps=1.0):
    """Analytic leaky ReLU: a bijection of R with slope in (a, 1)."""
    return 0.5 * ((1 + a) * x + (1 - a) * torch.sqrt(x * x + eps * eps))


def smooth_leaky_relu_inv(y, a=0.5, eps=1.0):
    """Closed-form inverse of smooth_leaky_relu, defined on all of R."""
    return ((1 + a) * y - (1 - a) * torch.sqrt(y * y + a * eps * eps)) / (2 * a)


def leaky_relu(x, a=0.5):
    """Piecewise-linear leaky ReLU: slope 1 for x > 0, a for x < 0."""
    return F.leaky_relu(x, a)


def leaky_relu_inv(y, a=0.5):
    """Exact inverse of leaky_relu, defined on all of R."""
    return F.leaky_relu(y, 1 / a)


def jump(x):
    """x + 1 for x > 0, x - 1 for x < 0 (x + sign(x)): slope 1, with a jump of 2 at 0 that cuts the input space along every hyperplane where a pre-activation is 0."""
    return x + torch.sign(x)


def jump_inv(y):
    """Right inverse of jump on its range |y| > 1 (y - sign(y)); the gap |y| <= 1 is not in the range and is sent to 0, the pre-image of the jump."""
    return torch.where(y.abs() > 1, y - torch.sign(y), torch.zeros_like(y))


ACTIVATIONS = {
    "smooth_leaky_relu": (smooth_leaky_relu, smooth_leaky_relu_inv),
    "leaky_relu": (leaky_relu, leaky_relu_inv),
    "jump": (jump, jump_inv),
}


class Arctan2(nn.Module):
    """x (..., 2m) -> angles (..., m): arctan2(first half, second half) of the last dim. The radius of each pair is dropped, so inverse returns unit pairs."""

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return torch.atan2(a, b)

    def inverse(self, theta):
        """angles (..., m) -> (sin theta, cos theta) (..., 2m), the unit-radius preimage under forward."""
        return torch.cat([torch.sin(theta), torch.cos(theta)], -1)


def linear_init(fan_in, fan_out):
    """(fan_in, fan_out) weights drawn like nn.Linear's default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    bound = 1 / math.sqrt(fan_in)
    return (torch.rand(fan_in, fan_out) * 2 - 1) * bound


def orthogonal_init(fan_in, fan_out):
    """(fan_in, fan_out) weights with orthonormal columns (tall) or rows (wide): every singular value is 1, so the matrix starts at the minimum of the orthogonality penalty and is far from singular."""
    return torch.nn.init.orthogonal_(torch.empty(fan_in, fan_out))


INITS = {"uniform": linear_init, "orthogonal": orthogonal_init}


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
        arctan2_after=0,
        radius_head=0,
        init="uniform",
        arctan2_layernorm=False,
        bias=False,
        arctan2_project=False,
        arctan2_x_hidden=(),
        arctan2_x_activation="",
    ):
        super().__init__()
        self.n = len(dims) - 1
        self.arctan2_after = arctan2_after
        self.arctan2 = Arctan2()
        if arctan2_project and not arctan2_after:
            raise ValueError("arctan2_project needs arctan2_after >= 1")
        self.project = arctan2_project
        n_angles = (
            dims[arctan2_after]
            if arctan2_project and arctan2_after
            else dims[arctan2_after] // 2
        )
        if arctan2_x_hidden and not arctan2_project:
            raise ValueError("arctan2_x_hidden needs arctan2_project")
        self.x_act = ACTIVATIONS[arctan2_x_activation or activation][0]
        if (
            arctan2_project and arctan2_x_hidden
        ):  # the x entry of every pair is an MLP of h instead of the single matrix P
            xd = [n_angles, *arctan2_x_hidden, n_angles]
            self.Px = nn.ParameterList(
                [
                    nn.Parameter(INITS[init](xd[j], xd[j + 1]))
                    for j in range(len(xd) - 1)
                ]
            )
            if bias:
                self.bx = nn.ParameterList(
                    [nn.Parameter(torch.zeros(xd[j + 1])) for j in range(len(xd) - 1)]
                )
        elif arctan2_project:
            self.P = nn.Parameter(INITS[init](n_angles, n_angles))
        for i in range(self.n):
            fan_in = n_angles if i == arctan2_after and i > 0 else dims[i]
            setattr(self, f"W{i + 1}", nn.Parameter(INITS[init](fan_in, dims[i + 1])))
        for i in range(self.n):
            if bias:
                setattr(self, f"b{i + 1}", nn.Parameter(torch.zeros(dims[i + 1])))
        self.bias = bias
        self.act, self.act_inv = ACTIVATIONS[activation]
        if arctan2_layernorm and not arctan2_after:
            raise ValueError("arctan2_layernorm needs arctan2_after >= 1")
        self.norm = (
            nn.LayerNorm(2 * n_angles, elementwise_affine=False)
            if arctan2_layernorm
            else None
        )  # no parameters: it only centres and scales each sample's features
        self.radius_head = None
        if arctan2_after and radius_head:
            m = n_angles  # number of pairs = number of angles
            self.radius_head = nn.Sequential(
                nn.Linear(2 * m, radius_head),
                nn.Tanh(),
                nn.Linear(radius_head, m),
                nn.Softplus(),
            )  # (sin, cos) of the angles -> m positive radii

    def weight_matrices(self):
        return [getattr(self, f"W{i + 1}") for i in range(self.n)]

    def biases(self):
        """The bias of every weight matrix, zeros when the model has none."""
        return [getattr(self, f"b{i + 1}") if self.bias else 0.0 for i in range(self.n)]

    def composed_matrices(self):
        """Weight matrices whose product is the linear chain: all of them, or only those after the Arctan2 (it cuts the chain)."""
        return self.weight_matrices()[self.arctan2_after :]

    def encode(self, x):
        x = self.hidden(x)
        return x @ self.weight_matrices()[-1] + self.biases()[-1]

    def decode(self, z, radii=None):
        """radii: (n, m) radii that replace the radius head's (or the unit radii without a head) at the Arctan2 layer, None = the model's own."""
        return self._decode(z, radii)

    def _decode(self, z, radii=None, record=None):
        """The decoder; record, a list, collects (label, tensor) after every step (see decoder_activations)."""
        Ws, bs = self.weight_matrices(), self.biases()
        note = (
            (lambda label, h: record.append((label, h)))
            if record is not None
            else (lambda label, h: None)
        )
        h = (z - bs[-1]) @ Ws[-1].T
        note(f"decoder: (z - b{self.n}) W{self.n}^T", h)
        for i in range(self.n - 1, 0, -1):
            if i == self.arctan2_after:
                h = unit = self.arctan2.inverse(h)
                note("decoder: Arctan2 inverse (unit pairs)", h)
                if radii is not None or self.radius_head is not None:
                    h = unit * (
                        self.radius_head(unit) if radii is None else radii
                    ).tile(2)
                    note("decoder: radius-scaled pairs", h)
                if self.project:
                    h = h.chunk(2, dim=-1)[0]
                    note(
                        "decoder: first half of the pairs", h
                    )  # the y half is h itself; the x half would be h P
            h = self.act_inv(h)
            note(f"decoder: layer {i} inverse activation", h)
            h = (h - bs[i - 1]) @ Ws[i - 1].T
            note(
                f"decoder: layer {i} output" + (" (reconstruction)" if i == 1 else ""),
                h,
            )
        return h

    def pairs(self, x):
        """(..., 2m) the pair entries fed to Arctan2: x itself, or with arctan2_project (y, x) = (h, h P), or (h, MLP(h)) with arctan2_x_hidden."""
        if not self.project:
            return x
        if not hasattr(self, "Px"):
            return torch.cat([x, x @ self.P], -1)
        h = x
        for j, M in enumerate(self.Px):
            h = h @ M + (self.bx[j] if self.bias else 0.0)
            if j < len(self.Px) - 1:
                h = self.x_act(h)  # no activation after the last matrix
        return torch.cat([x, h], -1)

    def true_radii(self, x):
        """(n, m) radii sqrt(a^2 + b^2) of the m pairs (a, b) that Arctan2 drops: the two halves of the activated output of W_arctan2_after, before the layer norm (the decoder inverts the activation, not the layer norm)."""
        for W, c in zip(self.weight_matrices()[: self.arctan2_after], self.biases()):
            x = self.act(x @ W + c)
        a, b = self.pairs(x).chunk(2, dim=-1)
        return torch.sqrt(a**2 + b**2)

    def predicted_radii(self, z):
        """(n, m) radii the radius head predicts from the latent code alone: the decoder up to the Arctan2, then the head on the unit pairs of the decoded angles."""
        Ws, bs = self.weight_matrices(), self.biases()
        h = (z - bs[-1]) @ Ws[-1].T
        for i in range(self.n - 1, self.arctan2_after, -1):
            h = (self.act_inv(h) - bs[i - 1]) @ Ws[i - 1].T
        return self.radius_head(self.arctan2.inverse(h))

    def radius_loss(self, x, z):
        """Mean squared relative error of the predicted radii against the true ones (the true radii are a constant target: no gradient flows into them)."""
        with torch.no_grad():
            r = self.true_radii(x)
        return ((self.predicted_radii(z) / r - 1) ** 2).mean()

    def encoder_steps(self):
        """[(label, fn)]: the encoder as a chain of functions h_(s-1) -> h_s (a layer with its bias and activation, the pair builder of arctan2_project, the layer norm, the Arctan2, the last matrix); applying them in order is encode."""
        Ws, bs, steps = self.weight_matrices(), self.biases(), []
        for i in range(1, self.n):
            steps.append(
                (f"layer {i}", lambda h, W=Ws[i - 1], b=bs[i - 1]: self.act(h @ W + b))
            )
            if i == self.arctan2_after:
                if self.project:
                    steps.append(("pairs", self.pairs))
                if self.norm is not None:
                    steps.append(("layer norm", self.norm))
                steps.append(("Arctan2", self.arctan2))
        steps.append(
            (f"layer {self.n} (output z)", lambda h, W=Ws[-1], b=bs[-1]: h @ W + b)
        )
        return steps

    def intermediates(self, x):
        """[x, h_1, ..., z]: the input and the output of every step of encoder_steps."""
        out = [x]
        for _, fn in self.encoder_steps():
            out.append(fn(out[-1]))
        return out

    def encoder_layers(self):
        """One function per weight matrix: the map from the input of the layer to its output, composed of the consecutive steps of encoder_steps that belong to it (the layer with its bias and activation, and after the Arctan2 layer the pair builder, layer norm and Arctan2)."""
        groups = []
        for label, fn in self.encoder_steps():
            if re.match(r"layer \d", label):
                groups.append([])  # not "layer norm"
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

    def decoder_activations(self, z, radii=None):
        """[(label, tensor)]: the decoder from z, step by step as in decode: (z - b_n) W_n^T, then at the Arctan2 layer the unit pairs (sin, cos), the radius-scaled pairs and the kept first half, and for every layer the inverse activation and the layer output. The last tensor is decode(z)."""
        record = []
        self._decode(z, radii, record)
        return record

    def hidden(self, x):
        """Activations feeding the last weight matrix."""
        for i, (W, c) in enumerate(zip(self.weight_matrices()[:-1], self.biases()), 1):
            x = self.act(x @ W + c)
            if i == self.arctan2_after:
                x = self.arctan2(
                    self.pairs(x) if self.norm is None else self.norm(self.pairs(x))
                )
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
        cfg["arctan2_after"],
        cfg["radius_head"],
        cfg.get("init", "uniform"),
        cfg.get("arctan2_layernorm", False),
        cfg.get("bias", False),
        cfg.get("arctan2_project", False),
        cfg.get("arctan2_x_hidden", []),
        cfg.get("arctan2_x_activation", ""),
    )
