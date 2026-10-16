"""GPU helpers shared by the torch implementations. Everything is float32 unless stated."""
import time
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float32


def to_t(x, dtype=DT):
    return torch.as_tensor(x, dtype=dtype, device=DEV)


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize()


def nbytes(*ts):
    return int(sum(t.numel() * t.element_size() for t in ts if t is not None))


class MemTimer:
    """Context manager: wall time and *extra* peak GPU memory (beyond what was allocated on entry)."""
    def __enter__(self):
        sync(); self.base = torch.cuda.memory_allocated() if DEV.type == "cuda" else 0
        if DEV.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.t0 = time.perf_counter(); return self

    def __exit__(self, *a):
        sync(); self.time = time.perf_counter() - self.t0
        self.peak = (torch.cuda.max_memory_allocated() - self.base) if DEV.type == "cuda" else 0


def sqdist(A, B):
    """Squared Euclidean distances (a, b) via one GEMM."""
    return ((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T).clamp_min_(0)


def knn_query(X, Y, k, chunk=4096):
    """k nearest rows of X for each row of Y (no self exclusion). Returns (dist (q,k), idx (q,k)), sorted ascending."""
    mu = X.mean(0); X = X - mu; Y = Y - mu          # centring keeps the GEMM-based distances accurate in fp32
    D, I = [], []
    for i in range(0, len(Y), chunk):
        s = sqdist(Y[i : i + chunk], X)
        v, j = torch.topk(s, k, dim=1, largest=False)
        D.append(v.sqrt_()); I.append(j)
    return torch.cat(D), torch.cat(I)


def knn_train(X, k, chunk=4096):
    """k nearest *other* training points for each training point."""
    D, I = knn_query(X, X, k + 1, chunk)
    return D[:, 1:], I[:, 1:]


def top_left(G, d):
    """Top-d left singular vectors of a batch G (B, p, m) via the eigendecomposition of the smaller Gram matrix."""
    B, p, m = G.shape
    if m <= p:
        w, V = torch.linalg.eigh(G.transpose(1, 2) @ G)
        V = V[..., -d:].flip(-1); w = w[..., -d:].flip(-1)
        return G @ V / w.clamp_min(1e-30).sqrt().unsqueeze(1)
    w, U = torch.linalg.eigh(G @ G.transpose(1, 2))
    return U[..., -d:].flip(-1)


def chunks(n, size):
    for i in range(0, n, size):
        yield slice(i, min(i + size, n))


def auto_chunk(per_item_bytes, budget=3e9, lo=16, hi=8192):
    return int(max(lo, min(hi, budget // max(per_item_bytes, 1))))


def self_tuned_dense(dist, idx, k_tune, n):
    """Dense symmetric self-tuned Gaussian kernel (LDLE Alg. 1). dist/idx: (n, k) training neighbours excl. self."""
    sigma = dist[:, k_tune - 1].clamp_min(1e-12)
    rows = torch.arange(n, device=dist.device).repeat_interleave(idx.shape[1])
    cols = idx.reshape(-1)
    vals = torch.exp(-(dist.reshape(-1) ** 2) / (sigma[rows] * sigma[cols]))
    K = torch.zeros(n, n, dtype=dist.dtype, device=dist.device)
    K[rows, cols] = vals
    K = torch.maximum(K, K.T)
    return K, sigma


def smallest_eigs(A, k, largest=False):
    """k extreme eigenpairs of a dense symmetric matrix (dense eigh on the GPU). Returns ascending-in-quality order:
    smallest first (largest=False) or largest first (largest=True)."""
    w, V = torch.linalg.eigh(A)
    if largest:
        return w[-k:].flip(0), V[:, -k:].flip(1)
    return w[:k], V[:, :k]
