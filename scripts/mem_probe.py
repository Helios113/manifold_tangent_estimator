import torch
for B in (1000, 4000, 8192):
    A = torch.randn(B, 20, 20, device="cuda"); A = A @ A.transpose(1, 2)
    torch.cuda.synchronize(); base = torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
    w, V = torch.linalg.eigh(A); torch.cuda.synchronize()
    print(B, "input MB", A.numel() * 4 / 1e6, "peak extra MB", (torch.cuda.max_memory_allocated() - base) / 1e6)
    del A, w, V
