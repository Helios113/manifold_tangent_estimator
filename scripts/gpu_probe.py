import torch, time
print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.mem_get_info())
for dt in (torch.float32, torch.float64):
    a = torch.randn(4096, 4096, device="cuda", dtype=dt); torch.cuda.synchronize()
    t = time.time(); 
    for _ in range(5): b = a @ a
    torch.cuda.synchronize(); print(dt, "matmul TFLOPs", 5 * 2 * 4096**3 / (time.time() - t) / 1e12)
for n in (2000, 8000, 16000):
    for dt in (torch.float32, torch.float64):
        a = torch.randn(n, n, device="cuda", dtype=dt); a = a + a.T
        torch.cuda.synchronize(); t = time.time()
        w, v = torch.linalg.eigh(a); torch.cuda.synchronize(); print("eigh", n, dt, round(time.time() - t, 2), "s")
        del a, w, v
for n in (16000,):
    a = torch.randn(n, n, device="cuda", dtype=torch.float64); a = a @ a.T / n + torch.eye(n, device="cuda", dtype=torch.float64)
    torch.cuda.synchronize(); t = time.time(); L = torch.linalg.cholesky(a); torch.cuda.synchronize(); print("chol", n, round(time.time() - t, 2))
