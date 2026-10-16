"""Check the torch ports against the numpy reference implementations (same data, same hyper-parameters)."""
import sys, time, numpy as np, torch
sys.path.insert(0, ".")
from tangent_est import sample_embedded, tangent_errors, lpca, ldle, lego, SignatureFunction, signature_tangent
from tangent_torch import LPCA, LDLE, LEGO, Signature, to_t

def err(Q, Qt): return tangent_errors(Q, Qt)["max_angle"].mean()

for name in ["circle", "sphere S2", "swiss roll", "clifford T2", "sphere S3"]:
    for sg in (0.0, 0.02):
        X, Qt, d = sample_embedded(name, 2000, 20, sg, seed=1)
        Xt = to_t(X); row = [f"{name:12s} s={sg}"]
        ref = {"lpca": lpca(X, d, k=20), "ldle": ldle(X, d, k=25, klv=20), "lego": lego(X, d, k=15, m=20, m0=60),
               "sig": signature_tangent(X, d, method="shape", bw_mult=1.5, alpha=1e-3)}
        mods = {"lpca": LPCA(d, k=20), "ldle": LDLE(d, k=25, klv=20), "lego": LEGO(d, k=15, m=20, m0=60), "sig": Signature(d, 1.5, 1e-3)}
        for k_, m in mods.items():
            t = time.time(); m.fit(Xt); Q = m.tangents_train().cpu().numpy(); torch.cuda.synchronize()
            # agreement of the subspaces between numpy and torch implementations
            agree = tangent_errors(Q, ref[k_])["max_angle"].mean()
            row.append(f"{k_}: np={err(ref[k_], Qt):.2f} torch={err(Q, Qt):.2f} (diff {agree:.2f}deg, {time.time()-t:.1f}s)")
        print(" | ".join(row), flush=True)
# out-of-sample sanity
X, Qt, d = sample_embedded("sphere S2", 4000, 20, 0.01, seed=3)
tr, te = slice(0, 2000), slice(2000, 4000)
for k_, m in {"lpca": LPCA(d), "ldle": LDLE(d), "lego": LEGO(d), "sig": Signature(d)}.items():
    m.fit(to_t(X[tr])); Q = m.tangents_query(to_t(X[te])).cpu().numpy()
    print("oos", k_, f"{err(Q, Qt[te]):.2f}", "in-sample", f"{err(m.tangents_train().cpu().numpy(), Qt[tr]):.2f}", flush=True)
