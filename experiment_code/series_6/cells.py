"""Series 6: series 5 repeated over intrinsic dimension k in {3, 10, 30, 100, 300} (log scale), D = 1000 (2k <= D for families A, B at k = 300).
Everything else as series 5: families A, B, D (diam/tau = 1), n in {1e3, 1e4}, noise {none, isotropic 0.01, isotropic 0.1, normal-only 0.1}, uniform sampling.
Infeasible (method, k, n) combinations are skipped and logged by the runner (plan.md steps 5 and 7)."""
import pathlib, importlib.util
_spec = importlib.util.spec_from_file_location("cells_series2", pathlib.Path(__file__).resolve().parent.parent / "series_2" / "cells.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
KS = (3, 10, 30, 100, 300)
D = 1000


def all_cells():
    out = []
    for k in KS:
        for n in (1000, 10000):
            for fam in ("A", "B", "D"):
                for noise, level in (("none", 0.0), ("iso", 0.01), ("iso", 0.1), ("normal", 0.1)):
                    out.append(_m.cell(fam, {}, k, D, n, "uniform", noise, level, dtau=1.0))
    return out
