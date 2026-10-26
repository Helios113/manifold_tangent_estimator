"""Series 5: easy-setting pre-study before the full campaign -- every method on k = 3 in D = 100 (families A, B, D at diam/tau = 1), n in {1e3, 1e4},
noise {none, isotropic 0.01, isotropic 0.1, normal-only 0.1} (levels are sigma sqrt(D) / tau_eff).  Same runner, grids, tuning and baselines as series 2."""
import pathlib, importlib.util
_spec = importlib.util.spec_from_file_location("cells_series2", pathlib.Path(__file__).resolve().parent.parent / "series_2" / "cells.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)


def all_cells():
    out = []
    for n in (1000, 10000):
        for fam in ("A", "B", "D"):
            for noise, level in (("none", 0.0), ("iso", 0.01), ("iso", 0.1), ("normal", 0.1)):
                out.append(_m.cell(fam, {}, 3, 100, n, "uniform", noise, level, dtau=1.0))
    return out
