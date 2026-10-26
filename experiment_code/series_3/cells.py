"""Series 3 (large k): plan.md step 7 -- only the near-flat family D is a fair test (diam / tau in {1e-3, 1e-2, 1e-1, 1}), plus family A as the demonstration of failure.
k in {50, 100, 300}, D = 1000, n = 1e4, uniform sampling (rejection with the exact volume density), noise {none, iso 0.1}."""
import pathlib, importlib.util
_spec = importlib.util.spec_from_file_location("cells_series2", pathlib.Path(__file__).resolve().parent.parent / "series_2" / "cells.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
cell = _m.cell


def all_cells():
    out = []
    for k in (50, 100, 300):
        for noise, level in (("none", 0.0), ("iso", 0.1)):
            for dt in (1e-3, 1e-2, 1e-1, 1.0):
                out.append(cell("D", {}, k, 1000, 10000, "uniform", noise, level, dtau=dt))
            out.append(cell("A", {}, k, 1000, 10000, "uniform", noise, level))
    return out
