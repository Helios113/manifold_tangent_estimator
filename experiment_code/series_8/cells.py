"""Series 8: the sphere of the autoencoder experiments with analytic tangent spaces.  S^100 (intrinsic dimension k = 100, living in R^101) embedded into D = 1000 by the
random Fourier-feature embedding (family S, tangent_est/generators.py); n = 1e4, uniform sampling, no noise.  Methods: LPCA, LDLE, LEGO, Guidotti, AL (+ MS-SVD and the baselines of the runner)."""
import pathlib, importlib.util
_spec = importlib.util.spec_from_file_location("cells_series2", pathlib.Path(__file__).resolve().parent.parent / "series_2" / "cells.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)


def all_cells():
    return [_m.cell("S", {}, 100, 1000, 10000, "uniform", "none", 0.0)]
