"""Worker: python run_cells.py SHARD NSHARDS [ONLY_SUBSTRING]  -> results/series_5/cells/<id>.csv (restartable)."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from tangent_est.bench import worker
from cells import all_cells
if __name__ == "__main__":
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    worker(all_cells(), ROOT / "results" / "series_5" / "cells", shard, nshards, sys.argv[3] if len(sys.argv) > 3 else None)
