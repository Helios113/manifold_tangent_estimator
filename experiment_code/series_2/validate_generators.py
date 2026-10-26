"""Series 2 (small k), step 2 of plan.md: validate every generator used (immersion, bottleneck, reach, non-triviality) for k in {1,2,5,10}, D in {300,1000}.
Family D is swept over diam/tau in {1, 3, 10, 30} to find the curvature at which global PCA fails (non-triviality). -> results/series_2/generator_validation.csv"""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tangent_est.validate import validate

specs = []
for D in (300, 1000):
    for k in (1, 2, 5, 10):
        specs += [dict(family="A", k=k, D=D), dict(family="B", k=k, D=D), dict(family="C", k=k, D=D, gen=dict(base="A")), dict(family="C", k=k, D=D, gen=dict(base="B"))]
        specs += [dict(family="D", k=k, D=D, diam_over_tau=t) for t in (1.0, 3.0, 10.0, 30.0)]
if __name__ == "__main__":
    validate(specs, ROOT / "results" / "series_2" / "generator_validation.csv")
