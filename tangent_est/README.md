# tangent_est

Tangent-space estimation benchmark; the protocol is in [`../plan.md`](../plan.md).

| file | what |
|---|---|
| `generators.py` | manifold generators with analytic Jacobians and tangent spaces |
| `validate.py` | generator validation: immersion, bottleneck, reach, non-triviality |
| `bench.py` | benchmark runner: one cell = one condition, evaluated on a validation and test draws |
| `aggregate.py` | tuning on the validation draw, mean and s.d. over test seeds, tables and figures |
| `plots.py` | figures of the protocol |
| `rate.py` | rate check of the Aamari-Levrard estimator |
| `metrics.py` | tangent-space error metrics |

The estimators themselves are in [`../tangent_torch/`](../tangent_torch/README.md). The campaigns that use this package are in [`../experiment_code/`](../experiment_code/README.md).
