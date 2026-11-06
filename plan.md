# Protocol of the tangent-space estimator benchmark

`tangent_est/` implements this protocol; `experiment_code/series_*` run the campaigns.

1. **Manifold generators** (`tangent_est/generators.py`): families A (product of circles), B (Fourier embedding of the torus), C (A or B restricted to a box, with boundary), D (near-flat, curvature knob `eps`), each with analytic Jacobian, sampler, and three noise models (isotropic, normal, tangent). Family S is the autoencoder's embedded sphere.
2. **Generator validation** (`tangent_est/validate.py`): immersion (Jacobian full rank), bottleneck, reach, and non-triviality (global PCA must fail on the curved families at small k); a manifold that fails is rejected.
3. **Cells**: one *cell* = one condition (family, k, D, n, sampling, noise model, noise level `nu = level * tau_eff`, curvature). The cell lists are in `experiment_code/series_*/cells.py`.
4. **Draws** (`tangent_est/bench.py`): every cell is evaluated on one validation draw (to tune the hyper-parameters of every method) and `N_TEST` test draws (fresh manifold instance, sample and noise each).
5. **Methods**: LPCA, multiscale SVD, Aamari-Levrard local polynomial of order m = 2, 3, 4, LDLE, LEGO, Guidotti's signature function (`tangent_torch/`), plus the baselines random, global PCA and oracle LPCA.
6. **Rate check** (`tangent_est/rate.py`): noise-free data, error vs n; the fitted slope is compared with the predicted `-(m-1)/k` (tangent) and `-(m-2)/k` (curvature).
7. **Metrics** (`tangent_est/metrics.py`, `bench.py`): principal angles between estimated and true tangent spaces, summarised as median / p90 / max of `e = mean sin^2(theta)`, the largest angle and `sin(largest angle)`; for order m >= 3 also the curvature error `||II o pi_T - T2 o pi_hat||_op`.
8. **Skipped / failed configurations** are rows with `status != "ok"` and a reason, collected in a failure log.
9. **Aggregation** (`tangent_est/aggregate.py`, `tangent_est/plots.py`): tune every method on the validation draw (lowest validation median error), report mean and s.d. over the test seeds, tables, failure log and the figures (error vs noise / n / diam-tau / scale parameter, rate check).
