# Tangent-Space Estimation Benchmark: Agent Protocol

## 0. Scope
- **Goal:** measure tangent-space estimation error of the methods under test (LPCA, multiscale-SVD LPCA, local polynomial fitting, LEGO, Guidotti signature function) on synthetic manifolds with exact ground truth.
- **Two regimes, run and reported separately:**
  - **Small-k:** k ∈ {1, 2, 5, 10}, D ∈ {300, 1000}.
  - **Large-k:** k ∈ {50, 100, 300}, D = 1000. Only the near-flat regime is testable here (see step 7).
- **Always give each method the true k.** Test dimension estimation as a separate experiment.

## 1. Manifold generators
Each generator returns `X_clean`, `theta`, and an analytic Jacobian `J(theta)`. The true tangent space at a point is `T = col(J)`.

- **A. Product of circles**
  - F(θ) = Q · (r cos θ₁, r sin θ₁, …, r cos θ_k, r sin θ_k, 0, …), with Q a random orthogonal D×D matrix.
  - Principal curvatures are 1/r; reach τ = r.
  - Requires 2k ≤ D.
- **B. Fourier embedding**
  - F(θ) = A · φ(θ), with θ on the k-torus T^k.
  - φ(θ) = (cos⟨ω_j, θ⟩, sin⟨ω_j, θ⟩) for j = 1..J; ω_j are integer frequency vectors; A is random D×2J; 2J ≤ D.
  - The curvature knob is the frequency magnitude |ω_j|.
- **C. With boundary:** same as A or B, but θ is restricted to a box.
- **D. Near-flat** (required for large k)
  - F(θ) = Q · [θ ; ε·g(θ)], where θ ∈ [0, 1]^k and g: R^k → R^(D−k) is smooth (e.g. low-frequency random Fourier features).
  - J = Q · [I_k ; ε·∂g].
  - The curvature knob is ε. Report diam/τ.

## 2. Generator validation (reject any manifold that fails)
1. **Immersion:** rank J(θ) = k at every sample (check singular values against a tolerance).
2. **Bottleneck:** for parameter pairs far apart, min ‖F(θ) − F(θ')‖ must be much larger than both the noise level and the largest neighbourhood scale used. Record the value.
3. **Reach:** record τ, exact (A) or estimated (B, D) as min(1/max curvature, bottleneck/2). Curvature comes from second derivatives of F.
4. **Non-triviality:** global PCA must fail on families B and D at small k. If it succeeds, increase the span of the embedding.

## 3. Sampling
- **Uniform on M:** sample with density proportional to √det(JᵀJ).
  - Small k: rejection sampling.
  - Large k: MCMC, or record the sampling as non-uniform.
- **Non-uniform:** add a separate condition with a known non-uniform density.
- **Sample sizes:** n ∈ {10³, 10⁴} for small k; as large as the methods' compute limits allow for large k.

## 4. Noise
Add noise to `X_clean` to get `X_noisy`. Always evaluate against the tangent at the clean point.

- **Three noise models, run separately:** (i) isotropic Gaussian, (ii) normal-only, (iii) tangential-only.
- **Level:** parameterise by σ√D/τ ∈ {0, 0.01, 0.1, 0.5}. Also record σ√D divided by the median nearest-neighbour distance.

## 5. Methods and hyperparameters
- **Scale parameters per method:**
  - LPCA: radius h or kNN count.
  - Local polynomial: h and degree.
  - LEGO: graph bandwidth and number of eigenvectors.
  - Guidotti: kernel width and α.
- **Tuning:** tune each method on an independent validation draw (new A, Q, samples and noise), then evaluate on a separate test draw.
- **Error-vs-scale curves:** additionally report error as a function of the scale parameter for every method.
- **Infeasibility:** skip a method and log the reason (do not silently drop it) when:
  - local polynomial degree ≥ 2 and the neighbourhood contains fewer than k(k+1)/2 points;
  - LPCA has fewer than 2k points per neighbourhood;
  - memory or time exceeds the budget (dense n×n solves; D×D Hessians at cost O(nD²) per point).

## 6. Metrics
- **Primary:** all k principal angles between the estimated and true tangent spaces, and
  e = ‖P_est − P_true‖²_F / (2k).
- **Small k only:** also report sin θ_max.
- **Aggregation:**
  - Report median, 90th percentile and maximum over points.
  - Report interior and boundary points separately (family C).
  - Report mean ± s.d. over at least 5 seeds.

## 7. Feasibility gate (compute and log before running anything)
- **Spacing ratio:** δ = median nearest-neighbour distance / diam. Note that n^(−1/k) ≈ 0.955 for k = 300, n = 10⁶.
- **Large k:** run only on family D (plus family A as a demonstration of failure). Sweep ε so that diam/τ ∈ {10⁻³, 10⁻², 10⁻¹, 1}.
- **Labelling:** label every result with its δ and diam/τ.
- **Do not extrapolate:** results from the near-flat regime must not be presented as evidence about generic curved high-k manifolds.

## 8. Baselines (run in every condition)
1. **Random subspace:** chance level E[e] = 1 − k/D.
2. **Global PCA:** a "too easy" detector.
3. **Oracle LPCA:** noise-free data with dense sampling; gives a lower bound.
4. **Low-dimensional reproduction:** a D = 3 configuration matching published low-dimensional results, as a sanity check.

## 9. Output
- **Per condition:** a table with columns family, k, D, n, noise model, σ√D/τ, diam/τ, δ, sampling, method, tuned hyperparameters, and metrics.
- **Plots:** error vs noise, error vs n, and error vs diam/τ, each per k.
- **Failure log:** skipped methods, rejected manifolds, and numerical failures (ill-conditioned solves, ∇u ≈ 0 for Guidotti).
- **Reproducibility:** save all seeds, generator parameters and code.



# Additions to the protocol: Aamari–Levrard local polynomial estimator

**Notation clash.** In the paper, *d* is the intrinsic dimension and *k* is the smoothness order. The protocol uses *k* for the intrinsic dimension. Below, *k* stays the intrinsic dimension and *m* denotes the paper's smoothness order.

## Step 5: method definition

**Method AL — Aamari–Levrard local polynomial estimator (Ann. Statist. 2019).**

At each sample point X_j, solve

  argmin over (π, T₂, …, T_{m−1}) of
  (1/(n−1)) · Σ_{i≠j} ‖x − π(x) − Σ_{s=2}^{m−1} T_s(π(x)^{⊗s})‖² · 1{‖x‖ ≤ h},  with x = X_i − X_j,

subject to:
- π is an orthogonal projector of rank k;
- each T_s is a symmetric s-linear tensor with ‖T_s‖_op ≤ t.

Outputs:
- **Tangent estimate:** T̂_j = im(π̂_j).
- **Curvature estimate (m ≥ 3):** T̂₂ ∘ π̂ estimates the second fundamental form II ∘ π_T. Score this as an extra curvature task.
- **Special case:** m = 2 is exactly LPCA.

**Implementation.** The paper gives two options and states no preference between them:
- **(a) Two-stage:** run LPCA in the same ball to get an orthonormal basis; express π and the tensors as coefficients in that basis; solve the least-squares problem under orthogonality constraints.
- **(b) Grassmann optimisation:** gradient descent over rank-k projectors. The objective is non-convex in π.

**Hyperparameters.**
- **m ∈ {2, 3, 4}.** All generator families are analytic, so any m is admissible.
- **h:** the theory sets h = (C · log n / (n−1))^{1/k} with C unspecified. Treat C as tuned (validation draw plus error-vs-h curve). Constraint: h ≤ h₀ = min(τ, 1/L⊥)/8.
- **t = 1/h**, as the paper prescribes for h → 0.
- The paper assumes the true k and m are known. Supply both.

## Step 4: noise models

- **Noise model (ii), normal-only, must match the paper's tubular model:** Z ∈ (T_Y M)^⊥, ‖Z‖ ≤ σ, E[Z | Y] = 0, σ < τ_min. Use bounded noise (e.g. uniform on a normal ball), not Gaussian, for the condition used to check the theory.
- **Guarantee validity:** the guarantee requires σ ≤ h/4. Log whether each condition satisfies it.
- **Isotropic and tangential noise** are outside the paper's model. Results there are empirical only.

## Step 6: metrics

- **Rate check for AL (main validation of the implementation):**
  - Noise-free case: fit the slope of log(max_j angle) against log n, for n spanning at least one decade.
  - Theory predicts slope ≈ −(m−1)/k, up to log factors (Theorems 2–3: upper and lower bounds match up to log n).
  - Report the fitted slope next to the predicted one.
- **Curvature error (m ≥ 3):** ‖II ∘ π_T − T̂₂ ∘ π̂‖_op, against the analytic second fundamental form from the generator. Predicted rate ≈ (1/n)^{(m−2)/k}.
- **High-noise regime:** for σ ≫ (1/n)^{m/k}, the paper states its estimator is likely suboptimal relative to the lower bound. Flag these cells.

## Step 7: feasibility gate

- **Unknowns per neighbourhood:**
  - p(k, m) = Σ_{s=2}^{m−1} C(k+s−1, s) coefficients per output coordinate, plus the projector.
  - Examples: m = 3 gives C(k+1, 2), which is 45,150 at k = 300; m = 4 adds C(k+2, 3), about 4.5×10⁶ at k = 300.
- **Rule:** run AL at order m only if the median number of points in B(X_j, h) is at least 2·p(k, m) (plus the LPCA requirement of more than 2k points for stage (a)). Otherwise fall back to m = 2 (= LPCA) and log the reason.
- **Expected outcome:**
  - m ≥ 3 is feasible only for small k, roughly k ≤ 10 depending on n.
  - At k ≥ 50, AL reduces to LPCA.
- **Reach:** Theorem 1 shows no estimator is uniformly consistent if the reach is not bounded below. The bottleneck/reach validation in step 2 is therefore a precondition for interpreting AL results, not optional.

## Step 9: output additions

- For each condition: m used, h, t, the σ ≤ h/4 flag, the fitted rate slope vs. the predicted −(m−1)/k, and the curvature error when m ≥ 3.

# Series 6: series 5 over intrinsic dimension

- k ∈ {3, 10, 30, 100, 300} (log scale), D = 1000, families A, B, D (diam/τ = 1), n ∈ {10³, 10⁴}, noise {none, iso 0.01, iso 0.1, normal 0.1}, uniform sampling: 120 cells, 5 test draws + 1 validation draw each (same runner, grids and tuning as series 5).
- Infeasible method/k/n combinations are skipped and logged (failure_log.csv), per the feasibility rules above.
- Outputs: results/series_6/{cells, summary_table.csv, failure_log.csv, tables.md, figures/} (error vs k per family, n, noise).
- Code: experiment_code/series_6/.

# Series 7: Guidotti scaling with the ambient dimension

- Question: how do time, GPU memory and accuracy of the signature function grow with D.
- Family A (product of circles), k = 2, n = 2000, no noise, uniform sampling; D ∈ {10, 30, 100, 300, 1000, 3000, 10000}; 3 seeds (new manifold and sample each); Guidotti with the true k, bw_mult 1.5, α = 1e-3, float64, default perturbations 2(D−k); tangents estimated at 20 points.
- Measured per (D, seed): fit time and peak memory, tangent-estimate time and peak memory, largest principal angle (median, p90, max).
- Outputs: results/series_7/{cells/D<D>.csv, summary.csv, guidotti_vs_D.png}. Code: experiment_code/series_7/ (run.py per D as a slurm array task, plot.py).

# Series 8: the autoencoder sphere with analytic tangents (replaces the earlier sphere tests)

- Manifold: the unit sphere S^100 (intrinsic dimension 100, in R^101) from auto_encoder/sphere.py, embedded into D = 1000 with the same random Fourier-feature embedding as the autoencoder runs (family S in tangent_est/generators.py). Ground truth: the analytic tangent space col(J_f(p) B(p)), B(p) an orthonormal basis of p^⊥. Checked beforehand (experiment_code/series_8/smoke.py → results/series_8/generator_check.txt): unit norm, orthonormal frame, finite differences along the sphere lie in the frame.
- n = 1e4, uniform, no noise; 5 test draws + 1 validation draw (new embedding and sample each); every method gets the true k = 100.
- Methods: LPCA, LDLE (new grid: graph neighbours = local view = (2k+2)·{1,2,4}, N = 2k eigenvectors), LEGO, Guidotti, AL (m = 2, 3, 4; m ≥ 3 skipped by the feasibility gate, p(100,3) = 5050 monomials), plus MS-SVD and the random / PCA / oracle-LPCA baselines of the runner. Grids and tuning as series 5 and 6.
- Outputs: results/series_8/{cells, summary_table.csv, failure_log.csv, tables.md, figures/bars.png, figures/scale_*.png}. Code: experiment_code/series_8/.
- Replaces the earlier sphere tests of these methods (the S^300-patch in R^2048 of the old series 2, kept only in results_backup_pre_rerun/); series 1 (paper reproductions) is untouched.
