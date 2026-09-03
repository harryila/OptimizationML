# Results policy

Commit manifests, compact JSON/CSV summaries, and final figures. Keep raw
checkpoints, W&B caches, and large tensors out of Git. Every result must name
the operator, normalizer, epsilon, coefficients, steps, dtype, seed, hardware,
software versions, and Git SHA.

Current outputs:

- `summaries/canonical_witness.json`: exact/high-precision Jordan witness;
- `summaries/bf16_witness.json`: backend-specific executable Keller--Jordan
  BF16 pairwise witness with full run provenance (record separately on every
  claimed backend; returned values are not universal);
- `summaries/floored_repair_certificate.json`: two-precision Arb scalar
  enclosure, exact finite-pair lower witness, and dimension-uniform
  full-matrix floored-repair certificate;
- `summaries/momentum_iqc_certificate.json`: exact closed-form sector region,
  rational rate-LMI replay, exact local instability control, and matched
  deterministic rank-one trajectories for the stylized non-Nesterov loop;
- `summaries/ema_nesterov_iqc_certificate.json`: exact rational LMI replay for
  the pinned EMA/Nesterov state-and-signal ordering, numerical repair-margin
  design scan, exact local threshold, and matched rank-one diagnostics;
- `summaries/structure_aware_stability_certificate.json`: full-step,
  dimension-independent P4 certificate for the repaired floored operator in
  the pinned deterministic quadratic loop;
- `summaries/nonquadratic_stability_certificate.json`: P5 arbitrary-pair
  incremental certificate for globally `1`-strongly-convex, `10`-smooth
  objectives at `eta=1/640000`;
- `summaries/nonquadratic_convergence_certificate.json`: P5 objective-gap and
  interpolation-storage certificate for trajectory-to-minimizer convergence
  at the full P4 step `eta=1/32000`;
- `summaries/nonquadratic_falsification.json` and
  `summaries/nonquadratic_convergence_falsification.json`: qualified float64
  changing-orientation implementation probes for the two P5 certificates;
- `summaries/P5_RESULTS.md` and `summaries/P5_FULL_STEP_RESULTS.md`: scoped
  human-readable summaries of the incremental and full-step P5 results;
- `summaries/pl_convergence_certificate.json`: exact dimension-independent P6
  value--momentum certificate for global function-gap convergence and momentum
  decay on differentiable globally `10`-smooth objectives satisfying the
  global PL inequality with constant `1`, at `eta=1/32000`;
- `summaries/pl_falsification.json`: 72-case CPU/float64 diagnostic on an
  analytic nonconvex PL family, including negative curvature, changing Hessian
  orientations, and nonunique minimizers;
- `summaries/P6_RESULTS.md`: scoped human-readable P6 theorem and diagnostic
  readout;
- `summaries/deficit_audit.{json,csv}`: 24-prefix spectral audit;
- `summaries/quadratic_lr_sweep.json` plus CSV tables: gain-matched quadratic
  sweep and sensitivity classifications;
- `summaries/quadratic_horizon_check.{json,csv}`: 750/3000-step robustness;
- `figures/*.svg`: source-hashed static figures;
- `summaries/RESULTS.md`: concise experiment-only readout.
