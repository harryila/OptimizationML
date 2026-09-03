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
  readout, including the distinct certificate-source and final-checkpoint
  revisions; the standalone reconstruction entry point is
  `scripts/reconstruct_pl_convergence.py`;
- `summaries/robust_dissipativity_certificate.json`: exact
  dimension-independent P7 `6 x 6` pathwise certificate at `beta=19/20` and
  `eta=1/32000`, proving
  `V_next <= (399960001/400000000)*V + ||xi||_F^2/2 + ||e||_F^2/2000000`
  for the repaired exact-real max-floor five-step Jordan operator; its source
  revision is `c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`;
- `summaries/robust_dissipativity_falsification.json`: 144-case,
  17,280-update CPU/float64 disturbed nonconvex-PL diagnostic with zero
  candidate violations, plus the flat-minimizer harmonic-drift construction
  showing that square-summable errors alone do not imply iterate convergence;
- `summaries/P7_RESULTS.md`: scoped human-readable P7 theorem, deterministic
  and bounded-second-moment stochastic consequences, diagnostic readout, and
  strict exclusion of full-state ISS and a BF16 deployment claim;
- `summaries/mixed_precision_certificate.json`: exact fixed-`2 x 2` P8
  forward-error certificate for the proposed FP32 serial-Horner kernel with
  BF16 stage-boundary storage, including the affine operator-error bound,
  arbitrary-real input adapter, five-stage invariant, locked finite-input
  domain, and exact P7 error-to-objective closure; the independent
  reconstruction entry point is `scripts/reconstruct_mixed_precision.py`;
- `summaries/mixed_precision_falsification.json`: deterministic 82-case P8
  fixed-`2 x 2` CPU diagnostic against a float64, non-exact evaluation of the
  ideal formula; it is falsification evidence, not the proof;
- `summaries/P8_RESULTS.md`: concise scoped P8 theorem, exact replay, and
  diagnostic readout;
- `summaries/scalable_mixed_precision_certificate.json`: canonical replay
  target for the exact P9 shape-parameterized recurrence, seven locked
  Transformer shapes, affine operator-error bounds, and the resulting P7
  rates; generate it with `scripts/certify_scalable_mixed_precision.py` and
  independently check it with the standard-library-only
  `scripts/reconstruct_scalable_mixed_precision.py`;
- `summaries/scalable_mixed_precision_diagnostic.json`: output target for the
  P9 CPU/native-matmul falsification probe. Its default run uses modest shapes;
  realistic shapes are an explicit expensive opt-in, and neither mode is the
  proof reference;
- `summaries/P9_RESULTS.md`: scoped P9 theorem, exact recurrence/replay,
  serial-normalizer and one-term-proof obstructions, qualified frontier
  evidence, and deployment exclusions;
- `summaries/deficit_audit.{json,csv}`: 24-prefix spectral audit;
- `summaries/quadratic_lr_sweep.json` plus CSV tables: gain-matched quadratic
  sweep and sensitivity classifications;
- `summaries/quadratic_horizon_check.{json,csv}`: 750/3000-step robustness;
- `figures/*.svg`: source-hashed static figures;
- `summaries/RESULTS.md`: concise experiment-only readout.
