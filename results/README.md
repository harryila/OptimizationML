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
  deterministic rank-one momentum trajectories;
- `summaries/deficit_audit.{json,csv}`: 24-prefix spectral audit;
- `summaries/quadratic_lr_sweep.json` plus CSV tables: gain-matched quadratic
  sweep and sensitivity classifications;
- `summaries/quadratic_horizon_check.{json,csv}`: 750/3000-step robustness;
- `figures/*.svg`: source-hashed static figures;
- `summaries/RESULTS.md`: concise experiment-only readout.
