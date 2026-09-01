# Task ledger

## Gate 1: theorem and witness

- [x] Separate current and fixed Frobenius normalization.
- [x] Derive the diagonal normalized-map Jacobian.
- [x] Construct an exact Jordan witness with all four fixed-scale local modes positive.
- [x] Verify a finite pairwise violation in high precision.
- [x] Independent proof review.
- [x] Fix the global repair domain via a positive Frobenius normalization floor.

## Gate 2: deficit audit

- [x] Implement Jordan and classical Newton--Schulz in float64.
- [x] Implement local-Jacobian and pairwise deficits.
- [x] Record the robust BF16 pair with backend-specific provenance.
- [x] Add Polar Express from a pinned upstream revision.
- [x] Add clean-room CANS from its published coefficient table.
- [x] Add a two-precision Arb upper certificate for the five-step Jordan map.

## Gate 3: evidence

- [x] Matrix-spectrum audit for Jordan, classical, Taylor, Polar Express, CANS.
- [x] Gain-matched controlled matrix quadratics for all five families.
- [x] Long-horizon sensitivity check for Jordan and classical controls.
- [x] Matrix-multiplication accounting for every audited prefix.

## Post-certificate gate

- [x] Derive a dimension-uniform full-matrix upper certificate for a floored normalizer.
- [x] Implement the floored architecture and corresponding certified constant repair.
- [x] Prove the simplified continuous-time contraction corollary.
- [ ] Run matched momentum quadratics.
- [ ] Run a small matched NanoGPT sweep only after the preceding gates pass.
- [ ] Run an accelerator throughput benchmark only for a certified design.

## Explicitly deferred

- Lean formalization;
- Gram Newton--Schulz implementation audit;
- language-model evidence before a full-matrix repair certificate;
- general circuit-designed optimizers.
