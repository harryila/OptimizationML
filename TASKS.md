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
- [x] Prove a dimension-independent stylized non-Nesterov quadratic momentum IQC.
- [x] Replay its exact rational rate LMI and matched rank-one boundary checks.
- [x] Match the pinned EMA/Nesterov state-and-signal ordering in a second IQC.
- [x] Replay the EMA/Nesterov exact rational LMI at the default `beta=0.95`.
- [x] Derive and replay the full-step P4 structure-aware quadratic certificate
      at `eta=1/32000`.
- [x] Prove P5 arbitrary-pair nonlinear incremental contraction at
      `eta=1/640000`.
- [x] Prove P5 full-step trajectory-to-minimizer convergence for every fixed
      differentiable globally `1`-strongly-convex, `10`-smooth objective at
      `eta=1/32000`.
- [x] Replay the P5 rational LMIs and run changing-orientation nonlinear
      falsification probes.
- [x] Add a standalone independent exact reconstruction of the P5 full-step
      matrix and determinant checks.
- [ ] Obtain independent human review of C7 and C8.
- [ ] Obtain independent human review of C9, C10, and C11.
- [ ] Obtain an independent parity audit against the pinned upstream EMA/Nesterov
      update (the automated local two-`lerp` regression exists; external sign-off
      is still pending).
- [x] Derive a less conservative architecture-aware momentum region for fixed
      quadratics and the strongly-convex/smooth nonlinear class.
- [x] Prove full-step global function-value convergence and momentum decay for
      every fixed differentiable globally `10`-smooth objective satisfying the
      global Polyak--Lojasiewicz (PL) inequality with constant `1`, without
      claiming a unique minimizer.
- [x] Replay the P6 rational value--momentum LMI and run the 72-case nonconvex,
      changing-orientation, nonunique-minimizer falsification grid.
- [ ] Extend the P6 result to input-to-state or mean-square robustness under
      stochastic-gradient noise.
- [ ] Treat BF16 and rounding as disturbances and certify an ultimate error
      neighborhood.
- [ ] Run a small matched NanoGPT sweep only after the preceding gates pass.
- [ ] Run an accelerator throughput benchmark only for a certified design.

## Explicitly deferred

- Lean formalization;
- Gram Newton--Schulz implementation audit;
- language-model evidence before a full-matrix repair certificate;
- general circuit-designed optimizers.
