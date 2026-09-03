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
- [ ] Obtain independent human review of C9, C10, C11, and C12.
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
- [x] Add a standalone standard-library reconstruction of the P6 `4 x 4` LMI,
      exact value cancellation, and Sylvester-minor checks.
- [ ] Obtain an independent human proof audit of C11, with particular attention
      to the directed smooth nonconvex interpolation step.
- [x] Extend P6 at the full step to an exact pathwise input-to-storage/output
      inequality under additive gradient and post-operator implementation
      errors, without claiming full-state ISS on nonunique minimizer sets.
- [x] Derive the bounded-input, square-summable-input, and conditional
      bounded-second-moment stochastic consequences of the P7 inequality.
- [x] Add an independent exact reconstruction of the P7 `6 x 6` LMI and its
      rational Sylvester-minor and function-cancellation checks.
- [x] Run the 144-case, 17,280-update disturbed nonconvex-PL falsification grid
      and record the flat-minimizer harmonic-drift counterexample to iterate
      convergence under merely square-summable errors.
- [ ] Obtain an independent human proof audit of C12, including disturbance
      placement, physical-unit gain conversion, and the stochastic corollary;
      the unsigned review packet is
      `theory/audits/P7_HUMAN_PROOF_AUDIT.md`.
- [x] Specify a proposed fixed-`2 x 2` mixed-precision repaired operator with
      FP32 scaled max-floor normalization and repair, a BF16-normalized stage
      input and five BF16 stage outputs, and a fixed serial-FP32-Horner kernel.
- [x] Prove the exact affine operator-error bound
      `||Rhat(s)-R(s)||_F <= (11/100000) ||s||_F + 347/100` for finite FP32
      inputs with maximum absolute entry at most `2^116` under the locked IEEE
      arithmetic contract.
- [x] Extend the operator interface to arbitrary real `2 x 2` inputs in the
      same range by an explicit entrywise FP32 cast, proving the affine bound
      `||Rhat_R(s)-R(s)||_F <= (1/5000) ||s||_F + 347/100`.
- [x] Close that affine error through the P7 post-operator port in an otherwise
      exact-real loop, obtaining an exact rate below one and the certified
      zero-gradient-noise objective-gap neighborhood `2.22630172325...`, with
      an exact sufficient initial-storage invariant that preserves the finite
      input range.
- [x] Add a standalone standard-library reconstruction of the fixed-`2 x 2`
      P8 certificate and its exact rational comparisons.
- [ ] Extend P8 to a dimension-scalable mixed-precision kernel certificate.
- [ ] Bound FP32 EMA/Nesterov and parameter-update rounding through appropriate
      internal ports or a compensated/higher-precision master-weight design.
- [ ] Bound the complete error of a pinned deployed BF16 backend by the C12
      disturbance model and certify its ultimate neighborhood; P8's proposed
      stage-boundary design is not literal upstream parity.
- [ ] Run a small matched NanoGPT sweep only after the preceding gates pass.
- [ ] Run an accelerator throughput benchmark only for a certified design.

## Explicitly deferred

- Lean formalization;
- Gram Newton--Schulz implementation audit;
- language-model evidence before a full-matrix repair certificate;
- general circuit-designed optimizers.
