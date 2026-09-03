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
- [x] Identify the executable serial-FP32 normalization obstruction on the
      all-ones `4096 x 11008` shape and replace the long serial reduction by a
      scale-free balanced FP32 normalizer.
- [x] Specify the P9 compensated two-term BF16 stage boundary and balanced
      FP32 thick-Horner proof-reference kernel for guarded arbitrary shapes.
- [x] Derive exact shape-parameterized affine coefficients `A_(r,c),B_(r,c)`,
      including subnormal crumbs, `2^116`/`2^52` overflow guards, and the full
      five-stage spectral-tube recurrence.
- [x] Certify the seven locked representative Transformer shapes, including
      `4096 x 11008` and `4096 x 14336`, while retaining the exact P7 rate
      `137425214491/137438953472<1` and a finite objective-gap neighborhood.
- [x] Add a standalone standard-library reconstruction of the P9 recurrence,
      operator bound, overflow gates, and P7 absorption.
- [ ] Obtain an independent human proof audit of C14 and a parity audit of any
      optimized BLAS/GPU implementation against the slow balanced reference;
      the current theorem does not transfer automatically to native kernels.
- [x] Specify the P10 CPU FP32 EMA/Nesterov operation graph, including exact
      binary32 scalar encodings, a single reused rounded gradient product, and
      exact algebraic momentum/signal residual ports.
- [x] Derive exact full-matrix rounding envelopes for the FP32 momentum,
      Nesterov-signal, and logical master-update residuals at
      `4096 x 11008`, retaining subnormal crumbs and overflow guards.
- [x] Implement the three-word FP32 compensated master and prove its two
      `TwoSum` cascades preserve the logical update up to the displayed
      step/low rounding port without a high-word-sized error term.
- [x] Close the concrete P10 rounding envelopes through P7 at the full
      `eta=1/32000` step, certifying `q_10<1`, the guarded `V<=1` invariant,
      and a finite subunit objective-gap neighborhood.
- [x] Add the exact actual-P9 witness showing ordinary FP32 parameter
      subtraction stalls at `W=2^30` while the compensated logical master
      moves, plus a deterministic outer-shell falsification diagnostic.
- [x] Add a standalone standard-library reconstruction of the P10 arithmetic
      envelopes, internal-port reduction, exact rate, and range comparisons.
- [ ] Obtain an independent human proof audit of C15, including the final
      gradient cast, residual placement, three-word master identity, range
      guards, and physical-units conversion; the unsigned review packet is
      `theory/audits/P10_HUMAN_PROOF_AUDIT.md`.
- [x] Add P11 pre-cast gradient/model-weight and post-P9 deployed-output
      discrepancy ports without changing the P10 step, repair, objective
      class, max-floor normalization, or locked `4096 x 11008` shape.
- [x] Derive the exact affine external sensitivities, including cast/EMA,
      Nesterov-signal cancellation, P9 magnitude/error, and master-rounding
      feedback, and reproduce the P10 `q10`, `D10`, and objective fraction at
      zero additional error.
- [x] Compute exact one-axis maxima on the declared `2^-40` budget grid,
      coordinatewise-maximal slope/intercept Pareto slices, and adjacent-grid
      rejection controls without calling certificate rejection instability.
- [x] Exhibit a jointly nonzero P11 budget with `q11<1`, a forward-invariant
      `V<=1` set, a subunit objective-gap neighborhood, and closed signal,
      `2^15` output, step, and master-word guards.
- [x] Add a standalone standard-library reconstruction of the P11 port
      reduction, grid boundaries, frontiers, exact fractions, and guards.
- [ ] Obtain an independent human proof audit of C16, including external-port
      placement, the exact signal cancellation, master-rounding feedback,
      grid-maximality claims, and representation premises; the unsigned packet
      is `theory/audits/P11_HUMAN_PROOF_AUDIT.md`.
- [ ] P10 human proof audit remains pending; P11 does not retroactively provide
      the external sign-off requested in
      `theory/audits/P10_HUMAN_PROOF_AUDIT.md`.
- [x] Prove the exact fixed-shape scaling law
      `delta(E_h,epsilon)=delta(E_h,1)/epsilon` for the exact-real
      additive-Frobenius-epsilon normalizer, including its ordinary positive
      Frechet derivative at zero.
- [x] Derive a dimension-uniform full-rectangular additive-epsilon upper
      certificate using four normalized-radius bands, outward-rounded Arb
      prefix covers, and the projection/anticommutator envelope.
- [x] Certify
      `delta(E_h,1)<=6602082433275499863/41641817600000000` in every finite
      shape and the exact rank-two strict lower
      `delta(E_h,1)>98823281/625000` for shapes with `min(m,n)>=2`.
- [x] Add an exact rational swapped-diagonal witness, two-precision primary
      replay, standalone standard-library directed-interval reconstruction,
      band-boundary checks, and under-repair negative control.
- [x] Quantify the pinned `epsilon=1e-7` exact-real constant-repair
      obstruction: necessary `rho>1,581,172,496` and certified sufficient
      `rho=6602082433275499863/4164181760`.
- [ ] Obtain an independent human proof audit of C18, including the origin,
      full rectangular spectral reduction, four-band interval bounds,
      pairwise promotion, exact witness, and exact-real/BF16 separation; the
      unsigned packet is
      `theory/audits/P12_ADDITIVE_EPSILON_HUMAN_PROOF_AUDIT.md`.
- [ ] Bound the complete error of a pinned deployed BF16 backend by the C12
      disturbance model and certify its ultimate neighborhood; P8--P10 are
      proposed proof-reference designs, not literal upstream parity.
- [ ] Specify how model forward/backward evaluation consumes the three-word
      logical master, or derive a separate reconstruction/error port for the
      represented model weights.
- [ ] Run a small matched NanoGPT sweep only after the preceding gates pass.
- [ ] Run an accelerator throughput benchmark only for a certified design.

## Explicitly deferred

- Lean formalization;
- Gram Newton--Schulz implementation audit;
- language-model evidence before a full-matrix repair certificate;
- general circuit-designed optimizers.
