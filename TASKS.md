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
- [x] Integrate a positive nonincreasing majorant of P12's pointwise
      full-matrix deficit into an exact radial correction with a closed-form
      logarithmic primitive.
- [x] Prove `E_(h,epsilon)+G_epsilon` globally monotone in every finite
      rectangular shape by bounding both radial and tangential repair modes
      and integrating the complete symmetric Jacobian.
- [x] Certify the deployed-scale correction magnitudes at unit norm and P11's
      exact signal guard with two-precision Arb arithmetic and an independent
      standard-library rational-log reconstruction.
- [x] Prove the exact identity
      `Lip(G_epsilon)=6602082433275499863/(41641817600000000*epsilon)` and the
      universal strict lower `Lip(C)>158.1172496/epsilon` for every globally
      Lipschitz passivator in shapes with `min(m,n)>=2`.
- [x] Add exact correction-only and full-operator pinned EMA/Nesterov Jury
      failures plus a finite rational one-step scalar-quadratic expansion
      control at `epsilon=1e-7` and `eta=1/32000`.
- [ ] Obtain an independent human proof audit of C19, including P12-envelope
      inheritance, radial/tangential derivatives, pairwise promotion, Arb and
      rational-log enclosures, universal stiffness lower bound, Jury algebra,
      and exact-real/BF16 separation; the unsigned packet is
      `theory/audits/P13_RADIAL_PASSIVATION_HUMAN_PROOF_AUDIT.md`.
- [x] Prove existence and uniqueness of the P13 resolvent after adding a
      positive shunt, together with zero preservation, firm resolvent bounds,
      Yosida cocoercivity, strong monotonicity, Lipschitzness, and the exact
      full-matrix incremental sector IQC.
- [x] At `lambda=1/1000` and `mu=1000`, derive the epsilon-independent sector
      `[500,1000]` and centered decomposition `Y=750*I+E` with
      `Lip(E)<=250`, explicitly without assuming a symmetric Jacobian.
- [x] Certify the pinned `beta=19/20`, `eta=1/32000` EMA/Nesterov loop on the
      full globally `10`-smooth, PL-`1` objective class with the exact rate
      `q14=249001/250000<1` and `D14=0`.
- [x] Add exact scalar controls showing that direct P13 evaluation and a
      finite under-regularized Yosida point fail the Jury test while the
      selected regularized point passes, plus sector-boundary witnesses.
- [x] Add an exact P14 generator, a standard-library-only independent
      reconstruction, exact LMI and source-manifest tests, and dedicated CI.
- [ ] Obtain an independent human proof audit of C20, including maximality and
      resolvent existence, the pulled-back sector identity, nonsymmetric
      centered residual bound, nonconvex interpolation/value flow, exact LMI,
      and negative controls; the unsigned packet is
      `theory/audits/P14_YOSIDA_STABILITY_HUMAN_PROOF_AUDIT.md`.
- [x] Introduce the exact graph residual
      `r=s-u_hat-lambda*B(u_hat)` with deployed output
      `Y_hat=(s-u_hat)/lambda`, and prove the sharp uniform solution/output
      bounds `||u_hat-J(s)||<=||r||/2` and `||Y_hat-Y||<=500||r||`.
- [x] At the computable rule
      `||r||<=||s||/250+rbar` at every oracle call, split relative and absolute output-error
      ports and certify the effective centered radius `252` in every finite
      matrix shape and for every `epsilon>0`.
- [x] Replay the exact pinned smooth-PL value--momentum certificate at
      `beta=19/20`, `eta=1/32000`, and
      `q15=249001/250000`, with the exact robust term `C15=5/2`.
- [x] Derive the exact ultimate bounds
      `limsup V<=625000*rbar^2/999` and
      `limsup(f-f*)<=6250000000000*rbar^2/312929757`, without claiming
      iterate convergence for persistent absolute error.
- [x] Add exact P14-recovery, adjacent frozen-certificate rejection, sharp
      residual-gain, zero-output stalling, and negative-output instability
      controls, together with independent reconstruction and dedicated CI.
- [ ] Obtain an independent human proof audit of C21, including the output
      convention, graph-residual identity, sharp solution/output gains,
      pointwise port split, exact `4 x 4` and `5 x 5` LMIs, ultimate bounds,
      controls, and solver/BF16 exclusions; the unsigned packet is
      `theory/audits/P15_INEXACT_YOSIDA_ROBUSTNESS_HUMAN_PROOF_AUDIT.md`.
- [x] Prove bi-orthogonal equivariance of the P14 resolvent and singular-vector
      preservation, including repeated and zero singular values, on every
      fixed finite rectangular matrix shape.
- [x] Reduce the resolvent graph equation to coupled singular values and prove
      its Jacobian is diagonal plus rank one with positive exact band margins,
      permitting a Sherman--Morrison Newton step.
- [x] Prove global exact-real convergence of safeguarded Armijo Newton from
      every finite start and finite termination for each positive P15
      threshold, without claiming a useful uniform iteration-count bound.
- [x] Implement a fail-closed FP64 reference solver that emits
      `Y_hat=1000*(s-u_hat)` only after recomputing and passing the actual
      candidate P15 residual rule at `kappa=1/250`. Dense calls reevaluate
      `B(u_hat)` on the stored reconstructed matrix with a second SVD; all 13
      declared calls pass the computed check.
- [x] Add an exact rational unequal-mode witness and an outward-rounded Arb
      enclosure proving algebraic noncollapse on the canonical `diag(3,4)`
      input.
- [x] Apply the frozen pre-certificate meaningful-fidelity gate. The locked operator has
      best-scalar departure about `5.879e-6` and retains only about
      `8.403e-5` of upstream Jordan shaping, so the overall P16 acceptance
      gate fails.
- [x] Run the required six-point sampled `(lambda,mu)` frontier. No declared
      point jointly passes the frozen P14 certificate and fidelity gates;
      this is diagnostic evidence, not a global impossibility theorem.
- [ ] Obtain an independent human proof audit of C22, including the
      stabilizer argument at repeated/zero singular values, exact radius-band
      margins, Sherman--Morrison denominator, Armijo convergence, residual
      convention, Arb enclosure, and fidelity classification; the unsigned
      packet is
      `theory/audits/P16_EQUIVARIANT_RESOLVENT_SOLVER_HUMAN_PROOF_AUDIT.md`.
- [x] Define the P17 C2 gate between the scaled passive Yosida fallback and
      `E_(h,epsilon)(J(S))`, with the passive region covering the exact/Arb
      unsafe raw-shape derivative band as a local/incremental safeguard (the
      pointwise PL theorem uses only the gate ceiling).
- [x] Derive a dimension-uniform origin-centered pointwise sector for that
      interface, including the exact full-matrix raw-shape gain upper
      `20191130443162880000000/26793221204801899863`.
- [x] Certify global exact-real one-trajectory smooth-PL convergence for the
      primary ceiling-`3/4`, divisor-`4096`, `eta=1/128000` design and the
      secondary ceiling-`1/8`, divisor-`8192`, full-`eta=1/32000` design.
- [x] Replay both exact `4 x 4` LMIs independently, prove the C2 gate joins,
      enclose the canonical `diag(3,4)` fidelity decisions after P15 residual
      inflation, and include an under-sized-passive-region Jury control.
- [ ] Obtain an independent human proof audit of C23, including the
      singular-mode pointwise bound, distinction from an incremental sector,
      PL use of the pointwise residual supply, gate coverage of the unsafe
      band, both exact LMIs, and exact-real/inexact-solver separation; the
      unsigned packet is
      `theory/audits/P17_SHAPE_PRESERVING_RESOLVENT_HUMAN_PROOF_AUDIT.md`.
- [x] Define the P18 ray projection
      `alpha=min(1,K*<X,S>/||X||^2)` with `K=1`, prove the full-matrix
      pointwise `[0,1]` shape-channel sector, and combine it with `Y/1024`
      under the ceiling-`3/4` gate to obtain the exact global sector
      `[125/1024,509/512]`.
- [x] Apply the predeclared `eta=1/50` test. An exact complex-skew boundary
      map gives a negative Schur--Cohn margin for the generic sector class;
      this is not claimed to be a counterexample to the structured P18 map.
- [x] Replay the mandated exact rational frontier and lock `eta=1/83`,
      `q=999598040401/1000000000000`. The exact comparison against P14 proves
      a certified Lyapunov-rate half-life ratio below ten; `eta=1/120`
      supplies a faster-rate alternative.
- [x] Add non-scale-invariant global amplitude and effective-update gates,
      canonical Arb fidelity checks, operating-annulus and broad-spectrum
      diagnostics, plus unprojected and small-`K` controls.
- [ ] Obtain an independent human proof audit of C24; the unsigned packet is
      `theory/audits/P18_SECTOR_PROJECTED_USEFUL_RATE_HUMAN_PROOF_AUDIT.md`.
- [x] Define the P19 moving Frobenius-ball shield and prove exactly, including
      at `S=0`, that it is equivalent to P18's full-matrix origin-centred
      pointwise sector `[125/1024,509/512]`.
- [x] Prove that every finite approximate or corrupted candidate is shielded
      into the P18 sector, while every exact P18 output is fixed globally.
- [x] Prove fixed-signal metric-projection nonexpansiveness and its modular
      candidate-error corollary, without claiming joint nonexpansiveness or
      that the P15 graph residual alone bounds the nonlinear P18 candidate.
- [x] Replay the exact P18 smooth-PL certificates at both `eta=1/83` and
      `eta=1/120` for arbitrary time-varying finite shielded candidates.
- [x] Implement the scaled/balanced inward-margin NumPy FP64 shield with a
      mandatory exact-as-stored dyadic postcheck, certified `S/2` fallback,
      positive-part clipping of P18's rounded inner product, and fail-closed
      nonfinite/unrepresentable semantics.
- [x] Verify normal bitwise inactivity on all 2,688 declared annulus calls,
      preserve all 2,176 informative fidelity passes, and add exact and
      numerical corrupted-candidate controls.
- [x] Add a standard-library-only independent reconstruction, exact
      provenance locks, dedicated tests, and P19 CI replay.
- [ ] Obtain an independent human proof audit of C25; the unsigned packet is
      `theory/audits/P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_HUMAN_PROOF_AUDIT.md`.
      The P18 and P19 packets are ready, but no external reviewer or delivery
      destination has been supplied.
- [ ] Derive a final P15-graph-residual-to-P18-candidate fidelity bound if one
      is needed beyond P19's stability shield. P19 establishes fixed-signal
      nonexpansiveness but does not manufacture this intervening nonlinear
      error estimate.
- [x] Freeze the P20 proof-reference arithmetic graph: contiguous CPU BF16 or
      FP32 inputs widened to FP32, FP32 round-to-nearest-even vector
      operations, maximum-scaled fixed balanced reductions, directed-inward
      FP64 scalar operations, gradual underflow, no FMA/reassociation, and an
      FP32 stored output.
- [x] Derive exact pass-through, radial-clip, and `S/2` fallback envelopes,
      with positive shape-specific inward margins for all seven frozen P9
      Transformer shapes. The tight `4096 x 14336` overall margin is
      `71710053325847/1152921504606846976`.
- [x] Prove every successful stored P20 output lies in P18's full-matrix
      pointwise sector `[125/1024,509/512]` without a runtime rational or
      big-integer postcheck, including the zero, normal-anchor, exactly
      halvable subnormal, and fail-closed all-subnormal cases.
- [x] Replay the P19 rates conditionally when every call along the trajectory
      succeeds and stored `S` is the abstract operator-port signal:
      `eta=1/83` has rate
      `999598040401/1000000000000`, and `eta=1/120` has rate
      `624350169/625000000`. Do not treat this as composition of the outer
      BF16/FP32 signal cast, momentum, parameter, or master-weight arithmetic.
- [x] Evaluate both the P18 candidate and the literal pinned five-stage
      upstream candidate. The P18 annulus study has `2671/2688` pass-throughs,
      `17` radial clips, no half fallbacks, and all `2176/2176` informative
      fidelity passes; the upstream candidate has `761` pass-throughs and
      `1927` clips.
- [x] Add one-ULP, arbitrary/adversarial, nonfinite, normal/subnormal, FTZ,
      sparse full-shape, and cross-platform decision controls, plus an exact
      generator, standard-library-only reconstruction, focused tests, and
      dedicated P20 CI.
- [ ] Obtain an independent human proof audit of C26, including the scaled
      balanced-norm envelope, all three stored-output branches, near-zero
      representability boundary, exact shape recurrence, conditional port
      identification, and distinction from P19 projection; the unsigned
      packet is
      `theory/audits/P20_SCALABLE_MIXED_PRECISION_SECTOR_SHIELD_HUMAN_PROOF_AUDIT.md`.
- [ ] Compose the P20 stored-signal boundary with FP32 EMA/Nesterov,
      parameter/master-weight rounding, aspect scaling, weight decay, and
      distributed semantics. This is the P21 integration gate.
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
