# P10 human proof-audit packet

## Status

**Independent human review is pending.** This file is a reviewer packet and
checklist. It does not record an approval, and no unchecked item below may be
cited as completed review.

The theorem under review is the exact finite-precision outer-shell reduction
for a proposed CPU FP32 EMA/Nesterov implementation, the P9 mixed-precision
repaired operator at shape `4096 x 11008`, and a three-word compensated FP32
master weight. The complete statement is in
`theory/finite_precision_outer_loop_certificate.md`. The theorem-facing code
is `src/passive_muon/outer_loop_roundoff_certificate.py`, and the executable
operation graph is `src/passive_muon/finite_precision_outer_loop.py`.

## Fixed claim to audit

The proposed shell uses the exact max-floor (`c=1`, no additive epsilon),
five-step Jordan operator and constant repair from P9. It receives one finite,
represented FP32 gradient sample `ghat_t`, reuses its rounded
`(1-beta)ghat_t` product in both outer updates, and executes

\[
\begin{aligned}
m_{t+1}&=\operatorname{fl}_{32}
 \bigl(\operatorname{fl}_{32}(\beta_{32}m_t)
      +\operatorname{fl}_{32}(a_{32}\widehat g_t)\bigr),\\
s_{t+1}&=\operatorname{fl}_{32}
 \bigl(\operatorname{fl}_{32}(\beta_{32}m_{t+1})
      +\operatorname{fl}_{32}(a_{32}\widehat g_t)\bigr),
\qquad a=1-\beta,
\end{aligned}
\]

under the locked CPU IEEE round-to-nearest, ties-to-even, gradual-underflow,
no-FTZ/DAZ contract. Relative to exact coefficients `beta=19/20` and
`a=1/20`, the executable residuals are defined algebraically by

\[
\widetilde r_t^m=m_{t+1}-\beta m_t-a\widehat g_t,
\qquad
\widetilde r_t^s=s_{t+1}-\beta m_{t+1}-a\widehat g_t.
\]

For the locked deterministic theorem, \(g_t=\nabla f(W_t)\) is evaluated at the
logical three-word master and `ghat_t=C32(g_t)`. The total ports are
`r^m=rtilde^m+a(ghat-g)` and `r^s=rtilde^s+a(ghat-g)`. Any error before this
one final cast is a separate P7 gradient-noise input and is not part of the
locked P10 rate.

The logical master weight is the real sum of three FP32 words. The theorem
must bound its logical update residual `r_t^W`, not merely the motion of its
high word. The resulting port-augmented P7 statement is a storage,
function-value, true-gradient, and momentum guarantee while all displayed
arithmetic guards hold. It is not full-state ISS in `W`, and the high-word
range guard is an explicit conditional premise rather than a consequence of
global PL storage.

At shape `4096 x 11008`, the locked exact claim is

\[
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776}
\]

on `V_t<=1` and the stated arithmetic domain, with objective-gap limsup at
most `399957341889/549755813888`. The rate is strictly below one and the
forcing preserves `V<=1`; only the high-word bound remains a rechecked
conditional premise.

## Reviewer checklist

### 1. Arithmetic contract and scalar encodings

- [ ] Confirm the exact binary32 encodings of `beta=19/20`, `a=1/20`, and
  `eta=1/32000`, including their signs and rational representation errors.
- [ ] Confirm the executable graph materializes every multiplication and
  addition separately and excludes FMA contraction and reassociation.
- [ ] Check that the runtime self-test really observes round-to-nearest,
  ties-to-even and gradual underflow on the claimed CPU backend.
- [ ] Verify that every input is finite, contiguous, CPU FP32 with exactly the
  locked matrix shape before any theorem-facing operation is called.

Reviewer notes:

> Pending.

### 2. EMA and Nesterov residual envelopes

- [ ] Derive the componentwise error for
  `fl32(fl32(beta32*m)+fl32(a32*g))`, retaining coefficient representation
  error, both multiplication errors, the final addition error, and every
  subnormal crumb.
- [ ] Lift the componentwise result to the Frobenius norm with the declared
  integer upper bound on `sqrt(4096*11008)`.
- [ ] Apply the same derivation to `r^s` with `m_next` as its first argument;
  confirm that reusing the already rounded gradient product is reflected in
  the executable graph and in the proof.
- [ ] Check every storage-derived bound on `||m||_F`, `||g||_F`, and
  `||m_next||_F`; a magnitude guard alone must not silently become a
  Lyapunov estimate.
- [ ] Confirm that the locked theorem includes the one final
  `ghat=C32(grad f(W))` cast in the total `r^m,r^s` ports, while any error in
  evaluating the gradient before that cast remains P7's separate input and
  is set to zero in the reported `q_10` result.

Reviewer notes:

> Pending.

### 3. Three-word compensated master update

- [ ] Replay the literal graph: rounded FP32 step, rounded `low-step`, an
  error-free `TwoSum` into `middle`, then an error-free `TwoSum` into `high`.
- [ ] Verify the `TwoSum` identities entry by entry without assuming an
  ordering of operand magnitudes and under the stated finite/no-overflow
  conditions.
- [ ] Prove that the logical real sum `high+middle+low` changes by the rounded
  pending term exactly, even when the high word itself does not move.
- [ ] Derive `r^W` relative to the ideal `-eta*Rhat(s)` update, retaining the
  FP32 encoding error in `eta`, rounded step multiplication, rounded
  `low-step`, and subnormal crumbs.
- [ ] Confirm the resulting `r^W` envelope is independent of the magnitude of
  the high word except through the separately stated range/invariant checks.

Reviewer notes:

> Pending.

### 4. Word, output, and overflow guards

- [ ] Recompute the exact high-, middle-, low-word, operator-output, rounded-
  step, pending, and `TwoSum` intermediate bounds.
- [ ] Check that the middle- and low-word separation is forward invariant
  under the locked operation graph and that all strict comparisons leave room
  for the following rounded operation.
- [ ] Confirm the high-word guard is checked after every update and is not
  inferred from objective gap on a class with nonunique minimizers.
- [ ] Verify the P9 signal/input guard and the P10 output guard are distinct
  and that the certificate proves or assumes each in the correct direction.
- [ ] Check all overflow exclusions against the actual FP32 maximum finite
  value and all underflow terms against gradual-underflow semantics.

Reviewer notes:

> Pending.

### 5. Ordinary FP32 subtraction obstruction

- [ ] Replay the exact initial binary32 value, objective/gradient state,
  repaired P9 operator output, and rounded step in the stalling witness.
- [ ] Verify that ordinary FP32 subtraction returns the initial bit pattern
  because the positive downward step is below the relevant half-spacing.
- [ ] Verify that the compensated logical master changes on the first update
  and eventually transfers the accumulated correction into a lower high-word
  value under the same operator signal.
- [ ] Distinguish the actual P9-operator witness from any synthetic arithmetic
  control and confirm neither is generalized to every ordinary FP32 update.

Reviewer notes:

> Pending.

### 6. Exact reduction to the P7 ports

- [ ] Put `a=1-beta` and verify that `xi_eff=r^m/a` makes P7's momentum state
  equal the implemented momentum state.
- [ ] Starting from the implemented and P7 Nesterov signals, recompute their
  exact difference and every coefficient multiplying `r^m` and `r^s`.
- [ ] Re-derive the effective post-operator error, including the P9 affine
  operator error, the Lipschitz motion between the two signal arguments, and
  the physical-units conversion of `r^W` by `1/eta`.
- [ ] Verify that the full-matrix P9 Lipschitz/error bounds are used; no
  diagonal or sampled estimate may carry this step.
- [ ] Check the signs of all ports by substituting the effective errors back
  into the physical parameter update.

Reviewer notes:

> Pending.

### 7. Port-augmented storage certificate

- [ ] Starting from P7's exact pathwise inequality, reproduce every Young
  split and the exact rational gains on the storage term, P9 intercept, and
  the three outer residual ports.
- [ ] Substitute the concrete arithmetic envelopes rather than illustrative
  free port budgets, and rebuild the final rational inequality.
- [ ] Recompute the exact `q_10` and verify `q_7<q_10<1` at shape
  `4096 x 11008` and full step `eta=1/32000`.
- [ ] Recompute the constant forcing and storage-to-function conversion,
  confirming that the stated objective neighborhood is finite and positive.
- [ ] Confirm the generator and standalone standard-library reconstruction
  agree on every canonical exact field without importing one another's
  theorem algebra.

Reviewer notes:

> Pending.

### 8. Scope and range closure

- [ ] Verify the precise initial-storage and arithmetic-range premises and
  check the one-step implication preserving the signal/output/word guards.
- [ ] Confirm the theorem concerns storage/function value, true gradient, and
  momentum, not full-state ISS or parameter convergence on the global PL
  class.
- [ ] Replay the rescaled flat-direction construction showing why bounded or
  merely square-summable parameter residuals do not imply bounded/convergent
  `W` when the minimizer set is nonunique, and check that its scale can be made
  to fit any positive port budget.
- [ ] Confirm that the logical three-word master is the mathematical `W` used
  by the theorem and that model-forward consumption or reconstruction of that
  master remains outside scope.
- [ ] Check the exclusions: literal upstream Muon, gradient computation,
  native BLAS/GPU/tensor-core parity, FTZ/DAZ, weight decay, aspect scaling,
  additive-epsilon/current normalization, and neural-network convergence.

Reviewer notes:

> Pending.

### 9. Provenance and executable parity

- [ ] Verify all source hashes, P9 dependency hashes, software/hardware data,
  operation-order identifiers, and Git-state fields in the canonical result.
- [ ] Run the small executable parity probes independently and confirm they
  test the named CPU reference graph rather than a vendor-fused substitute.
- [ ] Check that sampled residual ratios and trajectories, if reported, are
  labeled falsification diagnostics rather than global certificates.
- [ ] Confirm the committed canonical artifact was produced from a clean,
  identified source commit and that the reconstruction checks this provenance.

Reviewer notes:

> Pending.

## Required review record

The reviewer should add their name or stable anonymous identifier, date,
reviewed Git commit, independent calculation environment, and one of:

- approved without correction;
- approved after listed corrections; or
- not approved, with the first failed checklist item and a reproducible
  discrepancy.

Until that record is present and every relevant checkbox is resolved, the P10
human proof audit remains pending.
