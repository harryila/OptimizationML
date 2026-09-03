# P9 human proof-audit packet

## Status

**Independent human review is pending.** This file is a reviewer packet and
checklist.  It does not record an approval, and no unchecked item below may be
cited as completed review.

The theorem under review is the shape-parameterized affine forward-error
bound for the proposed balanced-FP32, two-term-BF16 five-stage repaired Jordan
operator, together with its zero-gradient-noise P7 closure.  The complete
statement and recurrence are in
`theory/scalable_mixed_precision_certificate.md`; the theorem-facing code is
`src/passive_muon/scalable_mixed_precision_certificate.py` and the executable
operation graph is `src/passive_muon/scalable_mixed_precision.py`.

## Fixed claim to audit

For every separately accepted shape `(r,c)`, every real matrix `s` of that
shape with `rc<=2^52` and `max_ij|s_ij|<=2^116` satisfies

\[
\lVert\widehat R_{r,c}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c}.
\]

The exact reference uses

\[
R_{r,c}(s)=\mathcal H_{q^{\circ5}}
\!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\]

with no epsilon,

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
 +\frac{4063}{2000}x^5,\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

The implementation takes an entrywise FP32 input adapter, uses a scale-free
balanced-FP32 max-floor normalizer, transposes once to
`min(r,c) x max(r,c)`, executes exactly five fixed thick-Horner stages with
balanced FP32 dot products, stores every boundary as two BF16 terms, and uses
FP32 repair/output arithmetic.  It excludes fused or reassociated execution.

For the seven locked representative dimensions, the claimed common values
are

\[
A_{r,c}=\frac{102465557}{549755813888},\qquad
q_{r,c}=\frac{137425214491}{137438953472}<1.
\]

The largest claimed table values are

\[
B_{r,c}=\frac{2179083213031}{1099511627776},\qquad
\limsup_t(f(W_t)-f_\star)
\le\frac{798350562999}{1099511627776}.
\]

## Reviewer checklist

### 1. Scale-free normalizer branches

- [ ] Confirm the identity
  `N(s)=z/max(1/sigma,||z||F)` with
  `sigma=max(1,maxabs(s))` and `z=s/sigma`.
- [ ] Check that the exact denominator is at least one on both `sigma=1` and
  `sigma>1` branches.
- [ ] Verify the separate norm-controlled, floor-controlled with
  `||z||F>=1/2`, and small-floor analyses.  In particular, ensure no relative
  rounding model is used at zero or for a subnormal-scale exact norm.
- [ ] Recompute the squared-norm, square-root, division, and subnormal-error
  propagation leading to the locked denominator bound `1/100000` and output
  bound `1/90000` for every accepted shape.
- [ ] Check that the scaled implementation cannot overflow before the
  branchwise error conclusions are invoked.

Reviewer notes:

> Pending.

### 2. Balanced pairwise gamma and crumb

- [ ] For a dot of length `k`, verify the maximum path has one rounded product
  plus `ceil(log2(k))` rounded additions and hence
  `gamma_k=L_k*u/(1-L_k*u)` with `L_k=1+ceil(log2(k))`.
- [ ] Verify that carrying an unpaired value unchanged does not increase this
  maximum depth.
- [ ] Re-derive the absolute underflow crumb
  `ceil(sqrt(o))*2*k*tau/(1-L_k*u)` for `o` output entries.  Check the count of
  product and addition crumbs and their propagation along a balanced path.
- [ ] Confirm every use of `gamma_d`, `gamma_p`, and the corresponding output
  dimension in the Gram, square-product, and final-product bounds.
- [ ] Confirm `rc<=2^52` keeps every relevant reduction in the required gamma
  regime.

Reviewer notes:

> Pending.

### 3. Two-term BF16 cancellation without Sterbenz

- [ ] Starting from
  `high=RN_b(Y)`, `residual=fl32(Y-fp32(high))`,
  `low=RN_b(residual)`, and
  `X=fl32(fp32(high)+fp32(low))`, derive the entrywise error without assuming
  the subtraction or reconstruction is exact.
- [ ] Verify that the first BF16 high error cancels algebraically and that the
  remaining FP32 subtraction, low BF16 cast, and FP32 reconstruction terms
  produce
  `||X-Y||F <= omega||Y||F+ceil(sqrt(rc))*chi`.
- [ ] Recompute
  `omega=282587406795009/18446744073709551616` and
  `chi=72341289546351105/1569275433846670190958947355801916604025588861116008628224`.
- [ ] Check gradual-underflow crumbs and verify that no line silently invokes
  Sterbenz's lemma, exact residual representation, or exact high-plus-low
  addition.
- [ ] Confirm the rank-71 one-term and rank-4,656,751 compensated
  boundary-only slope gates are characterized as sufficient normwise proof
  gates, not kernel impossibility theorems.

Reviewer notes:

> Pending.

### 4. Full-matrix spectral bounds

- [ ] Independently verify `0<=q(x)<121/100` on `[0,5/4]` and the locked
  full spectral-map Lipschitz bound `3000459/512000` on the spectral tube.
- [ ] Check all diagonal and off-diagonal singular-vector modes required for a
  rectangular full-matrix Lipschitz statement; a scalar or diagonal-only
  derivative argument is insufficient.
- [ ] Verify the use of `||AB||F<=||A||2||B||F`, the Gram spectral/Frobenius
  envelopes, and `b^2/(4c)` in the thick-Horner product bound.
- [ ] Confirm transpose-once orientation preserves the Frobenius error and the
  exact-real spectral map.

Reviewer notes:

> Pending.

### 5. Five-stage recurrence and outward grid

- [ ] Replay every line of equations (P9.13)--(P9.15) from the locked operation
  graph; check coefficient-representation errors, scalar rounding, diagonal
  additions, matrix products, boundary storage, and ideal-map propagation.
- [ ] Confirm every named upper quantity is rounded upward by
  `ceil(2^40*x)/2^40`, and every lower capacity is rounded downward.
- [ ] Verify that all five **inputs** satisfy the strict `S_j<5/4` tube gate
  before the corresponding polynomial range and Lipschitz facts are used.
- [ ] Recompute the seven representative shape audits and their exact
  `(A,B,q,D,G)` fields.
- [ ] Confirm the `4608 x 18432` negative control fails only the
  stage-five-input tube gate, with
  `S_4=10753627185/8589934592>5/4`, and that this is reported only as a
  sufficient-envelope failure.

Reviewer notes:

> Pending.

### 6. Repair shell and real-input adapter

- [ ] Re-derive the binary32 repair-multiply/final-add slope and intercept from
  `rho32`, including coefficient mismatch and subnormal crumbs.
- [ ] Verify the ideal repaired-map Lipschitz constant
  `K_R=336372400608849/260261360000` is global on the max-floor domain and is
  applied in the correct direction to absorb `C32(s)-s`.
- [ ] Check the arbitrary-real input rounding inequalities and confirm that
  monotone rounding preserves the inclusive `2^116` max-absolute guard.
- [ ] Recompute the exact common slope
  `102465557/549755813888` and every shape-dependent intercept.

Reviewer notes:

> Pending.

### 7. Overflow and finite-range guards

- [ ] Confirm the normalizer bound covers the complete balanced square-sum,
  not merely its final rounded value.
- [ ] Check that the stage envelope bounds absolute partial-dot sums and
  pre-add operands, and that the half-maximum tests leave sufficient room for
  the following FP32 or BF16 rounding operation.
- [ ] Check the final repair multiplication and addition at input magnitude
  `2^116` against the FP32 maximum finite value.
- [ ] Verify the explicit serial all-ones obstruction: beyond `2^24` unit
  squares, the RNE FP32 serial accumulator sticks at `2^24`; at
  `4096 x 11008`, the returned squared singular value is `43/16`.
- [ ] Confirm that this executable witness is scoped only to the specified
  serial reduction and does not overclaim against balanced or vendor-specific
  reductions.

Reviewer notes:

> Pending.

### 8. P7 affine absorption

- [ ] Starting from P7's pathwise inequality and
  `||s||F^2<=(1655544025/2600084)V`, apply Young's inequality to
  `(A||s||F+B)^2` with the locked integer `theta=3006`.
- [ ] Recompute the exact outward-rounded `q_(r,c)` and `D_(r,c)`, and verify
  `q_(r,c)<1` for each accepted representative shape.
- [ ] Verify the storage-to-function conversion and each exact ultimate
  function-gap bound.
- [ ] Check `D_(r,c)<=(1-q_(r,c))H_safe` with
  `H_safe=2^232/(1655544025/2600084)`, establishing the operator-input range
  invariant when gradient noise is zero.
- [ ] Confirm the theorem does not reuse this storage-only guard under noisy
  gradients, claim full-state ISS, or absorb FP32 EMA/Nesterov/parameter
  rounding into the post-operator port.

Reviewer notes:

> Pending.

## Required review record

The reviewer should add their name or stable anonymous identifier, date,
reviewed Git commit, independent calculation environment, and one of:

- approved without correction;
- approved after listed corrections; or
- not approved, with the first failed checklist item and a reproducible
  discrepancy.

Until that record is present and every relevant checkbox is resolved, the P9
human proof audit remains pending.
