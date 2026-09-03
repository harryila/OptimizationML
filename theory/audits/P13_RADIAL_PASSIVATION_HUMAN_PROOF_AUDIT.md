# P13 radial-passivation human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. No unchecked item below may be cited as completed
human review. Automated exact generation, independent machine reconstruction,
and tests do not constitute a human proof audit.

The theorem under review is `theory/radial_passivation_tradeoff.md`. The
theorem-facing module is expected at
`src/passive_muon/radial_passivation_tradeoff.py`; the canonical exact artifact
is expected at
`results/summaries/radial_passivation_tradeoff_certificate.json`; and the
independent replay entry point is expected at
`scripts/reconstruct_radial_passivation_tradeoff.py`.

## Fixed claims to audit

For every finite real matrix shape and every `epsilon>0`, define the exact-real
additive-normalized map

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right)
\]

using the exact Jordan coefficients and five stages locked in P12. With

\[
\begin{aligned}
a&=-199437/1250,\\
\Gamma&=-41528474059081/260261360000,\\
U&=6602082433275499863/41641817600000000,\\
z_0&=63/9937,
\end{aligned}
\]

define

\[
\widehat d(z)=
\begin{cases}
U,&z\le z_0,\\
-(a+\Gamma z)/(1+z)^2,&z\ge z_0,
\end{cases}
\qquad
p(z)=\int_0^z\widehat d(u)\,du
\]

and

\[
G_\epsilon(0)=0,
\qquad
G_\epsilon(M)=p(\lVert M\rVert_F/\epsilon)
                 M/\lVert M\rVert_F.
\]

The fixed main claim is that `E_(h,epsilon)+G_epsilon` is globally monotone
in every finite shape. The fixed tradeoff claims are

\[
\operatorname{Lip}(G_\epsilon)=U/\epsilon
\]

and, for every shape containing the P12 rank-two pair, every globally
Lipschitz correction `C` passivating that pair satisfies

\[
\operatorname{Lip}(C)\ge d_{\rm pair}/\epsilon
>158.1172496/\epsilon.
\]

At `epsilon=10^-7`, the fixed scalar-quadratic control claims that the pinned
`beta=19/20` EMA/Nesterov recurrence is locally unstable at `eta=1/32000`
for the correction alone and for the complete repaired map. A separate exact
one-step control starts at `m0=0`, `w0=1/389025000` and increases the quadratic
objective by a factor greater than `23,325,554`.

All claims concern a continuous exact-real surrogate and explicitly exclude
literal BF16 Muon.

## Reviewer checklist

### 1. Frozen P12 premises and constants

- [ ] Confirm the P13 operator uses exactly the P12 coefficient fractions,
  stage count, additive-denominator placement, and condition `epsilon>0`.
- [ ] Match `a`, `Gamma`, and `U` to the active fourth row of the frozen P12
  radius-band artifact.
- [ ] Check `t0=63/10000` and the exact conversion
  `z0=t0/(1-t0)=63/9937`.
- [ ] Verify that P13 consumes P12's full-rectangular pointwise derivative
  result rather than treating a diagonal computation as a matrix theorem.
- [ ] Confirm the rank-two lower witness and its hash are unchanged.

Reviewer notes:

> Pending.

### 2. Pointwise radial majorant

- [ ] Substitute `t=z/(1+z)` into P12's fourth-band lower Jacobian bound and
  recover `(a+Gamma*z)/(epsilon*(1+z)^2)`.
- [ ] Verify that using the global P12 upper `U` before `z0` covers all first
  three bands.
- [ ] Check exactly that the two definitions of `d_hat` agree at `z0`.
- [ ] Recompute `2a-Gamma=-41520717707831/260261360000<0`.
- [ ] Confirm `d_hat` is positive, continuous, and nonincreasing on the whole
  nonnegative line.
- [ ] Confirm `d_hat/epsilon` is only a certified pointwise deficit majorant,
  not a claim of equality with the exact local deficit.

Reviewer notes:

> Pending.

### 3. Closed-form primitive

- [ ] Integrate `-(a+Gamma*z)/(1+z)^2` independently.
- [ ] Verify the constant of integration makes `p` continuous at `z0` and
  that `p'(z)=d_hat(z)` on both branches.
- [ ] Recompute `U*z0`, `a-Gamma`, and the simplified logarithm argument in
  the theorem note.
- [ ] Check `p(z)=-Gamma*log(z)+O(1)` and that `-Gamma>0`.
- [ ] Verify the construction defines `G_epsilon(0)=0` separately rather than
  dividing by the zero norm.

Reviewer notes:

> Pending.

### 4. Full-matrix radial derivative

- [ ] Regard the complete matrix space as a Frobenius Hilbert space and
  independently derive the radial projector formula for `DG_epsilon`.
- [ ] Verify the radial eigenvalue is `d_hat(z)/epsilon` and every tangential
  eigenvalue is `p(z)/(epsilon*z)`.
- [ ] Use monotonicity of `d_hat` to prove
  `d_hat(z)<=p(z)/z<=U`.
- [ ] Check that `G_epsilon=(U/epsilon)M` throughout `z<=z0`, including at the
  origin.
- [ ] Verify Frechet differentiability at zero, differentiability across
  `z=z0`, and the exact global Lipschitz constant `U/epsilon`.
- [ ] Confirm all Frobenius tangential directions are included for arbitrary
  rectangular shape.

Reviewer notes:

> Pending.

### 5. Global monotonicity theorem

- [ ] Add the P12 symmetric-Jacobian lower bound to the radial repair's
  derivative lower bound and obtain a positive-semidefinite sum.
- [ ] Check the origin and the joining sphere separately.
- [ ] Integrate along an arbitrary matrix line segment, including segments
  through zero or tangent to the joining sphere.
- [ ] Confirm the resulting claim is global pairwise monotonicity in every
  finite shape.
- [ ] Verify the proof does not claim strong monotonicity or identify the
  exact residual passivity margin.

Reviewer notes:

> Pending.

### 6. Arb magnitude evaluations

- [ ] Recompute `||G_epsilon(M)||_F=p(||M||_F/epsilon)`.
- [ ] At `epsilon=10^-7` and norm one, independently enclose the logarithmic
  expression inside
  `[2571.857826470212145, 2571.857826470212146]`.
- [ ] Verify the exact P11 guard is
  `13872266672489/549755813888`, not the rounded `25.2335` display.
- [ ] At that exact guard, independently enclose the magnitude inside
  `[3086.959580254301425, 3086.959580254301426]`.
- [ ] Recompute the corresponding constant-conductance magnitudes and check
  that all displayed intervals are outward, not nearest-rounded assertions.
- [ ] Confirm these evaluations do not claim observed BF16 magnitudes or
  training-trajectory bounds.

Reviewer notes:

> Pending.

### 7. Universal Lipschitz obstruction

- [ ] Independently scale the locked P12 pair by `epsilon` and recover its
  exact pair deficit `d_pair/epsilon`.
- [ ] Starting only from monotonicity on that pair, derive the lower inner
  product required of the correction.
- [ ] Apply Cauchy--Schwarz and the correction's Lipschitz inequality with the
  correct direction of every sign.
- [ ] Confirm the conclusion is
  `Lip(C)>=d_pair/epsilon>98823281/(625000*epsilon)`; the first comparison is
  non-strict and the rational displayed comparison is strict.
- [ ] Verify that no radiality, oddness, differentiability, or linearity of
  `C` enters the proof.
- [ ] Check zero-padding scopes this lower bound only to shapes with
  `min(m,n)>=2`.
- [ ] Recompute the `0.270230608278...%` near-optimal stiffness comparison and
  confirm it is not described as exact minimality.

Reviewer notes:

> Pending.

### 8. Scalar pinned-loop Jury boundary

- [ ] Re-derive the two-state Jacobian for `f(w)=w^2/2` in state order
  `(w,m)` under the exact EMA/Nesterov update order.
- [ ] Check the trace and determinant and all three strict Jury expressions.
- [ ] Verify that only
  `2(1+beta)-theta*(1-beta)*(1+2beta)>0` supplies the positive-step upper
  boundary.
- [ ] Reproduce the correction-only critical step
  `1082687257600/63820130188329832009` at `epsilon=10^-7`.
- [ ] Reproduce its exact negative boundary expression at `eta=1/32000` and
  verify the characteristic polynomial has a real root below `-1`.
- [ ] Check the exact origin derivative of the complete repaired operator,
  the full critical-step fraction, and its separate negative boundary
  expression.
- [ ] Confirm the note says that the original additive-normalized operator's
  positive origin gain also contributes to the full-operator stiffness.
- [ ] Verify `f(w)=w^2/2` belongs to the earlier smooth/PL class.

Reviewer notes:

> Pending.

### 9. Exact one-step negative control

- [ ] Starting from `m0=0`, `w0=1/389025000`, reproduce
  `s1=epsilon/399` and normalized input `1/400`.
- [ ] Verify `1/399<z0`, so the repair is exactly linear at this signal.
- [ ] Prove the Jordan response is positive from the exact negative
  discriminant `-2594691/500000`, without relying on a float evaluation.
- [ ] Recompute
  `chi=257481214897744494657/53301526528000000` and prove `chi>2` exactly.
- [ ] Check that discarding the positive Jordan response gives the inequality
  in the safe direction.
- [ ] Verify `w1<-(chi-1)w0` and the strict objective growth factor greater
  than `23,325,554`.
- [ ] Confirm one-step growth is not overclaimed as divergence from every
  initialization; local instability is separately supplied by the Jury
  calculation.

Reviewer notes:

> Pending.

### 10. Independent reconstruction and negative controls

- [ ] Run the primary certificate generator at both locked Arb precisions and
  compare the logarithm interval signatures.
- [ ] Run the independent reconstruction and verify it does not import the
  theorem-facing module or primary generator.
- [ ] Reconstruct `d_hat`, the primitive coefficients, the pair lower bound,
  both Lipschitz endpoints, and every Jury fraction independently.
- [ ] Verify an exact boundary control sets `eta=eta_crit` and places an
  eigenvalue at `-1`.
- [ ] Verify an exact interior control, for example `eta=eta_crit/2`, makes
  every Jury expression strictly positive.
- [ ] Confirm corrupted constants, a sign-reversed correction, or a step just
  beyond the boundary are rejected by tests.
- [ ] Check that source hashes and the frozen P12 prerequisite artifact are
  bound in the canonical result.

Reviewer notes:

> Pending.

### 11. Exact-real/BF16 separation and claim discipline

- [ ] Verify the pinned upstream revision
  `f98f1cacc0263b04290753e32be8d498c1efc806` and its recorded formula/cast
  order against the P12 provenance packet.
- [ ] Confirm the theorem uses a continuous exact-real map, exact rational
  coefficients, and exact additive-epsilon placement.
- [ ] Confirm no Jacobian, radial correction, monotonicity, Lipschitz, or
  stability conclusion is transferred to literal BF16 Muon.
- [ ] Check the note does not call this a theorem for upstream, stochastic,
  weight-decayed, aspect-scaled, or neural-network training.
- [ ] Check the output-magnitude improvement is distinguished from the
  unavoidable differential-stiffness obstruction.
- [ ] Confirm “near-minimal” is always qualified as a witness-relative
  Lipschitz statement on shapes containing the rank-two pair.

Reviewer notes:

> Pending.

## Reviewer sign-off

Reviewer name:

> Pending.

Affiliation or contact:

> Pending.

Review date and timezone:

> Pending.

Reviewed Git commit:

> Pending.

Reviewed canonical artifact SHA-256:

> Pending.

Overall verdict (`approve`, `approve with listed corrections`, or `reject`):

> Pending.

Required corrections or residual concerns:

> Pending.
