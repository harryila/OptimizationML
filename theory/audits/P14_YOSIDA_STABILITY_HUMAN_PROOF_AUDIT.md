# P14 Yosida-stability human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. Automated rational replay, independent machine
reconstruction, and tests do not constitute a human proof audit.

The theorem under review is `theory/yosida_stability_certificate.md`. The
theorem-facing module is expected at `src/passive_muon/yosida_stability.py`,
the canonical artifact at
`results/summaries/yosida_stability_certificate.json`, and the independent
entry point at `scripts/reconstruct_yosida_stability.py`.

## Fixed claim

For P13's exact-real full-matrix monotone, continuous, zero-preserving map
`A_epsilon`, for every `epsilon>0`, define

\[
B=A_\epsilon+1000I,
\qquad
Y=(1000)(I-(I+B/1000)^{-1}).
\]

The first claim is that the resolvent exists uniquely on every finite matrix
space and that `Y` is zero preserving, `1/1000`-cocoercive, at most
`1000`-Lipschitz, and at least `500`-strongly monotone.

The second claim concerns the exact pinned EMA/Nesterov recurrence with
`beta=19/20`, `eta=1/32000`, and `Y` in the update. For every differentiable,
globally `10`-smooth objective satisfying the global PL inequality with
constant `1`, the locked exact storage proves

\[
\mathcal V_{t+1}\le\frac{249001}{250000}\mathcal V_t,
\qquad D_{14}=0.
\]

The theorem is exact-real and implicit. It does not claim that a finite-step
solver or BF16 kernel evaluates the resolvent exactly.

## Reviewer checklist

### 1. P13 premise and scope

- [ ] Confirm `A_epsilon` is exactly the P13 map
  `E_(h,epsilon)+G_epsilon`, with the same five Jordan stages, coefficient
  fractions `6889/2000`, `-191/40`, and `4063/2000`, additive normalization
  `M/(||M||_F+epsilon)`, radial correction, and arbitrary `epsilon>0`;
  `epsilon=10^-7` is pinned only for the reported scalar controls.
- [ ] Confirm P13 proves global monotonicity on the complete Frobenius matrix
  space, rather than only on diagonal or sampled directions.
- [ ] Verify continuity, full domain, and `A_epsilon(0)=0`.
- [ ] Confirm P13's pending human review is disclosed and is not treated as
  completed by P14.

Reviewer notes:

> Pending.

### 2. Shifted operator and resolvent existence

- [ ] Derive `1000`-strong monotonicity of `B=A_epsilon+1000I`.
- [ ] Check that `I+lambda B`, with `lambda=1/1000`, is continuous,
  coercive, and `2`-strongly monotone.
- [ ] Supply or verify the finite-dimensional surjectivity argument.
- [ ] Verify strict monotonicity gives uniqueness, so the inverse is globally
  defined and single valued.
- [ ] Check that the proof establishes a mathematical resolvent without
  asserting a method or complexity for computing it.

Reviewer notes:

> Pending.

### 3. Resolvent and Yosida inequalities

- [ ] Starting from `x_i=u_i+lambda*b_i`, derive firm nonexpansiveness of the
  resolvent using monotonicity of `B`.
- [ ] Derive the sharper resolvent contraction
  `Lip(J)<=1/(1+lambda*mu)=1/2`.
- [ ] Derive `lambda`-cocoercivity of the Yosida operator.
- [ ] Derive `Lip(Y)<=1/lambda=1000` without assuming differentiability.
- [ ] Derive strong monotonicity
  `mu/(1+lambda*mu)=500`, checking the Cauchy--Schwarz direction.
- [ ] Reproduce the pulled-back identity
  `<v-m*x,M*x-v>=(<v,u>-mu*||u||^2)/(lambda*(1+lambda*mu))`
  with `x=u+lambda*v`.
- [ ] Confirm `J(0)=Y(0)=0` follows from `B(0)=0` and uniqueness.

Reviewer notes:

> Pending.

### 4. Joint sector and centered residual reduction

- [ ] Starting from the pulled-back **joint** sector identity, independently
  verify

  \[
  \langle\Delta Y-500\Delta s,1000\Delta s-\Delta Y\rangle_F
  =250^2\|\Delta s\|_F^2-\|\Delta Y-750\Delta s\|_F^2.
  \]

- [ ] Deduce, without combining only the separate strong/cocoercive bounds,

  \[
  \|\Delta Y-750\Delta s\|_F\le250\|\Delta s\|_F.
  \]

- [ ] Confirm the result applies to arbitrary pairs, not only increments from
  zero.
- [ ] Verify `Y=750I+E_Y`, `E_Y(0)=0`, and `Lip(E_Y)<=250`.
- [ ] Confirm no step interprets the bounds as Loewner ordering of a
  nonsymmetric Jacobian.
- [ ] Confirm no gradient/cyclic-monotonicity property of `Y` is assumed.

Reviewer notes:

> Pending.

### 5. Pinned recurrence and normalization

- [ ] Check the EMA and Nesterov ordering gives
  `s_(t+1)=beta^2*m_t+(1-beta^2)*g_t`.
- [ ] Verify the normalized coordinates and signal
  `p=beta^2*z+(1-beta^2)*u`.
- [ ] Recompute
  `alpha=eta*gamma*L=15/64` and `d/gamma=1/3`.
- [ ] Verify the step selector is `-alpha*(p+v/3)`.
- [ ] Check the residual IQC is
  `||p||_F^2-||v||_F^2>=0` with the stated normalization.
- [ ] Confirm every lifted scalar matrix is tensored with the identity, making
  the result independent of matrix shape.

Reviewer notes:

> Pending.

### 6. Smooth nonconvex interpolation and PL supply

- [ ] Re-derive both directed `(-1,1)` smooth interpolation inequalities
  between `W_t` and `W_(t+1)`.
- [ ] Confirm negative curvature is allowed and convexity is nowhere assumed.
- [ ] Check the PL convention
  `0.5*||grad f||^2 >= f-f_star` for constant `1`.
- [ ] Verify the PL supply is applied at the next iterate with normalized
  condition ratio `1/10`.
- [ ] Confirm a finite infimum and global, not local, PL inequality are
  assumed.

Reviewer notes:

> Pending.

### 7. Exact storage and multipliers

- [ ] Recompute positive definiteness of

  \[
  P=10^{-6}\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix}.
  \]

- [ ] Verify `c_F=313243/10^6`, reverse interpolation multiplier
  `622414/10^6`, and residual multiplier `27530/10^6`.
- [ ] Derive the remaining interpolation and PL multipliers from `c_F` and
  `q=249001/250000`; check every multiplier is nonnegative.
- [ ] Verify the weighted function coefficients are exactly
  `(c_F*q,-c_F)`.
- [ ] Confirm storage normalization has not been changed between generation
  and reconstruction.

Reviewer notes:

> Pending.

### 8. Exact 4-by-4 LMI

- [ ] Independently rebuild the transition and selection matrices in
  `chi=(z,u,v,u_next)` order.
- [ ] Check every supply enters the LMI with the correct sign.
- [ ] Recompute the complete rational LMI from the published fractions.
- [ ] Use exact leading principal minors of `-LMI` to establish strict
  negative definiteness.
- [ ] Verify `tau=499/500`, `q=tau^2=249001/250000<1`.
- [ ] Confirm the final inequality has `D14=0`, with no omitted constant or
  solve-error term.

Reviewer notes:

> Pending.

### 9. Convergence consequences

- [ ] Verify `V_t<=q^t V_0` and the resulting global objective-gap bound.
- [ ] Check that positive definiteness of `P` yields geometric decay of
  normalized momentum and gradient.
- [ ] Verify the operator-output/update sequence is absolutely summable.
- [ ] Check the argument that the iterate limit is a global minimizer.
- [ ] Confirm no unique minimizer or arbitrary-pair trajectory contraction is
  claimed.

Reviewer notes:

> Pending.

### 10. Exact positive and negative controls

- [ ] Match the explicit P13 origin Jury failure fraction to the frozen P13
  artifact.
- [ ] Verify adding `1000I` cannot repair the explicit `lambda=0` Jury
  failure at the fixed positive step.
- [ ] Recompute the selected Yosida worst-gain scalar margin
  `2467/640>0` at curvature `10`.
- [ ] At `lambda=1/100000`, recompute the upper gain `100000` and the exact
  negative worst-curvature Jury margin `-101/160`.
- [ ] At that same finite lambda, independently evaluate the actual shifted
  P13 origin slope and its strict negative curvature-`10` Jury fraction.
- [ ] Confirm the scalar pass is presented only as a boundary control; the
  nonlinear full-class claim comes from the exact LMI.
- [ ] Verify no failed numerical solver status is presented as an
  impossibility result.
- [ ] Check the abstract sector witnesses: `A=0` at the lower endpoint,
  positive scalar gain approaching the upper endpoint, and `A=1000S` with
  `S^T=-S`, `S^T*S=I`, giving `Y=600I+200S` and centered norm `250`.
- [ ] Confirm the witnesses establish uniform sharpness over the abstract
  monotone class, not attainment by the locked P13 map.

Reviewer notes:

> Pending.

### 11. Independent reconstruction and provenance

- [ ] Run the primary generator and inspect every exact fraction and matrix.
- [ ] Run the independent reconstruction and confirm it imports no project or
  third-party numerical package.
- [ ] Verify the reconstruction builds all mathematics before reading the
  canonical artifact.
- [ ] Confirm corrupted parameters, matrices, or rate fields are rejected.
- [ ] Verify source files are hashed at the clean source commit recorded in
  the artifact, using full Git history rather than mutable worktree contents.
- [ ] Match the P13 canonical hash, source commit, artifact commit, and
  checkpoint tag to their distinct provenance roles.
- [ ] Confirm the seed is null and no sampled evidence is used as proof.

Reviewer notes:

> Pending.

### 12. Scope exclusions

- [ ] Confirm the result is for exact real arithmetic and a mathematically
  exact resolvent.
- [ ] Confirm no claim is made for finite iteration, an inexact proximal
  solve, BF16, weight decay, aspect scaling, stochastic gradients, or a full
  neural-network training system.
- [ ] Confirm literal upstream Muon does not contain this Yosida operator.
- [ ] Verify `D14=0` is not described as robustness to implementation error.
- [ ] Confirm solve-error ports and nonzero `D14` are explicitly deferred to
  P15.

Reviewer notes:

> Pending.

## Required reviewer record

- Reviewer name: _pending_
- Affiliation or role: _pending_
- Date: _pending_
- Commit or tag reviewed: _pending_
- Primary generator command: _pending_
- Independent reconstruction command: _pending_
- Verdict: _pending_
- Required corrections: _pending_
- Signature or durable approval reference: _pending_

Until these fields and the checklist are completed by an independent human,
the repository must continue to describe the P14 human audit as pending.
