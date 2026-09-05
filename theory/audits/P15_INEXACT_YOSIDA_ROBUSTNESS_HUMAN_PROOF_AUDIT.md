# P15 inexact-Yosida robustness human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. Exact rational replay, an independently implemented
machine reconstruction, and unit tests do not constitute a human proof audit.

The theorem under review is
`theory/inexact_yosida_robustness_certificate.md`. The theorem-facing module
is `src/passive_muon/inexact_yosida_robustness.py`, the canonical artifact is
`results/summaries/inexact_yosida_robustness_certificate.json`, and the
independent entry point is
`scripts/reconstruct_inexact_yosida_robustness.py`.

## Fixed claim

For P14's shifted P13 operator

\[
B=A_\epsilon+1000I,
\qquad J=(I+B/1000)^{-1},
\qquad Y=1000(I-J),
\]

an oracle returns `u_hat` with graph residual

\[
r=s-\widehat u-B(\widehat u)/1000
\]

and deploys `Y_hat(s)=1000*(s-u_hat)`. If

\[
\|r\|_F\le\frac1{250}\|s\|_F+\bar r
\]

then the exact pinned EMA/Nesterov recurrence at `beta=19/20` and
`eta=1/32000`, for every differentiable globally `10`-smooth objective with
finite infimum satisfying the global PL inequality with constant `1`, obeys

\[
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t+\frac52\bar r^2.
\]

The theorem holds for every `epsilon>0` and every fixed finite real matrix
shape. It concerns an exact-real residual oracle, not an implemented solver
or a finite-precision residual calculation.

## Reviewer checklist

### 1. Inherited P13--P14 premise

- [ ] Confirm `A_epsilon` is P13's exact-real additive-epsilon Jordan map plus
  radial passivator, with normalization `M/(||M||_F+epsilon)`, five stages,
  and coefficient fractions `6889/2000`, `-191/40`, and `4063/2000`.
- [ ] Verify P13 supplies continuity, full domain, zero preservation, and
  full-matrix monotonicity for every `epsilon>0`.
- [ ] Verify P14's shift `B=A_epsilon+1000I` is `1000`-strongly monotone and
  its resolvent exists uniquely on every finite matrix space.
- [ ] Re-derive `Lip(J)<=1/2` and P14's exact joint sector for `Y`.
- [ ] Confirm pending human review of P13 and P14 is disclosed rather than
  silently treated as complete.

Reviewer notes:

> Pending.

### 2. Residual and deployed-output convention

- [ ] Confirm the graph residual is exactly
  `r=s-u_hat-lambda*B(u_hat)`, with `lambda=1/1000`.
- [ ] Confirm the deployed output is exactly
  `Y_hat=(s-u_hat)/lambda`, not `B(u_hat)`.
- [ ] Check that these two approximate outputs differ by `r/lambda`.
- [ ] Verify the residual is computable after an exact-real evaluation of
  `B(u_hat)` and that no finite-precision computability claim is smuggled in.
- [ ] Confirm `s-r=(I+lambda*B)(u_hat)` implies `u_hat=J(s-r)`.
- [ ] Derive the solution-error estimate
  `||u_hat-J(s)||_F<=||r||_F/2`.

Reviewer notes:

> Pending.

### 3. Sharp residual-to-output gain

- [ ] Derive
  `Y_hat(s)-Y(s)=(J(s)-J(s-r))/lambda` exactly.
- [ ] Use `Lip(J)<=1/(1+lambda*mu)` to derive
  `||Y_hat-Y||<=||r||/(lambda*(1+lambda*mu))=500||r||`.
- [ ] Verify the direction of every subtraction; in particular, do not use
  the different `1000*||r||` class bound appropriate to output `B(u_hat)`.
- [ ] Check that the abstract boundary `A=0`, `B=1000I` attains gain `500`.
- [ ] Confirm this sharpness is over the admissible monotone-operator class,
  not claimed as attainment by the specific P13 map.

Reviewer notes:

> Pending.

### 4. Stopping rule and error split

- [ ] Verify the locked rule
  `||r||<=kappa*||s||+rbar`, with `kappa=1/250` and `rbar>=0`.
- [ ] Recompute `500*kappa=2`.
- [ ] Check the pointwise collinear decomposition of the output error into
  terms bounded by `2||s||` and `500*rbar`, including the zero-bound case.
- [ ] Combine only the relative solve-error term with P14's centered residual
  to obtain radius `250+2=252`.
- [ ] Confirm the decomposition is used as a trajectory supply and is not
  misrepresented as an incremental Lipschitz property of an arbitrary oracle
  selection.
- [ ] Verify that the stopping condition is a posteriori and carries no
  iteration-count or solver-convergence guarantee.

Reviewer notes:

> Pending.

### 5. Pinned recurrence and objective class

- [ ] Check the EMA/Nesterov state and signal ordering, including
  `s_(t+1)=beta*m_(t+1)+(1-beta)*g_t`.
- [ ] Confirm `beta=19/20` and `eta=1/32000` are unchanged from P14.
- [ ] Confirm the theorem covers every differentiable globally `10`-smooth
  objective with finite infimum satisfying the **global** PL inequality with
  constant `1`.
- [ ] Verify convexity, strong convexity, and uniqueness are nowhere assumed.
- [ ] Confirm every lifted scalar matrix is tensored with the
  identity on the ambient Frobenius space.

Reviewer notes:

> Pending.

### 6. Exact storage and relative-error LMI

- [ ] Recompute positive definiteness of

  \[
  P=10^{-6}\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix}.
  \]

- [ ] Verify the function storage is `313243/10^6` and all P14
  directed-interpolation, PL, and residual multipliers are preserved exactly.
- [ ] Rebuild the normalized dynamics with centered radius `252`, rather than
  `250` or an incorrectly scaled physical value.
- [ ] Verify the exact `4 x 4` LMI is strictly negative definite at radius
  `252`, using exact leading principal minors of its negative.
- [ ] Check exact function-value cancellation at
  `tau=499/500`, `q15=tau^2=249001/250000`.

Reviewer notes:

> Pending.

### 7. Absolute-error port and 5-by-5 LMI

- [ ] Add the physical output-error coordinate `e_abs` with the exact sign
  induced by `W_+=W-eta*(750s+e_rel+e_abs)`.
- [ ] Independently rebuild the complete `5 x 5` rational LMI and verify
  strict negative definiteness with gain `gamma_abs=1/100000`.
- [ ] Confirm the port inequality is
  `V_+<=q15*V+(1/100000)||e_abs||^2`.
- [ ] Substitute `||e_abs||<=500*rbar` and recompute
  `(1/100000)*500^2=5/2=C15`.
- [ ] Confirm no dependence on matrix dimension or `epsilon` enters these
  constants.

Reviewer notes:

> Pending.

### 8. Ultimate-bound consequences

- [ ] Unroll the scalar recurrence and verify
  `C15/(1-q15)=625000/999`.
- [ ] From the function term in storage, independently derive
  `limsup(f-f*) <= 6250000000000/312929757*rbar^2`.
- [ ] Check the decimal `19972.533324787` only reports the exact fraction and
  does not replace it.
- [ ] At `rbar=0`, verify geometric decay of objective gap, gradient, and
  momentum and absolute summability of updates.
- [ ] Check the resulting convergence to some global minimizer without a
  uniqueness claim.
- [ ] For persistent `rbar>0`, confirm that no iterate-convergence claim is
  made.

Reviewer notes:

> Pending.

### 9. Exact controls

- [ ] Set `kappa=rbar=0` and confirm exact recovery of every appropriate P14
  operator, storage, multiplier, and rate fraction.
- [ ] At `kappa=3/500`, recompute centered radius `253` and verify rejection
  of the unchanged frozen-storage `4 x 4` certificate.
- [ ] Confirm that this rejection is not called instability or impossibility.
- [ ] For `A=0`, `B=1000I`, check that `u_hat=s` gives `r=-s`, `Y_hat=0`,
  and satisfies the `kappa=1`, `rbar=0` rule.
- [ ] Verify this supplies a real loose-tolerance stalling control at a
  nonstationary point without claiming it occurs for the locked P13 map.
- [ ] At the same abstract boundary, check that `kappa=2`, `r=-2s`, and
  `u_hat=3s/2` give `Y_hat=-500s` and that the scalar curvature-`1`
  characteristic has `p(1)=-1/1280<0`.
- [ ] Confirm no sampled pass is used as evidence for the global theorem.

Reviewer notes:

> Pending.

### 10. Independent reconstruction and provenance

- [ ] Run the primary generator and inspect every exact fraction, matrix,
  principal minor, stopping-rule field, and control.
- [ ] Run the reconstruction and verify it imports neither project modules
  nor third-party numerical packages.
- [ ] Confirm the reconstruction builds its mathematics before reading the
  canonical artifact and rejects corrupted rate, gain, or matrix fields.
- [ ] Verify frozen source hashes are read at the recorded source commit using
  full Git history, not from the mutable worktree.
- [ ] Distinguish P14's source commit, artifact commit, artifact hash, and
  annotated checkpoint tag, and do the same for P15's two-commit artifact.
- [ ] Confirm the seed is null and no sampling is part of the proof.

Reviewer notes:

> Pending.

### 11. Scope exclusions

- [ ] Confirm P15 specifies an exact-real residual oracle rather than an
  algorithm for producing `u_hat`.
- [ ] Confirm no convergence rate or iteration count is claimed for a
  fixed-point, Newton, Krylov, or other solve.
- [ ] Confirm residual and `B(u_hat)` evaluation are not rounded in the
  theorem.
- [ ] Confirm no BF16/FP32, accelerator, upstream parity, weight decay,
  aspect scaling, stochastic-gradient, or neural-network theorem is claimed.
- [ ] Verify the practical upstream provenance is not confused with presence
  of the proposed P13--P15 architecture in upstream code.

Reviewer notes:

> Pending.

## Sign-off

Reviewer name:

> Pending.

Affiliation or contact:

> Pending.

Date:

> Pending.

Verdict:

> Pending.
