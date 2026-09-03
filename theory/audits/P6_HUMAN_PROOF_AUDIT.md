# P6 independent human proof-audit packet

Status: **pending external human review**. This packet identifies the exact
claims and proof transitions to audit. It is not a completed audit and must not
be represented as one until the reviewer section is signed.

## Immutable revisions

- certificate generation/source commit:
  `a8f650f6c60dcbc5d2f83647fd367348f4c67548`;
- final P6 checkpoint commit:
  `ef88d8f5b26148af0ec1ca70b506048938bf9bef`;
- annotated checkpoint tag: `p6-checkpoint`.

Primary proof: `theory/pl_convergence_certificate.md`. Canonical exact
artifact: `results/summaries/pl_convergence_certificate.json`. Independent
code-path replay: `scripts/reconstruct_pl_convergence.py`.

## Claim under review

For every finite real matrix shape and every fixed differentiable objective
with finite infimum, globally `10`-Lipschitz gradient, and global PL constant
`1`, the displayed deterministic exact-real repaired max-floor five-step
Jordan EMA/Nesterov loop at `eta=1/32000` obeys

\[
f(W_t)-f_\star\le C
\left(\frac{399960001}{400000000}\right)^t,
\qquad m_t\to0.
\]

Its geometrically summable updates imply convergence of each trajectory to
some trajectory-dependent global minimizer. The claim is not arbitrary-pair
contraction and does not assert a unique minimizer.

## Priority audit: directed nonconvex interpolation

Check the following step without relying on the certificate generator. For a
differentiable one-smooth `F`, set

\[
H(x)=F(x)+\frac12\lVert x\rVert^2.
\]

Verify that `H` is convex and two-smooth, apply the convex smooth
interpolation inequality to the ordered pair `(i,j)`, and expand it to

\[
F_i-F_j-\langle u_j,x_i-x_j\rangle
-\frac{\lVert u_i-u_j\rVert^2
+2\langle x_i-x_j,u_i-u_j\rangle
-\lVert x_i-x_j\rVert^2}{4}\ge0.
\]

Audit both directions `(current,next)` and `(next,current)`. In particular,
check every sign after substituting `x_current=0`, `x_next=d`, and confirm that
no convexity of `F` or fixed Hessian orientation is used.

## Complete checklist

- [ ] Confirm the objective normalization
      `F=(f-f_star)/L`, `u=grad(f)/L`, `z=m/L`, and normalized PL constant
      `k=1/10`.
- [ ] Confirm the pinned EMA/Nesterov ordering and the selectors for
      `z_next`, `p`, and `d`.
- [ ] Confirm the full-matrix residual supply follows from the certified
      decomposition `R(s)=gamma s+E(s)`, `E(0)=0`, and
      `Lip(E)<=K_E`.
- [ ] Independently derive both directed nonconvex interpolation supplies.
- [ ] Confirm the next-state PL supply and its direction.
- [ ] Confirm exact cancellation of both function-value coefficients.
- [ ] Confirm positive definiteness of `P` and negative definiteness of the
      `4 x 4` LMI from the exact rational leading principal minors.
- [ ] Confirm that the supplies yield `V_next<=q V`, rather than the reverse
      inequality.
- [ ] Confirm that storage coercivity gives function-gap, gradient, and
      momentum decay.
- [ ] Check the summability argument for `W_next-W` and that its limit is a
      global minimizer without asserting uniqueness.
- [ ] Confirm all exclusions: stochastic gradients, BF16, additive epsilon,
      exact current normalization, weight decay, aspect scaling, and complete
      neural-network training.

## Reviewer record

- reviewer name:
- affiliation or relevant expertise:
- revision audited:
- date:
- verdict (`pass`, `pass with corrections`, or `fail`):
- corrections or comments:
- signature or auditable acknowledgment:
