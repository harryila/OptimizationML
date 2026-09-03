# P7 independent human proof-audit packet

Status: **pending external human review**. This packet identifies the exact
C11--C12 claims and proof transitions to audit. It is not a completed audit
and must not be represented as one until the reviewer record is signed.

## Immutable revision

- certificate and falsification source commit:
  `c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`;
- final P7 checkpoint commit:
  `30b55e53f50525ea980dc41f3201fdd160d4a75e`;
- annotated checkpoint tag: `p7-checkpoint`;
- annotated tag object:
  `7516ac9c4b021cd3c536f046303fb502bcf3f8b0`.

Primary proofs: `theory/pl_convergence_certificate.md` and
`theory/robust_dissipativity_certificate.md`. Canonical exact artifacts:
`results/summaries/pl_convergence_certificate.json` and
`results/summaries/robust_dissipativity_certificate.json`. Independent
code-path replays: `scripts/reconstruct_pl_convergence.py` and
`scripts/reconstruct_robust_dissipativity.py`.

## Claims under review

For every finite real matrix shape and every fixed differentiable objective
with finite infimum, globally `10`-Lipschitz gradient, and global PL constant
`1`, the disturbed repaired max-floor five-step Jordan EMA/Nesterov loop at
`beta=19/20` and `eta=1/32000` obeys, pathwise,

\[
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2.
\]

Here the same noisy gradient is used in both EMA/Nesterov occurrences, and
`e_t` is added after the repaired operator and before multiplication by
`eta`. Setting both disturbances to zero recovers C11. The claim is
input-to-storage/output stability, not full-state ISS in `W` and not a BF16 or
unrepaired-upstream theorem.

## Priority audit: disturbance placement and units

- [ ] Re-derive
      `m_next=beta*m+(1-beta)*(grad f(W)+xi)` and
      `s_next=beta*m_next+(1-beta)*(grad f(W)+xi)`, checking that one identical
      `xi` is reused.
- [ ] Check that `e` enters only through
      `W_next=W-eta*(R(s_next)+e)`.
- [ ] With `L=10`, `w=xi/L`, `h=e/(gamma*L)`, verify that the normalized input
      penalties convert exactly to `||xi||_F^2/2` and
      `||e||_F^2/2000000`.
- [ ] Confirm that both directed nonconvex interpolation supplies contain the
      true gradients, not the noisy measurements.

## Complete C11--C12 checklist

- [ ] Independently reconstruct the C11 directed smooth-nonconvex
      interpolation formula and its two signs.
- [ ] Confirm the global PL normalization and next-point PL supply.
- [ ] Confirm the repaired operator decomposition, full-matrix residual IQC,
      and pinned EMA/Nesterov selectors.
- [ ] Confirm exact function-value cancellation in the `4 x 4` and `6 x 6`
      LMIs.
- [ ] Confirm positive definiteness of the storage and negative definiteness
      of both LMIs from exact rational Sylvester minors.
- [ ] Confirm the pathwise inequality holds without stochastic independence or
      unbiasedness assumptions.
- [ ] Derive the bounded-input convolution and ultimate-neighborhood constants.
- [ ] Check that square-summable disturbances imply storage, objective-gap,
      momentum, and true-gradient convergence, but not necessarily iterate
      convergence.
- [ ] Verify the flat-minimizer harmonic-drift counterexample.
- [ ] Check that absolute summability is sufficient for iterate convergence to
      some trajectory-dependent global minimizer without uniqueness.
- [ ] Derive the conditional-second-moment expectation bound and confirm that
      unbiasedness is optional for that energy inequality.
- [ ] Confirm all exclusions: full-state ISS, persistent-noise almost-sure
      convergence, BF16 error bounds, additive epsilon, exact current
      normalization, weight decay, aspect scaling, and full neural training.

## Reviewer record

- reviewer name:
- affiliation or relevant expertise:
- revision audited:
- date:
- verdict (`pass`, `pass with corrections`, or `fail`):
- corrections or comments:
- signature or auditable acknowledgment:
