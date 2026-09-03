# P11 human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. No unchecked item below may be cited as completed
human review. The P10 audit also remains pending; reviewing P11 does not
retroactively approve P10.

The theorem under review is the exact two-port implementation-margin result in
`theory/implementation_margin_certificate.md`. The theorem-facing code is
`src/passive_muon/implementation_margin_certificate.py`; the canonical exact
artifact is
`results/summaries/implementation_margin_certificate.json`; and the
independent replay entry point is
`scripts/reconstruct_implementation_margin.py`.

## Fixed claim to audit

The claim concerns the P10 proof-reference optimizer at matrix shape
`4096 x 11008`. Its objective is any differentiable, globally `10`-smooth
function satisfying the global Polyak--Lojasiewicz inequality with constant
one, including nonconvex functions and nonunique minimizers. It uses
`beta=19/20` and `eta=1/32000`.

The exact target operator is

\[
R(s)=\mathcal H_{q^{\circ5}}
\!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\]

with no additive epsilon, exactly five stages,

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

The reference implementation is P9's balanced-FP32, two-term-BF16 kernel
inside P10's separately rounded FP32 EMA/Nesterov graph and three-word
compensated FP32 master.

The two new pathwise input contracts are

\[
\lVert\zeta_t\rVert_F\le a_g\sqrt{V_t}+b_g,
\qquad
\lVert\nu_t\rVert_F\le a_R\sqrt{V_t}+b_R.
\]

Here `zeta_t` is added to the exact gradient before the one final FP32 cast.
The deployed kernel output is a finite contiguous FP32 tensor, and `nu_t` is
its real difference from P9's finite contiguous FP32 reference output. No
separate `U9+nu` arithmetic operation is claimed.

For all disturbances satisfying those contracts, P11 claims

\[
V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).
\]

At zero added error it must reproduce every P10 envelope and

\[
q_{10}=\frac{549700907325}{549755813888},\qquad
D_{10}=\frac{2162331}{1099511627776}
\]

exactly. At

\[
a_g=b_g=\frac1{4096},\qquad
a_R=\frac1{128},\qquad b_R=\frac18,
\]

it claims

\[
q_{11}=\frac{274850515349}{274877906944},\qquad
D_{11}=\frac{1254603}{549755813888},
\]

and exact positive invariance slack
`53528587/549755813888` while all signal, output, step, and conditional master
guards close.

Reported maxima and frontiers are exact only on the nonnegative `2^-40`
budget grid. Adjacent rejected controls demonstrate failure of the sufficient
unit-storage invariance check, not actual dynamical instability.

## Reviewer checklist

### 1. Frozen dependencies and scope

- [ ] Confirm that P11 changes none of the P10 shape, objective class,
  `beta`, `eta`, normalizer, epsilon rule, polynomial coefficients, stage
  count, repair, P9 kernel, EMA/Nesterov graph, or compensated-master graph.
- [ ] Confirm the three P10 provenance roles are distinct and correct:
  theorem/source `2d62b566e5a34895470d4eeb4b76789add826cbe`, exact
  artifact `6e7ea000692a269cbd3766146eb7423733d18694`, and
  diagnostic/final checkpoint `3246972d44fc6e04c15cf4e205615acbb879df36`.
- [ ] Check that P9 and P10 exact artifacts replay before using their
  constants; a copied decimal must not replace an exact dependency field.
- [ ] Verify that the claimed matrix domain is only `4096 x 11008`, even
  though the P7 storage inequality itself is dimension independent.
- [ ] Confirm that both external budgets are in physical Frobenius units and
  are pathwise; no stochastic independence or unbiasedness is assumed.

Reviewer notes:

> Pending.

### 2. External-port placement

- [ ] Verify that `y=g+zeta` is formed conceptually before the single final
  gradient cast and that the actual graph consumes `C32(y)` in both EMA and
  Nesterov occurrences.
- [ ] Confirm that `zeta` may represent model-weight error only through its
  induced gradient error; under `10`-smoothness, check the conversion
  `||Delta g||_F<=10||Delta W||_F`.
- [ ] Verify that the deployed output and P9 reference output are both finite,
  contiguous FP32 tensors on the locked shape.
- [ ] Confirm that `nu=Udep-U9` is defined over the reals after the two output
  graphs and that the proof does not silently introduce an unmodeled rounded
  `U9+nu` operation.
- [ ] Check the signs and time indices of both external ports by substituting
  them back into the implemented logical state update.

Reviewer notes:

> Pending.

### 3. FP32 residual sensitivities

- [ ] Starting from P10's exact residual graph, derive the added
  gradient sensitivity
  `k_m=C_g(1+u)+(1-beta)u`, including the cast perturbation and coefficient
  representation error.
- [ ] Derive
  `k_s=C_m*bar(a)*(1+u)+k_m` and check that the already rounded gradient
  product is reused exactly where the implementation says it is.
- [ ] Derive the direct signal coefficient
  `k_sig=bar(a)*(1+bar(beta))*(1+u)` without replacing the executable graph by
  algebraically equivalent real arithmetic.
- [ ] Recompute the P9 output sensitivity
  `k_U=(K_R+A_32)k_sig` from the full-matrix repaired Lipschitz constant and
  P9 binary32-input slope.
- [ ] Derive `k_Wg=C_U*k_U` and `k_WR=C_U`, retaining the master residual's
  dependence on the deployed output.
- [ ] Confirm that each augmented slope and intercept is rounded outward once
  on `2^-40` and never rounded inward.

Reviewer notes:

> Pending.

### 4. Exact signal cancellation and P7 port reduction

- [ ] Define `r_m,r_s` relative to `y=g+zeta` and verify
  `xi_eff=zeta+r_m/(1-beta)` reproduces the implemented momentum update.
- [ ] Rebuild P7's nominal signal and show exactly that
  `s_actual-s_nominal=r_s-r_m`; no explicit `zeta` term may remain.
- [ ] Re-derive the bound on `R(s_actual)-R(s_nominal)` from P9/P4's global
  full-matrix Lipschitz constant, not a diagonal or sampled estimate.
- [ ] Derive
  `k_xig=1+k_m/(1-beta)` and
  `k_eg=A_32*k_sig+K_R(k_m+k_s)+k_Wg/eta`.
- [ ] Derive `k_eR=1+k_WR/eta`, checking that the master residual and direct
  operator difference use the same physical units.
- [ ] Substitute the effective ports into P7 and confirm the implemented
  logical update exactly, including every sign.

Reviewer notes:

> Pending.

### 5. Storage inequality

- [ ] Begin from P7's exact pathwise inequality with
  `q7=399960001/400000000`, gradient gain `1/2`, and operator gain
  `1/2000000`.
- [ ] Rebuild the four affine coefficients `X_1,X_0,E_1,E_0`, including
  P10's nonzero base envelopes and all external increments.
- [ ] Verify that the full augmented squares retain base/external and
  gradient/operator cross terms.
- [ ] Apply Young parameters `theta_g=1` and `theta_e=837`; independently
  derive the formulas for `q11` and `D11`.
- [ ] Check every outward `2^-40` rounding and reproduce the zero-budget P10
  rate, forcing, and function-gap fraction exactly.
- [ ] Recompute the all-positive profile, including the exact rate, forcing,
  positive invariance slack, and subunit objective-gap neighborhood.
- [ ] Confirm that `D11<=1-q11` proves forward invariance of `V<=1` and that
  the claimed limsup follows from the exact P7 storage-to-value constant.

Reviewer notes:

> Pending.

### 6. Axis maxima and Pareto slices

- [ ] Verify that every acceptance predicate is monotone under increasing a
  nonnegative budget coordinate.
- [ ] Recompute the four one-axis integer searches on the `2^-40` grid and
  confirm each accepted endpoint is certified.
- [ ] Check the adjacent `+1`-tick point on every axis and confirm that its
  exact invariance slack is `-1/2^40`.
- [ ] Rebuild all nine slope-frontier rows and all nine intercept-frontier
  rows from exact integer arithmetic.
- [ ] For every frontier row, increase `a_g` or `a_R` (respectively `b_g` or
  `b_R`) by one tick while holding the other displayed coordinate fixed, and
  confirm both adjacent controls fail.
- [ ] Confirm that nonzero endpoint crumbs are caused by outward-rounding
  plateaus and are preserved in the canonical artifact.
- [ ] Verify the wording: these are grid maxima and two-dimensional slices,
  not unrestricted-real optima or a complete four-dimensional frontier.
- [ ] Confirm that rejected controls show only failure of this sufficient
  certificate, not an unstable optimizer trajectory.

Reviewer notes:

> Pending.

### 7. Signal, output, step, and word guards

- [ ] Recompute the all-positive profile's exact actual-signal,
  P9-reference-output, and deployed-output Frobenius bounds on `V<=1`.
- [ ] Verify that the deployed output is strictly below `2^15` and remains
  inside P10's broader runtime domain.
- [ ] Recompute the rounded-step bound using the exact represented FP32
  learning rate, FP32 unit roundoff, and gradual-underflow crumb; confirm it
  is below `2`.
- [ ] Recheck every pre-cast-gradient and EMA-intermediate finite-range bound
  after adding `a_g+b_g` on the unit-storage set.
- [ ] Confirm that the `2^15` output bound is sufficient to reuse P10's
  middle- and low-word invariant proof.
- [ ] Confirm that `maxabs(high)<=2^30` remains a conditional, per-update
  premise rather than a consequence of global PL storage.
- [ ] Verify that overflow, RNE, gradual-underflow, contiguity, and no-FTZ/DAZ
  premises match the P9/P10 arithmetic contracts.

Reviewer notes:

> Pending.

### 8. Independent reconstruction and artifact integrity

- [ ] Run `scripts/certify_implementation_margin.py` and compare every exact
  field with the canonical JSON.
- [ ] Run `scripts/reconstruct_implementation_margin.py --require-canonical`
  and verify that it does not import the P11 theorem module or its generator.
- [ ] Confirm the reconstruction independently rebuilds sensitivities,
  augmented envelopes, `q11`, `D11`, guards, maxima, frontiers, and adjacent
  controls from standard-library rational arithmetic.
- [ ] Verify all source-file hashes, Python/Torch/platform fields, branch,
  Git SHA, dirty-state record, and upstream revision/hash in the canonical
  manifest.
- [ ] Confirm that no sampled diagnostic is being used as evidence for a
  global bound.

Reviewer notes:

> Pending.

### 9. Claim boundaries

- [ ] Confirm P11 does not claim literal upstream Muon, current-plus-epsilon
  normalization, an unrepaired map, GPU/BLAS/tensor-core parity, or a full
  neural-network arithmetic theorem.
- [ ] Confirm no measured production gradient or kernel error is reported;
  the result is an admissible-error margin for a proposed reference design.
- [ ] Confirm no claim is made for weight decay, aspect scaling, stochastic
  rounding, FTZ/DAZ, or neural-network training convergence.
- [ ] Confirm the result controls P7 storage and its function-value,
  true-gradient, and momentum consequences, not full-parameter ISS,
  arbitrary-pair contraction, or a unique minimizer.
- [ ] Check that P11 does not represent its exact automated replay as an
  independent human proof audit.

Reviewer notes:

> Pending.

## Required review record

The reviewer should record:

- name or stable anonymous identifier;
- date;
- exact reviewed Git commit;
- independent calculation environment;
- commands or notebook used for reconstruction;
- every correction requested; and
- one of `approved without correction`, `approved after listed corrections`,
  or `not approved`, with the first failed item and a reproducible discrepancy.

Until that record is present and every relevant checkbox is resolved, P11's
human proof audit is pending.

