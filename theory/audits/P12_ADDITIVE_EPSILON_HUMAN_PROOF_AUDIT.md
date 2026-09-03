# P12 additive-epsilon human proof-audit packet

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. No unchecked item below may be cited as completed
human review. Automated exact generation, independent machine reconstruction,
and tests do not constitute a human proof audit.

The theorem under review is
`theory/additive_epsilon_deficit_certificate.md`. The theorem-facing code is
`src/passive_muon/additive_epsilon_deficit.py`; the canonical exact artifact is
`results/summaries/additive_epsilon_deficit_certificate.json`; and the
independent replay entry point is
`scripts/reconstruct_additive_epsilon_deficit.py`.

## Fixed claim to audit

For every finite matrix shape `m x n`, every `epsilon>0`, and exact real
arithmetic, define

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right),
\]

where `h` is five exact repetitions of

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5.
\]

The fixed upper claim is

\[
\delta_{m,n}(E_{h,\epsilon})
\le\frac1\epsilon
\frac{6602082433275499863}{41641817600000000}.
\]

For `2 x 2`, and for every larger shape into which the pair can be
zero-padded, the fixed strict lower claim is

\[
\delta_{m,n}(E_{h,\epsilon})
>\frac1\epsilon\frac{98823281}{625000}.
\]

The lower witness uses

\[
\tau=\frac{8974467}{10^9},\qquad
u_-=\frac{803760}{1136689},\qquad
u_+=\frac{803761}{1136689},
\]

and its exact quotient has canonical SHA-256
`de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336`.
The theorem also claims the exact fixed-shape scaling law

\[
\delta_{m,n}(E_{h,\epsilon})
=\delta_{m,n}(E_{h,1})/\epsilon.
\]

It concerns a continuous exact-real surrogate. It does not certify the
discontinuous pinned BF16 implementation.

## Reviewer checklist

### 1. Operator, domain, and scaling

- [ ] Confirm the domain is each fixed finite real matrix space with the
  Frobenius inner product, and that the pairwise deficit is the authoritative
  quantity.
- [ ] Confirm the exact coefficient fractions, five-stage composition,
  additive-denominator placement, and condition `epsilon>0`.
- [ ] Starting from
  `E_(h,epsilon)(epsilon X)=E_(h,1)(X)`, independently derive the single
  numerator factor and squared denominator factor in the pairwise quotient.
- [ ] Verify that the change of variables is bijective for every fixed shape
  and therefore gives equality, not merely an upper bound.
- [ ] Confirm that any restricted-domain version would require the domain to
  scale with `epsilon`.

Reviewer notes:

> Pending.

### 2. Differentiability at zero

- [ ] Verify
  `||N_epsilon(M)-M/epsilon||_F = r^2/(epsilon(epsilon+r))` and hence the
  Frechet derivative `DN_epsilon(0)=I/epsilon`.
- [ ] Verify that the five-stage polynomial derivative at zero is
  `(6889/2000)^5 I`.
- [ ] Recompute the exact origin gain
  `15516041187205853449/32000000000000000/epsilon` and confirm it is positive.
- [ ] Check continuity of the nonzero derivative formula as `M` tends to zero.
- [ ] Confirm the proof never differentiates the Frobenius norm alone at the
  origin or silently removes the additive epsilon.

Reviewer notes:

> Pending.

### 3. Full rectangular tangent reduction

- [ ] Independently derive the active singular-value, difference, sum, and
  rectangular null-side modes of `D H_h(Z)`.
- [ ] Check the limiting formulas at repeated and zero singular values.
- [ ] Verify that oddness of `h` makes `h'` even and turns every mode into a
  secant average over a subinterval of `[-t,t]`.
- [ ] Confirm the scalar prefix enclosure therefore bounds the entire matrix
  tangent space for arbitrary finite rectangular shape, not only diagonal
  perturbations.
- [ ] Verify that `A_Z=D H_h(Z)` is Frobenius-self-adjoint and that the radial
  projector lies in its diagonal singular-value block.

Reviewer notes:

> Pending.

### 4. Additive-normalizer and projection algebra

- [ ] For nonzero `M`, independently derive
  `DN_epsilon=(1-t)/epsilon*(I-tP)` with
  `t=||M||_F/(epsilon+||M||_F)` and `P=U tensor U`.
- [ ] Check the exact decomposition `I-tP=(1-t)I+tQ`, `Q=I-P`.
- [ ] Verify the order of composition and the identity
  `Sym(DE)=(1-t)/epsilon*((1-t)A+t Sym(AQ))`.
- [ ] Re-derive the projection/anticommutator envelope `Gamma(a,b)`, including
  the piecewise conditions `a+b>0` and `b+3a>=0`.
- [ ] Check those conditions separately for all four locked bands.
- [ ] Confirm the projection bound is dimension independent and applies on
  the complete Frobenius tangent space.

Reviewer notes:

> Pending.

### 5. Arb scalar proof and four bands

- [ ] Re-evaluate the five-stage value/derivative recurrence without
  expanding the composed polynomial.
- [ ] Confirm outward-rounded Arb arithmetic proves the global strict upper
  `h'(s)<4848763/10000` on `[0,1]`.
- [ ] Confirm the three prefix strict lower bounds:
  `-504/5` through `1/200`, `-77877/500` through `3/500`, and `-6379/40`
  through `63/10000`.
- [ ] Confirm the inherited global strict lower `-199437/1250` on `[0,1]`.
- [ ] Verify the four bands are contiguous and cover `[0,1]`, with shared
  endpoints handled safely.
- [ ] Replay both 160-bit and 224-bit Arb passes and confirm identical leaf
  counts, depths, and trace hashes.
- [ ] Check that a prefix bound through each right endpoint really covers all
  derivative secants available at every radius in that band.

Reviewer notes:

> Pending.

### 6. Exact band maximization and global upper

- [ ] Recompute all four exact values of `Gamma(a,b)` from the locked rational
  slope bounds.
- [ ] Rebuild
  `phi_a(t)=-(1-t)((1-t)a+t Gamma(a,b))` for every band.
- [ ] Check both endpoints and the stationary point of each quadratic using
  exact arithmetic.
- [ ] Confirm the four maximizing radii are respectively
  `0`, `1/200`, `3/500`, and `63/10000`.
- [ ] Reproduce every band upper in the theorem table and confirm the fourth
  is active.
- [ ] Reproduce exactly
  `6602082433275499863/41641817600000000` and its outward-safe decimal.
- [ ] Verify that integrating the symmetric-Jacobian bound along arbitrary
  line segments proves the global pairwise inequality, including segments
  through zero.

Reviewer notes:

> Pending.

### 7. Exact swapped-diagonal lower witness

- [ ] Verify exactly that
  `(803760/1136689)^2+(803761/1136689)^2=1`.
- [ ] Check `r=tau/(1-tau)=8974467/991025533` and that both swapped matrices
  have exact norm `r` and normalized radius `tau` at `epsilon=1`.
- [ ] Independently derive the pairwise quotient
  `-[h(tau*u_high)-h(tau*u_low)]/[r(u_high-u_low)]`.
- [ ] Evaluate all five stages in exact rational arithmetic and reproduce the
  canonical fraction hash
  `de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336`.
- [ ] Confirm the exact quotient is strictly larger than
  `98823281/625000=158.1172496` and smaller than the global upper.
- [ ] Verify that scaling both pair members by `epsilon` scales the witnessed
  deficit by exactly `1/epsilon`.
- [ ] Check the zero-padding claim only for shapes with at least two singular
  directions; do not apply the lower bracket to scalar-only shapes.

Reviewer notes:

> Pending.

### 8. Repair conclusion and deployed epsilon

- [ ] Starting from the pairwise upper, verify that adding
  `(bar_delta_1/epsilon)M` gives a globally monotone map in every finite shape.
- [ ] Verify that any repair no larger than the strict pair lower endpoint
  leaves the locked pairwise gap negative.
- [ ] At `epsilon=10^-7`, reproduce the strict necessary endpoint
  `1,581,172,496` and sufficient endpoint
  `6602082433275499863/4164181760`.
- [ ] Check the comparison with the prior max-floor `c=1` upper
  `41528474059081/260261360000`.
- [ ] Verify the ordinary-unit-signal comparison used to classify the global
  deployed-epsilon repair as catastrophic.
- [ ] Confirm the text does not call the sufficient endpoint minimal or infer
  that every training trajectory attains the worst case.

Reviewer notes:

> Pending.

### 9. Exact-real/BF16 separation

- [ ] Verify the pinned KellerJordan/Muon revision
  `f98f1cacc0263b04290753e32be8d498c1efc806` and audited-file SHA-256
  `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
- [ ] Check the pinned operation order: BF16 cast, one-time orientation, BF16
  norm, Python-float epsilon addition, division, five BF16 stages, orientation
  reversal.
- [ ] Confirm the exact-real theorem uses exact rational coefficients and a
  continuous map, while the pinned BF16 map is backend-specific and
  discontinuous.
- [ ] Inspect the recorded backend witness and verify that epsilon absorption
  is reported only for that tested backend and pair.
- [ ] Confirm no exact-real Jacobian, pairwise, or repair conclusion is
  transferred to literal BF16 Muon.

Reviewer notes:

> Pending.

### 10. Independent reconstruction and integrity

- [ ] Run the primary generator twice at the locked Arb precisions and compare
  every proof-cover signature.
- [ ] Run
  `scripts/reconstruct_additive_epsilon_deficit.py --require-canonical` and
  verify that it imports neither the theorem module nor the generator.
- [ ] Confirm the reconstruction uses only standard-library exact fractions
  and explicitly directed `Decimal` endpoint intervals.
- [ ] Verify that the canonical artifact is not read until the independent
  interval and algebraic reconstruction has completed.
- [ ] Reproduce the upper fraction, lower strict fraction, witness hash,
  deployed-epsilon values, band table, prior-artifact hashes, and upstream
  provenance independently.
- [ ] Verify every source snapshot hash, software field, hardware field, Git
  SHA, branch, and dirty-state record in the canonical manifest.
- [ ] Confirm no sampled grid supplies any global upper or monotonicity claim.

Reviewer notes:

> Pending.

### 11. Boundary and negative controls

- [ ] Check the positive derivative at the origin and the `t->1` limiting
  boundary.
- [ ] Verify monotonicity of the affine identity control `h(s)=s` by its
  radial and tangential derivative eigenvalues.
- [ ] Recheck the exact band-boundary maximizers and the active boundary
  `t=63/10000`.
- [ ] Verify the locked under-repair pair is a genuine negative pairwise gap,
  rather than only a failed sufficient test.
- [ ] Confirm none of these controls is described as evidence about optimizer
  training stability.

Reviewer notes:

> Pending.

## Required review record

The reviewer should record:

- name or stable anonymous identifier;
- date;
- exact reviewed Git commit;
- independent calculation environment;
- commands, notebook, or derivation used for reconstruction;
- every correction requested; and
- one of `approved without correction`, `approved after listed corrections`,
  or `not approved`, with the first failed item and a reproducible
  discrepancy.

Reviewer identity:

> Pending.

Reviewed commit:

> Pending.

Independent environment and commands:

> Pending.

Corrections requested:

> Pending.

Verdict and signature:

> Pending.

Until this record is completed and every relevant checkbox is resolved, the
P12 additive-epsilon human proof audit remains pending.
