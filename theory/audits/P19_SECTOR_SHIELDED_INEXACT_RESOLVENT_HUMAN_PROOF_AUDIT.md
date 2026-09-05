# P19 human proof audit: sector-shielded inexact resolvent

## Status

**Independent human review is pending.** This file is an unsigned reviewer
packet, not an approval record. The requested external review cannot be sent
automatically without a reviewer and delivery destination.

The theorem under review is
`theory/sector_shielded_inexact_resolvent.md`. The primary exact generator is
`scripts/certify_sector_shielded_inexact_resolvent.py`; the independent
standard-library reconstruction is
`scripts/reconstruct_sector_shielded_inexact_resolvent.py`; and the canonical
artifacts are
`results/summaries/sector_shielded_inexact_resolvent_certificate.json` and
`results/summaries/p19_sector_shielded_study.json`.

## Locked claim

Fix any positive finite real rectangular matrix shape. Retain P18's
additive normalization `U/(||U||_F+epsilon)` at
`epsilon=1/10000000`, five Jordan stages with coefficients `6889/2000`,
`-191/40`, and `4063/2000`, P13 radial repair, `mu=1000`,
`lambda=1/1000`, the P17 C2 gate with ceiling `3/4`, P18 ray gain `K=1`,
passive divisor `1024`, and `beta=19/20`.

For every finite candidate `C(S)`, P19 exposes its Euclidean projection onto

\[
\mathcal D_S=\left\{U:
\left\|U-\frac{1143}{2048}S\right\|_F
\le\frac{893}{2048}\|S\|_F\right\}.
\]

The claimed exact-real result is that every shielded candidate satisfies the
P18 pointwise sector `[125/1024,509/512]`, exact P18 outputs are fixed, and
the P18 smooth-PL certificates replay at `eta=1/83` and `eta=1/120`.
Every successful return of the locked binary64 reference implementation
passes an exact-as-stored dyadic disk check. It may fail closed rather than
return an output for a nonfinite signal or an unrepresentable subnormal case.

## Reviewer checklist

### 1. Inherited operator and provenance

- [ ] Confirm the matrix domain is every fixed positive finite real
      rectangular shape with the Frobenius inner product.
- [ ] Verify the additive-epsilon placement, five exact polynomial
      coefficients/stage count, radial repair, shunt, resolvent, gate, ray
      projection, and passive divisor against the frozen P18 artifact.
- [ ] Check that the KellerJordan/Muon revision is provenance for the Jordan
      comparator only; upstream contains none of the repair, resolvent, ray
      projection, or shield.
- [ ] Verify the P18 source commit, canonical artifact hash, artifact commit,
      and annotated checkpoint tag have distinct and correct roles.

Reviewer notes:

> Pending.

### 2. Disk and pointwise-sector identity

- [ ] Recompute
      `center=(125/1024+509/512)/2=1143/2048` and
      `radius=(509/512-125/1024)/2=893/2048`.
- [ ] Expand
      `<U-mS,MS-U>` and verify exact cancellation of the cross terms.
- [ ] Confirm the remaining expression is
      `radius^2*||S||^2-||U-center*S||^2` in the full Frobenius space.
- [ ] Check that this is an origin-centred **pointwise** sector, not an
      incremental Jacobian or Lipschitz statement.

Reviewer notes:

> Pending.

### 3. Projection formula and zero input

- [ ] Derive the metric projection onto the closed ball `D_S` for an inside
      candidate, outside candidate, and zero displacement.
- [ ] At `S=0`, verify `D_0={0}` and every finite candidate projects to zero.
- [ ] Confirm the theorem accepts every finite exact-real candidate and does
      not assume continuity, determinism, or a graph-residual tolerance.
- [ ] Check the exact rational radial and tangential corruption controls and
      confirm their projected sector supply is zero on the boundary.

Reviewer notes:

> Pending.

### 4. Identity on P18 and modular nonexpansiveness

- [ ] Use P18's global sector theorem to verify `T18(S)` belongs to `D_S` for
      every finite matrix shape.
- [ ] Apply the fixed-point property of metric projection to prove
      `Pi_(D_S)(T18(S))=T18(S)`.
- [ ] Prove fixed-`S` projection nonexpansiveness for two arbitrary
      candidates.
- [ ] Set the second candidate to `T18(S)` and derive the displayed modular
      fidelity inequality.
- [ ] Confirm no joint nonexpansiveness in `(S,C)`, incremental-sector claim,
      or final graph-residual-to-P18 candidate bound is inferred.

Reviewer notes:

> Pending.

### 5. Smooth--PL specialization

- [ ] Verify the pinned EMA/Nesterov state and signal ordering and
      `beta=19/20`.
- [ ] Confirm the objective class is differentiable, globally `10`-smooth,
      bounded below, and satisfies the **global** PL inequality with constant
      `1`; convexity and uniqueness are not assumed.
- [ ] Audit the P6/P18 value--momentum derivation and confirm it uses only the
      pointwise sector supply at the current signal.
- [ ] Check that this permits arbitrary finite time-varying candidates after
      shielding without turning the conclusion into arbitrary-pair
      contraction.

Reviewer notes:

> Pending.

### 6. Maximum-step exact replay

- [ ] At `eta=1/83`, replay the exact storage
      `P=((97/125,-151/500),(-151/500,17/100))`, function weight `1`,
      reverse-interpolation multiplier `3459/500`, and residual multiplier
      `9/250`.
- [ ] Verify exact storage positivity and strict negativity of the complete
      `4 x 4` LMI by rational Sylvester minors.
- [ ] Recompute `tau=999799/1000000` and
      `q=999598040401/1000000000000`.
- [ ] Confirm the approximately `1724.0734` number is described only as a
      certified Lyapunov-rate half-life.

Reviewer notes:

> Pending.

### 7. Faster-rate exact replay

- [ ] At `eta=1/120`, replay
      `P=((599/1000,-231/625),(-231/625,657/2500))`, function weight `1`,
      reverse-interpolation multiplier `2177/200`, and residual multiplier
      `53/2500`.
- [ ] Verify the exact `4 x 4` LMI and storage Sylvester minors.
- [ ] Recompute `tau=24987/25000`,
      `q=624350169/625000000`, and the certified Lyapunov-rate half-life of
      approximately `666.314`.
- [ ] Check that the smaller step having a faster certified rate is reported
      as a property of the finite certificate frontier, not as a claim about
      every observed trajectory.

Reviewer notes:

> Pending.

### 8. Binary64 reference implementation

- [ ] Inspect the scaled/balanced norm construction and confirm unscaled
      extreme entries are not squared during projection.
- [ ] Verify inside candidates are exact-checked and returned bit for bit.
- [ ] Check that outside projection uses a common scale and a radius moved 32
      binary64 ULPs inward.
- [ ] Confirm every successful stored output is converted to exact rational
      dyadics and checked against the original, not inward, P18 disk.
- [ ] Verify the `S/2` fallback itself is exact-checked.
- [ ] Check that nonfinite candidates use the safe fallback, while nonfinite
      signals are rejected without returning an update.
- [ ] Reproduce the least-positive-subnormal representability obstruction and
      confirm the routine raises instead of emitting an uncertified output.
- [ ] Confirm this is a locked NumPy/CPU reference, not a theorem for every
      FP64 backend or compiler.

Reviewer notes:

> Pending.

### 9. P18 rounded inner-product behavior

- [ ] Verify the exact P18 shape channel has nonnegative inner product with
      its signal.
- [ ] Inspect the binary64 positive-part clipping introduced for a rounded
      negative inner product and verify it returns the zero shape branch.
- [ ] Confirm the final disk shield, rather than this local clip alone, is the
      authoritative exact stored-output safety postcondition.

Reviewer notes:

> Pending.

### 10. Diagnostics, controls, and claim boundaries

- [ ] Reproduce all `2688` annulus calls and verify the shield is inactive
      and bitwise identical on every guarded P18 candidate.
- [ ] Reproduce `2176/2176` informative annulus fidelity passes and the
      canonical departure/retention values.
- [ ] Verify all `2690` computed P15 residual checks, while retaining the
      stated limitation that they do not alone bound the nonlinear P18
      candidate error.
- [ ] Reproduce all finite corrupted-candidate, zero, nonfinite-candidate,
      nonfinite-signal, and one-ULP escape controls.
- [ ] Confirm sampled fidelity evidence is not used to prove global safety or
      stability.
- [ ] Confirm no BF16, literal upstream, weight-decay, aspect-scaling,
      stochastic-gradient, represented-model-state, throughput, or neural
      training claim appears.

Reviewer notes:

> Pending.

### 11. Independent reconstruction and artifacts

- [ ] Generate a fresh canonical exact artifact and verify every source hash
      and Git provenance field.
- [ ] Run the standard-library-only reconstruction and confirm it imports no
      project theorem module or third-party numerical package.
- [ ] Compare every reconstructed shield field, control, exact LMI entry, and
      Sylvester minor with the canonical artifact.
- [ ] Run the dedicated P19 workflow and full repository CI from a clean
      checkout.
- [ ] Confirm machine replay is described as exact reconstruction and unit
      cross-checking, not an independent human audit.

Reviewer notes:

> Pending.

## Reviewer record

- Reviewer name:
- Affiliation or relationship:
- Commit reviewed:
- Artifact hashes checked:
- Date:
- Verdict:
- Corrections required:
- Signature or review reference:
