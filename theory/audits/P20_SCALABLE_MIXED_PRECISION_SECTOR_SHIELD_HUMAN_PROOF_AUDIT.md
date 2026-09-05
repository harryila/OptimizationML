# P20 human proof audit: scalable mixed-precision sector shield

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. No unchecked item below may be cited as completed
review.

The theorem under review is
`theory/scalable_mixed_precision_sector_shield.md`. The exact generator is
`scripts/certify_scalable_sector_shield.py`; the independent standard-library
reconstruction is `scripts/reconstruct_scalable_sector_shield.py`; and the
theorem-facing runtime is `src/passive_muon/scalable_sector_shield.py`. The
canonical artifacts are
`results/summaries/scalable_sector_shield_certificate.json` and
`results/summaries/p20_scalable_sector_shield_study.json`.

## Locked claim

For each of the seven separately certified matrix shapes, consider finite
stored FP32 or BF16 signal/candidate inputs. BF16 inputs are widened exactly
to FP32. Under the locked CPU, round-to-nearest-ties-to-even, gradual-underflow
operation graph, every successful stored FP32 return `U` satisfies

\[
 \left\|U-\frac{1143}{2048}S\right\|_F
 \le \left(\frac{893}{2048}-\Delta_{a,b}\right)\|S\|_F,
 \qquad \Delta_{a,b}>0.
\]

The algorithm returns an inward-screened candidate unchanged, otherwise uses
a directed inward radial clip, and reserves `S/2` for exceptional arithmetic
and guarded subnormal handling. It performs no runtime exact-rational
postcheck. A nonfinite signal or an all-subnormal signal without an exactly
representable half fails closed without an update.

The corresponding P19 smooth-PL rates are inherited only when the stored
signal is identified with the abstract sector port in the otherwise-real
outer loop. P20 does not yet compose the cast into that port or the outer
momentum/parameter arithmetic.

## Reviewer checklist

### 1. Scope, dependencies, and provenance

- [ ] Confirm the full-matrix domain is each fixed positive finite real
      rectangular space with the Frobenius norm and inner product.
- [ ] Verify P19's sector lower bound `125/1024`, upper bound `509/512`, center
      `1143/2048`, and radius `893/2048` against the frozen P19 artifact.
- [ ] Verify the P19 source commit, artifact commit/hash, annotated checkpoint
      tag, and tag object have their stated distinct roles.
- [ ] Check the frozen P10 and P11 artifact hashes and confirm P20 reuses only
      their relative-plus-absolute-crumb ledger convention, not their complete
      outer-loop disturbance theorem.
- [ ] Confirm no diagonal or sampled calculation is used as a full-matrix
      containment proof.

Reviewer notes:

> Pending.

### 2. Frozen arithmetic contract

- [ ] Confirm accepted inputs are contiguous CPU FP32 or BF16 tensors, BF16
      widening to FP32 is exact, and every successful output is FP32.
- [ ] Check every materialization boundary: center multiply, displacement
      subtraction, norm division/square/tree/square-root, clip displacement
      multiply, center addition, and half fallback.
- [ ] Verify the two FP32 norm parts have at most 48 significant bits in their
      FP64 product, making that product exact.
- [ ] Confirm the exact scalar operation order and every `nextafter` direction
      for the pass-through screen and radial clip.
- [ ] Confirm FMA, reassociation, stochastic rounding, FTZ/DAZ, a BF16 output
      cast, native BLAS, tensor-core reductions, and unspecified compiler
      transformations are excluded.
- [ ] Reproduce the runtime RNE and gradual-underflow self-checks.

Reviewer notes:

> Pending.

### 3. Scale-free balanced norm enclosure

- [ ] For `n=ab`, recompute `h_n=ceil(sqrt(n))`,
      `L_n=1+ceil(log2(n))`, and `g_n=L_n*2^-24/(1-L_n*2^-24)`.
- [ ] Starting with `v=x/maxabs(x)`, verify `||v||F>=1` and derive the FP32
      division, square, pairwise-sum, square-root, and underflow-crumb bounds.
- [ ] Check the exact outward dyadic square-root enclosures and prove
      `ell_n||x||F <= Nhat_n(x) <= U_n||x||F` for every nonzero finite stored
      FP32 matrix of the certified shape.
- [ ] Verify the aligned `2^20` blocks followed by the zero-padded global tree
      have the same maximum path depth as the declared adjacent tree.
- [ ] Check that zero is handled separately and no relative-error statement is
      applied where the exact norm vanishes.

Reviewer notes:

> Pending.

### 4. Pass-through branch

- [ ] Re-derive the FP32 displacement envelope for
      `Dhat=fl32(C-fl32((1143/2048)S))`.
- [ ] Verify the screen coefficient `891/2048` is exactly representable and
      the final FP64 threshold is rounded inward.
- [ ] From the norm enclosure and normal anchor `maxabs(S)>=2^-126`, derive

      `R_pass=(k*U_n/ell_n+gamma*u*(1+u)+h*tau*(2+u)/sigma_min)/(1-u)`.
- [ ] Confirm an accepted candidate is already a stored FP32 value and is
      returned bit for bit, with no omitted final-cast term.
- [ ] Check that removing the `1/1024` screen buffer makes the largest-shape
      sufficient bound escape the P19 disk.

Reviewer notes:

> Pending.

### 5. Directed radial-clip branch

- [ ] Verify the clip coefficient is exactly `890/2048=445/1024`.
- [ ] Check that division then multiplication are each followed by a binary64
      `nextafter` toward zero, and that the FP64-to-FP32 conversion is followed
      by an FP32 `nextafter` toward zero.
- [ ] Prove the stored FP32 scale cannot exceed
      `(890/2048)Nhat(S)/Nhat(Dhat)`.
- [ ] Pay separately for `fl32(alpha32*Dhat)` and
      `fl32(fl32(gamma*S)+Q)` and re-derive

      `R_clip=(890/2048)*(U_n/ell_n)*(1+u)^2`
      `+ gamma*u*(2+u) + (h*tau/sigma_min)*(3+2u)`.
- [ ] Confirm the minimum clip-only margin is exactly
      `53997772719001/576460752303423488` and positive.
- [ ] Verify that using the pass-through coefficient in the clip path would
      make the largest-shape sufficient bound escape the disk.
- [ ] Confirm this rounded contraction along `Dhat` is not claimed to be
      P19's metric projection or nonexpansive.

Reviewer notes:

> Pending.

### 6. Half fallback and representability boundary

- [ ] Re-derive the normal-anchor half bound
      `R_half=|1143/2048-1/2|+h*tau/2^-126`.
- [ ] Confirm this covers rounded odd subnormal coordinates in a tensor having
      at least one normal anchor; entrywise exact halving is not assumed there.
- [ ] Verify zero signal returns zero, the only member of the zero disk.
- [ ] Audit the bit-level exact-halving rule for nonzero all-subnormal signals.
- [ ] Reproduce successful even-subnormal halving and failure of the least
      positive and maximum odd FP32 subnormals.
- [ ] Confirm the all-subnormal failure set is contained in
      `||S||F<h_n*2^-126` and is described only as an input-space
      representability region, not an objective or Lyapunov neighborhood.
- [ ] Check that nonfinite candidates use the guarded half while nonfinite
      signals return no update.

Reviewer notes:

> Pending.

### 7. Shape recurrence and exact margins

- [ ] For all seven shapes, recompute the exact pass-through, clip, and half
      radii, take their maximum, and round it upward on the `2^-60` grid.
- [ ] Verify every `Delta_(a,b)=893/2048-R_(a,b)` is strictly positive.
- [ ] In particular, recompute
      `25250274108291/144115188075855872` at `4096 x 11008` and
      `71710053325847/1152921504606846976` at `4096 x 14336`.
- [ ] Check the latter is the overall minimum margin and verify the generated
      FP64 hexadecimal shape table matches every exact rational value.
- [ ] Confirm the three small diagnostic shapes are not presented as headline
      Transformer theorem targets.

Reviewer notes:

> Pending.

### 8. Runtime implementation

- [ ] Confirm configuration accepts only the seven certified Transformer
      shapes and three explicitly separated diagnostic shapes.
- [ ] Inspect the chunked balanced reduction and reproduce its bitwise match
      to the global adjacent tree control.
- [ ] Verify ordinary finite rejections take the radial clip, while exceptional
      overflow/nonfinite arithmetic alone reaches the half fallback.
- [ ] Check every successful branch emits a finite contiguous FP32 output.
- [ ] Confirm neither runtime configuration nor per-output checking imports or
      invokes `Fraction`, arbitrary precision, or big integers.
- [ ] Reproduce all seeded adversarial candidates, the original-sector
      boundary, and the one-ULP escape controls using offline exact checks.

Reviewer notes:

> Pending.

### 9. Conditional smooth-PL specialization

- [ ] Verify the pointwise disk is exactly equivalent to P18's pointwise
      sector `[125/1024,509/512]` in the full Frobenius space.
- [ ] Confirm P18/P19's value--momentum proof uses only that current-signal
      pointwise sector supply, not incremental monotonicity.
- [ ] At `beta=19/20`, replay the exact P19 LMIs and rates
      `999598040401/1000000000000` at `eta=1/83` and
      `624350169/625000000` at `eta=1/120`.
- [ ] Check the objective class is differentiable, globally `10`-smooth,
      bounded below, and global-PL-`1`, without convexity or minimizer
      uniqueness.
- [ ] Confirm the displayed half-lives are Lyapunov-bound half-lives.
- [ ] Audit the explicit qualification that stored-signal/abstract-port
      identification is required and that P20 does not compose FP32/BF16
      input casts, EMA/Nesterov rounding, master weights, or parameter updates.

Reviewer notes:

> Pending.

### 10. Candidate definitions and provenance

- [ ] Verify the P18 comparator's additive normalization
      `U/(||U||F+1e-7)`, coefficients `6889/2000`, `-191/40`, `4063/2000`,
      five stages, P13 repair, `(lambda,mu)=(1/1000,1000)`, C2 gate ceiling
      `3/4`, ray gain `K=1`, and passive divisor `1024`.
- [ ] Verify KellerJordan/Muon revision
      `f98f1cacc0263b04290753e32be8d498c1efc806` and audited `muon.py` hash
      `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
- [ ] Check the canonical `2 x 2` comparator follows the pinned BF16 cast,
      transpose, additive-`1e-7` norm placement, coefficient order, and exactly
      five Jordan stages.
- [ ] Confirm large-shape upstream comparisons are labelled packed
      singular-coordinate formula diagnostics, not literal backend runs.
- [ ] Confirm the shield theorem is candidate-independent but does not certify
      an unshielded upstream candidate.

Reviewer notes:

> Pending.

### 11. Study, fidelity, and negative controls

- [ ] Reproduce P18 annulus counts: `2671/2688` pass-through, `17/2688`
      radial clips, zero half fallbacks, and `2176/2176` informative fidelity
      passes.
- [ ] Reproduce upstream annulus counts: `761/2688` pass-through and
      `1927/2688` radial clips, with every stored output passing the offline
      exact disk check.
- [ ] Verify all `14/14` P18 operating Transformer spectra pass through and all
      seven deliberately flat boundary stresses clip.
- [ ] Reproduce the canonical `diag(3,4)` shaping and amplitude metrics, while
      retaining their sampled/falsification classification.
- [ ] Reproduce finite adversarial, one-ULP, nonfinite, zero, normal-boundary,
      subnormal, and FTZ controls.
- [ ] Confirm no finite diagnostic is used as proof of global containment,
      global fidelity, or training behavior.

Reviewer notes:

> Pending.

### 12. Reconstruction, cross-platform replay, and claim boundary

- [ ] Generate a fresh exact artifact and confirm all P10/P11/P19 provenance
      checks and all exact contract checks pass.
- [ ] Run the standard-library reconstruction and verify it imports neither
      the project package nor a numerical library and matches every exact
      reconstruction field.
- [ ] Reproduce the default decision digest
      `998ef020d1b642cf923a0789dfdd489b2b949a2aed0b3948cff67b88584a5c9c`
      on both the locked macOS and Ubuntu workflow jobs; record and inspect
      the actual runner architectures.
- [ ] Archive and inspect the platform-specific full study records; do not
      infer GPU, BLAS, compiler, or unspecified-platform parity.
- [ ] Run the dedicated P20 workflow and full repository CI from a clean
      checkout.
- [ ] Confirm automated exact reconstruction and unit cross-checks are not
      described as independent human proof review.
- [ ] Confirm no claim covers a BF16 output, FTZ/DAZ, full optimizer
      integration, aspect scaling, weight decay, distributed reductions,
      stochastic gradients, throughput, or neural-network training.

Reviewer notes:

> Pending.

## Reviewer record

- Reviewer name or stable anonymous identifier:
- Affiliation or relationship:
- Commit reviewed:
- Artifact hashes checked:
- Hardware/software environment:
- Date:
- Verdict:
- Corrections required:
- Signature or review reference:
