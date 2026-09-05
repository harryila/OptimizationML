# P18 human proof audit: sector-projected useful rate

Status: **pending independent human signature**.

This packet is a checklist, not evidence that an independent human audit has
already occurred.  The reviewer should work from the theorem, generator,
standard-library reconstruction, and committed artifacts without relying on
the implementation author's prose summary.

## Locked claim

For every fixed finite real rectangular matrix space, lock additive
`epsilon=1/10000000`, five Jordan stages with coefficients `6889/2000`,
`-191/40`, and `4063/2000`, the P14 graph `lambda=1/1000`, `mu=1000`, the P17
quintic gate with ceiling `3/4`, projection gain `K=1`, passive divisor
`c=1024`, and `beta=19/20`.

The claimed exact-real result is an origin-centred pointwise sector
`[125/1024,509/512]` and global smooth-PL-1 convergence at `L=10`,
`eta=1/83`, with rate `999598040401/1000000000000`.  It is not an incremental
sector or an approximate-solver theorem.

## Required checks

- [ ] Verify bi-orthogonal equivariance, sign preservation, and
      `inner_product(X(S),S) >= 0`, including repeated, zero, and rectangular
      null modes.
- [ ] Check both branches of
      `alpha=min(1,K<X,S>/||X||^2)` and the `X=0` convention.
- [ ] Confirm `||Z||^2 <= K<Z,S>` is full-matrix and dimension-independent.
- [ ] Reconstruct the Minkowski/sector-disk blend and the exact endpoints
      `125/1024` and `509/512`; do not treat them as derivative bounds.
- [ ] Verify that the P6 value--momentum theorem needs only the pointwise
      residual norm supply and never differentiates the gate or projection.
- [ ] Replay the exact `2 x 2` storage positivity and `4 x 4` negative-LMI
      Sylvester minors for `eta=1/83`.
- [ ] Verify the exact rate comparison `q18^10 < q14` and its half-life
      interpretation.
- [ ] Check the full exact Pareto table independently, distinguishing a
      finite searched frontier from a global optimality theorem.
- [ ] Reconstruct the complex-skew Schur--Cohn margin at `eta=1/50`; confirm
      that it obstructs the generic sector abstraction but is not claimed to
      be the structured P18 map.
- [ ] Check the exact amplitude and effective-update gates.
- [ ] Audit the Arb canonical enclosure, including projection status,
      computed graph residual, root-error inflation, fidelity, and amplitude.
- [ ] Check that every numerical solver result fails closed on the actual P15
      graph-residual postcondition.
- [ ] Confirm the broad-grid and operating-annulus results are sampled
      diagnostics and do not become global fidelity claims.
- [ ] Verify the unprojected and `K=1/100` negative controls.
- [ ] Confirm source hashes, upstream revision, epsilon placement, cast/order
      qualifications, and exact-real/BF16 separation.

## Reviewer record

- Reviewer name:
- Affiliation or relationship:
- Commit reviewed:
- Artifact hashes checked:
- Date:
- Verdict:
- Corrections required:
- Signature or review reference:
