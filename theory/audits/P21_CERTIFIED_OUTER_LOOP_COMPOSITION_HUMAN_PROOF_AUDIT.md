# P21 human proof audit: certified outer-loop composition

## Status

**Independent human review is pending.** This is an unsigned reviewer packet,
not an approval record. No unchecked item may be cited as completed review.

The theorem under review is `theory/certified_outer_loop_composition.md`.
The exact theorem modules are
`src/passive_muon/certified_outer_loop_composition_certificate.py` and
`src/passive_muon/certified_outer_loop_roundoff.py`; the stored runtime is
`src/passive_muon/certified_outer_loop_composition.py`. The generator and
independent reconstruction are `scripts/certify_outer_loop_composition.py`
and `scripts/reconstruct_outer_loop_composition.py`. The empirical protocol
is `theory/p21_shadow_trace_protocol.md` with its machine-readable companion
`experiments/training/p21_shadow_trace_protocol.json`.

## Locked claim

For the P20 pointwise sector at the actual stored FP32 Nesterov signal, an
exact dimension-independent `7 x 7` LMI proves

\[
 \mathcal V_{t+1}\le q\mathcal V_t
 +G_m\lVert a_t^m\rVert_F^2
 +G_s\lVert a_t^s\rVert_F^2
 +G_h\lVert h_t\rVert_F^2.
\]

The exact base pairs are `eta=1/120`,
`q=624350169/625000000`, gains `(16384,1024,512)`, and `eta=1/83`,
`q=999598040401/1000000000000`, gains `(32768,2048,512)`. Setting all ports
to zero recovers the frozen P18/P19 `4 x 4` matrices exactly.

For the seven P20 Transformer shapes, the locked CPU FP32 arithmetic ledger
absorbs these ports on `V<=1`. Under the explicit pathwise source budget
`||zeta||_F <= (sqrt(V)+1)/4096`, both rates remain contractive and give finite
objective neighborhoods. The primary theorem sets weight decay to zero.
Nonzero decay is certified only through separately bounded logical and stored
decay ports. The all-subnormal zero wrapper is covered by an absolute update
port relative to the conceptual `S/2` sector point.

The result does not cover an arbitrary neural loss, unmodified upstream Muon,
an unspecified accelerator/backend, or the currently blocked real-gradient
shadow trace.

## Reviewer checklist

### 1. Scope and frozen dependencies

- [ ] Confirm the objective class is differentiable, globally `10`-smooth,
      bounded below, and global-PL-`1`; no convexity, unique-minimizer, or
      arbitrary-pair contraction claim is made.
- [ ] Verify the P20 sector endpoints `125/1024` and `509/512`, center
      `1143/2048`, and radius `893/2048` against its frozen artifact.
- [ ] Verify the P20 source/artifact commits, artifact SHA-256, annotated tag,
      and P10/P11 ledger dependencies recorded in the P21 artifact.
- [ ] Confirm the global algebraic LMI is dimension independent while the
      concrete rounding closure is limited to seven fixed shapes and their
      transpose orientations.
- [ ] Check that sampled matrices and the synthetic trace are never used as
      proof of a full-matrix inequality.

Reviewer notes:

> Pending.

### 2. Stored signal and port definitions

- [ ] Starting from the definitions of `a_m` and `a_s`, derive
      `z_next=beta*z+(1-beta)*u+a_m`.
- [ ] Derive exactly
      `p=beta^2*z+(1-beta^2)*u+beta*a_m+a_s` for
      `p=s_hat/L`.
- [ ] Confirm the P20 sector is applied directly to this stored `p`, not to an
      reconstructed ideal signal.
- [ ] Verify `T/L=gamma*p+K_T*v`, `||v||<=||p||`, with
      `gamma=1143/2048` and `K_T=893/2048`.
- [ ] Check the sign and units of the aggregate `h` port in
      `Delta W=-eta*gamma*L*(p+(K_T/gamma)*v+h)`.
- [ ] Confirm no step invokes or assumes a bound on
      `T(s_hat)-T(s)` or incremental Lipschitzness of P20.

Reviewer notes:

> Pending.

### 3. Exact `7 x 7` LMI

- [ ] Rebuild the matrix in variable order
      `(z,u,v,u_next,a_m,a_s,h)`.
- [ ] Check both directed nonconvex smooth-interpolation supplies, the
      next-sample PL supply, and the stored-signal sector residual supply.
- [ ] Verify exact function-value cancellation with function-storage
      coefficient one.
- [ ] Recompute the leading minors of both storage matrices and all seven
      leading minors of each negated LMI.
- [ ] Confirm the exact gains `(16384,1024,512)` at `1/120` and
      `(32768,2048,512)` at `1/83` are subtracted on only the three port
      diagonals.
- [ ] Delete the port rows/columns from the unpenalized matrix and verify the
      recovered `4 x 4` block equals the frozen P18/P19 LMI entry for entry.
- [ ] Confirm zero-port recovery is an exact-real specialization rather than
      a claim that FP32 arithmetic has zero error.

Reviewer notes:

> Pending.

### 4. Gradient storage and FP32 EMA/Nesterov graph

- [ ] Audit the accepted contiguous CPU BF16/FP32 gradient types and the one
      final storage into FP32.
- [ ] Verify stored beta words
      `15938355/16777216` and `13421773/268435456`.
- [ ] Check that the same materialized rounded `bg` tensor is reused in the
      momentum and Nesterov additions.
- [ ] Re-derive the P10 relative-plus-absolute-underflow-crumb envelopes for
      momentum and signal roundoff, including coefficient representation.
- [ ] Verify every pre-cast source error is collected in `zeta` before the
      final FP32 storage boundary.
- [ ] Confirm that `||zeta|| <= noise + L||middle+low||` follows when the
      gradient is evaluated at the high word, and that static word guards do
      not themselves imply the frozen `1/4096` budget.
- [ ] Verify the robust decomposition: external/stochastic gradient error is
      bounded by `(sqrt(V)+1)/8192`, represented-master distance by
      `(sqrt(V)+1)/81920`, and `L=10` makes the latter a second `1/8192`
      gradient term, summing to `1/4096`.

Reviewer notes:

> Pending.

### 5. Candidate, aspect scaling, and shield order

- [ ] Verify additive epsilon is exactly `1e-7`, with the denominator
      `||S||_F+epsilon` and not a max floor.
- [ ] Verify five BF16 Jordan stages and coefficients `6889/2000`, `-191/40`,
      and `4063/2000` against pinned revision
      `f98f1cacc0263b04290753e32be8d498c1efc806` and the audited source hash.
- [ ] Check the aspect factor `sqrt(max(1,rows/columns))` is applied in the
      candidate's stored dtype before orientation and P20.
- [ ] Confirm signal and candidate are transposed together when only the
      reverse orientation is certified, and output is transposed back.
- [ ] Verify candidate quality, BF16 arithmetic error, and even a nonfinite
      candidate are irrelevant to sector safety after a successful/fallback
      P20 call, while they remain relevant to fidelity.
- [ ] Confirm no post-shield aspect scaling occurs.

Reviewer notes:

> Pending.

### 6. Learning-rate and compensated-master arithmetic

- [ ] Reproduce the primary FP32 eta word `0x3c088889` as
      `8947849/1073741824` and the secondary word `0x3c4565c8` as
      `1617081/134217728`.
- [ ] Audit the exact operation order: stored operator multiply, optional
      stored decay products, subtraction into the low word, middle `TwoSum`,
      then high `TwoSum`.
- [ ] Verify the logical master is the exact sum `high+middle+low` and the
      master residual is computed against that logical sum and mathematical
      eta.
- [ ] Confirm the represented model and reconstruction port refer to the
      pre-update high word, not the returned next state.
- [ ] Re-derive the shape-scaled master crumbs and eta-representation term in
      the `h` envelope.
- [ ] Check output max `64`, combined-step max `1`, lower-word invariants, and
      the conditional high-word guard `2^30`.

Reviewer notes:

> Pending.

### 7. Affine absorption and seven-shape results

- [ ] Independently enclose the storage-dual norms on the frozen `2^-80`
      square-root grid and report all positive rounding directions.
- [ ] Recompute each affine port envelope for both operating points and all
      seven P20 shapes.
- [ ] Apply Young's inequality with all three parameters equal to one and
      verify the reported rate, forcing, and objective values are rounded
      upward on the `2^-40` grid.
- [ ] Reproduce the tight robust primary fields
      `q_bar=1098368546995/1099511627776`,
      `D=14267/274877906944`, and objective bound
      `274464109/549755813888`.
- [ ] Reproduce the tight robust secondary fields
      `q_bar=549534922135/549755813888`,
      `D=57049/549755813888`, and objective bound
      `2839673189/1099511627776`.
- [ ] Confirm `q_bar<1`, `D<=1-q_bar`, and every finite-range guard for all
      fourteen shape/rate cases.
- [ ] Check the zero-source profile separately and confirm it retains a tiny
      nonzero forcing due to stored rounding.

Reviewer notes:

> Pending.

### 8. All-subnormal completion

- [ ] Verify P20's exception is caught only for
      `NearZeroUnrepresentable`; nonfinite signals still abort.
- [ ] Derive `||S||_F<h_n*2^-126` for the all-subnormal region and
      `||0-S/2||_F<h_n*2^-127` for `h_n=ceil(sqrt(mn))`.
- [ ] Check the resulting parameter-displacement contribution
      `eta*h_n*2^-127` and its normalization into `h`.
- [ ] Confirm the wrapper records zero as an output-disturbance branch and
      never calls it a successful positive-sector return.
- [ ] Reproduce least-subnormal, odd-subnormal, exactly halvable, zero, and
      nonfinite controls.

Reviewer notes:

> Pending.

### 9. Weight-decay qualification

- [ ] Confirm the main smooth-PL profile sets `weight_decay=0`.
- [ ] Audit the separation between exact logical decay displacement, stored
      rounded decay-step error, and total master arithmetic residual.
- [ ] Verify `1/131072` is separately assumed for the exact logical decay
      displacement and the stored rounded decay-step norm, not derived from
      an arbitrary runtime coefficient. Confirm only the former enters `D`
      and the latter closes the range guard without double counting.
- [ ] Reproduce the robust conditional objective bounds about `0.26604` and
      `0.33128` and the forward-invariance checks.
- [ ] Re-derive the centered strongly-convex corollary from
      `||W-W*||<=||grad f(W)||/ell` and the inverse-storage gradient bound.
- [ ] Confirm ordinary zero-centered decay inherits exact convergence only
      when the relevant minimizer is zero.
- [ ] Reproduce the exact scalar nonzero-minimizer negative control, including
      next iterate `11999/12000` and gap `1/288000000`.

Reviewer notes:

> Pending.

### 10. Generator and independent reconstruction

- [ ] Run the canonical generator from a clean source commit and verify its
      source snapshot excludes its own output.
- [ ] Confirm the canonical artifact records exact fractions, matrices,
      leading minors, shape/rate profiles, guards, upstream source identity,
      P20 provenance, environment, lockfile, branch, and clean Git state.
- [ ] Inspect the reconstruction and confirm it imports neither the project
      theorem modules nor NumPy/SymPy/Torch.
- [ ] Run the reconstruction with `--require-canonical` and compare every
      reconstruction field exactly.
- [ ] Run the focused tests, full suite, lint, formatting, and dedicated P21
      CI on a clean checkout.

Reviewer notes:

> Pending.

### 11. Shadow-trace protocol and evidence boundary

- [ ] Verify the 24 capture steps and early/middle/late phase assignment were
      frozen before any real-gradient result.
- [ ] Check nearest-rank quantiles, zero-denominator semantics, shape coverage,
      P16/P18 fidelity gates, and all mild-intervention thresholds against the
      machine-readable protocol.
- [ ] Confirm a real observer must copy the candidate actually produced by
      the backend and must pass a trace-on/off noninterference test.
- [ ] Confirm no real-gradient result is present and the synthetic runner is
      labelled infrastructure-only.
- [ ] Verify the current blockers: no pinned trainer/instrumentation patch,
      dataset, tokenizer, checkpoint, trace fixture, CUDA/MPS device, and no
      P20 certificate for vanilla GPT-2 fused shape `768 x 2304`.
- [ ] Confirm sampled trace results, when eventually available, are never used
      as proof of global passivity, stability, or fidelity.

Reviewer notes:

> Pending.

### 12. Claim boundaries

- [ ] Confirm P21 is not described as stability of unmodified upstream Muon.
- [ ] Confirm no global neural-loss PL, arbitrary-initial-state finite-
      precision, generic bounded-variance stochastic, global fidelity,
      distributed, GPU, FTZ/DAZ, throughput, or training-quality theorem is
      claimed.
- [ ] Confirm the primary/secondary labels are respectively faster certified
      rate at `1/120` and maximum numerical step at `1/83`.
- [ ] Confirm every reported half-life elsewhere is explicitly a certified
      Lyapunov-bound quantity, not an observed objective-halving promise.

Reviewer notes:

> Pending.

## Sign-off

Reviewer name:

Review date:

Reviewed commit:

Decision: pending / approved / approved with corrections / rejected

Signature or verifiable approval reference:
