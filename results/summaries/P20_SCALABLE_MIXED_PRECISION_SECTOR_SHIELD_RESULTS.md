# P20 scalable mixed-precision sector-shield results

## Verdict

P20 is a positive, shape-parameterized finite-precision containment result.
It replaces P19's per-output exact-rational check with a statically certified
FP32/BF16-input proof-reference shield. Every successful stored FP32 return
from the locked arithmetic graph lies in the original P19 disk and therefore
in P18's pointwise sector `[125/1024,509/512]`.

The shield is memory-scalable and covers all seven declared Transformer
shapes, including `4096 x 11008` and `4096 x 14336`. It is not yet a complete
optimizer theorem or a GPU/backend-parity result. In particular, inheritance
of the P19 rates is conditional on identifying the stored matrix with the
abstract operator-port signal; FP32/BF16 signal casting and outer-loop state
rounding remain a P21 integration problem.

## Locked construction

P20 retains the exact P18/P19 disk

\[
 \mathcal D_S=\left\{U:
 \left\|U-\frac{1143}{2048}S\right\|_F
 \le \frac{893}{2048}\|S\|_F\right\}.
\]

For a finite stored candidate `C`, it materializes in FP32

\[
 \widehat D=\operatorname{fl}_{32}
 \left(C-\operatorname{fl}_{32}\left(\frac{1143}{2048}S\right)\right).
\]

The candidate is returned bit for bit when the locked scaled/balanced norm
screen verifies

\[
 \widehat N(\widehat D)\le
 \operatorname{down}_{64}\left(\frac{891}{2048}\widehat N(S)\right).
\]

A rejected finite candidate is contracted along the computed displacement
ray. The scalar is formed sequentially downward using

\[
 q_{64}=\operatorname{down}_{64}
   \left(\frac{\widehat N(S)}{\widehat N(\widehat D)}\right),\qquad
 a_{64}=\operatorname{down}_{64}\left(\frac{890}{2048}q_{64}\right),
\]

followed by a downward FP32 conversion, one FP32 displacement multiply, and
one FP32 addition to the materialized center. Only nonfinite candidates,
failed clip arithmetic, and the guarded subnormal branch use `fl32(S/2)`.
Nonfinite signals and unrepresentable all-subnormal signals fail closed
without an update.

This pass-through/radial-clip/fallback map is not P19's exact metric
projection. P20 does not claim fixed-input nonexpansiveness or global identity
on P18. Its guarantee is stored-output containment for every successful call.

## Frozen arithmetic contract

- The signal is finite; the candidate may also be nonfinite, in which case it
  takes the guarded fallback. Inputs are contiguous CPU `torch.float32` or
  `torch.bfloat16`; every finite BF16 value is widened exactly once to FP32.
  The output remains FP32.
- Arithmetic is IEEE-754 round-to-nearest, ties-to-even, with gradual
  underflow. FTZ, DAZ, stochastic rounding, FMA contraction, and reassociation
  are excluded.
- Each norm uses maximum scaling, materialized FP32 division and squaring, a
  fixed adjacent balanced FP32 tree, a materialized FP32 square root, and an
  exact FP64 product of the two FP32 norm parts.
- Reductions use aligned blocks of `2^20` entries followed by the same
  zero-padded adjacent tree. No BLAS, tensor-core, or unspecified reduction is
  part of the proof graph.
- The exceptional exact-halving bit guard uses bounded-size int32 blocks and
  is not evaluated for normal-anchored signals.
- The pass-through threshold and radial-clip scalar use the exact operation
  order above. There is no final BF16 cast and no runtime `Fraction` or
  big-integer postcheck.

The exact artifact locks the P10 outer-loop and P11 implementation-margin
artifacts because P20 reuses their relative-plus-absolute-underflow-crumb
ledger convention. It does not compose their disturbance ports into this
shield theorem.

## Exact shape margins

For shape `a x b`, the exact proof bounds the worst of the pass-through,
radial-clip, and half-fallback branches by `R_(a,b)` and certifies

\[
 \|U-(1143/2048)S\|_F
 \le \left(\frac{893}{2048}-\Delta_{a,b}\right)\|S\|_F,
 \qquad \Delta_{a,b}>0.
\]

| Shape | Exact `Delta_(a,b)` | Decimal lower margin |
| --- | ---: | ---: |
| `768 x 768` | `1019531105057811/1152921504606846976` | `8.84302271216e-4` |
| `768 x 3072` | `456959092551631/576460752303423488` | `7.92697665410e-4` |
| `768 x 50257` | `33874569209731/144115188075855872` | `2.35052041787e-4` |
| `3072 x 12288` | `35059980289411/144115188075855872` | `2.43277483501e-4` |
| `4096 x 4096` | `562014638485407/1152921504606846976` | `4.87469993612e-4` |
| `4096 x 11008` | `25250274108291/144115188075855872` | `1.75208973082e-4` |
| `4096 x 14336` | `71710053325847/1152921504606846976` | `6.21985564839e-5` |

The last shape is the overall tight case. For the radial-clip branch alone,
the minimum margin is

\[
 \frac{53997772719001}{576460752303423488}
 =9.36712039861\times10^{-5}.
\]

The complete proof includes norm scaling, balanced summation, multiplication,
subtraction, underflow crumbs, directed scalar construction, both rounded
clip vector operations, candidate storage, and the exceptional half fallback.
An independent standard-library reconstruction rebuilds every exact field
using only `Fraction` and integer square roots.

## Conditional P19 rate inheritance

The stored output satisfies the same full-matrix origin-centred pointwise
sector used by P18/P19. Hence, subject to every call along the trajectory
successfully emitting an output and exact identification of stored `S` with
the abstract operator-port signal, P19's otherwise-real-arithmetic
EMA/Nesterov theorem with `beta=19/20` retains both exact rates:

| Role | `eta` | Exact rate `q` | Certified Lyapunov-rate half-life |
| --- | ---: | ---: | ---: |
| maximum step | `1/83` | `999598040401/1000000000000` | about `1724.0734` |
| faster certified rate | `1/120` | `624350169/625000000` | about `666.314` |

The objective class is differentiable, globally `10`-smooth, bounded below,
and satisfies the global PL inequality with constant `1`. The conclusion is
function-value, gradient, momentum, and trajectory convergence to some global
minimizer; it is not uniqueness or arbitrary-pair contraction. The half-life
is for the certified Lyapunov bound, not every observed objective trajectory.

P20 alone does not justify silently replacing the theorem's real signal with
an FP32/BF16 cast or compose finite-precision EMA, Nesterov, parameters,
master weights, aspect scaling, weight decay, or distributed reductions.

## Candidate and activation study

The guarded P18 candidate retains additive normalization
`U/(||U||_F+1e-7)`, five Jordan stages with exact coefficients `6889/2000`,
`-191/40`, and `4063/2000`, P13's radial repair, `mu=1000`,
`lambda=1/1000`, P17's C2 gate with ceiling `3/4`, ray gain `K=1`, and
passive divisor `1024`.

The upstream comparator pins KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806` and audited `muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
The canonical and annulus `2 x 2` cases execute the literal clean-room BF16
matrix graph: one optional transpose, additive-`1e-7` Frobenius
normalization, and exactly five pinned Jordan stages. Large-shape upstream
records are packed singular-coordinate formula diagnostics, not literal
backend parity.

The frozen deterministic study produced:

| Candidate/diagnostic | Pass-through | Radial clip | Half fallback | Result |
| --- | ---: | ---: | ---: | --- |
| guarded P18 annulus | `2671/2688` | `17/2688` | `0` | `2176/2176` informative fidelity passes |
| pinned upstream annulus | `761/2688` | `1927/2688` | `0` | every returned value passes offline exact disk check |
| guarded P18 operating spectra | `14/14` | `0` | `0` | inactive on all declared operating cases |
| guarded P18 flat-boundary stresses | `0/7` | `7/7` | `0` | intentional boundary activation |
| pinned upstream shape spectra | `3/21` | `18/21` | `0` | all seven shapes exercised |

On `diag(3,4)`, both candidates pass through unchanged. Guarded P18 retains a
best-scalar departure of about `0.0362830` and an upstream-shaping retention
fraction of about `0.5185695`. Across the full annulus, the P18 clip changes
have maximum relative magnitude about `0.001458` and minimum candidate/output
cosine above `0.9999999979`.

Five deliberately corrupted finite candidates, including the one-ULP escape,
are radially clipped and pass offline exact-as-stored disk checks. Nonfinite
candidates use the safe half fallback; nonfinite signals are rejected. The
least FP32 subnormal and the maximum odd subnormal reproduce the unavoidable
near-zero representability failure. The exact theorem, not these samples,
establishes global shape-specific containment.

The default discrete decision digest is
`998ef020d1b642cf923a0789dfdd489b2b949a2aed0b3948cff67b88584a5c9c`.
The dedicated workflow requires both macOS and Ubuntu runs to reproduce it,
records each runner architecture, and archives their full study records. That
is parity evidence for the locked CPU proof-reference graph, not a claim about
unspecified platforms or accelerators.

## Evidence classification

| Statement | Evidence | Scope |
| --- | --- | --- |
| Stored-output containment | Exact rational shape recurrence | Every successful call under the locked graph and certified shapes |
| No runtime exact postcheck needed | Static inward-margin proof plus generated runtime table | Locked CPU FP32/BF16-input implementation |
| P19 rates survive | Exact sector implication | Conditional abstract-port identification; otherwise-real outer loop |
| P18 fidelity is normally preserved | Deterministic study | Declared canonical, annulus, and operating spectra only |
| Upstream can be used as a candidate | Candidate-independent shield theorem | Shielded output only; not unmodified upstream stability |
| Cross-platform decisions agree | Dedicated locked-digest CI gate | macOS/Ubuntu proof-reference executions after the workflow passes |

## Reproduction

```bash
uv run --locked python scripts/certify_scalable_sector_shield.py \
  --output results/summaries/scalable_sector_shield_certificate.json

uv run --locked python scripts/reconstruct_scalable_sector_shield.py \
  --canonical results/summaries/scalable_sector_shield_certificate.json \
  --require-canonical

uv run --locked python \
  experiments/mixed_precision/run_p20_scalable_sector_shield_study.py \
  --output results/summaries/p20_scalable_sector_shield_study.json
```

The independent replay imports neither the project theorem module nor a
numerical library. Automated reconstruction and cross-platform tests do not
replace the pending human proof audit.

## Remaining limitations

P20 does not certify unshielded upstream Muon, arbitrary GPU/BLAS/tensor-core
execution, FTZ/DAZ, a BF16 final output, stochastic rounding, global candidate
fidelity, exact P18 identity, projection nonexpansiveness, outer-loop casts,
optimizer-state or master-weight arithmetic, aspect scaling, weight decay,
distributed semantics, stochastic gradients, throughput, or neural-network
training. Calls may fail closed in the explicitly bounded all-subnormal input
region; that region is not an objective or Lyapunov neighborhood.
