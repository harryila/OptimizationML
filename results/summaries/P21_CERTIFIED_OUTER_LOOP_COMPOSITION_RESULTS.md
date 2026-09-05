# P21 certified outer-loop composition results

## Verdict

P21 is a positive stored-computation composition result. It connects P20's
stored FP32 sector directly to the FP32 EMA/Nesterov recurrence and the
compensated three-word FP32 master update. The proof is built around the
actual stored Nesterov signal, so it does not require the unavailable bound
`||T(s_hat)-T(s)||`.

The strongest unconditional algebraic statement is the exact pathwise
`7 x 7` port inequality. The complete finite-precision rate is an invariant-
domain result under explicit gradient-source, range, and optional decay-port
premises. It is not a theorem for arbitrary neural training or a native GPU
implementation, and the P20 supply remains pointwise rather than incremental.

## Stored graph

The locked CPU proof-reference path is

`stored BF16/FP32 gradient -> FP32 EMA/Nesterov -> five-stage BF16 Muon ->`
`aspect scaling -> P20 shield -> compensated three-word FP32 update`.

Aspect scaling occurs before the shield. The candidate retains additive
normalization `U/(||U||_F+1e-7)`, exactly five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`, and Keller--Jordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806` (`muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`).
The P20 shield is candidate-independent: candidate BF16 error affects
fidelity, while every successful shield output remains in the pointwise
sector `[125/1024,509/512]` relative to the stored FP32 signal.

## Exact stored-signal certificate

With `L=10`, `beta=19/20`, normalized momentum `z=m/L`, true gradient
`u=grad f(W)/L`, stored signal `p=s_hat/L`, sector residual `v`, and the three
ports `(a_m,a_s,h)`, the lifted variable order is

`(z,u,v,u_next,a_m,a_s,h)`.

Here `a_m` and `a_s` contain the complete momentum and stored-signal errors
relative to the true-gradient recurrence. The output-equivalent port `h`
contains master arithmetic, the all-subnormal wrapper, and any separately
bounded update. Exact rational replay gives

\[
 \mathcal V_{t+1}\le q\mathcal V_t
 +G_m\lVert a_t^m\rVert_F^2
 +G_s\lVert a_t^s\rVert_F^2
 +G_h\lVert h_t\rVert_F^2.
\]

| result | `eta` | exact base `q` | exact port gains `(G_m,G_s,G_h)` |
| --- | ---: | ---: | ---: |
| primary, faster rate | `1/120` | `624350169/625000000` | `(16384,1024,512)` |
| secondary, maximum step | `1/83` | `999598040401/1000000000000` | `(32768,2048,512)` |

Both `7 x 7` matrices are strictly negative definite by exact Sylvester
checks. Their zero-port leading blocks recover the corresponding P18/P19
`4 x 4` LMIs exactly. Thus the original rates survive at zero error, while
the new ports quantify the stored arithmetic rather than relabeling it as
exact.

The objective class is every differentiable globally `10`-smooth objective
with finite infimum satisfying the global PL inequality with constant `1`.
PL does not imply a unique minimizer, and the result is one-trajectory
storage convergence rather than arbitrary-pair contraction.

## Seven-shape finite-precision closure

The concrete port ledger includes final gradient storage, the reused rounded
`bg` term, FP32 EMA and Nesterov operations, stored learning-rate error, FP32
step multiplication, and compensated master accumulation. Model evaluation
uses the pre-update FP32 high word while the logical state is the exact sum of
three FP32 words. The resulting model-reconstruction discrepancy is exposed
before gradient storage; it is not assumed small merely because the lower
words satisfy range guards.

P21 reports two gradient-source premises:

| profile | pre-cast source bound |
| --- | --- |
| zero combined pre-cast source | `||zeta||_F=0` |
| locked robust source | `||zeta||_F <= (sqrt(V)+1)/4096` |

The robust row combines a conditional external/stochastic contribution
`(sqrt(V)+1)/8192` with a represented-master distance budget
`(sqrt(V)+1)/81920`; global `L=10` smoothness turns the latter into another
`1/8192` gradient contribution. These are premises checked by a future
integration, not consequences of the static three-word range guards.

For the robust profile, the exact shape recurrence certifies all seven P20
Transformer shapes at both operating points. The outward dyadic reports are:

| shape | primary `D` upper | primary objective-gap upper | secondary `D` upper | secondary objective-gap upper |
| --- | ---: | ---: | ---: | ---: |
| `768 x 768` | `5.18803e-8` | `4.99029e-4` | `1.03761e-7` | `2.58240e-3` |
| `768 x 3072` | `5.18812e-8` | `4.99037e-4` | `1.03761e-7` | `2.58240e-3` |
| `768 x 50257` | `5.18958e-8` | `4.99177e-4` | `1.03768e-7` | `2.58258e-3` |
| `3072 x 12288` | `5.18949e-8` | `4.99169e-4` | `1.03767e-7` | `2.58255e-3` |
| `4096 x 4096` | `5.18867e-8` | `4.99090e-4` | `1.03763e-7` | `2.58246e-3` |
| `4096 x 11008` | `5.18976e-8` | `4.99195e-4` | `1.03769e-7` | `2.58260e-3` |
| `4096 x 14336` | `5.19030e-8` | `4.99247e-4` | `1.03772e-7` | `2.58267e-3` |

The rounded rate upper bound is shape-independent at the displayed grid:

\[
 \bar q_{1/120}=\frac{1098368546995}{1099511627776}
 \approx0.998960374086,
\]

\[
 \bar q_{1/83}=\frac{549534922135}{549755813888}
 \approx0.999598200242.
\]

In every table row `D<=1-q_bar`, so `V<=1` is forward invariant. On the
tight `4096 x 14336` profile the exact objective-gap bounds are
`274464109/549755813888` at `1/120` and
`2839673189/1099511627776` at `1/83`. These finite neighborhoods are caused
by the explicitly retained absolute errors; they are not exact convergence
claims.

With the zero combined pre-cast source premise, the tight-shape primary reported rate
is `1098368443773/1099511627776`, the forcing is
`13/549755813888`, and the objective-gap upper bound is
`62517/274877906944` (about `2.275e-7`). The corresponding secondary values
are `68691854719/68719476736`, `13/1099511627776`, and
`323421/1099511627776` (about `2.942e-7`). Concrete FP32 rounding therefore
still gives a tiny neighborhood even when the external source is zero.

The unit-ball guard audit bounds the worst robust stored signal by about
`36.992`, P20 output by `36.775`, and the rounded operator step by `0.307`,
inside the locked output and total-step maxima `64` and `1`. The P10 middle
and low word guards replay; the high-word range `|high|<=2^30` remains a
premise.

## All-subnormal completion

If P20 reports that no positive-sector FP32 output is representable for a
nonzero all-subnormal signal, P21 returns zero. It compares that return to the
conceptual exact sector point `S/2`, adding the strict parameter-displacement
bound

\[
 \eta\lceil\sqrt{mn}\rceil2^{-127}
\]

to the output-equivalent port. This is not a successful P20 sector call and
does not pretend that zero satisfies the positive sector for nonzero `S`.
Nonfinite signals still abort without an update.

## Weight-decay scope

The primary smooth-PL result has `weight_decay=0`. Nonzero decoupled decay is
an explicit conditional port: the runtime exposes both the exact logical
decay displacement and stored decay-step rounding, but no arbitrary configured
coefficient is automatically certified.

Under separately predeclared bounds of `1/131072` on both the exact logical
decay displacement and stored rounded decay-step norm, the robust tight-shape
profiles remain forward invariant. The logical displacement enters `D`; the
stored-step bound closes the runtime guard and is not double-counted. Their
outward objective-gap neighborhoods are
`146255189439/549755813888` (about `0.26604`) at `eta=1/120` and
`364238397541/1099511627776` (about `0.33128`) at `eta=1/83`.

An exact-real corollary covers decay centered at a true minimizer on an
`ell`-strongly convex objective. The artifact checks `wd=1/1000000`, `ell=1`
at the primary rate and finds it contractive. Ordinary decay toward zero has that
interpretation only when the relevant minimizer is zero. A scalar quadratic
with minimizer one supplies the exact negative control.

## Shadow-trace status

The frozen 256-step protocol samples 24 optimizer steps in early, middle,
and late phases, observes every intended Muon matrix, and leaves training
unchanged. It predeclares shape-coverage, nonfinite/dead-zone, P16/P18
fidelity, amplitude/effective-step, activation, correction, cosine, and
output-amplitude gates. The synthetic CPU runner tests this observer and its
decision logic only. Its deterministic `2 x 2` fixture contains 144
observations: 126 pass through, 18 activate the shield, all 144 are successful
P20 calls, and there are no nonfinite or dead-zone events. All synthetic gate
checks pass. The correction p95 is about `0.18444`, cosine p05 about
`0.999813`, and output/candidate amplitude p05 about `0.81639`. These values
validate the frozen metric and aggregation machinery, not expected behavior
on training gradients.

The real-gradient study is **blocked and has not been run**. This checkout has
no NanoGPT trainer/instrumentation patch, data, tokenizer, checkpoint, or
real-gradient trace, and the current host has no CUDA or MPS accelerator.
Vanilla GPT-2's fused `768 x 2304` QKV matrix is also outside P20's certified
shape table. No synthetic observation is reported as neural-training
evidence.

## Evidence classification

| statement | evidence | scope |
| --- | --- | --- |
| stored-signal port inequality | exact rational `7 x 7` LMI | arbitrary finite matrix space satisfying the P20 pointwise supply |
| zero-port P18/P19 recovery | entrywise exact matrix equality | exact-real specialization only |
| concrete FP32 neighborhoods | exact outward port envelopes | seven shapes, stored CPU graph, `V<=1`, explicit source/range premises |
| all-subnormal completion | exact absolute disturbance bound | P21 zero wrapper, not a successful P20 return |
| nonzero decay neighborhood | conditional bounded update port | only when both declared decay bounds are met |
| shadow observer correctness | deterministic synthetic CPU tests | infrastructure only; no real gradients |

## Reproduction

```bash
uv run --locked python scripts/certify_outer_loop_composition.py \
  --output results/summaries/certified_outer_loop_composition_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_composition.py \
  --canonical results/summaries/certified_outer_loop_composition_certificate.json \
  --require-canonical
uv run --locked python experiments/training/run_p21_synthetic_shadow_trace.py \
  --output results/summaries/p21_synthetic_shadow_trace.json
```

See `../../theory/certified_outer_loop_composition.md` for the theorem and
`../../theory/p21_shadow_trace_protocol.md` for the empirical protocol. The
human proof audit remains pending and unsigned.

## Remaining limitations

P21 does not certify global PL for a neural loss, arbitrary initial states
outside the storage/range invariant, an arbitrary pre-cast model-gradient
error, generic stochastic-gradient variance, arbitrary weight decay,
unmodified upstream Muon, global fidelity, native GPU/tensor-core execution,
FTZ/DAZ, distributed reductions, production shape inventory, throughput, or
training loss. The precise next empirical prerequisite is a pinned trainer
and a fully P20-covered model shape inventory for the already frozen shadow
protocol.
