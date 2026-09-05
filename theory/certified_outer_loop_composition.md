# Certified stored-signal outer-loop composition

## Scope and result

P21 composes the P20 stored-output sector with the actual proof-reference
outer arithmetic.  The locked order is

\[
 \widehat g_t\longrightarrow \widehat m_{t+1}
 \longrightarrow \widehat s_{t+1}
 \longrightarrow \text{five-stage BF16 Muon candidate}
 \longrightarrow \text{aspect scale}
 \longrightarrow \text{P20 shield}
 \longrightarrow \text{compensated FP32 master update}.
\]

The central proof choice is that the sector supply is imposed directly at
the stored FP32 signal \(\widehat s_{t+1}\).  P21 never compares an operator
call at \(\widehat s\) with a call at an ideal signal \(s\): P20 proves a
full-matrix **pointwise** sector, not an incremental Lipschitz property.  The
result therefore removes P20's conditional identification of the stored
signal with its abstract port without inventing an unavailable regularity
theorem.

There are three levels of conclusion.

1. An exact dimension-independent `7 x 7` LMI gives a pathwise dissipativity
   inequality for arbitrary momentum, stored-signal, and output-equivalent
   ports.
2. Exact shape-specific FP32 envelopes close those ports on the storage unit
   ball for all seven P20 Transformer shapes, and for their transposes through
   a simultaneous signal/candidate orientation step.
3. A frozen shadow-trace protocol specifies how to measure intervention on
   real training gradients without changing training. No real-gradient trace
   is claimed in this repository: the required trainer, dataset, tokenizer,
   checkpoint, supported complete shape inventory, and accelerator are not
   available.

This is a CPU proof-reference result for differentiable globally
`10`-smooth, global-PL-`1` objectives with finite infimum. It does not prove
that a neural-network loss satisfies global PL, certify unmodified upstream
Muon, or establish GPU/tensor-core/distributed parity.

## Frozen operator and stored arithmetic

The candidate is the pinned Keller--Jordan/Muon graph at revision
`f98f1cacc0263b04290753e32be8d498c1efc806` and audited `muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
It uses additive normalization

\[
 U\mapsto \frac{U}{\lVert U\rVert_F+10^{-7}}
\]

and exactly five Jordan stages with coefficients `6889/2000`, `-191/40`,
and `4063/2000`. The candidate is stored in BF16. The aspect multiplier

\[
 \sqrt{\max(1,\operatorname{rows}/\operatorname{columns})}
\]

is applied in the candidate's stored dtype **before** the P20 shield. Scaling
after the shield is excluded because it need not preserve the certified
sector.

The theorem-facing runtime accepts a contiguous CPU BF16 or FP32 gradient,
stores it in FP32, and reuses the same rounded `(1-beta)*gradient` term in the
two P10 EMA/Nesterov additions. The FP32 coefficient words are

\[
 \widehat\beta=\frac{15938355}{16777216},\qquad
 \widehat{(1-\beta)}=\frac{13421773}{268435456},
\]

around the mathematical value \(\beta=19/20\). The stored Nesterov tensor is
passed unchanged to P20. If the certified orientation is transposed, signal
and aspect-scaled candidate are transposed together, copied contiguously,
shielded, and the FP32 output is transposed back.

The logical parameter is represented by the exact sum of three FP32 words
`high + middle + low`; model evaluation uses the pre-update `high` word. The
primary and secondary mathematical learning rates and their stored FP32 words
are

| role | mathematical `eta` | stored FP32 word | exact stored value |
| --- | ---: | ---: | ---: |
| primary, faster certified rate | `1/120` | `0x3c088889` | `8947849/1073741824` |
| secondary, maximum step | `1/83` | `0x3c4565c8` | `1617081/134217728` |

The stored update materializes `fl32(eta32*output)` and optional decoupled
decay before the public P10 two-`TwoSum` three-word accumulator. The exact
master-residual object compares the logical before/after sums to the
mathematical `eta`; no ordinary FP32 parameter subtraction is substituted.

The P20 arithmetic contract remains in force: CPU round-to-nearest,
ties-to-even, gradual underflow, fixed balanced reductions, explicit
materialization boundaries, and no FMA/reassociation. P21 does not extend the
claim to FTZ/DAZ or an unspecified backend.

## Stored-signal `7 x 7` certificate

Let \(g_t=\nabla f(W_t)\), \(L=10\), and normalize

\[
 z_t=\frac{m_t}{L},\qquad u_t=\frac{g_t}{L},\qquad
 p_t=\frac{\widehat s_{t+1}}{L}.
\]

Define the total normalized arithmetic ports relative to the true gradient:

\[
\begin{aligned}
 a_t^m&=\frac{\widehat m_{t+1}-\beta m_t-(1-\beta)g_t}{L},\\
 a_t^s&=\frac{\widehat s_{t+1}-\beta\widehat m_{t+1}-(1-\beta)g_t}{L}.
\end{aligned}
\]

Then the actual stored recurrence is represented exactly by

\[
\begin{aligned}
 z_{t+1}&=\beta z_t+(1-\beta)u_t+a_t^m,\\
 p_t&=\beta^2z_t+(1-\beta^2)u_t+\beta a_t^m+a_t^s.
\end{aligned}
\tag{P21.1}
\]

P20 places its stored output \(T_t\) in the pointwise sector
`[125/1024,509/512]`. Equivalently, with

\[
 \gamma=\frac{1143}{2048},\qquad K_T=\frac{893}{2048},
\]

write

\[
 \frac{T_t}{L}=\gamma p_t+K_Tv_t,qquad
 \lVert v_t\rVert_F\le\lVert p_t\rVert_F.
\tag{P21.2}
\]

The aggregate normalized output-equivalent port \(h_t\) is defined by

\[
 W_{t+1}-W_t=-\eta\gamma L
 \left(p_t+\frac{K_T}{\gamma}v_t+h_t\right).
\tag{P21.3}
\]

For example, a logical master residual \(r_t^W\) contributes
`-r_t^W/(eta*gamma*L)` to \(h_t\), whereas an additional intended update
`-eta*d_t` contributes `d_t/(gamma*L)`. This sign convention is locked.

For

\[
 F_t=\frac{f(W_t)-f^\star}{L},\qquad
 \mathcal V_t=
 \left\langle
 \begin{bmatrix}z_t\\u_t\end{bmatrix},
 (P\otimes I)
 \begin{bmatrix}z_t\\u_t\end{bmatrix}
 \right\rangle_F+F_t,
\]

the lifted variable order is exactly

`(z_t, u_t, v_t, u_(t+1), a_t^m, a_t^s, h_t)`.

Exact directed smooth-interpolation, next-sample PL, and sector supplies give

\[
 \mathcal V_{t+1}\le q\mathcal V_t
 +G_m\lVert a_t^m\rVert_F^2
 +G_s\lVert a_t^s\rVert_F^2
 +G_h\lVert h_t\rVert_F^2.
\tag{P21.4}
\]

The two exact certificates are:

| role | `eta` | base `q` | `(G_m,G_s,G_h)` |
| --- | ---: | ---: | ---: |
| primary | `1/120` | `624350169/625000000` | `(16384,1024,512)` |
| secondary | `1/83` | `999598040401/1000000000000` | `(32768,2048,512)` |

The storage matrices are the frozen P18/P19 matrices:

\[
 P_{1/120}=\begin{bmatrix}599/1000&-231/625\\-231/625&657/2500\end{bmatrix},
 \quad
 P_{1/83}=\begin{bmatrix}97/125&-151/500\\-151/500&17/100\end{bmatrix}.
\]

Exact Sylvester checks prove both storage matrices positive definite and both
`7 x 7` dissipation matrices strictly negative definite. Setting
`a_m=a_s=h=0` recovers the corresponding P18/P19 `4 x 4` LMI entry for entry,
including exact function-value cancellation. This is an exact-real recovery,
not a claim that concrete FP32 rounding vanishes.

The core inequality is dimension independent: every scalar matrix is tensored
with the identity on the finite matrix space. Its use of the stored signal is
global. The concrete finite-precision closure below additionally assumes the
declared storage unit ball and range guards.

## Exact finite-precision port closure

Every error before final FP32 gradient storage is collected in \(\zeta_t\).
This includes stochastic/model-gradient error and evaluating the gradient at
the represented high word instead of the logical three-word master. If the
gradient is evaluated at `high`, smoothness gives

\[
 \lVert\zeta_t\rVert_F
 \le \lVert\text{gradient noise}_t\rVert_F
   +L\lVert\text{middle}_t+\text{low}_t\rVert_F.
\tag{P21.5}
\]

The static three-word range guards do not make the right side small. P21
therefore reports two explicit profiles rather than silently deriving a model
reconstruction guarantee:

- zero combined pre-cast source: \(\lVert\zeta_t\rVert_F=0\);
- locked robust source premise:
  \(\lVert\zeta_t\rVert_F\le(\sqrt{\mathcal V_t}+1)/4096\).

The robust premise is itself the sum of two predeclared affine budgets. The
external/stochastic gradient contribution is at most
`(sqrt(V)+1)/8192`. The represented-master distance is at most
`(sqrt(V)+1)/81920`; multiplication by `L=10` in (P21.5) contributes another
`(sqrt(V)+1)/8192`. Their sum is the displayed `1/4096` source budget. The
runtime exposes the reconstruction discrepancy exactly, but neither component
budget follows from the static three-word guards.

The exact P10 relative-plus-underflow-crumb recurrences reduce gradient cast,
reused `bg`, EMA rounding, stored-signal rounding, learning-rate storage,
FP32 step rounding, and three-word accumulation into affine bounds

\[
 \lVert a_t^j\rVert_F\le A_j\sqrt{\mathcal V_t}+B_j,
 \qquad j\in\{m,s,h\}.
\]

Applying exact Young inequalities with all three parameters equal to one
yields

\[
 \mathcal V_{t+1}\le \bar q\mathcal V_t+D.
\tag{P21.6}
\]

For the worst declared shape, `4096 x 14336`, the outward dyadic reports are:

| source profile | `eta` | `q_bar` upper | `D` upper | objective-gap ultimate upper |
| --- | ---: | ---: | ---: | ---: |
| zero combined pre-cast source | `1/120` | `1098368443773/1099511627776` | `13/549755813888` | `62517/274877906944` (`2.275e-7`) |
| zero combined pre-cast source | `1/83` | `68691854719/68719476736` | `13/1099511627776` | `323421/1099511627776` (`2.942e-7`) |
| robust `1/4096` source | `1/120` | `1098368546995/1099511627776` | `14267/274877906944` | `274464109/549755813888` (`4.993e-4`) |
| robust `1/4096` source | `1/83` | `549534922135/549755813888` | `57049/549755813888` | `2839673189/1099511627776` (`2.583e-3`) |

All seven shapes pass at both rates. In every robust case, `q_bar<1` and
`D<=1-q_bar`; consequently `V<=1` is forward invariant. The objective bound
uses `f(W)-f* <= 10 V`. On the largest-shape robust profile, the unit-ball
guards give stored-signal norm below `36.992`, shield-output norm below
`36.775`, rounded operator step below `0.307`, the locked output maximum
`64`, and the locked combined-step maximum `1`. The three-word lower-word
invariants replay exactly; the high-word range remains an explicit
conditional guard (`|high|<=2^30`).

These are sufficient worst-case envelopes, not measured typical errors. A
stochastic gradient is covered only when each realized pre-cast error obeys
the stated pathwise affine premise. P21 does not derive that premise from
unbiasedness or a variance bound.

## All-subnormal wrapper

P20 can fail to represent any positive-sector output for a nonzero
all-subnormal stored signal. P21 closes the runtime call by returning zero and
measuring the difference from the exact interior sector point `S/2` as an
absolute output/update disturbance. For an `a x b` signal, with
\(h_{ab}=\lceil\sqrt{ab}\rceil\),

\[
 \lVert S\rVert_F<h_{ab}2^{-126},\qquad
 \lVert 0-S/2\rVert_F<h_{ab}2^{-127},
\]

so the parameter-displacement contribution is strictly below

\[
 \eta h_{ab}2^{-127}.
\tag{P21.7}
\]

This branch is not reclassified as a successful P20 sector return: zero lies
outside the positive sector for nonzero `S`. It is safe only through the
explicit absolute port in (P21.4). A nonfinite signal still aborts without an
update; a nonfinite candidate is already handled by P20's guarded fallback.

## Weight decay

The primary smooth-PL result sets `weight_decay=0`. For nonzero decoupled
decay, P21 exposes both its exact logical displacement and its stored rounded
decay step. A theorem user must supply a separate per-step bound; arbitrary
configured runtime decay is not automatically certified.

As a declared conditional profile, the exact logical decay displacement and
the stored rounded decay-step norm are each bounded by `1/131072`. The logical
displacement enters the LMI forcing; the stored-step bound closes the runtime
range guard without being counted a second time in `D`. Combined with the
robust gradient-source premise, the profile remains forward invariant on the
storage unit ball. Its outward objective-neighborhood bounds on the tight
shape are approximately `0.26604` at `eta=1/120` and `0.33128` at `eta=1/83`.
These are deliberately reported as bounded-port results, not exact
convergence to the unregularized minimizer.

There is a separate exact-real strong-convexity corollary for decay centered
at the true minimizer:

\[
 -\eta\,\mathrm{wd}\,(W_t-W^\star).
\]

Strong convexity with constant \(\ell\) gives
`||W-W*||<=||grad f(W)||/ell`, allowing this update to enter (P21.4) as a
zero-intercept output port. For example, `wd=1/1000000` and `ell=1` remains
contractive at the primary operating point recorded in the exact artifact.
Ordinary decay toward zero inherits
this corollary only when the relevant minimizer is zero. The exact scalar
control `f(w)=(w-1)^2/2`, started at its minimizer with `eta=1/120` and
`wd=1/100`, moves to `11999/12000` and creates gap `1/288000000`; global PL
alone cannot justify a zero-centered exact-convergence claim.

## Shadow-mode gradient trace

The frozen protocol observes all intended Muon matrix parameters at 24
predeclared optimizer steps: `0,...,7`, `124,...,131`, and `248,...,255`.
Training must remain bitwise unchanged; the stored candidate is copied before
aspect scaling, and the P20 result is recorded but never applied. Nearest-rank
quantiles and all zero-denominator cases are defined in advance.

The protocol retains the P16/P18 shaping, annulus, amplitude, and effective-
step gates and adds predeclared mild-intervention gates: activation at most
`1/4` overall and `1/2` per phase/role cell with at least 20 observations;
relative-correction median at most `1/20`, p95 at most `1/4`, and maximum at
most one; cosine p05 at least `19/20` and minimum at least `4/5`; and
output/candidate amplitude p05 at least `3/4` and p95 at most `5/4`.

No real-gradient result has been produced. This checkout contains no pinned
NanoGPT trainer or instrumentation patch, dataset, tokenizer, checkpoint, or
real-gradient fixture; the current host has neither CUDA nor MPS. Moreover,
vanilla GPT-2 has a fused `768 x 2304` QKV matrix absent from P20's certified
shape table. The supplied synthetic CPU runner exercises only the observer
and gate logic. It contains no model, data, backward pass, accelerator, or
real gradient and cannot satisfy the empirical gate.

## Reproduction and evidence boundary

```bash
uv run --locked python scripts/certify_outer_loop_composition.py \
  --output results/summaries/certified_outer_loop_composition_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_composition.py \
  --canonical results/summaries/certified_outer_loop_composition_certificate.json \
  --require-canonical
uv run --locked python experiments/training/run_p21_synthetic_shadow_trace.py \
  --output results/summaries/p21_synthetic_shadow_trace.json
```

The generator and independent standard-library reconstruction carry the exact
LMI and rounding claims. The synthetic trace is an infrastructure diagnostic,
not evidence about training gradients. Automated replay does not replace the
pending independent human proof audit.

## Not established

P21 does not establish global PL for a neural loss, arbitrary-initialization
finite-precision convergence outside the declared invariant domain, a small
represented-model reconstruction port from static word guards alone,
automatic certification of an arbitrary weight-decay coefficient, generic
bounded-variance stochastic convergence, unmodified-upstream stability,
global candidate fidelity or mild intervention, native accelerator parity,
FTZ/DAZ semantics, distributed reductions, throughput, or training quality.
The ordinary upstream candidate remains heavily clipped on P20's synthetic
annulus; only the blocked real-gradient shadow study can determine whether
that intervention is mild in the intended operating distribution.
