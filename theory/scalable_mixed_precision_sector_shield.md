# Scalable mixed-precision sector-shield certificate

## Scope and result

This note specifies the P20 proof-reference shield.  It replaces P19's
runtime exact-rational postcheck with a shape-parameterized rounding proof.
The result is a certificate for a locked arithmetic graph, not a statement
about every implementation that happens to use FP32 or BF16.

Let

\[
 m=\frac{125}{1024},\qquad M=\frac{509}{512},\qquad
 \gamma=\frac{1143}{2048},\qquad r=\frac{893}{2048}.
\]

For a stored signal `S`, P19's certified set is

\[
 \mathcal D_S=\{U:\|U-\gamma S\|_F\le r\|S\|_F\}.
\tag{P20.1}
\]

P20 returns an arbitrary finite stored candidate unchanged when a
conservative rounded-norm comparison proves that it lies in a strictly inward
disk. A rejected finite candidate is contracted along its computed
displacement ray with a second, more conservative inward coefficient. Only
exceptional arithmetic uses the interior fallback `fl32(S/2)`. For every
successful call satisfying the range and representability guards below, the
stored FP32 output lies in (P20.1), without a runtime big-integer or rational
check. Consequently it satisfies the full-matrix origin-centred
**pointwise** sector `[125/1024,509/512]`. This is not an incremental sector
or a Jacobian bound.

The proof applies to every finite matrix entry, not merely to diagonal or
sampled matrices.  The exact shape recurrence is evaluated separately for
the seven declared Transformer shapes.  Numerical candidate studies are
falsification and fidelity evidence only; they do not establish the global
containment theorem.

## Locked arithmetic contract

The theorem-facing implementation has the following complete contract.

- The signal is finite; the candidate may contain finite values, infinities,
  or NaNs. Both are contiguous, strided CPU tensors in binary32 or bfloat16.
  Every finite bfloat16 value is widened exactly, entrywise, to binary32
  before any arithmetic. The returned tensor is binary32.
- Every arithmetic operation is IEEE-754 round-to-nearest, ties-to-even.
  Gradual underflow is required.  FTZ and DAZ are rejected by a runtime
  self-check and receive no positive containment claim.
- `gamma=1143/2048`, `r=893/2048`, and the inward comparison constant are
  exactly representable dyadic constants.
- Differences are formed by a materialized binary32 multiplication followed
  by a materialized binary32 subtraction.  Contraction into an FMA is not
  permitted.
- Each norm uses maximum scaling, materialized binary32 division and square,
  a fixed adjacent balanced binary32 addition tree, and a materialized
  binary32 square root.  An unpaired value is carried to the next level.
  Reassociation and unspecified BLAS, tensor-core, or reduction semantics are
  excluded.
- The two scaled norm parts are compared using the locked binary64 scalar
  graph. The acceptance threshold is rounded one binary64 value inward with
  `nextafter(...,-inf)`. A clip scale is formed, in order, by a binary64
  division followed by `nextafter` toward zero, a binary64 multiplication
  followed by `nextafter` toward zero, a ties-to-even conversion to binary32,
  and one binary32 `nextafter` toward zero. No full-size binary64 tensor is
  created.
- A nonfinite signal is rejected. A nonfinite candidate or a nonfinite
  candidate-displacement or clip calculation takes the guarded interior
  fallback. The all-subnormal exact-halving bit guard uses bounded-size int32
  blocks and is skipped entirely for normal-anchored signals.

This graph is memory-scalable, but it is intentionally a correctness-first
reference.  P20 makes no throughput or native accelerator-parity claim.

The candidate is otherwise unconstrained.  In particular, the containment
theorem does not rely on a resolvent stopping rule or on the candidate being
close to P18.  The two candidate families evaluated later are the P18
construction and the pinned five-step Keller--Jordan Muon orthogonalizer.
Candidate provenance does not change the shield theorem. P20 reuses P9's
scale-free balanced reduction and the P10/P11
relative-plus-absolute-underflow-crumb ledger convention; the exact artifact
locks all three dependencies. It does **not** yet compose the P10/P11 outer
arithmetic and implementation-error ports with this shield.

## Scaled balanced-norm enclosure

Put

\[
 u=2^{-24},\qquad \tau=2^{-150},\qquad \sigma_{\min}=2^{-126}.
\]

For shape `a x b`, let `n=ab`,

\[
 h_n=\lceil\sqrt n\rceil,\qquad
 L_n=1+\lceil\log_2 n\rceil,\qquad
 g_n=\frac{L_nu}{1-L_nu}.
\tag{P20.2}
\]

For a nonzero binary32 vector `x`, the locked norm routine first takes
`a=maxabs(x)` and forms `z=fl32(x/a)`.  At least one component of `z` is
exactly one.  Thus the balanced square sum is bounded away from zero even
when much smaller components underflow.  P9's relative-plus-crumb lemmas,
applied to this scaled graph, give exact shape-dependent factors

\[
 \ell_n\|x\|_F\le \widehat N_n(x)\le U_n\|x\|_F.
\tag{P20.3}
\]

The canonical certificate defines `ell_n` and `U_n` with outward rational
square-root enclosures.  The `h_n tau` term inside the normalized calculation
is dimensionless because `||x/a||_F>=1`; crumbs introduced when the scale is
restored are tracked separately.  No invalid relative-error model is applied
at zero.

## Inward acceptance and stored-output theorem

Let

\[
 k=r-\frac1{1024}=\frac{891}{2048}
\tag{P20.4}
\]

and form

\[
 \widehat D=\operatorname{fl}_{32}
 \bigl(C-\operatorname{fl}_{32}(\gamma S)\bigr).
\tag{P20.5}
\]

The candidate is returned unchanged only when the locked rounded comparison
proves

\[
 \widehat N_n(\widehat D)
 \le \operatorname{down}_{64}\bigl(k\widehat N_n(S)\bigr).
\tag{P20.6}
\]

Here `down64` is the locked one-step inward binary64 rounding operation.
Combining (P20.3), (P20.5), the FP32 multiplication and subtraction envelopes,
and the normal-anchor guard `maxabs(S)>=sigma_min` gives

\[
 \|C-\gamma S\|_F\le R^{\rm pass}_{a,b}\|S\|_F.
\tag{P20.7}
\]

More explicitly, the unchanged-candidate branch uses

\[
 R^{\rm pass}_{a,b}=
 \frac{kU_n/\ell_n+\gamma u(1+u)
       +h_n\tau(2+u)/\sigma_{\min}}{1-u}.
\]

The pass-through bound includes norm reduction, maximum scaling,
multiplication, subtraction, underflow crumbs, input widening, comparison
rounding, and the stored-output semantics. A returned candidate is already a
stored FP32 value, so there is no unmodelled final cast.

When (P20.6) rejects a finite candidate and both norms are positive and
finite, P20 forms

\[
 \begin{aligned}
 q_{64}&=\operatorname{down}_{64}
   \left(\widehat N_n(S)/\widehat N_n(\widehat D)\right),\\
 a_{64}&=\operatorname{down}_{64}\left((890/2048)q_{64}\right),\\
 a_{32}&=\operatorname{down}_{32}(a_{64}),\\
 Q&=\operatorname{fl}_{32}(a_{32}\widehat D),\\
 U&=\operatorname{fl}_{32}
   \left(\operatorname{fl}_{32}(\gamma S)+Q\right).
 \end{aligned}
\tag{P20.8}
\]

The directed construction gives
`a32 <= (890/2048) Nhat(S)/Nhat(Dhat)`. Paying separately for the
scaled-norm envelope and the two materialized FP32 vector operations yields

\[
 \frac{\|U-\gamma S\|_F}{\|S\|_F}
 \le \frac{890}{2048}\frac{U_n}{\ell_n}(1+u)^2
 +\gamma u(2+u)+\frac{h_n\tau}{\sigma_{\min}}(3+2u).
\tag{P20.9}
\]

The right-hand side is `R^(clip)_(a,b)`. With

\[
 R^{\rm half}_{a,b}=|\gamma-1/2|+h_n\tau/\sigma_{\min},
\]

the authoritative shape formula is

\[
 R_{a,b}=\operatorname{ceil}_{2^{-60}}
 \max\{R^{\rm pass}_{a,b},R^{\rm clip}_{a,b},R^{\rm half}_{a,b}\},
 \qquad \Delta_{a,b}=r-R_{a,b}.
\]

All quantities in `R_(a,b)` and `Delta_(a,b)` are generated with exact
`Fraction` arithmetic and directed square-root rounding.

This is a rounded contraction along `Dhat`, not the exact P19 metric
projection. Its extra inward coefficient is what keeps the stored result
inside the original disk after all clip arithmetic.

If the clip graph cannot produce a finite result, the shield returns
`fl32(S/2)`. Under the same normal-anchor guard,

\[
 \|\operatorname{fl}_{32}(S/2)-S/2\|_F\le h_n\tau
 \le h_nu\|S\|_F.
\tag{P20.10}
\]

The exact audit checks that this error fits strictly inside (P20.1) for every
certified shape.  This permits odd subnormal entries in an otherwise
normal-anchored tensor; entrywise exact halving is not assumed.

At `S=0`, the shield returns zero, the only member of `D_0`.  For a nonzero
all-subnormal signal, an exactly representable `S/2` is returned only when
its stored bits pass the locked sufficient rule.  Otherwise the call fails
closed without emitting an update. This exception is necessary: the least
positive subnormal has no representable output in the present sector, whose
lower gain is positive and whose upper gain is strictly below one.

The all-subnormal failure set is an explicit **input-space representability
dead zone**.  It is contained in

\[
 \|S\|_F<h_n\,2^{-126}.
\tag{P20.11}
\]

Equation (P20.11) is not an objective-error or Lyapunov neighborhood. A small
Nesterov signal can arise by cancellation even when the storage is not small.
P20 therefore makes no objective-neighborhood claim for a call that fails
closed.  Such a claim would require an additional output-disturbance port.

## Certified shapes

The exact recurrence is run for the same seven theorem targets as P9. The
displayed values are rigorous lower margins after taking the worst of the
pass-through, radial-clip, and half-fallback branches and rounding the radius
outward on the frozen `2^-60` grid.

| Shape | Role | certified `Delta_(m,n)` |
| --- | --- | ---: |
| `768 x 768` | square attention block | `8.84302271216e-4` |
| `768 x 3072` | fused projection / MLP block | `7.92697665410e-4` |
| `768 x 50257` | vocabulary-scale matrix | `2.35052041787e-4` |
| `3072 x 12288` | large MLP block | `2.43277483501e-4` |
| `4096 x 4096` | large square block | `4.87469993612e-4` |
| `4096 x 11008` | required Transformer MLP shape | `1.75208973082e-4` |
| `4096 x 14336` | required wide Transformer MLP shape | `6.21985564839e-5` |

The committed exact artifact is authoritative for every margin.  The largest
shape is the tight case, so replacing the balanced tree, enabling FTZ, adding
a final BF16 cast, or reassociating (P20.5) does not inherit the result.

## Conditional interface consequence for the P18/P19 loop

Every successful P20 return lies in the same exact P18 pointwise sector as a
real matrix. Therefore, when every call along a trajectory succeeds and the
stored matrix `S` is identified with the signal at the abstract operator
port, the P19 smooth-PL LMI needs no new numerical optimization. For the
pinned otherwise-real-arithmetic EMA/Nesterov ordering with `beta=19/20`, the
two sector implications remain

\[
 \mathcal V_{t+1}\le
 \frac{999598040401}{1000000000000}\mathcal V_t
 \quad\text{at }\eta=\frac1{83},
\tag{P20.12}
\]

and

\[
 \mathcal V_{t+1}\le
 \frac{624350169}{625000000}\mathcal V_t
 \quad\text{at }\eta=\frac1{120}.
\tag{P20.13}
\]

These are certified Lyapunov rates of the abstract sector interconnection,
not fixed observed-objective halving times. P20 alone does not prove that an
FP32/BF16 cast of an otherwise real signal can be silently substituted into
that interconnection. Nor does it compose finite-precision momentum,
Nesterov, parameter, master-weight, aspect-scaling, weight-decay, or
distributed arithmetic. Those input and outer-loop semantics are the P21
integration problem. Subject to the stated port identification, the usual
objective-gap, gradient, momentum, and trajectory-to-some-global-minimizer
conclusions follow on differentiable globally `10`-smooth, global-PL-`1`
objectives with finite infimum. PL does not imply a unique minimizer, and no
arbitrary-pair contraction is claimed.

Unlike P19's exact metric projection, the P20 pass-through/clip/fallback map
is not claimed to be fixed-input nonexpansive and is not globally the identity
on exact P18. Candidate quality affects fidelity and shield activation, not
the guarded pointwise-sector safety theorem. Normal inactivity is a declared
diagnostic, not a global identity result.

## Candidate diagnostics and provenance

The literal small-matrix upstream comparator follows the operation order in
KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, audited `muon.py` SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`:
cast the complete input to BF16, transpose once if tall, divide by the BF16
Frobenius norm plus the Python value `1e-7`, and apply exactly five Jordan
stages with literals corresponding to `6889/2000`, `-191/40`, and
`4063/2000` in the pinned order.  P20 does not copy or certify the surrounding
upstream optimizer.

The P18 comparator retains additive normalization
`U/(||U||_F+1e-7)`, the same five exact Jordan coefficients and stages,
P13's radial repair, `mu=1000`, `lambda=1/1000`, the P17 C2 gate with ceiling
`3/4`, ray gain `K=1`, and passive divisor `1024`. Its FP64 reference result
is converted once to the P20 FP32 candidate boundary. These details identify
the fidelity comparator; P20's candidate-independent containment theorem does
not depend on them.

Large-shape diagnostics use explicitly identified packed singular-coordinate
surrogates to avoid pretending that a CPU scalar study executed the pinned
backend's enormous dense matrix products.  They are formula/fidelity checks,
not literal backend parity.  Both families remain arbitrary candidates from
the safety theorem's perspective.

In the frozen study, the shield is bitwise inactive on `2671/2688` P18
annulus cases; the remaining `17` use the radial clip, not the half fallback.
All `2176/2176` informative P18 annulus cases still pass the fidelity gates.
It is inactive on all `14/14` declared operating Transformer spectra and
clips all seven deliberately flat boundary stresses. The pinned upstream
candidate is inactive on `761/2688` annulus cases and clipped on `1927`; this
is a safeguarded candidate result, not a theorem about unmodified upstream
Muon.

Adversarial outside candidates, one-ULP boundary controls, zero and subnormal
signals, nonfinite candidates, and the excluded FTZ mode are included as
falsification controls. Dedicated CI is configured to require a shared
discrete-decision digest for the locked small proof-reference graph on macOS
arm64 and Ubuntu x86_64. A passing comparison is useful parity evidence but
does not generalize the theorem to an unspecified backend.

## What is not proved

P20 does not prove that unshielded upstream Muon is stable, that the shield is
usually inactive on all training distributions, or that sampled fidelity is
global.  It does not cover native GPU/tensor-core reductions, FTZ/DAZ,
stochastic rounding, compiler reassociation, a BF16 output cast, aspect-ratio
scaling, weight decay, distributed reductions, FP32 EMA/Nesterov and master
weight composition, stochastic-gradient convergence, throughput, or neural
training.  Those require separate integration and parity certificates.

The exact generator and standard-library-only reconstruction, rather than
the displayed decimal summaries or sampled study, are the authoritative
rounding proof.  Independent human review remains required.
