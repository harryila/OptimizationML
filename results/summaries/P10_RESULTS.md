# P10 finite-precision outer-loop result

## Result

P10 closes the outer arithmetic ports left open by P9 for one proposed CPU
proof-reference optimizer shell at shape `4096 x 11008`. It combines:

- the P9 balanced-FP32, compensated-BF16 repaired operator;
- a separately rounded FP32 EMA/Nesterov graph at `beta=19/20`;
- the full step `eta=1/32000`; and
- a three-word FP32 master parameter updated by two error-free `TwoSum`
  cascades.

The operator remains the repaired exact max-floor map with `c=1`, no additive
epsilon, coefficients `(6889/2000,-191/40,4063/2000)`, exactly five Jordan
stages, and `rho=210177835339081/260261360000`.

The realized shell obeys the exact real port equations

\[
\begin{aligned}
m_{t+1}&=\beta m_t+(1-\beta)g_t+r_t^m,\\
s_{t+1}&=\beta m_{t+1}+(1-\beta)g_t+r_t^s,\\
W_{t+1}&=W_t-\eta\widehat R(s_{t+1})+r_t^W,
\end{aligned}
\]

where `W` is the exact real sum of the high, middle, and low FP32 master
words. The executable gradient argument is one represented, finite FP32
sample. In the locked deterministic certificate it is the entrywise FP32 cast
of the exact gradient at the logical master, and that last cast is absorbed in
the outer residual envelopes. Noisy or inexact evaluation before the cast
remains P7's separate gradient-error input and is zero in the locked result.

## Exact rounding envelopes

At `4096 x 11008`, with `ceil(sqrt(4096*11008))=6715`, exact rational error
propagation proves

\[
\begin{aligned}
\lVert \widetilde r_t^m\rVert_F
&\le C_m\lVert m_t\rVert_F+C_g\lVert \widehat g_t\rVert_F+b_{\rm ema},\\
\lVert \widetilde r_t^s\rVert_F
&\le C_m\lVert m_{t+1}\rVert_F+C_g\lVert \widehat g_t\rVert_F+b_{\rm ema},\\
\lVert r_t^W\rVert_F
&\le C_U\lVert\widehat R(s_{t+1})\rVert_F
 +2^{-24}\lVert L_t\rVert_F+b_W,
\end{aligned}
\]

Here the tilded residuals are relative to the represented gradient. The total
theorem ports add
\((1/20)(\widehat g_t-\nabla f(W_t))\). After exact storage reduction,

\[
\begin{aligned}
\lVert r_t^m\rVert_F
&\le\frac{3222635}{1099511627776}\sqrt{V_t}
 +\frac1{1099511627776},\\
\lVert r_t^s\rVert_F
&\le\frac{522283}{137438953472}\sqrt{V_t}
 +\frac1{1099511627776}.
\end{aligned}
\]

with

\[
\begin{aligned}
C_m&=1.251697559823128\times10^{-7}\ldots,\\
C_g&=6.705522803684974\times10^{-9}\ldots,\\
C_U&=5.209585939719435\times10^{-12}\ldots.
\end{aligned}
\]

The committed exact fractions retain coefficient-representation errors and
positive gradual-underflow crumbs. The coarse operation-domain profile
`maxabs(operator output)<=2^15`, `maxabs(low)<=2^-16` gives

\[
\lVert r_t^W\rVert_F\le
\frac{686215821091555326528019070148118101046615043379139}
{598631070650737835296229307480589524851069969602969600}
=0.0011463083938252805\ldots.
\]

That max-entry envelope is not used to close the rate. Substituting the
storage-derived Frobenius output bound instead gives

\[
\lVert r_t^W\rVert_F
\le \frac{93403}{549755813888}\sqrt{V_t}
 +\frac{6727}{1099511627776},
\]

which is about `1.76017238118e-7` when `V_t<=1`.

## Compensated master and raw-FP32 obstruction

The master update forms `step=fl32(eta32*U)`, rounds `low-step`, then applies
`TwoSum(middle,pending)` and `TwoSum(high,middle_candidate)`. Under the locked
RNE, gradual-underflow and no-overflow contract, the logical real sum changes
by the pending term exactly. Its update error therefore has no term
proportional to the high-word magnitude.

The guard audit assumes `maxabs(high)<=2^30`, `maxabs(middle)<=2^7`,
`maxabs(low)<=2^-16`, and the certificate output profile above. It proves the
lower-word guards forward invariant and checks the high guard after every
step. The high guard is conditional: PL value storage cannot bound travel
along an unbounded flat minimizer set.

An executable exact witness uses the actual P9 repaired operator with
`m=g=s=diag(64,0)`. Its `(1,1)` output is `13231315/256`, and the rounded
positive step is `13548867/8388608`, below the downward half-spacing at
`W_(1,1)=2^30`. Ordinary FP32 subtraction retains bit pattern `0x4e800000`
for all 64 repeated updates. The compensated logical master moves on the
first update and its high word moves on update 20. This is a specific
operation-graph obstruction, not a universal finite-precision impossibility.

## Port-augmented storage result

For the full smooth nonconvex-PL class from P7, with global smoothness `10`,
PL constant `1`, and zero error before the locked final gradient cast, the
exact port reduction with Young parameters `theta_g=1`, `theta_e=837` proves
on `V_t<=1`

\[
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776}.
\]

The rate and forcing are

\[
q_{10}=0.9999001255437179\ldots<1,
\qquad
D_{10}=1.966628587979\ldots\times10^{-6}.
\]

The exact check `D_10<=1-q_10` makes `V<=1` invariant, apart from the
separately conditional high-word range premise. On this invariant set the
signal norm is below `25.233459`, the repaired P9 output norm is below
`32614.736<2^15`, and the rounded-step max-absolute entry is below
`1.019211<2`, closing the
P9, output, step, middle-word, and low-word gates. P7's storage-to-function
conversion then yields

\[
\limsup_t(f(W_t)-f_\star)
\le \frac{399957341889}{549755813888}
=0.727518166766\ldots<1.
\]

This is a finite nonvacuous worst-case objective-gap neighborhood, not a
measured loss. The theorem controls the P7 storage and its function-value,
true-gradient, and momentum outputs; it does not turn the PL class into a
full-state or unique-parameter guarantee.

## CPU operation-graph diagnostic

A deterministic CPU diagnostic covers shapes `1 x 1`, `2 x 3`, and
`8 x 16`, with nine EMA cases and six master-update cases. It performs

- 810 exact EMA residual-entry checks;
- 270 exact logical-master residual checks;
- 544 exact `TwoSum` identities; and
- 270 exact logical-update identities.

All candidate-violation counts are zero. The same run replays the exact actual-
P9 `2 x 2` stalling witness, including its compensated high-word move at
iteration 20. The finite exact identities check the named operation graph,
while the sampled case grid is falsification evidence only; neither replaces
the global rational envelopes or storage certificate.

## Evidence and limitations

The exact rational certificate and its standalone standard-library
reconstruction are authoritative. Executable residual identities, `TwoSum`
edge probes, small-shape operator parity, and the stalling witness validate
the locked operation graph but do not replace the global proof.

P10 is not literal upstream Muon. It locks the same EMA/Nesterov algebraic
ordering, but not upstream `torch.lerp_` bit semantics. It assumes a
represented FP32 gradient sample and does not analyze gradient computation,
model-forward consumption of the three-word master, native BLAS/GPU/tensor-
core execution, weight decay, aspect scaling, current-plus-epsilon
normalization, stochastic rounding, FTZ/DAZ, or neural-network convergence.
The result controls storage, function value, true gradient, and momentum; it
does not establish full-state ISS or parameter convergence on a nonunique PL
minimizer set.

## Reproduce

```bash
uv run --locked python scripts/certify_outer_loop_roundoff.py \
  --output results/summaries/outer_loop_roundoff_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py \
  --require-canonical
uv run --locked python \
  experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py \
  --output results/summaries/finite_precision_outer_loop_diagnostic.json
```

The first two commands are the theorem evidence. The third is the scoped CPU
diagnostic and exact finite witness replay.
