# P10 finite-precision outer-loop certificate

## Status and scope

P10 supplies a finite-precision arithmetic shell around the P9 repaired
mixed-precision operator. It locks a CPU FP32 EMA/Nesterov graph and a
three-word compensated FP32 master-weight update, derives exact residual
envelopes for their internal ports, and closes those ports through the P7
storage inequality at shape `4096 x 11008`.

This is a theorem for a **proposed proof-reference implementation**. It is not
literal upstream Muon and does not claim bit parity with upstream
`torch.lerp_`, BLAS, GPU, tensor-core, fused, or compiler-reassociated
execution. The executable shell receives one finite represented FP32 gradient
sample. For the locked deterministic certificate this sample is the entrywise
binary32 cast of the exact gradient at the logical master weight, and that
final cast error is absorbed into the displayed outer residuals. Any noisy or
inexact gradient error *before* that cast remains P7's separate gradient-error
input and is set to zero in the locked `q_10` result. P10 does not analyze
gradient computation or a model forward pass.

The exact-real reference remains the P9 operator

\[
R(s)=\mathcal H_{q^{\circ5}}
\!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\tag{P10.1}
\]

with no additive epsilon, exactly five Jordan stages,

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\tag{P10.2}
\]

The P9 adapter uses a balanced FP32 normalizer and thick-Horner products with
two-term BF16 stage boundaries. Because the P10 signal is already binary32,
the relevant separately proved shape-specific P9 bound is the binary32-input
bound

\[
\lVert\widehat R(s)-R(s)\rVert_F
\le A\lVert s\rVert_F+B,
\quad
A_{32}=\frac{60114853}{549755813888},
\quad
B_{32}=\frac{1089541606515}{549755813888}.
\tag{P10.3}
\]

P9's larger all-real adapter slope is not applied a second time; P10 accounts
for its own final gradient cast explicitly before forming the FP32 signal.

## Locked FP32 EMA/Nesterov graph

Put

\[
\beta=\frac{19}{20},\qquad a=1-\beta=\frac1{20},
\qquad \eta=\frac1{32000}.
\tag{P10.4}
\]

The exact binary32 runtime values and bit patterns are

| Scalar | Binary32 value | Bits | Representation residual |
| --- | --- | --- | --- |
| `beta32` | `15938355/16777216` | `0x3f733333` | `-1/83886080` |
| `a32` | `13421773/268435456` | `0x3d4ccccd` | `1/1342177280` |
| `eta32` | `8589935/274877906944` | `0x3803126f` | `51/34359738368000` |

For one represented FP32 gradient sample `ghat_t`, the graph materializes

```text
bg     = fl32(a32 * ghat_t)
bm     = fl32(beta32 * m_t)
m_next = fl32(bm + bg)
bs     = fl32(beta32 * m_next)
s_next = fl32(bs + bg)
```

and reuses the same rounded `bg` tensor in the two additions. Every operation
is separate; FMA contraction and reassociation are excluded. Define the exact
real residuals

\[
\begin{aligned}
\epsilon_t^m&=m_{t+1}-\beta m_t-a \widehat g_t,\\
\epsilon_t^s&=s_{t+1}-\beta m_{t+1}-a \widehat g_t.
\end{aligned}
\tag{P10.5}
\]

These are the executable graph's exact algebraic residuals. For the locked
certificate, put \(g_t=\nabla f(W_t)\),
\(\widehat g_t=C_{32}(g_t)\), and absorb the final
cast into

\[
r_t^m=\epsilon_t^m+a(\widehat g_t-g_t),
\qquad
r_t^s=\epsilon_t^s+a(\widehat g_t-g_t).
\tag{P10.6}
\]

Thus the realized state and signal satisfy the desired true-gradient port
equations

\[
m_{t+1}=\beta m_t+a g_t+r_t^m,
\qquad
s_{t+1}=\beta m_{t+1}+a g_t+r_t^s
\tag{P10.7}
\]

exactly over the reals.

## Exact EMA/Nesterov envelopes

Let `u=2^-24`, `tau=2^-150`, `n=4096*11008=45088768`, and
`h_n=ceil(sqrt(n))=6715`. Applying the entrywise binary32 model

\[
|\operatorname{fl}_{32}(x)-x|\le u|x|+\tau
\tag{P10.8}
\]

to the locked graph gives

\[
\begin{aligned}
\lVert \epsilon_t^m\rVert_F
&\le C_m\lVert m_t\rVert_F+C_g\lVert \widehat g_t\rVert_F+b_{\rm ema},\\
\lVert \epsilon_t^s\rVert_F
&\le C_m\lVert m_{t+1}\rVert_F+C_g\lVert \widehat g_t\rVert_F+b_{\rm ema},
\end{aligned}
\tag{P10.9}
\]

where

\[
\begin{aligned}
C_m&=\frac{2955487301599231}{23611832414348226068480}
   =1.251697559823128\times10^{-7}\ldots,\\
C_g&=\frac{2533274891059201}{377789318629571617095680}
   =6.705522803684974\times10^{-9}\ldots,\\
b_{\rm ema}
&=\frac{168988514875}
{11972621413014756705924586149611790497021399392059392}.
\end{aligned}
\tag{P10.10}
\]

These are full Frobenius-norm upper envelopes, not observed residual maxima.
The last term retains the exact gradual-underflow crumb and is positive even
when a relative-error model alone would be invalid.

## Three-word compensated master

Represent the logical parameter by the exact real sum

\[
W=H+M+L,
\tag{P10.11}
\]

where each word is a finite FP32 matrix. For
`U=Rhat(s_next)`, one update executes

```text
step                    = fl32(eta32 * U)
pending                 = fl32(L - step)
middle_candidate, Lnext = TwoSum(M, pending)
Hnext, Mnext            = TwoSum(H, middle_candidate)
```

Knuth `TwoSum` is evaluated entrywise with every operation separately
materialized. Under round-to-nearest, gradual underflow, and no overflow in
the named intermediates, each returned pair sums exactly to its two inputs.
Consequently the logical master changes by `pending-L` exactly. Defining

\[
r_t^W=(H_{t+1}+M_{t+1}+L_{t+1})
      -(H_t+M_t+L_t)+\eta U_t,
\tag{P10.12}
\]

gives the third requested port equation

\[
W_{t+1}=W_t-\eta\widehat R(s_{t+1})+r_t^W.
\tag{P10.13}
\]

The exact Frobenius envelope is

\[
\lVert r_t^W\rVert_F
\le C_U\lVert U_t\rVert_F+u\lVert L_t\rVert_F+b_W,
\tag{P10.14}
\]

with

\[
\begin{aligned}
C_U&=\frac{50384023616225331}
{9671406556917033397649408000}
=5.209585939719435\times10^{-12}\ldots,\\
b_W&=\frac{225318017595}
{23945242826029513411849172299223580994042798784118784}.
\end{aligned}
\tag{P10.15}
\]

Unlike ordinary rounded subtraction, (P10.14) has no term proportional to
`||H||_F`: the high-word magnitude appears only in the explicit finite-range
guard below.

## Arithmetic guards

The executable outer-shell domain requires

- at most `2^52` entries and the P9 signal bound `maxabs(s)<=2^116`;
- `maxabs(H)<=2^30`, checked again after every update;
- `maxabs(M)<=2^7` and `maxabs(L)<=2^-16`;
- runtime `maxabs(U)<=2^16` and `maxabs(step)<=2`;
- the tighter certificate profile `maxabs(U)<=2^15`.

Under these premises, the exact guard audit gives `maxabs(pending)<3`,
`maxabs(middle_candidate)<2^8`, `maxabs(M_next)<=2^6`, and
`maxabs(L_next)<=2^-17`; the two lower-word guards are therefore preserved.
The high-word `TwoSum` cannot overflow. The coarse max-entry certificate
profile alone gives

\[
\lVert r_t^W\rVert_F
\le
\frac{686215821091555326528019070148118101046615043379139}
{598631070650737835296229307480589524851069969602969600}
=0.0011463083938252805\ldots.
\tag{P10.16}
\]

This coarse value is an operation-domain envelope; it is **not** inserted into
the closed-loop certificate. The P7 storage bounds the complete operator-
output Frobenius norm much more tightly. After substituting that bound, the
master port obeys

\[
\lVert r_t^W\rVert_F
\le \frac{93403}{549755813888}\sqrt{V_t}
   +\frac{6727}{1099511627776}.
\tag{P10.17}
\]

At `V_t<=1` the right side is approximately
`1.76017238118e-7`, not `0.0011463`.

The high-word bound is deliberately a conditional range premise. The global
PL class may have unbounded flat minimizer directions, so the value--momentum
storage cannot make `maxabs(H)<=2^30` forward invariant.

## Reduction to P7

Use the true-gradient notation \(g_t=\nabla f(W_t)\) for this algebra and put
`a=1-beta`. P7 sees the effective gradient error

\[
\xi_t^{\rm eff}=\frac{r_t^m}{a}=20r_t^m,
\tag{P10.18}
\]

which reproduces the implemented momentum state. Its corresponding nominal
Nesterov signal is

\[
s_t^0=\beta^2m_t+(1-\beta^2)g_t+(1+\beta)r_t^m,
\tag{P10.19}
\]

whereas the implemented signal is

\[
s_{t+1}=\beta^2m_t+(1-\beta^2)g_t+\beta r_t^m+r_t^s.
\tag{P10.20}
\]

Hence `s_(t+1)-s_t^0=r_t^s-r_t^m`. The actual logical parameter update is
exactly P7's parameter update with

\[
e_t^{\rm eff}=
R(s_{t+1})-R(s_t^0)
+\bigl(\widehat R(s_{t+1})-R(s_{t+1})\bigr)
-\frac{r_t^W}{\eta}.
\tag{P10.21}
\]

Using (P10.3) and the exact global repaired-operator Lipschitz bound

\[
K_R=\frac{336372400608849}{260261360000},
\tag{P10.22}
\]

gives

\[
\begin{aligned}
\lVert e_t^{\rm eff}\rVert_F
\le{}&A_{32}\lVert s_{t+1}\rVert_F+B_{32}
 +K_R\bigl(\lVert r_t^m\rVert_F+\lVert r_t^s\rVert_F\bigr)\\
&
 +\eta^{-1}\lVert r_t^W\rVert_F,
\end{aligned}
\tag{P10.23}
\]

The certificate bounds `s_(t+1)` directly through the five-operation FP32
signal graph, rather than re-expanding it through the residual ports. This is
an exact full-matrix port reduction; no sampled or diagonal estimate enters
it.

## Port-augmented P7 closure

Let `f` be differentiable, globally `10`-smooth, and satisfy the global PL
inequality with constant one under the P6--P7 convention. The locked result
uses the exact gradient at the logical three-word master, casts it once to
FP32 at the shell boundary, and has no earlier gradient noise. The final cast,
both outer state updates, the P9 operator, and the compensated master update
are all included in the residual reduction above.

Exact storage supplies and the arithmetic envelopes reduce to

\[
\begin{aligned}
\lVert r_t^m\rVert_F
&\le \frac{3222635}{1099511627776}\sqrt{V_t}
       +\frac1{1099511627776},\\
\lVert r_t^s\rVert_F
&\le \frac{522283}{137438953472}\sqrt{V_t}
       +\frac1{1099511627776},\\
\lVert r_t^W\rVert_F
&\le \frac{93403}{549755813888}\sqrt{V_t}
       +\frac{6727}{1099511627776}.
\end{aligned}
\tag{P10.24}
\]

All outward-rounded intercepts are positive, so subnormal effects are not
silently dropped. They imply the exact effective P7 envelopes

\[
\begin{aligned}
\lVert\xi_t^{\rm eff}\rVert_F
&\le\frac{16113175}{274877906944}\sqrt{V_t}
    +\frac5{274877906944},\\
\lVert e_t^{\rm eff}\rVert_F
&\le\frac{9288413563}{549755813888}\sqrt{V_t}
    +\frac{2179298479615}{1099511627776}.
\end{aligned}
\tag{P10.25}
\]

With locked Young parameters `theta_g=1` and `theta_e=837`, exact rational
absorption and outward rounding on P9's fixed `2^-40` grid prove, whenever
the arithmetic guards and `V_t<=1` hold,

\[
\boxed{
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776}.}
\tag{P10.26}
\]

Thus

\[
q_{10}=\frac{549700907325}{549755813888}
=0.9999001255437179\ldots<1,
\qquad
D_{10}=\frac{2162331}{1099511627776}
=1.966628587979\ldots\times10^{-6}.
\tag{P10.27}
\]

The exact comparison `D_10<=1-q_10` has positive numerator `107650795` on
the common denominator `2^40`. Hence `V_0<=1` makes `V_t<=1` forward
invariant, subject to the separately checked high-word premise. On this
storage set the certificate proves

\[
\lVert s_{t+1}\rVert_F
\le\frac{13872240500025}{549755813888}
=25.233458472985\ldots,
\qquad
\lVert\widehat R(s_{t+1})\rVert_F
\le\frac{8965070341576875}{274877906944}
=32614.735907\ldots<2^{15},
\tag{P10.28}
\]

and the entrywise maximum magnitude of the rounded parameter step is bounded
by

\[
\frac{70039619545}{68719476736}
=1.019210606\ldots<2.
\tag{P10.29}
\]

These inequalities
close the P9 input domain, the certificate operator-output profile, and the
lower-word invariants. They do not prove the conditional high word remains
below `2^30`; that guard must be checked at each call.

Using P7's exact storage-to-function conversion, (P10.26) gives the finite,
strictly subunit objective-gap neighborhood

\[
\boxed{
\limsup_{t\to\infty}(f(W_t)-f_\star)
\le \frac{399957341889}{549755813888}
=0.727518166766\ldots<1.}
\tag{P10.30}
\]

This is a worst-case certified bound, not a sampled loss or a tightness claim.

## Ordinary FP32 subtraction can stall

The executable negative control uses the actual P9 repaired operator on a
`2 x 2` state with represented `m=g=s=diag(64,0)`. Here `r^m=r^s=0` and

\[
\widehat R(s)_{11}=\frac{13231315}{256},
\qquad
\operatorname{fl}_{32}(\eta_{32}\widehat R(s)_{11})
=\frac{13548867}{8388608}>0.
\tag{P10.31}
\]

At `W_(11)=2^30`, this step is below the downward half-spacing `32`.
Ordinary FP32 subtraction therefore returns the unchanged bit pattern
`0x4e800000` on every one of 64 repeated updates. The three-word update puts
the first lost step in its middle word, changes its logical value immediately,
and transfers enough accumulated correction to move the high word on update
20. After 64 updates the high word is `1073741696<2^30` and the exact logical
value is

\[
\frac{140737474806461}{131072}.
\tag{P10.32}
\]

This is an explicit obstruction to the specified ordinary FP32 parameter
subtraction, not a claim that every finite-precision update or every optimizer
state stalls.

## Consequences and limitations

The closed P10 inequality is a pathwise input-to-storage/output result. It
controls objective gap, true-gradient norm, and momentum under its stated
port and range premises. It is not full-state ISS in `W`. For example, on
`f(x,y)=x^2/2`, errors entirely in the flat `y` direction can move parameters
while objective gap, gradient, momentum, and P7 storage remain zero. For any
positive `epsilon` small enough to fit the port budget, the harmonic choice
`r_t^W=(0,epsilon/(t+1))` is square summable but has divergent partial sums.
Absolute summability or additional geometry is needed for an iterate-
convergence conclusion. This is an obstruction to inferring a state bound
from the certified port envelopes; it is not a claim that the locked kernel
necessarily realizes that adversarial sequence.

The logical theorem parameter is the real sum of three FP32 words. P10 does
not specify how a neural-network forward pass consumes that expansion and
makes no storage, compression, or throughput claim. It also excludes gradient
evaluation, literal upstream `torch.lerp_` parity, current-plus-epsilon or
unrepaired normalization, weight decay, aspect-ratio scaling, stochastic
rounding, FTZ/DAZ, native BLAS/GPU/tensor-core kernels, and complete neural
training.

## Reproduction

```bash
uv run --locked python scripts/certify_outer_loop_roundoff.py \
  --output results/summaries/outer_loop_roundoff_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py \
  --require-canonical
uv run --locked python \
  experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py \
  --output results/summaries/finite_precision_outer_loop_diagnostic.json
```

The first two commands rebuild the exact certificate. The standalone
reconstruction must rebuild the arithmetic and port reduction before reading
the canonical JSON. The third command checks executable small-shape cases and
the exact stalling witness; it is falsification/operation-graph evidence, not
the global proof.
