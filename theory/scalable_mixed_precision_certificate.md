# P9 shape-parameterized mixed-precision certificate

## Status and scope

P9 certifies a proposed, shape-locked proof-reference implementation of the
repaired max-floor Jordan operator.  It is a shape-parameterized exact
forward-error theorem, not a sampled error estimate and not a claim about an
existing BLAS, GPU, tensor-core, or upstream Muon kernel.

For fixed positive integers `r,c`, put

\[
p=\min\{r,c\},\qquad d=\max\{r,c\},\qquad n=rc.
\]

The implementation transposes once when `r>c`, evaluates every polynomial
stage on a `p x d` matrix, and restores the original orientation once at the
output.  The exact-real reference is

\[
R_{r,c}(s)=\mathcal H_{q^{\circ5}}
\!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\tag{P9.1}
\]

with no additive epsilon, exactly five stages, and

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\tag{P9.2}
\]

Below, `(a_q,b_q,c_q)` denotes the three polynomial coefficients in this
order; the subscript avoids confusing `c_q` with the public column count `c`.

Let `C32` denote entrywise binary32 round-to-nearest, ties-to-even conversion,
and let `Khat_(r,c)` be the locked binary32 kernel specified below.  The
all-real adapter is

\[
\widehat R_{r,c}(s)=\widehat K_{r,c}(C_{32}(s)).
\]

For every shape whose exact audit gates pass, every real `r x c` input with
`max_ij |s_ij| <= 2^116` satisfies

\[
\boxed{
\lVert\widehat R_{r,c}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c}.}
\tag{P9.3}
\]

The domain also requires `n<=2^52`.  The quantities `A_(r,c)` and `B_(r,c)`
are the exact rationals produced by the recurrence below.  The seven locked
representative Transformer dimensions all pass.  On that table,

\[
A_{r,c}=\frac{102465557}{549755813888}
=0.00018638376241142396\ldots,
\tag{P9.4}
\]

while `B_(r,c)` is shape dependent and at most

\[
\frac{2179083213031}{1099511627776}
=1.9818646369740236\ldots.
\tag{P9.5}
\]

Neither coefficient is claimed minimal.

## Locked arithmetic contract

The executable contract assumes IEEE round-to-nearest, ties-to-even
arithmetic, gradual underflow, and no FTZ/DAZ behavior.  Write

\[
u=2^{-24},\quad \tau=2^{-150},\qquad
u_b=2^{-8},\quad \tau_b=2^{-134}.
\]

The real coefficients in (P9.2) are rounded once to the exact FP32 dyadics

\[
a_{32}=\frac{902955}{262144},\quad
b_{32}=-\frac{10013901}{2097152},\quad
c_{32}=\frac{8520729}{4194304},
\]

and the repair coefficient is

\[
\rho_{32}=\frac{13231137}{16384}.
\]

The normalizer first forms `sigma=max(1,maxabs(s))`, then

\[
z=\operatorname{fl}_{32}(s/\sigma),\qquad
\widehat d=\max\!\left\{
\operatorname{fl}_{32}(1/\sigma),
\operatorname{fl}_{32}\!\left(\sqrt{
\operatorname{pair}_{32}\sum z_{ij}^2}\right)
\right\},
\]

and returns `fl32(z/dhat)`.  Both the norm sum and every matrix dot product use
fixed adjacent balanced FP32 additions.  An unpaired final value is carried
unchanged to the next tree level.

At each boundary, including the initial normalized state, an FP32 matrix `Y`
is stored as two BF16 tensors:

\[
h=\operatorname{RN}_b(Y),\quad
r_Y=\operatorname{fl}_{32}(Y-\operatorname{fp32}(h)),\quad
\ell=\operatorname{RN}_b(r_Y),\quad
X=\operatorname{fl}_{32}(\operatorname{fp32}(h)+\operatorname{fp32}(\ell)).
\tag{P9.6}
\]

One stage then evaluates in FP32

\[
G=XX^T,\quad T=c_{32}G+b_{32}I,\quad D=TG,
\quad E=D+a_{32}I,\quad Y=EX,
\tag{P9.7}
\]

before applying (P9.6) again.  FP32 products are materialized separately and
each dot is reduced by the locked balanced tree.  `torch.matmul`, FMA
contraction, reassociation, native BF16 matrix multiplication, and unspecified
accumulator semantics are excluded.  After stage five, the two BF16 terms are
reconstructed in FP32, `fl32(rho32*s)` is formed, and one FP32 addition returns
the result.  This intentionally slow operation graph is the theorem-facing
reference, not a performance kernel. Two BF16 tensors also occupy the same
nominal storage as one FP32 tensor, so P9 makes no state-compression claim.

## Exact two-term boundary bound

For a matrix `Y` of shape `r x c`, the error-feedback boundary (P9.6) obeys

\[
\lVert X-Y\rVert_F
\le \omega\lVert Y\rVert_F+\lceil\sqrt n\rceil\chi,
\tag{P9.8}
\]

where

\[
\begin{aligned}
\omega={}&uu_b+u_b(1+u)u_b
 +u(1+u_b)\bigl(1+(1+u)u_b\bigr)\\
={}&\frac{282587406795009}{18446744073709551616},\\[2mm]
\chi={}&\frac{72341289546351105}
{1569275433846670190958947355801916604025588861116008628224}.
\end{aligned}
\tag{P9.9}
\]

The derivation includes FP32 subtraction and reconstruction rounding, both
BF16 casts, and FP32/BF16 subnormal crumbs.  It does not invoke Sterbenz's
lemma or assume exact high-plus-low reconstruction.

The ordinary one-term BF16 boundary has only the generic slope estimate
`u_b sqrt(p) (121/100)` before crumbs.  Requiring this estimate to fit between
the exact polynomial bound `121/100` and the proof tube `5/4` gives

\[
\frac{121}{100}\left(1+2^{-8}\sqrt p\right)<\frac54,
\]

which holds through `p=71` and fails at `p=72`.  This is a **normwise proof
obstruction**, not an executable instability witness and not an impossibility
theorem for every one-term BF16 kernel.  For comparison, the boundary-only
slope gate is `4,693,632` if subtraction and reconstruction are idealized as
exact, and `4,656,751` under the Sterbenz-free slope in (P9.9).  The full
five-stage certificate is more restrictive because it also propagates FP32
stage-body and normalization errors.

## Exact shape recurrence

All theorem calculations use `Fraction`.  To keep the five-stage expressions
finite and independently replayable, every named upper bound is rounded
outward on the fixed dyadic grid

\[
\lceil x\rceil_{40}=\frac{\lceil2^{40}x\rceil}{2^{40}}.
\tag{P9.10}
\]

This outward rounding is part of the theorem, not display formatting.  Put

\[
h_k=\lceil\sqrt k\rceil,\quad
L_k=1+\lceil\log_2 k\rceil,\quad
\gamma_k=\frac{L_ku}{1-L_ku},\quad
\kappa(o,k)=\frac{\lceil\sqrt o\rceil\,2k\tau}{1-L_ku}.
\tag{P9.11}
\]

The exact branchwise normalizer audit proves

\[
\lVert\widehat N(s)-N(s)\rVert_F<\bar\epsilon_N,
\qquad \bar\epsilon_N=\frac1{90000}.
\tag{P9.12}
\]

It separately treats the norm-controlled branch, the floor-controlled branch
with exact norm at least `1/2`, and the small-floor branch.  It checks the
derived denominator errors against `1/100000`; it does not apply an invalid
global relative-error model at zero or near underflow.

Define `S_j,F_j,E_j` as certified spectral, Frobenius, and forward-error
envelopes for the input to stage `j+1`.  The initialization is

\[
\begin{aligned}
\beta_0&=\left\lceil\omega(1+\bar\epsilon_N)+h_n\chi\right\rceil_{40},\\
S_0=F_0&=\left\lceil1+\bar\epsilon_N+\beta_0\right\rceil_{40},\\
E_0&=\left\lceil\bar\epsilon_N+\beta_0\right\rceil_{40}.
\end{aligned}
\tag{P9.13}
\]

For `j=0,...,4`, define the following exact outward recurrence:

\[
\begin{aligned}
e_G&=\lceil\gamma_dF_j^2+\kappa(p^2,d)\rceil_{40},\\
G_F&=\lceil S_jF_j+e_G\rceil_{40},
&G_2&=\lceil S_j^2+e_G\rceil_{40},\\
e_C&=\left\lceil |c_{32}|e_G+|c_{32}-c_q|S_jF_j
 +u|c_{32}|G_F+p\tau\right\rceil_{40},\\
C_F&=\lceil |c_q|S_jF_j+e_C\rceil_{40},\\
e_T&=\left\lceil e_C+|b_{32}-b_q|h_p
 +u(C_F+|b_{32}|h_p)+p\tau\right\rceil_{40},\\
T_F&=\lceil h_p|b_q|+e_T\rceil_{40},\\
e_D&=\left\lceil e_TG_2+|b_q|e_G+\gamma_pT_FG_F
 +\kappa(p^2,p)\right\rceil_{40},\\
D_F&=\left\lceil h_p\frac{b_q^2}{4c_q}+e_D\right\rceil_{40},\\
e_A&=\left\lceil e_D+|a_{32}-a_q|h_p
 +u(D_F+|a_{32}|h_p)+p\tau\right\rceil_{40},\\
A_F&=\lceil h_p|a_q|+e_A\rceil_{40},\\
e_Y&=\lceil e_AS_j+\gamma_pA_FF_j+\kappa(n,p)\rceil_{40},\\
Y_2&=\left\lceil\frac{121}{100}+e_Y\right\rceil_{40},
&Y_F&=\lceil |a_q|F_j+e_Y\rceil_{40},\\
\beta_{j+1}&=\lceil\omega Y_F+h_n\chi\rceil_{40},\\
S_{j+1}&=\lceil Y_2+\beta_{j+1}\rceil_{40},
&F_{j+1}&=\lceil Y_F+\beta_{j+1}\rceil_{40},\\
E_{j+1}&=\left\lceil
\frac{3000459}{512000}E_j+e_Y+\beta_{j+1}
\right\rceil_{40}.
\end{aligned}
\tag{P9.14}
\]

Exact polynomial analysis supplies `0<=q(x)<121/100`,
`0<q(x)/x<=a_q` away from zero, and the displayed Lipschitz constant on
`[0,5/4]`. This is a full rectangular spectral-map bound, not a diagonal-only
one. In singular-vector coordinates the derivative modes are `q'(x)`,
`q(x)/x`, and the two divided differences

\[
\frac{q(x)-q(y)}{x-y},\qquad
\frac{q(x)+q(y)}{x+y}
=\frac{q(x)-q(-y)}{x-(-y)}.
\]

Because `q` is odd, the mean-value theorem bounds both divided differences
by `sup_{|z|<=5/4}|q'(z)|=3000459/512000`; the null-space mode follows from
the same bound at zero. Thus the induced Frobenius derivative norm is bounded
over the entire spectral-norm ball. The audit accepts a shape only if every stage input satisfies
`S_j<5/4`; in particular, `S_4<5/4` is required before using the fifth-stage
bounds.

Let `E_5` be the final value from (P9.14), let

\[
K_R=\frac{336372400608849}{260261360000},\qquad
e_\rho=|\rho_{32}-\rho|+u|\rho_{32}|,
\]

and define

\[
\begin{aligned}
A^{32}_{r,c}
&=\left\lceil e_\rho+u(\rho+e_\rho)\right\rceil_{40},\\
B^{32}_{r,c}
&=\left\lceil(1+u)E_5+u|a_q|^5+(2+u)h_n\tau\right\rceil_{40},\\
A_{r,c}
&=\left\lceil A^{32}_{r,c}(1+u)+K_Ru\right\rceil_{40},\\
B_{r,c}
&=\left\lceil B^{32}_{r,c}
 +(A^{32}_{r,c}+K_R)h_n\tau\right\rceil_{40}.
\end{aligned}
\tag{P9.15}
\]

Equations (P9.10)--(P9.15) are the promised exact formulas for the affine
bound in (P9.3).  Under the locked arithmetic contract, the slope formula is
shape independent and equals (P9.4) for every accepted shape; the subscript is
retained to match the shape-indexed operator theorem.  The intercept carries
the shape dependence.

## Overflow and finite-range guards

The exact audit requires all of the following:

- `n<=2^52`, so every locked pairwise reduction remains in its stated gamma
  regime;
- finite FP32 kernel inputs with `maxabs<=2^116`;
- the scaled normalizer's pairwise square sum below the FP32 finite limit;
- every absolute partial-dot and pre-add stage envelope below half the FP32
  maximum finite value;
- every value sent to a BF16 boundary below half the BF16 maximum finite
  value; and
- the repair shell below half the FP32 maximum finite value.

The half-maximum checks leave room for the following rounded addition.  The
kernel rejects out-of-domain and nonfinite values.  These guards, together
with the explicit subnormal crumbs in (P9.8), (P9.11), and (P9.15), are part
of the result.

The familiar serial FP32 norm accumulation cannot simply be extended to the
large shapes.  On an all-ones input with `n>2^24`, the sequential sum of unit
squares reaches `2^24` and then every further `+1` is absorbed by
ties-to-even.  The returned normalized rank-one matrix has

\[
\sigma_1^2=\frac{n}{2^{24}}.
\]

It leaves the `5/4` spectral tube whenever
`n>25*2^20=26,214,400`.  For example, at `4096 x 11008`, the exact returned
value is `sigma_1^2=43/16>25/16`.  This is an executable obstruction to the
specified serial reduction semantics.  P9 avoids it with the locked balanced
tree; it does not claim that every alternative norm implementation fails.

## P7 closure

Let `f` be differentiable, globally `L=10` smooth, and satisfy the global PL
inequality with constant one.  Put the all-real adapter in P7's otherwise
exact-real EMA/Nesterov loop with `beta=19/20` and `eta=1/32000`, set gradient
noise to zero, and define

\[
e_t=\widehat R_{r,c}(s_{t+1})-R_{r,c}(s_{t+1}).
\]

P7 gives

\[
V_{t+1}\le q_7V_t+\gamma_R\lVert e_t\rVert_F^2,
\quad q_7=\frac{399960001}{400000000},\quad
\gamma_R=\frac1{2000000},
\]

and

\[
\lVert s_{t+1}\rVert_F^2\le C_sV_t,
\qquad C_s=\frac{1655544025}{2600084}.
\]

With the locked integer Young parameter `theta=3006`, define exactly

\[
\begin{aligned}
q_{r,c}&=\left\lceil q_7+\gamma_R(1+\theta)C_sA_{r,c}^2
\right\rceil_{40},\\
D_{r,c}&=\left\lceil\gamma_R(1+1/\theta)B_{r,c}^2
\right\rceil_{40},\\
G_{r,c}&=\left\lceil
\frac{10}{13533/50000}\frac{D_{r,c}}{1-q_{r,c}}
\right\rceil_{40}.
\end{aligned}
\tag{P9.16}
\]

Then

\[
V_{t+1}\le q_{r,c}V_t+D_{r,c},\qquad
\limsup_t(f(W_t)-f_\star)\le G_{r,c}.
\tag{P9.17}
\]

Every representative shape has the same certified rate

\[
q_{r,c}=\frac{137425214491}{137438953472}
=0.9999000357565819\ldots<1.
\tag{P9.18}
\]

Let

\[
H_{\rm safe}=\frac{2^{232}}{C_s}.
\]

The exact shape audit also checks
`D_(r,c)<=(1-q_(r,c))H_safe`.  Thus `V_0<=H_safe` makes the
`||s_(t+1)||_F<=2^116` input guard forward invariant.  This implication uses
zero gradient noise.  P7 remains available with gradient noise, but its
signal contains an additional direct noise term and the storage-only guard
does not automatically follow.

## Certified representative dimensions

The rows below are fixed theorem targets, not a sample used to infer a global
shape statement.  The exact recurrence is rerun separately for each row.

| Public shape | Oriented shape | `B_(r,c)` | `S_4` upper bound | `G_(r,c)` |
| --- | --- | ---: | ---: | ---: |
| `768 x 768` | `768 x 768` | `1.057373530` | `1.230419191` | `0.206682025` |
| `768 x 3072` | `768 x 3072` | `1.100936249` | `1.232303162` | `0.224063265` |
| `768 x 50257` | `768 x 50257` | `1.188552191` | `1.236074406` | `0.261145507` |
| `3072 x 12288` | `3072 x 12288` | `1.814441125` | `1.245091019` | `0.608599542` |
| `4096 x 4096` | `4096 x 4096` | `1.937451049` | `1.245905113` | `0.693916766` |
| `4096 x 11008` | `4096 x 11008` | `1.981864637` | `1.247804532` | `0.726095607` |
| `4096 x 14336` | `4096 x 14336` | `1.981864637` | `1.247804532` | `0.726095607` |

The largest intercept and neighborhood in the table have the exact values
in (P9.5) and

\[
G_{4096,11008}=G_{4096,14336}
=\frac{798350562999}{1099511627776}.
\]

The proof frontier is shape- and reduction-depth dependent; `4608` is not a
universal rank cutoff. As one locked negative control, `4608 x 18432` fails
only the fifth-stage-input tube check, with

\[
S_4=\frac{10753627185}{8589934592}
=1.25188696954865\ldots>\frac54.
\]

This means the current outward recurrence cannot justify reusing its
`[0,5/4]` fifth-stage bounds.  It is not an observed instability, an
executable counterexample, or an impossibility theorem for that shape.  The
positive theorem is exactly the per-shape set for which every recorded gate
passes.

## Scope exclusions

P9 does not cover literal upstream Muon, current-plus-epsilon normalization,
an unrepaired operator, native BF16 matmul, tensor-core or vendor-BLAS
accumulation, GPU execution, stochastic rounding, FTZ/DAZ, compiler fusion or
reassociation, weight decay, or aspect-ratio scaling.  A production kernel
must match the locked operation graph or receive its own forward-error and
parity audit.

The surrounding P7 loop remains exact real arithmetic.  FP32 EMA and
Nesterov state rounding require additional internal disturbance ports, and
FP32 parameter subtraction can stall at large binades.  Thus P9 is not an
end-to-end finite-precision training-convergence theorem, a neural-network
convergence result, or a performance claim.

The exact rational recurrence and its independent reconstruction are the
proof.  Small executable parity cases and any sampled diagnostics are useful
falsification only; they cannot establish the global bounds or their
tightness.

## Reproduction

```bash
uv run --locked python scripts/certify_scalable_mixed_precision.py \
  --output results/summaries/scalable_mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py \
  --require-canonical
uv run --locked python \
  experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py \
  --output results/summaries/scalable_mixed_precision_diagnostic.json
```

The first two commands rebuild the exact certificate.  The reconstruction
uses only the standard library for its algebra and reads the canonical JSON
only after rebuilding the result.  The third command is a modest CPU
diagnostic whose native matrix-product semantics are intentionally not used
as proof.
