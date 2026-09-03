# P8 fixed-2x2 mixed-precision operator certificate

## Status and scope

P8 instantiates the post-operator disturbance port in C12 for one proposed,
shape-locked implementation of the repaired max-floor Jordan operator.  It is
an exact forward-error certificate, not a sampled error estimate.  For every
finite binary32 input `s` of shape `2 x 2` satisfying

\[
\max_{i,j}|s_{ij}|\le 2^{116},
\]

the locked kernel below obeys

\[
\left\lVert \widehat R(s)-R(s)\right\rVert_F
\le \frac{11}{100000}\lVert s\rVert_F+\frac{347}{100}.
\tag{P8.1}
\]

Here the exact-real reference is

\[
R(s)=\mathcal H_{q^{\circ5}}
\!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\]

with no additive epsilon, exactly five stages,

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

The implementation is a deliberately certifiable design.  It stores the
input and output of each Jordan stage in BF16, but evaluates the complete
Horner stage in FP32.  It is not the literal upstream Muon kernel, not a
native all-BF16-intermediate kernel, and not a dimension-independent result.

For an arbitrary real input in the same shape and range, let `C32(s)` be its
entrywise binary32 round-to-nearest, ties-to-even conversion and define the
real-input adapter

\[
\widehat R_{\mathbb R}(s)=\widehat R(C_{32}(s)).
\]

The same certificate proves

\[
\left\lVert\widehat R_{\mathbb R}(s)-R(s)\right\rVert_F
\le\frac1{5000}\lVert s\rVert_F+\frac{347}{100}.
\tag{P8.2}
\]

For the P7 consequence, `\widehat R` is placed inside the otherwise exact-real
EMA/Nesterov loop through this all-real input adapter.  P8 does not include
FP32 rounding in the momentum state, Nesterov signal, gradient evaluation, or
parameter update.  Thus (P8.2) instantiates C12's post-operator error input; it
does not certify an entire FP32 optimizer end to end.

## Locked arithmetic contract

The certificate assumes IEEE round-to-nearest, ties-to-even arithmetic with
gradual underflow and no FTZ/DAZ behavior.  The unit roundoffs and half the
least positive subnormal are

\[
u_{32}=2^{-24},\quad t_{32}=2^{-150},\qquad
u_b=2^{-8},\quad t_b=2^{-134}.
\]

The three real polynomial coefficients are rounded once to the exact FP32
dyadics

\[
a_{32}=\frac{902955}{262144},\quad
b_{32}=-\frac{10013901}{2097152},\quad
c_{32}=\frac{8520729}{4194304},
\]

and the repair coefficient is rounded once to

\[
\rho_{32}=\frac{13231137}{16384}.
\]

The max-floor normalizer uses a scaled row-major FP32 Frobenius norm.  It first
forms `alpha=max(abs(s_ij))`.  The zero input is returned as zero.  If
`alpha<=1/2`, the `2 x 2` Frobenius norm is at most one and the normalizer
returns `s` directly.  Otherwise it divides by `alpha`, squares and sums the
four ratios in the locked order, takes a correctly rounded FP32 square root,
rescales by `alpha`, applies the floor at one, and performs FP32 division.
This scaling and the input bound exclude overflow in the norm and repair
shell.  Subnormal absolute errors are retained in the proof rather than
discarded.

At a stage boundary, let `X` denote the BF16-stored `2 x 2` matrix, converted
exactly to FP32.  A stage evaluates the algebraically equivalent Horner form

\[
Y=\bigl(a_{32}I+(b_{32}I+c_{32}XX^T)XX^T\bigr)X
\]

in this fixed order:

\[
A=XX^T,\quad T=c_{32}A+b_{32}I,\quad
D=TA,\quad E=D+a_{32}I,\quad Y=EX.
\]

Each length-two dot product uses two separately rounded FP32 multiplications
followed by one separately rounded FP32 addition.  FMA contraction,
reassociation, and native BF16 matrix multiplication are excluded.  Scalar
products and diagonal additions are likewise separately rounded FP32
operations.  Only the completed `Y` is rounded once to BF16 before the next
stage.  After stage five, the stored BF16 matrix is converted exactly to FP32,
`fl32(rho32*s)` is formed, and one FP32 addition returns `widehat R(s)`.

This arithmetic contract is the theorem-facing specification.  The committed
parity tests check the specified operation order and representative IEEE
behavior on the tested backend; finite probes do not exhaustively establish
the contract for every input or backend operation.

## Normalization and initial boundary

On the `alpha>1/2` branch, a standard product/sum analysis with

\[
\gamma_4=\frac{4u_{32}}{1-4u_{32}}
\]

and explicit subnormal crumbs proves that the derived relative norm error is
less than `2^-19`.  Consequently,

\[
\left\lVert\widehat N(s)-N(s)\right\rVert_F
<\frac1{500000}.
\]

Combining this with the first BF16 boundary cast gives total error below
`1/250` and places the initial stored matrix strictly inside

\[
\lVert X\rVert_2<\frac54,\qquad
\lVert X\rVert_F<\frac74.
\tag{P8.3}
\]

The direct `alpha<=1/2` branch obeys the same conclusion without relying on a
relative model near zero.  This branch split is essential: a global relative
floating-point norm-error claim would be false for subnormal inputs.

## Five-stage invariant

For a singular value `x` in `[0,5/4]`, exact Sturm-chain arithmetic proves

\[
0\le q(x)<\frac{121}{100}.
\tag{P8.4}
\]

Positivity follows because the quadratic factor in `x^2` has negative
discriminant `-2594691/500000` and positive leading coefficient.  For the
upper inequality, the Sturm chain of `121/100-q(x)` has three sign variations
at both endpoints and hence no root in the interval; its endpoint values are
positive.  This replaces a sampled spectral bound.

Assume (P8.3) at one stage boundary.  Using `sqrt(2)<99/70`, the FP32 dyadic
coefficient errors, `gamma4`, and the explicit underflow terms, exact rational
propagation gives

\[
\begin{array}{c|c}
\text{quantity}&\text{strict error upper bound}\\ \hline
XX^T&1/20000\\
c_{32}XX^T+b_{32}I&1/20000\\
(c_{32}XX^T+b_{32}I)XX^T&1/20000\\
(c_{32}XX^T+b_{32}I)XX^T+a_{32}I&1/20000\\
Y&1/20000.
\end{array}
\]

The bounds use the Frobenius norm and deliberately overbound every
length-two separate-multiply/add dot product by `gamma4`.  Before the boundary cast,

\[
\lVert Y\rVert_F<\frac{239587}{140000},\qquad
\lVert Y\rVert_2<\frac{24201}{20000}.
\]

After the one BF16 round, including gradual-underflow terms,

\[
\lVert X_+\rVert_F<1.718021<\frac74,\qquad
\lVert X_+\rVert_2<1.216735<\frac54.
\]

The invariant therefore closes for all five stages.  The exact rational
recurrence and every strict comparison are replayed by the certificate; none
of these inequalities is inferred from random tests.

## Operator forward error

Equation (P8.4) and `sqrt(2)<99/70` give

\[
\lVert\mathcal H_{q^{\circ5}}(N(s))\rVert_F
<\frac{11979}{7000}.
\]

The computed final stage satisfies `||X_5||_F<7/4`.  The intentionally simple
triangle bound therefore yields

\[
\left\lVert \widehat{\mathcal H}(s)
-\mathcal H_{q^{\circ5}}(N(s))\right\rVert_F
<\frac{24229}{7000}.
\]

For the FP32 repair multiply and final add, exact arithmetic gives the affine
slope

\[
A_0=
\frac{1025348293983610756598467}
{9376903711319508102676480000}
=0.00010934828015199111\ldots
<\frac{11}{100000}.
\]

The complete constant term, including the final-add and subnormal crumbs, is
`3.461285819...<347/100`.  Combining these facts proves (P8.1).  The intercept
is conservative: it bounds the ideal and computed polynomial outputs by
separate invariant envelopes rather than asserting a small five-stage
relative error.

## Arbitrary-real input adapter

For every real `2 x 2` matrix in the stated range, entrywise IEEE conversion
obeys

\[
\lVert C_{32}(s)-s\rVert_F
\le u_{32}\lVert s\rVert_F+2t_{32},
\qquad
\lVert C_{32}(s)\rVert_F
\le(1+u_{32})\lVert s\rVert_F+2t_{32}.
\]

Rounding is monotone and `2^116` is exactly representable, so the converted
matrix remains inside the binary32 kernel guard.  The exact repaired reference
map has the global Frobenius Lipschitz constant

\[
K_R=\rho+\frac{4848763}{10000}
=\frac{336372400608849}{260261360000}.
\]

Apply (P8.1) at `C32(s)` and use this Lipschitz bound between `C32(s)` and
`s`.  Before rounding to the displayed simple coefficient, the resulting
slope is

\[
\frac{29321583773060684572869055679171}
{157318338976009032452353483079680000}
=0.000186383761511315\ldots<\frac1{5000}.
\]

The exact unrounded intercept plus the cast's subnormal term remains below
`347/100`.  This proves (P8.2) without assuming that an exact-real outer-loop
signal is already binary32-representable.

## Closing the P7 disturbance port

Set gradient noise to zero and let

\[
e_t=\widehat R_{\mathbb R}(s_{t+1})-R(s_{t+1}).
\]

P7 proves

\[
V_{t+1}\le q_7V_t+\frac1{2000000}\lVert e_t\rVert_F^2,
\qquad q_7=\frac{399960001}{400000000}.
\]

In the P7 conditioned coordinates,

\[
\frac{s}{L}=\frac{361}{400}z+\frac{39}{400}u,
\]

and Cauchy--Schwarz in its exact quadratic storage gives

\[
\left\lVert\frac{s}{L}\right\rVert_F^2
\le \kappa V,\qquad
\kappa=\frac{66221761}{10400336}.
\]

Since `L=10`,

\[
\lVert s\rVert_F^2\le
\frac{1655544025}{2600084}V.
\tag{P8.5}
\]

Apply Young's inequality with `theta=5124` to the arbitrary-real adapter bound
(P8.2):

\[
(A\lVert s\rVert_F+B)^2
\le(1+\theta)A^2\lVert s\rVert_F^2
 +(1+1/\theta)B^2,
\]

where `A=1/5000` and `B=347/100`.  Substitution of (P8.5) gives the exact
pathwise recursion

\[
V_{t+1}\le q_8V_t+B_V,
\]

\[
q_8=
\frac{41597186684695561}{41601344000000000}
=0.9999000677645309\ldots<1,
\qquad
B_V=\frac{4936769}{819840000000}.
\tag{P8.6}
\]

The finite-input guard can be made invariant rather than assumed at every
step.  With the signal-storage coefficient in (P8.5), define

\[
H_{\rm safe}
=\frac{2^{232}}{1655544025/2600084}
=\frac{2600084\,2^{232}}{1655544025}.
\]

The exact replay checks

\[
B_V\le(1-q_8)H_{\rm safe}.
\]

Therefore `V_0<=H_safe` implies inductively that `V_t<=H_safe` and
`||s_(t+1)||_F<=2^116` for every step, so every adapter call stays in the
certified range.  The safe storage radius is approximately `1.08394e67`.
This invariant is a zero-gradient-noise result: with gradient noise, the
operator signal contains an additional direct noise term and (P8.5) alone no
longer bounds it.

Finally, P7's storage satisfies

\[
V_t\ge\frac{13533}{50000}\frac{f(W_t)-f_\star}{10}.
\]

Thus the proposed real-input adapter, when embedded in the exact-real outer
loop with zero gradient noise and `V_0<=H_safe`, obeys

\[
\limsup_{t\to\infty}\bigl(f(W_t)-f_\star\bigr)
\le
\frac{462392438350000000}{207695315294468001}
=2.22630172325\ldots.
\tag{P8.7}
\]

The number in (P8.7) is a worst-case certified neighborhood, not an observed
training loss and not a claim that the bound is tight.

The earlier candidate in which every polynomial intermediate was rounded to
BF16 is not certified.  A cancellation-aware normwise propagation already
leaves the useful spectral tube after its first stage.  That is a failure of
the attempted upper-bound strategy, not an executable instability proof or an
impossibility theorem for every all-BF16 kernel.

## Exclusions and the outer-loop obstruction

P8 does not cover arbitrary matrix shapes, GPU tensor-core kernels, native
BF16 accumulation, FTZ/DAZ modes, stochastic gradients, weight decay,
aspect-ratio scaling, current-plus-epsilon normalization, an unrepaired map,
or the literal upstream Muon implementation.

It also cannot be promoted to a full FP32 parameter-convergence theorem merely
by noting that the outer operations use FP32.  FP32 parameter subtraction has
a translation-dependent resolution floor.  For example, take

\[
f(w)=\tfrac12\bigl(w-(2^{30}-64)\bigr)^2,\qquad
w=2^{30},\quad m=64.
\]

The true gradient and steady EMA state are both `64`.  At `eta=1/32000`, the
real repaired step is far below the downward half-spacing `32` at `2^30`, so
round-to-nearest FP32 subtraction returns the unchanged parameter.  The
positive function gap can therefore persist.  A full finite-precision
optimizer theorem needs additional parameter-magnitude/binade assumptions,
a higher-precision or compensated master weight, or extra disturbance ports
and an ultimate-neighborhood statement.  FP32 EMA/Nesterov rounding likewise
requires separate internal ports; it is not represented by C12's single
post-operator error.

P7 remains the submission cutoff and the general robust theorem.  P6 is its
zero-disturbance corollary.  P8 is a fixed-shape constructive instantiation of
one P7 input port, not a replacement headline theorem.

## Reproduction and review

Run the canonical exact certificate and independent standard-library
reconstruction with:

```bash
uv run --locked python scripts/certify_mixed_precision.py
uv run --locked python scripts/reconstruct_mixed_precision.py
uv run --locked python \
  experiments/mixed_precision/run_mixed_precision_falsification.py
```

The canonical machine-readable artifact is
`results/summaries/mixed_precision_certificate.json`.  Unit tests additionally
check parity of the explicit kernel and exact identities, but sampled parity
tests do not replace the arithmetic proof.  The falsification result is
`results/summaries/mixed_precision_falsification.json`; its float64 reference
is not exact.  The P7 human audit of C11--C12, including disturbance placement
and physical-unit conversion, remains pending in
`theory/audits/P7_HUMAN_PROOF_AUDIT.md`.
