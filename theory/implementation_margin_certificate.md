# P11 certified implementation-margin theorem

## Status and scope

P11 quantifies how much error may be added at the two interfaces left open by
the frozen P10 arithmetic shell. It is an exact rational robustness theorem,
not a measurement of a model, gradient implementation, CUDA kernel, BLAS
backend, or production optimizer.

The objective class is every differentiable, globally `L=10`-smooth function
that satisfies the global Polyak--Lojasiewicz inequality with constant one.
The objective may be nonconvex and its set of global minimizers may be
nonunique. The matrix shape is fixed to `4096 x 11008`.

The exact target operator is

\[
R(s)=\mathcal H_{q^{\circ 5}}
 \!\left(\frac{s}{\max\{1,\lVert s\rVert_F\}}\right)+\rho s,
\tag{P11.1}
\]

where there is no additive epsilon,

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\qquad
\rho=\frac{210177835339081}{260261360000},
\tag{P11.2}
\]

and there are exactly five Jordan/Newton--Schulz polynomial stages. The P9
proof-reference implementation uses the balanced-FP32, two-term-BF16 stage
kernel; the P10 shell uses separately rounded FP32 EMA/Nesterov arithmetic and
a three-word compensated FP32 master. The optimizer constants remain

\[
\beta=\frac{19}{20},\qquad a=1-\beta=\frac1{20},
\qquad \eta=\frac1{32000}.
\tag{P11.3}
\]

P11 changes none of the objective class, operator, floor, repair, iteration
count, shape, learning rate, momentum, or P10 arithmetic graph.

## External-port contract

Let `g_t=grad f(W_t)` be evaluated at the exact logical sum of the three master
words. The real value presented to P10's one final gradient cast is

\[
y_t=g_t+\zeta_t,\qquad \widehat g_t=C_{32}(y_t),
\tag{P11.4}
\]

where `C32` is entrywise binary32 round-to-nearest, ties-to-even conversion.
The pre-cast error obeys

\[
\lVert\zeta_t\rVert_F
\le a_g\sqrt{V_t}+b_g.
\tag{P11.5}
\]

This port can include gradient-computation error and error caused by evaluating
the gradient at an approximate model weight. In the latter case, global
`10`-smoothness gives the separate conversion
`||Delta g||_F <= 10 ||Delta W||_F`. P11 does not itself certify a model-forward
reconstruction.

Let

\[
U_t^9=\widehat R_{4096,11008}(s_{t+1})
\tag{P11.6}
\]

be the finite, contiguous FP32 output of the P9 reference graph. The deployed
output supplied to the compensated master is another finite, contiguous FP32
tensor `U_t^dep`, and

\[
\nu_t=U_t^{\rm dep}-U_t^9,
\qquad
\lVert\nu_t\rVert_F\le a_R\sqrt{V_t}+b_R.
\tag{P11.7}
\]

The difference in (P11.7) is interpreted over the reals. The theorem does not
require executing a separate rounded addition `U9+nu`; `nu` describes the
end-to-end difference of two represented outputs.

All four budget coefficients are nonnegative and have physical Frobenius
units. The theorem applies pathwise to every error realization satisfying
(P11.5) and (P11.7); no independence, zero-mean, or stochastic assumption is
used.

## Exact reduction to the P7 ports

Define the P10 arithmetic residuals relative to `y_t`, not `g_t`, by

\[
\begin{aligned}
m_{t+1}&=\beta m_t+a y_t+r_t^m,\\
s_{t+1}&=\beta m_{t+1}+a y_t+r_t^s.
\end{aligned}
\tag{P11.8}
\]

P7 therefore sees the effective gradient perturbation

\[
\xi_t^{\rm eff}=\zeta_t+\frac{r_t^m}{a}.
\tag{P11.9}
\]

Its nominal Nesterov signal, formed with (P11.9), is

\[
s_t^0=\beta m_{t+1}+a(g_t+\xi_t^{\rm eff})
     =\beta m_{t+1}+a y_t+r_t^m.
\]

Consequently,

\[
s_{t+1}-s_t^0=r_t^s-r_t^m.
\tag{P11.10}
\]

The explicit `zeta_t` term cancels exactly. Bounding the signal mismatch by
an additional direct `||zeta_t||` term would discard this cancellation and
produce a needlessly weak margin.

If the logical compensated-master residual is `r_t^W`, the effective P7
post-operator error is

\[
e_t^{\rm eff}
=R(s_{t+1})-R(s_t^0)
 +(U_t^9-R(s_{t+1}))+\nu_t-\frac{r_t^W}{\eta}.
\tag{P11.11}
\]

Substitution of (P11.9)--(P11.11) into the P7 loop reproduces the implemented
logical update, including the signs and the physical-units conversion of the
master residual.

### Incremental sensitivities

Put `u=2^-24`. Let `C_m,C_g,C_U` denote P10's exact nonnegative EMA-momentum,
EMA-gradient, and master-output residual coefficients; let `bar(beta)` and
`bar(a)` denote its exact represented-state norm coefficients; let `A_32` be
P9's binary32-input error slope; and let `K_R` be the certified global
Lipschitz constant of the repaired exact operator. These are the exact
rationals stored in the P10 certificate. Define

\[
\begin{aligned}
k_m&=C_g(1+u)+au,\\
k_s&=C_m\bar a(1+u)+k_m,\\
k_{\rm sig}&=\bar a(1+\bar\beta)(1+u),\\
k_U&=(K_R+A_{32})k_{\rm sig},\\
k_{Wg}&=C_Uk_U,\qquad k_{WR}=C_U,\\
k_{\xi g}&=1+\frac{k_m}{a},\\
k_{eg}&=A_{32}k_{\rm sig}+K_R(k_m+k_s)
                 +\frac{k_{Wg}}{\eta},\\
k_{eR}&=1+\frac{k_{WR}}{\eta}.
\end{aligned}
\tag{P11.12}
\]

Every operation in (P11.12) is exact rational arithmetic. Numerically, only
for orientation,

\[
\begin{array}{c|c}
\text{coefficient}&\text{decimal value}\\ \hline
k_m&9.68575544213481\times10^{-9}\\
k_s&1.59442444536141\times10^{-8}\\
k_{\rm sig}&9.75000239536190\times10^{-2}\\
k_U&1.26013019725499\times10^{2}\\
k_{\xi g}&1.00000019371511\\
k_{eg}&6.47939385600949\times10^{-5}\\
k_{eR}&1.00000016670675
\end{array}
\tag{P11.13}
\]

The canonical JSON records the unreduced exact fraction for every coefficient
in (P11.12).

For a nonnegative rational `x`, define the theorem-facing outward grid
rounding

\[
\lceil x\rceil_{40}=\frac{\lceil 2^{40}x\rceil}{2^{40}}.
\tag{P11.14}
\]

If a P10 quantity has the affine envelope
`||z||_F <= z_1 sqrt(V)+z_0`, P11 augments it by

\[
\begin{aligned}
z_1^{11}&=\left\lceil z_1+k_g a_g+k_R'a_R\right\rceil_{40},\\
z_0^{11}&=\left\lceil z_0+k_g b_g+k_R'b_R\right\rceil_{40},
\end{aligned}
\tag{P11.15}
\]

with the corresponding sensitivity from (P11.12). In particular,

\[
\begin{aligned}
X_1&=\left\lceil X_1^{10}+k_{\xi g}a_g\right\rceil_{40},&
X_0&=\left\lceil X_0^{10}+k_{\xi g}b_g\right\rceil_{40},\\
E_1&=\left\lceil E_1^{10}+k_{eg}a_g+k_{eR}a_R\right\rceil_{40},&
E_0&=\left\lceil E_0^{10}+k_{eg}b_g+k_{eR}b_R\right\rceil_{40},
\end{aligned}
\tag{P11.16}
\]

where

\[
\begin{aligned}
X_1^{10}&=\frac{16113175}{274877906944},&
X_0^{10}&=\frac5{274877906944},\\
E_1^{10}&=\frac{9288413563}{549755813888},&
E_0^{10}&=\frac{2179298479615}{1099511627776}.
\end{aligned}
\tag{P11.17}
\]

Thus

\[
\lVert\xi_t^{\rm eff}\rVert_F\le X_1\sqrt{V_t}+X_0,
\qquad
\lVert e_t^{\rm eff}\rVert_F\le E_1\sqrt{V_t}+E_0.
\tag{P11.18}
\]

The full squares in (P11.18) are retained. In particular, the proof does not
drop the base/external or gradient/operator cross terms hidden inside these
augmented coefficients.

## The storage theorem

P7 supplies the pathwise inequality

\[
V_{t+1}\le
q_7V_t+\frac12\lVert\xi_t^{\rm eff}\rVert_F^2
       +\frac1{2000000}\lVert e_t^{\rm eff}\rVert_F^2,
\qquad
q_7=\frac{399960001}{400000000}.
\tag{P11.19}
\]

Use Young parameters `theta_g=1` and `theta_e=837`, exactly as in P10. Since

\[
(x_1\sqrt V+x_0)^2
\le(1+\theta)x_1^2V+(1+\theta^{-1})x_0^2,
\tag{P11.20}
\]

define

\[
\boxed{\begin{aligned}
q_{11}(a_g,a_R)
&=\left\lceil q_7
 +\frac12(1+1)X_1^2
 +\frac1{2000000}(1+837)E_1^2
 \right\rceil_{40},\\
D_{11}(b_g,b_R)
&=\left\lceil
 \frac12(1+1)X_0^2
 +\frac1{2000000}\left(1+\frac1{837}\right)E_0^2
 \right\rceil_{40}.
\end{aligned}}
\tag{P11.21}
\]

Then every admissible disturbance realization satisfies

\[
\boxed{V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).}
\tag{P11.22}
\]

This is dimension independent after the shape-specific P9/P10 envelope has
been instantiated, but the present certificate is claimed only at
`4096 x 11008`.

If

\[
q_{11}<1,\qquad D_{11}\le1-q_{11},
\tag{P11.23}
\]

then `V_t<=1` implies `V_(t+1)<=1`. Iteration gives the usual ultimate-storage
bound `limsup V_t <= D11/(1-q11)`. The P7 storage-to-value conversion yields

\[
\limsup_t(f(W_t)-f_\star)
\le
\left\lceil
\frac{10}{13533/50000}\frac{D_{11}}{1-q_{11}}
\right\rceil_{40}.
\tag{P11.24}
\]

This is a function-value/storage statement. It is not full-parameter ISS,
arbitrary-pair contraction, or convergence to a unique minimizer.

## Zero-error replay and a jointly nonzero budget

At `a_g=a_R=b_g=b_R=0`, every augmented envelope equals its P10 value
exactly, and (P11.21) replays

\[
q_{11}=q_{10}=\frac{549700907325}{549755813888},
\qquad
D_{11}=D_{10}=\frac{2162331}{1099511627776}.
\tag{P11.25}
\]

The function-gap bound also replays exactly as
`399957341889/549755813888`.

The all-positive profile

\[
a_g=b_g=\frac1{4096},\qquad
a_R=\frac1{128},\qquad b_R=\frac18
\tag{P11.26}
\]

gives

\[
\begin{aligned}
q_{11}&=\frac{274850515349}{274877906944},\\
D_{11}&=\frac{1254603}{549755813888},\\
1-q_{11}-D_{11}&=\frac{53528587}{549755813888}>0,\\
\limsup_t(f(W_t)-f_\star)
&\le\frac{930325132219}{1099511627776}<1.
\end{aligned}
\tag{P11.27}
\]

This proves that a jointly nonzero gradient-and-operator margin exists at the
full P10 step. The profile is a certified design point, not an empirically
estimated error level.

## Exact grid maxima and Pareto slices

The reporting grid is exactly `G={k/2^40 : k is a nonnegative integer}`.
"Maximum" below means the largest accepted point in `G`, not the supremum over
all real or rational budgets. Monotonicity of every nonnegative envelope and
guard makes an accepted point plus its rejected adjacent point a complete
one-dimensional grid certificate.

With the other three coordinates zero, the exact one-axis maxima are:

| Varied coordinate | Largest accepted value | Adjacent rejected value |
| --- | ---: | ---: |
| `a_g` | `10815225547 / 2^40` | `10815225548 / 2^40` |
| `a_R` | `513245498810 / 2^40` | `513245498811 / 2^40` |
| `b_g` | `10879487718 / 2^40` | `10879487719 / 2^40` |
| `b_R` | `13351103462525 / 2^40` | `13351103462526 / 2^40` |

At every accepted axis endpoint, `1-q11-D11=0`; at the adjacent point it is
exactly `-1/2^40`. The only failed check at each adjacent control is
unit-storage forward invariance. This rejects the sufficient certificate; it
does not exhibit an unstable trajectory.

The following are exact coordinatewise-maximal slices. Table entries are
integer ticks over the common denominator `2^40`. At each row, adding one tick
to either displayed coordinate while fixing the other fails the invariance
check.

### Slope slice: `b_g=b_R=0`

| Row | `2^40 a_g` | `2^40 a_R` |
| ---: | ---: | ---: |
| 0 | 0 | 513245498810 |
| 1 | 1351903193 | 508734339015 |
| 2 | 2703806386 | 495772587382 |
| 3 | 4055709580 | 473693173922 |
| 4 | 5407612773 | 441184385742 |
| 5 | 6759515966 | 395798829226 |
| 6 | 8111419160 | 332578695399 |
| 7 | 9463322353 | 238696271394 |
| 8 | 10815225547 | 1295 |

Every accepted slope row has

\[
q_{11}=\frac{1099509465445}{1099511627776},
\qquad
D_{11}=\frac{2162331}{1099511627776},
\qquad 1-q_{11}-D_{11}=0.
\tag{P11.28}
\]

### Intercept slice: `a_g=a_R=0`

| Row | `2^40 b_g` | `2^40 b_R` |
| ---: | ---: | ---: |
| 0 | 2622 | 13351103462525 |
| 1 | 1359935964 | 13231702254350 |
| 2 | 2719871929 | 12867815091068 |
| 3 | 4079807894 | 12240945421027 |
| 4 | 5439743859 | 11314491418802 |
| 5 | 6799679823 | 10020390941550 |
| 6 | 8159615788 | 8222316916289 |
| 7 | 9519551753 | 5577370189287 |
| 8 | 10879487718 | 8583 |

Every accepted intercept row has

\[
q_{11}=\frac{549700907325}{549755813888},
\qquad
D_{11}=\frac{54906563}{549755813888},
\qquad 1-q_{11}-D_{11}=0.
\tag{P11.29}
\]

The small nonzero endpoint coordinates arise from outward-rounding plateaus;
they are not transcription errors. These two tables are slices of a
four-dimensional acceptance set, not a claim that one slice dominates every
possible allocation of slope and intercept budget.

## Arithmetic guards

For a proposed budget, the same exact propagation augments P10's actual-signal,
P9-reference-output, deployed-output, master-residual, and EMA-intermediate
envelopes. On `V<=1` the certificate requires:

1. the signal Frobenius bound to remain inside P9's input domain;
2. every gradient-cast and EMA intermediate to stay finite;
3. `||Udep||_F<=2^15`, which also bounds every output entry;
4. the rounded FP32 step bound
   `ceil_40((1+2^-24)*eta32*||Udep||_F+2^-150)<=2`;
5. P10's middle- and low-word invariants; and
6. the high-word premise `maxabs(H)<=2^30`, checked again after each update.

For (P11.26), the exact unit-storage bounds are

\[
\begin{aligned}
\lVert s_{t+1}\rVert_F
&\le\frac{13872266672489}{549755813888}
 =25.2335060804\ldots,\\
\lVert U_t^9\rVert_F
&\le\frac{8965087254758081}{274877906944}
 =32614.7974365\ldots,\\
\lVert U_t^{\rm dep}\rVert_F
&\le\frac{8965123761980097}{274877906944}
 =32614.9302490\ldots<2^{15},\\
\max_{ij}|\operatorname{fl}_{32}(\eta_{32}U_{t,ij}^{\rm dep})|
&\le\frac{1120640590271}{1099511627776}
 =1.01921667944\ldots<2.
\end{aligned}
\tag{P11.30}

All EMA intermediates are below half the maximum finite binary32 value, and
the P10 middle/low master-word checks remain true because the tighter
`2^15` output profile still holds. The high-word guard remains conditional:
global PL storage cannot bound travel along an unbounded flat minimizer set.

## Exactness argument

The proof has four layers.

1. P9 supplies a full-matrix, shape-specific global forward-error and
   Lipschitz certificate for (P11.1), not a sampled or diagonal bound.
2. P10 supplies exact binary32 residual envelopes for the EMA/Nesterov graph
   and logical three-word master update.
3. Equations (P11.8)--(P11.18) compose the new pathwise ports with those frozen
   envelopes, preserving the exact signal cancellation.
4. Equations (P11.19)--(P11.24) absorb the augmented affine envelopes into the
   exact P7 dissipation inequality. Every theorem-facing rounding is outward
   on (P11.14), and every comparison is performed with integer/rational
   arithmetic.

The boundary search uses exact integer bisection on monotone predicates. Its
accepted points and adjacent failures are replayed independently from the
canonical JSON by the standalone standard-library reconstruction. The tables
are therefore exhaustive on their declared one-dimensional grid rays, even
though they are not unrestricted-real optima.

## Provenance and limitations

The three frozen P10 commits have distinct roles:

- theorem/source commit:
  `2d62b566e5a34895470d4eeb4b76789add826cbe`;
- exact-artifact commit:
  `6e7ea000692a269cbd3766146eb7423733d18694`;
- diagnostic/final checkpoint:
  `3246972d44fc6e04c15cf4e205615acbb879df36`.

Independent human review of P10 and P11 remains pending. Exact automated
replay is not represented as human approval.

P11 does not certify literal upstream Muon, current-plus-epsilon
normalization, an unrepaired operator, a full neural-network forward/backward
pass, native GPU/tensor-core execution, an observed production error budget,
weight decay, aspect scaling, stochastic rounding, FTZ/DAZ, or model training.
It does not make the high-word premise unconditional and does not turn the
global nonconvex-PL result into full-state ISS. The adjacent rejected grid
controls show only where this sufficient certificate stops proving the claim.

