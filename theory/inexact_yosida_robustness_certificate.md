# P15 inexact-Yosida robustness certificate

## Scope

This note replaces P14's exact resolvent evaluation by an exact-real oracle
whose error is checked through a computable graph residual. The ambient space
is any fixed finite real matrix space with the Frobenius inner product. All
operator gains and lifted certificates are independent of the matrix shape
and of the additive-normalization parameter `epsilon>0`.

P15 does **not** specify an iterative resolvent algorithm, prove a complexity
bound, or analyze residual evaluation in finite precision. In particular, it
is not a BF16 kernel or a literal upstream-Muon theorem. It is a robustness
theorem for any exact-real oracle that returns a candidate satisfying the
stated stopping rule.

## P14 operator and exact resolvent

Let

\[
A_\epsilon=E_{h,\epsilon}+G_\epsilon,
\qquad B=A_\epsilon+\mu I,
\]

where `E_(h,epsilon)` is the exact-real five-stage Jordan map with additive
Frobenius normalization `M/(||M||_F+epsilon)`, and `G_epsilon` is P13's radial
passivator. Its exact stage coefficients are `6889/2000`, `-191/40`, and
`4063/2000`. P13 proves that `A_epsilon` is continuous, full-domain,
zero-preserving, and monotone on every finite rectangular matrix space.

P14 fixes

\[
\lambda=\frac1{1000},\qquad \mu=1000,
\qquad J=(I+\lambda B)^{-1},
\qquad Y=\lambda^{-1}(I-J).
\]

The exact resolvent exists uniquely, fixes zero, and satisfies

\[
\operatorname{Lip}(J)\le\frac1{1+\lambda\mu}=\frac12.
\]

The exact Yosida map obeys the full-matrix incremental sector

\[
\langle\Delta Y-500\Delta s,1000\Delta s-\Delta Y\rangle_F\ge0.
\]

Equivalently, `Y(s)=750*s+E_Y(s)`, where `E_Y(0)=0` and
`Lip(E_Y)<=250`. This is a joint IQC for a possibly nonsymmetric map, not a
Loewner ordering.

The practical Jordan formula remains traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`. That upstream code contains
neither the P13 radial correction nor the P14--P15 resolvent architecture.

## Computable inexact-resolvent rule

Given an input `s`, an oracle returns a candidate `u_hat`. Define its graph
residual by

\[
r=s-\widehat u-\lambda B(\widehat u).
\]

The deployed approximate output in this theorem is exactly

\[
\widehat Y(s)=\frac{s-\widehat u}{\lambda}.
\]

It is important that this is **not** defined as `B(u_hat)`: those two values
differ by `r/lambda` away from an exact solve. The residual is computable from
the candidate and an exact-real evaluation of `B`.

Because

\[
s-r=\widehat u+\lambda B(\widehat u),
\]

one has the exact identity

\[
\widehat u=J(s-r),\qquad
\widehat Y(s)-Y(s)=\frac{J(s)-J(s-r)}{\lambda}.
\]

The P14 resolvent contraction therefore gives both

\[
\boxed{\|\widehat u-J(s)\|_F\le\frac12\|r\|_F}
\]

and

\[
\boxed{
\|\widehat Y(s)-Y(s)\|_F
\le \frac1{\lambda(1+\lambda\mu)}\|r\|_F
=500\|r\|_F.}
\]

By contrast, `B(u_hat)=Y(s-r)` has only the class-uniform bound
`||B(u_hat)-Y(s)||_F<=1000*||r||_F`; that graph-form output is not the one
used by the locked storage certificate.

The factor `500` is sharp over the admissible operator class: for the
abstract boundary base `A=0`, so `B=1000*I`, the resolvent is `J=I/2` and
equality holds.

P15 locks the verifiable stopping condition

\[
\|r\|_F\le\kappa\|s\|_F+\bar r,
\qquad \kappa=\frac1{250},\quad \bar r\ge0.
\]

This is an a posteriori rule. P15 does not claim how many iterations any
particular solver needs to meet it.

## Relative and absolute error ports

Let `nu=Y_hat(s)-Y(s)`. The preceding two inequalities imply

\[
\|\nu\|_F\le2\|s\|_F+500\bar r.
\]

Pointwise, `nu` can be split collinearly as
`nu=nu_rel+nu_abs` with

\[
\|\nu_{\rm rel}\|_F\le2\|s\|_F,
\qquad
\|\nu_{\rm abs}\|_F\le500\bar r.
\]

(When the right-hand side is zero, take both terms to be zero; otherwise
scale `nu` in the proportions of the two bounds.) Combining the relative
term with P14's centered residual gives

\[
\widehat Y(s)=750s+e_{\rm rel}+e_{\rm abs},
\qquad
\|e_{\rm rel}\|_F\le252\|s\|_F,
\qquad
\|e_{\rm abs}\|_F\le500\bar r.
\]

This reduction is pointwise and sufficient for the trajectory-to-minimizer
PL certificate below. An arbitrary oracle satisfying only the residual rule
need not define an incrementally Lipschitz or monotone approximate map, so P15
makes no such claim.

## Pinned EMA/Nesterov loop

Let `f` be differentiable, globally `L=10` smooth, bounded below, and satisfy
the global Polyak--Lojasiewicz inequality with constant `1`:

\[
\frac12\|\nabla f(W)\|_F^2\ge f(W)-f^\star.
\]

The recurrence is

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{32000}\widehat Y(s_{t+1}).
\end{aligned}
\]

At iteration `t`, the stopping rule is applied to this oracle call as
`||r_(t+1)||_F <= ||s_(t+1)||_F/250+rbar`. This indexing convention has no
effect on the pointwise certificate.

The state and signal ordering is the pinned real-arithmetic EMA/Nesterov
ordering, with the orthogonalizer replaced by the proposed inexact Yosida
oracle.

## Exact robust value--momentum certificate

P15 retains P14's storage

\[
\mathcal V_t=
\begin{bmatrix}m_t/10\\ \nabla f(W_t)/10\end{bmatrix}^{\!T}
(P\otimes I)
\begin{bmatrix}m_t/10\\ \nabla f(W_t)/10\end{bmatrix}
+\frac{313243}{10^6}\frac{f(W_t)-f^\star}{10},
\]

with

\[
P=\frac1{10^6}
\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix}.
\]

The relative solve tolerance changes only the centered residual radius from
`250` to `252`. With the same directed-interpolation and PL supplies, the
same exact storage, multipliers, and

\[
\tau=\frac{499}{500},\qquad
q_{15}=\tau^2=\frac{249001}{250000}
\]

remain feasible. The exact `4 x 4` relative-error LMI is strictly negative
definite. Adding the physical output-error coordinate `e_abs` gives an exact
`5 x 5` rational LMI with gain

\[
\gamma_{\rm abs}=\frac1{100000}.
\]

Its exact Sylvester replay yields

\[
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t
+\frac1{100000}\|e_{{\rm abs},t}\|_F^2.
\]

Since `||e_abs,t||_F<=500*rbar`, the advertised oracle-level result is

\[
\boxed{
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t
+\frac52\bar r^2.}
\]

Thus `q15=249001/250000<1` and `C15=5/2`. Iteration gives

\[
\mathcal V_t\le q_{15}^t\mathcal V_0
+\frac{5}{2}\frac{1-q_{15}^t}{1-q_{15}}\bar r^2,
\qquad
\limsup_t\mathcal V_t\le\frac{625000}{999}\bar r^2.
\]

Because the function term in the storage is nonnegative,

\[
\limsup_t(f(W_t)-f^\star)
\le\frac{6250000000000}{312929757}\bar r^2.
\]

At `rbar=0`, the relative-residual oracle therefore retains P14's full rate:
objective gap, gradient, and momentum converge geometrically, the updates are
absolutely summable, and the iterates converge to some trajectory-dependent
global minimizer. Uniqueness and arbitrary-pair contraction are not claimed.
For nonzero `rbar`, P15 claims the displayed storage and objective
neighborhoods, not iterate convergence.

## Exact controls and sharp boundaries

Four controls delimit the result.

1. With `kappa=0` and `rbar=0`, the solve is exact: the deployed operator and
   every applicable P14 storage, multiplier, and rate quantity are recovered.
   Restricting the zero absolute-error port gives the P14 `4 x 4` LMI.
2. Increasing the relative tolerance to `kappa=3/500` gives centered radius
   `253`. The unchanged frozen-storage `4 x 4` LMI then fails its exact
   definiteness check. This is rejection by this sufficient certificate, not
   a proof of instability or impossibility.
3. Loose residual rules can genuinely destroy convergence. In the abstract
   boundary case `A=0`, `B=1000*I`, choosing `u_hat=s` gives `r=-s` and
   `Y_hat=0`. It satisfies the `kappa=1`, `rbar=0` rule but can stall at a
   nonstationary point. This control is over the uniform admissible monotone-
   operator class; it is not asserted to be a trajectory of the specific P13
   base map.
4. At the same abstract boundary, `kappa=2`, `r=-2s`, and
   `u_hat=3s/2` give `Y_hat=-500s`. On the scalar curvature-`1` objective,
   the pinned EMA/Nesterov characteristic has `p(1)=-1/1280<0`, proving an
   unstable loose-tolerance control.

The factor-`500` residual-to-output gain is attained by the same `A=0`
boundary operator. Sampled trajectories are not used to prove any global
statement.

## Reproducibility and provenance

The primary generator writes every exact fraction, storage entry, LMI,
leading principal minor, stopping-rule field, control, and frozen-source hash
to `results/summaries/inexact_yosida_robustness_certificate.json`. A separate
standard-library-only program reconstructs the mathematics before comparing
with that artifact. Full Git history is required because source hashes are
checked at the recorded source commit, not against a mutable worktree.

The P14 source commit, canonical artifact hash, artifact commit, and annotated
checkpoint tag retain distinct provenance roles. P15 likewise uses separate
source and artifact commits. Automated replay is not an independent human
proof audit; the P15 reviewer packet remains unsigned.

## Exclusions

P15 certifies a residual **criterion**, not a solver. It does not establish:

- a convergent finite-step method for computing `u_hat`;
- a uniform iteration count or runtime for reaching the tolerance;
- finite-precision evaluation of `B(u_hat)` or of the graph residual;
- BF16, FP32, accelerator, or literal upstream-Muon parity;
- weight decay, aspect scaling, stochastic gradients, or neural-network
  convergence.

Those implementation questions remain separate. In particular, the exact-
real residual theorem must not be described as a certified deployed kernel.
