# EMA/Nesterov quadratic-loop IQC certificate

## Status and exact scope

This note records the final branch-`p3` discrete-time experiment/result. It is
an artifact theorem and proof replay, not a paper draft.

The pinned KellerJordan/Muon revision updates its momentum buffer by an EMA and
then constructs a Nesterov signal:

\[
m_{t+1}=\beta m_t+(1-\beta)g_t,
\qquad
s_{t+1}=\beta m_{t+1}+(1-\beta)g_t.
\]

The loop certified here uses that exact state-and-signal ordering, the pinned
default `beta=0.95`, a deterministic quadratic gradient, and

\[
W_{t+1}=W_t-\eta R(s_{t+1}),
\qquad R=F_{h,c}+\rho I.
\]

It is a real-arithmetic model in which the upstream orthogonalizer is replaced
by the repaired floored Jordan map. Weight decay is omitted. Consequently this
is not a theorem for the complete deployed optimizer: it does not include the
upstream BF16/current-plus-epsilon kernel, transpose/aspect multiplier,
distributed execution, stochastic gradients, nonquadratic objectives, or
neural-network convergence.

The earlier `momentum_iqc_certificate.md` studies a different, stylized
non-Nesterov recurrence. Its scaling cannot be absorbed into this one because
the floor makes `R` nonlinear and nonhomogeneous.

## Full-matrix operator sector

Let

\[
f(W)=\tfrac12\langle W-W_\star,H(W-W_\star)\rangle,
\qquad \ell I\preceq H\preceq LI,
\]

and use the full-matrix floored-repair certificate

\[
R=F_{h,c}+\rho I,
\qquad
\rho=\frac{\bar\delta_1}{c}+\mu,
\qquad
\bar\delta_1=\frac{41528474059081}{260261360000}.
\]

Then `R` is globally `mu`-strongly monotone for every fixed finite matrix
shape, and the rigorous global Lipschitz bound is

\[
K=\rho+\frac{4848763}{10000c}.
\]

No diagonal-only certificate is used in this step.

## Conditioned EMA/Nesterov recurrence

For two trajectories, set

\[
y=H^{1/2}\Delta W,
\qquad z=H^{-1/2}\Delta m,
\qquad a=1-\beta.
\]

The transformed Nesterov input and next state are

\[
\begin{aligned}
z_{t+1}&=\beta z_t+a y_t,\\
p_{t+1}&=\beta z_{t+1}+a y_t
         =\beta^2z_t+a(1+\beta)y_t,\\
y_{t+1}&=y_t-\alpha u(p_{t+1}),
\end{aligned}
\qquad
\alpha=\eta KL.
\]

Here `u` is the transformed incremental output difference divided by `K*L`.
Strong monotonicity, the quadratic curvature bounds, and Lipschitzness give
the two separate incremental IQCs

\[
\langle p,u\rangle\ge\nu\lVert p\rVert^2,
\qquad
\lVert u\rVert^2\le\lVert p\rVert^2,
\qquad
\nu=\frac{\mu\ell}{KL}.
\]

The inequalities remain separate because a strongly monotone Lipschitz map
need not be a gradient and may have a skew component.

With `chi=(y,z,u)`, define

\[
T=\begin{bmatrix}
1&0&-\alpha\\
1-\beta&\beta&0
\end{bmatrix},
\qquad
E=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix},
\qquad
q=\begin{bmatrix}1-\beta^2\\\beta^2\\0\end{bmatrix},
\qquad e_u=\begin{bmatrix}0\\0\\1\end{bmatrix}.
\]

Thus `p=q^T chi`. Symmetric matrices for the nonnegative IQCs are

\[
Q_\mu=\tfrac12(qe_u^\top+e_uq^\top)-\nu qq^\top,
\qquad
Q_L=qq^\top-e_ue_u^\top.
\]

For fixed `(alpha,beta,nu,tau)`, the dimension-independent feasibility test is

\[
P\succ0,\quad \lambda_\mu,\lambda_L\ge0,
\quad
T^\top PT-\tau^2E^\top PE
+\lambda_\mu Q_\mu+\lambda_LQ_L\prec0.
\]

The plus signs follow from writing both IQCs as nonnegative. A feasible point
proves global incremental exponential contraction in the storage `P tensor I`
for the specified quadratic loop, uniformly over finite matrix dimensions.
At equilibrium, `m=H(W-W_star)` and `s=m`; then `R(m)=0`. Strong
monotonicity with `R(0)=0` forces `m=0`, and positive definiteness of `H`
forces `W=W_star`. Thus every trajectory converges exponentially to the unique
equilibrium `(W_star,0)`.

## Exact locked representative

The committed rational replay uses

- `c=1`, `ell=1`, and `L=10`;
- the pinned default `beta=19/20`;
- `mu=648`, hence `rho=bar_delta_1+648`;
- `K=336372400608849/260261360000`;
- `nu=208209088000/4152745686529`;
- `alpha=1/400` and
  `eta=65065340/336372400608849 = 1.9343245724746989e-7`;
- `tau^2=99999/100000`.

An exact feasible point is

\[
P=\begin{bmatrix}
68309/100000&-315367/1000000\\
-315367/1000000&31691/100000
\end{bmatrix},
\qquad
\lambda_\mu=\frac{1847}{10^6},
\qquad
\lambda_L=\frac{43}{500000}.
\]

The leading principal minors of `P` are positive. The leading principal
minors of the symmetric LMI matrix alternate strictly in sign, so exact
Sylvester checks establish `P` positive definite and the LMI negative
definite. The machine-readable result stores the complete rational matrices
and minors. Floating-point solver status is not part of the proof.

## Numerical design search and hard decision rule

The exact replay above is authoritative. A separate 60-digit stationarity
solve, corroborated by a 1,101-point logarithmic scan over repair margins from
`1e-4` to `1e7`, locates the complex-skew design

\[
\mu\approx648.024,
\qquad
\eta_{\mathrm{sector}}\approx2.09708214366158\times10^{-7}.
\]

At the nearby exact choice `mu=648`, the complex-skew constant-slope necessary
boundary for the reduced sector class is
`alpha approximately 0.00271035452475484`, or
`eta approximately 2.09708214294051e-7`. This boundary is a statement about
the reduced strongly-monotone/one-Lipschitz sector, not about realizability by
the Jordan map.

For the linked curvature-`L` scalar mode, let

\[
g_0=\rho+\frac{h'(0)}c,
\qquad q=\eta Lg_0.
\]

The EMA/Nesterov zero-linearization has state matrix

\[
A(q)=\begin{bmatrix}
1-q(1-\beta)(1+\beta)&-q\beta^2\\
1-\beta&\beta
\end{bmatrix}.
\]

Jury's criterion gives the exact loss-of-local-stability threshold

\[
q_{\mathrm{local}}=
\frac{2(1+\beta)}{(1-\beta)(1+2\beta)}.
\]

At the locked parameters this yields

\[
\eta_{\mathrm{local}}
=\frac{8120154432000000000000000}
{3901919808117690731741568607}
=0.002081066457364538\ldots.
\]

The locked local threshold is `10,758.62x` the exact locked certified step. At
the numerical stationary design, the corresponding threshold is
`0.002081027708...`, about `9,923.44x` its complex-skew necessary boundary. It remains
thousands of times larger after optimizing `mu`. Under the predeclared hard
rule, this result is retained as an appendix/proof of principle: it proves
that the repaired full-matrix operator can be embedded in the pinned
EMA/Nesterov ordering, but it is not a headline practical-stability claim.

## Matched rank-one controls

The deterministic float64 checks use `H=diag(1,10)` and initialize only the
curvature-10 rank-one mode. This subspace is invariant, so the full repository
floored Jordan matrix operator reduces exactly to its scalar response. The
operator, floor, repair, EMA/Nesterov ordering, objective, initialization, and
operation order are held fixed; only `eta` changes.

- At the exact locked `eta`, position decreases from `1` to approximately
  `7.64e-113` after 100,000 updates.
- At `eta=1/400=0.0025`, the exact local Jury margin is negative and the
  trajectory settles into a nonzero period-two orbit. The retained float64
  tail has zero period-two residual.
- At `eta=1/200=0.005`, position crosses magnitude `1e100` in 264 updates.

These trajectories are matched diagnostics. They neither certify a global
boundary nor make the reduced complex-skew boundary realizable by Jordan.

## Provenance, parity, and prior art

The pinned source is KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, file `muon.py`, whose fetched
bytes have SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
Its `muon_update` defaults are `beta=0.95`, `ns_steps=5`, and
`nesterov=True`. The two `lerp_` calls implement the displayed EMA and
Nesterov equations, including in-place mutation of the gradient buffer. A
local regression checks the literal two-`lerp` sequence against the algebraic
recurrence; independent external parity sign-off remains an unchecked task.

The generic IQC setup is prior art. See Lessard, Recht, and Packard
(<https://arxiv.org/abs/1408.3595>) and Zhang, Bao, Lessard, and Grosse,
*A Unified Analysis of First-Order Methods for Smooth Games via Integral
Quadratic Constraints*, JMLR 22(103), 2021
(<https://jmlr.org/papers/v22/20-1068.html>). The research contribution here is
the certified full-matrix repaired floored-Muon sector and its connection to
this pinned signal ordering, not the generic one-step-memory LMI method.

Reproduce the artifact with

```bash
uv run --locked python scripts/certify_ema_nesterov_stability.py
uv run --locked pytest -q tests/test_ema_nesterov_iqc.py \
  tests/test_upstream_momentum.py
```
