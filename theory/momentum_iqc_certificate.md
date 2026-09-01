# Discrete-time momentum IQC certificate

## Status and scope

This note records the branch-`p3` result. It is an artifact theorem and proof
replay, not a paper draft.

Let

\[
f(W)=\tfrac12\langle W-W_\star,H(W-W_\star)\rangle,
\qquad \ell I\preceq H\preceq LI,
\]

and consider the deterministic real-arithmetic loop

\[
m_{t+1}=\beta m_t+H(W_t-W_\star),\qquad
W_{t+1}=W_t-\eta R(m_{t+1}),
\]

where `0 <= beta < 1`, `R(0)=0`, and `R` is globally `mu`-strongly
monotone and `K`-Lipschitz in the Frobenius norm. For the floored Jordan map,

\[
R=F_{h,c}+\rho I,\qquad
\rho=\frac{\bar\delta_1}{c}+\mu,
\]

the previous full-matrix certificate supplies

\[
\bar\delta_1=\frac{41528474059081}{260261360000},
\quad
K=\rho+\frac{4848763}{10000c}.
\]

The result covers every fixed finite matrix shape and the full tangent space.
It does not cover nonquadratic or stochastic objectives, BF16 execution, or
neural-network convergence.

## The theorem

Define

\[
\nu=\frac{\mu\ell}{KL},\qquad \alpha=\eta KL.
\]

The loop is globally incrementally exponentially stable whenever

\[
0<\alpha<\alpha_\star(\beta,\nu)
=\frac{2(1-\beta)^2(1+\beta)\nu}
{(1+\beta)^2-4\beta\nu^2}.
\]

In particular, every trajectory converges exponentially to `(W_star, 0)`.
The boundary is sharp for the reduced class of arbitrary `nu`-strongly
monotone, one-Lipschitz maps. It need not be necessary for the linked
floored-Jordan/quadratic system when `ell < L`; the two scalar bounds used to
form that reduced sector can be conservative simultaneously.

## Reduction to a dimension-independent feedback loop

For two trajectories, set

\[
y=H^{1/2}\Delta W,\qquad z=H^{-1/2}\Delta m.
\]

The transformed nonlinear output difference

\[
\Delta S_H=H^{1/2}\bigl(R(m^1)-R(m^2)\bigr),
\qquad \Delta m=H^{1/2}v,
\]

satisfies

\[
\langle v,\Delta S_H\rangle\ge\mu\ell\lVert v\rVert^2,
\qquad
\lVert\Delta S_H\rVert\le KL\lVert v\rVert.
\]

After dividing this output difference by `K*L`, denote it by `u`. For a
nonlinear `R`, `u` can depend on the base pair as well as on its difference
`v`; the proof uses only the two incremental inequalities

\[
\langle v,u\rangle\ge\nu\lVert v\rVert^2,
\qquad
\lVert u\rVert^2\le\lVert v\rVert^2.
\]

These are two separate incremental quadratic constraints. The usual single
gradient/cocoercive sector product is not used: a strongly monotone Lipschitz
map may contain a skew-symmetric part and need not be a gradient.

Eliminating the intermediate variables gives the heavy-ball recurrence

\[
v_{t+1}=(1+\beta)v_t-\beta v_{t-1}-\alpha u_t.
\]

With `e_t=v_t-v_(t-1)` and lifted vector `chi=(v,e,u)`, the state transition
and state selector are

\[
T=\begin{bmatrix}1&\beta&-\alpha\\0&\beta&-\alpha\end{bmatrix},
\qquad
E=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}.
\]

Let `Q_mu` and `Q_L` denote the symmetric matrices representing

\[
q_\mu=\langle v,u\rangle-\nu\lVert v\rVert^2,
\qquad
q_L=\lVert v\rVert^2-\lVert u\rVert^2.
\]

For fixed `(alpha,beta,nu,tau)`, the following is a `3 x 3`,
dimension-independent SDP feasibility test:

\[
P\succ0,\quad \lambda_\mu,\lambda_L\ge0,\quad
T^\top PT-\tau^2E^\top PE
+\lambda_\mu Q_\mu+\lambda_LQ_L\prec0.
\]

The plus signs are essential because both quadratic constraints are written
as nonnegative. A feasible point proves contraction in the storage
`P tensor I` in the conditioned `(H^(1/2) Delta W, H^(-1/2) Delta m)`
coordinates.

## Closed-form strict certificate for the whole region

The SDP region above has an exact rational construction. Put

\[
d=1-\beta,\quad s=1+\beta,\quad
D=s^2-4\beta\nu^2,\quad
\Delta=2\nu d^2s-\alpha D.
\]

The strict bound `alpha < alpha_star` is exactly `Delta > 0`. Define

\[
\begin{aligned}
A_p&=\alpha^2(s^2+4\beta\nu^2)+8\alpha\beta\nu s+2d^2s^2,\\
A_q&=\alpha^2(s^2-4\nu^2)-4\alpha\nu ds+2d^2s^2,\\
A_r&=2\beta ds^2+2\alpha\nu ds-\alpha^2D,\\
A_L&=2\alpha^2\nu s+\alpha(s^2+4\beta\nu^2)+2\nu d^2s.
\end{aligned}
\]

For any rational `epsilon > 0`, choose

\[
P=\frac{\epsilon}{\alpha\Delta}
\begin{bmatrix}
dA_p/s&\beta A_q/s\\
\beta A_q/s&A_r
\end{bmatrix},
\quad
\lambda_\mu=\frac{2\epsilon A_p}{s\Delta},
\quad
\lambda_L=\frac{\epsilon A_L}{\Delta}.
\]

Direct expansion gives the exact factorization

\[
T^\top PT-E^\top PE+\lambda_\mu Q_\mu+\lambda_LQ_L
=-\epsilon I-\gamma nn^\top\prec0,
\]

where

\[
\gamma=\frac{2\beta\epsilon d^2s(\alpha\nu+s)}{\alpha\Delta},
\qquad
n=\left(\frac{2\alpha\nu}{ds},-1,-\frac{\alpha}{d}\right).
\]

The implementation expands and compares both sides in exact rational
arithmetic. Positivity of `P` follows by restricting the identity to the
admissible constant slope `u=nu*v`. Its closed-loop matrix is Schur by Jury's
criterion, and the restriction gives a strict discrete Lyapunov equation
`P-A_nu^T P A_nu = Q` with `Q` positive definite. At `beta=0`, the formula
reduces directly to a positive diagonal storage. This is a sufficiency proof
for the full nonlinear incremental sector, not merely for constant slopes.

The same complex constant slope

\[
U(v)=\nu v+\sqrt{1-\nu^2}\,Jv,
\qquad J^\top=-J,\quad J^\top J=I,
\]

has a unit-circle characteristic root at `alpha=alpha_star` and is unstable
above it. Thus the boundary is exact for the reduced sector description. This
is not a claim that the skew witness is realizable by the actual Jordan map.

## Locked representative certificate

The committed replay uses

- `c=1`;
- `mu=bar_delta_1`, hence `rho=2*bar_delta_1`;
- `ell=1`, `L=10`, and `beta=9/10`;
- `nu=41528474059081/2092515133879300`;
- `alpha=1/10000` and
  `eta=6506534/523128783469825`;
- `tau^2=99999/100000`.

For conditioning only, set `d=1/10`, `r=d*z`, `v_hat=d*v`, and
`u_hat=d*u`. This simultaneous input/output scaling preserves both IQCs.
In lifted coordinates `(y,r,u_hat)`, the exact storage and nonnegative IQC
multipliers are

\[
P=\frac1{10^6}
\begin{bmatrix}739113&-260843\\-260843&260887\end{bmatrix},
\quad
\lambda_\mu=\frac{957}{10^6},
\quad
\lambda_L=\frac{11}{10^6}.
\]

The exact leading principal minors of `P` are positive and those of the LMI
alternate in sign, so `P` is positive definite and the LMI is negative
definite by Sylvester's criterion. No floating-point solver status is part of
the proof. The full matrices and minors are stored in the machine-readable
result.

The analytic strict supremum at this configuration is

\[
\eta_\star=
\frac{1026784428047408433965200}
{39501520231843908972134883541201}
=2.59935420718\ldots\times10^{-8}.
\]

The locked point is `0.478493...` of this sufficient boundary.

## Matched actual-operator checks and conservatism

The deterministic CPU check uses `H=diag(1,10)` and initializes only the
curvature-10 rank-one mode. That subspace is invariant, and the repository's
full floored Jordan matrix operator reduces exactly to the recorded scalar
recurrence. The operator, floor, repair, momentum, objective, initialization,
and operation order are held fixed; only `eta` changes.

- At the locked certified `eta`, the position decreases from `1` to about
  `2.37e-38` after 100,000 updates.
- At `eta=1/2000`, the optimizer is locally unstable by an exact Jury sign and
  the trajectory approaches a nonzero period-four orbit.
- At `eta=1/500`, the same trajectory crosses magnitude `1e100` in 159 updates.

The actual zero-linearization has gain

\[
\rho+\frac{h'(0)}c
\]

and loses local Schur stability in the curvature-`L` mode at

\[
\eta_{\mathrm{local}}
=\frac{2(1+\beta)}{L(\rho+h'(0)/c)}
=0.000472633706612\ldots.
\]

This is about `18,183` times the global sector-certified supremum. Therefore
the theorem is clean, global, and dimension-uniform, but its step-size region
is extremely conservative for the actual floored Jordan loop. The trajectory
outside the region is a matched falsification check, not evidence that the
global IQC boundary is tight for Jordan.

Reproduce everything with

```bash
uv run --locked python scripts/certify_momentum_stability.py
uv run --locked pytest -q tests/test_momentum_iqc.py
```

The IQC formulation follows the standard optimization-algorithm framework of
Lessard, Recht, and Packard: <https://arxiv.org/abs/1408.3595>.
