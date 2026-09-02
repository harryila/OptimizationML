# Nonquadratic EMA/Nesterov stability certificate

This is an artifact proof/result note, not a paper draft. All theorem claims
below are in deterministic real arithmetic.

## Statement and exact scope

Let the ambient space be `R^(m x n)`, for arbitrary fixed finite positive
`m,n`, with the Frobenius inner product. Define

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad h=q^{\circ 5},
\]

and let `H_h` be the rectangular singular-value map induced by `h`. The
normalizer and repaired operator are

\[
N_1(M)=\frac{M}{\max\{1,\lVert M\rVert_F\}},\qquad
F_{h,1}=\mathcal H_h\circ N_1,\qquad
R(M)=F_{h,1}(M)+\rho M,
\]

with no additive epsilon and with the exact constant conductance

\[
\rho=\frac{210177835339081}{260261360000}
     =\bar\delta_1+648.
\]

Let `f:R^(m x n)->R` be any fixed differentiable, globally
`1`-strongly convex, `10`-smooth scalar objective. It need not be quadratic.
In particular, its local Hessian orientations may change with the point and
along the trajectory. For

\[
g_t=\nabla f(W_t)
\]

consider the pinned real-arithmetic EMA/Nesterov ordering

\[
\begin{aligned}
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{640000}R(s_{t+1}).
\end{aligned}
\]

The exact rational LMI below proves global incremental exponential stability.
For any two trajectories, put

\[
x_t=\begin{bmatrix}\Delta W_t\\ \Delta m_t/10\end{bmatrix},
\qquad
V_t=\langle x_t,(P\otimes I)x_t\rangle_F,
\]

where

\[
P=\frac1{10^{10}}
\begin{bmatrix}
6682146216&-1788501566\\
-1788501566&3317853784
\end{bmatrix}.
\]

Then

\[
V_{t+1}<\left(\frac{99999}{100000}\right)^2 V_t
\]

whenever the two states differ. Thus the common storage contracts for the
whole nonlinear function class, every finite matrix shape, and every pair of
initial conditions. The loop converges globally to the unique equilibrium
`(W_star,0)`, where `W_star` is the unique minimizer of `f`.

This is a theorem for the repaired max-floored operator. It is not a theorem
for exact current normalization, additive-epsilon normalization, BF16,
stochastic gradients, weight decay, aspect-ratio scaling, or a complete neural
network training system.

## 1. The p4 operator decomposition

The exact full-matrix p4 certificate establishes the global incremental split

\[
R(s)=\gamma s+\mathcal E(s),
\]

where

\[
\gamma=\frac{505021761888849}{520522720000},\qquad
K_E=\frac{251582619905461}{520522720000},\qquad
\operatorname{Lip}(\mathcal E)\le K_E.
\]

These are dimension-uniform constants derived from the certified derivative
interval of the five-step Jordan polynomial, the radial structure of the
Frobenius floor, and the constant repair. They are not fitted to the current
input. The same proof also gives

\[
-d\lVert\Delta s\rVert_F^2
\le \langle\Delta s,\Delta e\rangle_F
\le d\lVert\Delta s\rVert_F^2,
\qquad
d=\frac{167723039328849}{520522720000},
\]

for `Delta e=E(s_1)-E(s_2)`. At the nondifferentiable floor boundary, the
pairwise statements follow by integrating the almost-everywhere derivative
or the equivalent Clarke bounds along the line segment. No differentiability
of the floor at `||s||_F=1` is assumed.

The two centered inner-product inequalities are valid extra IQCs. They must
not be collapsed into `||Delta e||<=d||Delta s||`, which would incorrectly
discard the certified skew component. Their optimal multipliers were
numerically zero in this certificate, so the exact locked replay uses only the
independent norm bound `||Delta e||<=K_E||Delta s||`.

## 2. A pointwise IQC for the whole objective class

For two positions, define

\[
w=\Delta W,\qquad g=\nabla f(W_1)-\nabla f(W_2).
\]

Every differentiable `ell`-strongly convex, `L`-smooth scalar potential
satisfies the exact interpolation inequality

\[
\langle g-\ell w,Lw-g\rangle_F\ge0.
\]

For the locked `ell=1,L=10` class this follows, for example, by applying
cocoercivity to the convex function
`f(W)-||W||_F^2/2`. Crucially, this is a pairwise, pointwise inequality. The
proof never diagonalizes a Hessian and never assumes that Hessians at
different points commute. The same IQC can therefore be applied at every time
step even when its effective secant orientation changes arbitrarily.

The scalar-potential assumption matters: this interpolation IQC does not hold
for every generic strongly monotone, Lipschitz game operator.

## 3. Scaled difference dynamics

Use the dimensionless variables

\[
z=\frac{\Delta m}{L},\qquad
u=\frac{\Delta g}{L},\qquad
v=\frac{\Delta e}{K_E L},\qquad L=10,
\]

and put

\[
p=\frac{\Delta s}{L}=\beta^2z+(1-\beta^2)u.
\]

With

\[
\alpha=\eta\gamma L,
\qquad r=\frac{K_E}{\gamma},
\qquad k=\frac\ell L=\frac1{10},
\]

the difference dynamics are

\[
\begin{aligned}
w_+&=w-\alpha p-\alpha r v,\\
z_+&=\beta z+(1-\beta)u.
\end{aligned}
\]

For `chi=(w,z,u,v)`, write `[w_+,z_+]^T=T chi` and
`[w,z]^T=X chi`, with

\[
T=
\begin{bmatrix}
1&-\alpha\beta^2&-\alpha(1-\beta^2)&-\alpha r\\
0&\beta&1-\beta&0
\end{bmatrix},
\qquad
X=\begin{bmatrix}1&0&0&0\\0&1&0&0\end{bmatrix}.
\]

The two IQCs used in the exact replay are

\[
q_f(\chi)=\langle u-kw,w-u\rangle_F\ge0,
\qquad
q_E(\chi)=\lVert p\rVert_F^2-\lVert v\rVert_F^2\ge0.
\]

They hold for vectorized matrices of every finite dimension.

## 4. Exact common-storage LMI

Let `Q_f,Q_E` be the symmetric `4 x 4` scalar matrices representing the two
IQCs above. The exact locked multipliers are

\[
\lambda_f=\frac{194722270}{10^{10}},\qquad
\lambda_E=\frac{136225077}{10^{10}}.
\]

The authoritative matrix is

\[
\mathcal M=
T^TPT-\tau^2X^TPX+\lambda_fQ_f+\lambda_EQ_E,
\qquad \tau=\frac{99999}{100000}.
\]

The exact `Fraction` replay in
`src/passive_muon/nonquadratic_stability.py` gives strictly positive leading
principal minors

\[
\begin{array}{c|cc}
&1&2\\ \hline
P&6.682146216\!\times10^{-1}&1.89716462564\!\times10^{-1}
\end{array}
\]

and

\[
\begin{array}{c|cccc}
&1&2&3&4\\ \hline
-\mathcal M
&1.93385847\!\times10^{-3}
&3.18141698\!\times10^{-5}
&1.75231070\!\times10^{-9}
&3.42893905\!\times10^{-13}.
\end{array}
\]

The displayed decimals are readability aids; the committed integer-ratio
values and exact determinants are authoritative. Sylvester's criterion proves
`P` positive definite and `M` negative definite without relying on a solver
tolerance.

For an admissible trajectory difference,

\[
\begin{aligned}
V_+-\tau^2V
&=\langle\chi,
(T^TPT-\tau^2X^TPX)\otimes I\,\chi\rangle_F\\
&=\langle\chi,(\mathcal M\otimes I)\chi\rangle_F
  -\lambda_fq_f(\chi)-\lambda_Eq_E(\chi)\\
&<0.
\end{aligned}
\]

This common `P tensor I` argument is precisely what makes the certificate
dimension independent and robust to changing local curvature orientations.

## 5. Equilibrium and convergence

Strong convexity gives a unique `W_star` with `grad f(W_star)=0`; since the
odd spectral map and the linear repair both vanish at zero,
`(W_star,0)` is an equilibrium. Conversely, any equilibrium obeys `m=g` and
`s=g`, while `R(g)=0`. Strong monotonicity of the repaired operator and
`R(0)=0` force `g=0`, hence the equilibrium is unique. Applying the incremental
contraction to a trajectory and this equilibrium proves global exponential
convergence.

## 6. Search diagnostics and classification

A floating-point SDP search (CVXPY 1.9.2, Clarabel 0.11.1, with an independent
SCS sign check) was used only for discovery. At rate one, the basic static-IQC
feasibility boundary was approximately

\[
\eta=1.69390537084\times10^{-6}
     =0.05420497\,\eta_{p4}.
\]

Adding the two valid centered residual IQCs changed the numerical boundary by
less than `1e-11` relative and gave multipliers near zero. This search does not
prove an impossibility result for larger steps or for richer dynamic/cyclic
IQCs.

The locked exact point is

\[
\eta=\frac1{640000}=0.05\,\eta_{p4}
\]

and is approximately `8.077755` times the locked p3 step. Under the declared
gate, it is a **strong nonlinear extension** (`0.01 eta_p4 <= eta < 0.1
eta_p4`), not an excellent-rate recovery of p4. The p4 quadratic theorem at
`eta=1/32000` remains the guaranteed stronger-step fallback.

The deterministic nonlinear trajectory experiment accompanying this note is
a falsification probe only. Its sampled success cannot replace the common
storage and exact LMI proof above.
