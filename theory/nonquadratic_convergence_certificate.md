# Full-step nonquadratic convergence certificate

This is an artifact proof/result note, not a paper draft. All theorem claims
below are in deterministic real arithmetic.

## Statement

Use the same repaired max-floored five-step Jordan operator as p4 and the first
p5 certificate:

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

where

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

There is no additive epsilon. Let `f:R^(m x n)->R` be any fixed
differentiable objective that is globally `ell=1` strongly convex and `L=10`
smooth, where `m,n` are arbitrary fixed finite positive integers. Let
`W_star` be its unique minimizer. For the pinned EMA/Nesterov loop

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{32000}R(s_{t+1}),
\end{aligned}
\]

the exact certificate below proves global exponential convergence to
`(W_star,0)` at rate

\[
\tau=\frac{2499}{2500}.
\]

This retains the full p4 learning rate for the whole nonlinear function
class. It does not require a fixed Hessian or a common Hessian eigenbasis.

The guarantee is trajectory-to-minimizer convergence. It is not arbitrary-pair
incremental contraction. The earlier p5 common-storage result retains that
stronger incremental property at the smaller step `eta=1/640000`.

## 1. Normalized variables and p4 residual

Translate the minimizer to zero and define

\[
w=W-W_\star,\qquad
F(w)=\frac{f(W_\star+w)-f(W_\star)}{L},\qquad
u=\nabla F(w)=\frac{\nabla f(W_\star+w)}{L},
\]

with `k=ell/L=1/10`. The full-matrix p4 analysis supplies fixed exact
constants

\[
R(s)=\gamma s+\mathcal E(s),\qquad
\gamma=\frac{505021761888849}{520522720000},\qquad
\operatorname{Lip}(\mathcal E)\le
K_E=\frac{251582619905461}{520522720000}.
\]

Because `R(0)=0`, also `E(0)=0`. Put

\[
z=\frac{m}{L},\qquad
p=\frac{s}{L}=\beta^2z+(1-\beta^2)u,
\qquad
v=\frac{\mathcal E(s)}{K_E L}.
\]

Then `||v||_F<=||p||_F`, and with

\[
\alpha=\eta\gamma L,
\qquad r=\frac{K_E}{\gamma},
\]

the exact state transition is

\[
w_+=w-\alpha p-\alpha r v,
\qquad
z_+=\beta z+(1-\beta)u.
\]

The residual inequality is full-matrix and dimension uniform. It includes
the max-floor boundary through the earlier pairwise/Clarke proof; it is not a
sampled diagonal bound.

## 2. Exact smooth strongly-convex interpolation

For any two samples `(x_i,u_i,F_i)` and `(x_j,u_j,F_j)` from a fixed
`k`-strongly convex, `1`-smooth scalar potential, define

\[
\Phi(d,r)=
\frac{\lVert r\rVert_F^2-2k\langle d,r\rangle_F
      +k\lVert d\rVert_F^2}{2(1-k)}.
\]

The exact interpolation inequality is

\[
I_{ij}=F_i-F_j-\langle u_j,x_i-x_j\rangle_F
       -\Phi(x_i-x_j,u_i-u_j)\ge0.
\]

This inequality applies to every differentiable strongly-convex/smooth
potential in the stated class. It handles changing local curvature
orientations directly and does not diagonalize a Hessian.

Use the three actual samples

\[
(x_0,u_0,F_0)=(0,0,0),\quad
(x_1,u_1,F_1)=(w,u,F(w)),\quad
(x_2,u_2,F_2)=(w_+,u_+,F(w_+)).
\]

The locked interpolation multipliers are

\[
\begin{aligned}
\lambda_{01}&=\frac{67589}{25000000},\\
\lambda_{12}&=\frac{17111528143163}{62500000000000},\\
\lambda_{20}&=\frac{155434393163}{62500000000000},\\
\lambda_{21}&=\frac{1203}{2500000},
\end{aligned}
\]

with `lambda_02=lambda_10=0`. Every selected multiplier is strictly positive.

For transparency, let

\[
c=\frac{27081630}{100000000},\qquad
h=\lambda_{01},\qquad b=\lambda_{21}.
\]

They satisfy exactly

\[
\lambda_{12}=b+h+c\tau^2,
\qquad
\lambda_{20}=h-c(1-\tau^2)>0.
\]

Therefore, the function-value part of
`lambda_01 I_01 + lambda_12 I_12 + lambda_20 I_20 + lambda_21 I_21`
is exactly

\[
c\tau^2F(w)-cF(w_+).
\]

It cancels the function-value part of the one-step Lyapunov difference. This
is a dissipativity argument with explicit potential storage, not a hard
pointwise IQC after the potential terms are discarded.

## 3. Exact `5 x 5` LMI

Let

\[
\chi=(w,z,u,v,u_+).
\]

Write `[w_+,z_+]^T=T chi` and `[w,z]^T=X chi` using the transition in
Section 1. For each selected edge, let `Q_ij` be the symmetric scalar matrix
whose quadratic form is

\[
-\langle u_j,x_i-x_j\rangle_F
-\Phi(x_i-x_j,u_i-u_j).
\]

Let `Q_E` represent `||p||_F^2-||v||_F^2`. The locked storage and residual
multiplier are

\[
P=\frac1{100000000}
\begin{bmatrix}
495723&-3085119\\
-3085119&72422647
\end{bmatrix},
\qquad
\lambda_E=\frac{5066256}{100000000}.
\]

The exact matrix

\[
\mathcal M=T^TPT-\tau^2X^TPX
+\sum_{(i,j)\in\{(0,1),(1,2),(2,0),(2,1)\}}
\lambda_{ij}Q_{ij}+\lambda_EQ_E
\]

is strictly negative definite. The exact leading principal minors used by
Sylvester's criterion are positive for `P`:

\[
\frac{495723}{100000000},\qquad
\frac{1319180629731}{500000000000000},
\]

and are approximately

\[
2.8439669197\!\times10^{-4},\quad
3.8020219692\!\times10^{-6},\quad
1.7217116441\!\times10^{-8},\quad
2.8006830348\!\times10^{-11},\quad
8.8426632819\!\times10^{-16}
\]

for `-M`. The displayed values are readability aids. The committed exact
integer ratios and determinant replay are authoritative. No floating-point
solver status carries a theorem claim.

## 4. Lyapunov proof

Define

\[
V(w,z)=
\left\langle
\begin{bmatrix}w\\z\end{bmatrix},
(P\otimes I)
\begin{bmatrix}w\\z\end{bmatrix}
\right\rangle_F+cF(w).
\]

The exact identity `trace(P)+c=1` merely fixes the certificate scale. Since
`P` is positive definite, `c>0`, and

\[
\frac{k}{2}\lVert w\rVert_F^2\le F(w)
\le\frac12\lVert w\rVert_F^2,
\]

`V` is globally equivalent to the squared state norm.

Adding the nonnegative interpolation residuals and the nonnegative residual
IQC to `V_+-tau^2 V` cancels the two objective values and leaves exactly
`<chi,(M tensor I)chi>`:

\[
V_+-\tau^2V
+\sum_{(i,j)}\lambda_{ij}I_{ij}
+\lambda_E(\lVert p\rVert_F^2-\lVert v\rVert_F^2)
=\langle\chi,(\mathcal M\otimes I)\chi\rangle_F<0.
\]

Hence

\[
V_{t+1}<\tau^2V_t
\]

away from the equilibrium. Iteration proves global exponential convergence of
`W_t` to `W_star` and `m_t` to zero. The same `P`, `c`, and rate apply to all
finite matrix shapes and every fixed objective in the class, including
nonlinear objectives whose Hessian orientations change along the trajectory.

## 5. Classification and limits

The certified step is exactly

\[
\eta=\eta_{p4}=\frac1{32000},
\]

which is about `161.5551` times the locked p3 step. It therefore exceeds the
predeclared `0.1 eta_p4` excellent-result threshold and preserves the full p4
step. Its certified rate `2499/2500` also has forty times the decrement
`1-tau` of the p4 fallback rate `99999/100000`. This is the major nonlinear
theorem sought by p5.

The distinction between the two p5 results is important:

- `eta=1/32000`: global exponential convergence to the unique minimizer using
  objective-gap/interpolation storage;
- `eta=1/640000`: arbitrary-pair global incremental exponential contraction
  using a common quadratic storage.

Neither result covers a time-varying objective, stochastic gradients, BF16,
weight decay, aspect-ratio scaling, additive-epsilon normalization, exact
current normalization, or a complete neural-network training system.
