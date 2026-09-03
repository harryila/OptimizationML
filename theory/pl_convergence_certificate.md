# Smooth-PL function-value convergence certificate

This is an artifact proof/result note, not a paper draft. All claims below are
for deterministic exact real arithmetic.

## Statement and scope

Use the repaired max-floored five-step Jordan operator

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

The normalizer is the exact max floor with floor `c=1`; there is no additive
epsilon. The polynomial has the displayed exact rational coefficients and is
composed for exactly five Newton--Schulz steps.

Let `f:R^(m x n)->R` be fixed and differentiable, with globally
`L=10`-Lipschitz gradient in the Frobenius norm and finite infimum
`f_star=inf_W f(W)>-infinity`. Assume the global Polyak--Lojasiewicz inequality
with `ell_PL=1`, using the convention

\[
\frac12\lVert\nabla f(W)\rVert_F^2
\ge \ell_{\rm PL}\bigl(f(W)-f_\star\bigr)
\quad\text{for every }W.
\]

No convexity or unique minimizer is assumed. The matrix dimensions `m,n` are
arbitrary fixed finite positive integers. For arbitrary finite initial
`(W_0,m_0)`, apply the pinned EMA/Nesterov loop

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{32000}R(s_{t+1}).
\end{aligned}
\]

The exact certificate below proves

\[
f(W_t)-f_\star\le Cq^t,
\qquad
q=\left(\frac{19999}{20000}\right)^2
 =\frac{399960001}{400000000}<1,
\qquad
m_t\longrightarrow0,
\]

where `C` is finite and depends on the initial state. This retains the full p5
learning rate `eta=1/32000`.

The headline claim is function-value convergence and momentum decay. It does
not claim convergence to a unique or preselected minimizer. Because the
certified state and hence the update increments decay geometrically, finite
dimensional completeness additionally implies that `W_t` converges to some
trajectory-dependent global minimizer. This corollary does not make that
minimizer unique.

## 1. Normalized dynamics and residual supply

Define

\[
F(W)=\frac{f(W)-f_\star}{L},\qquad
u=\nabla F(W)=\frac{\nabla f(W)}L,\qquad
z=\frac mL,
\qquad k=\frac{\ell_{\rm PL}}L=\frac1{10}.
\]

The p4 full-matrix certificate supplies

\[
R(s)=\gamma s+\mathcal E(s),\qquad
\gamma=\frac{505021761888849}{520522720000},\qquad
\operatorname{Lip}(\mathcal E)\le
K_E=\frac{251582619905461}{520522720000}.
\]

Since `R(0)=0`, also `E(0)=0`. Put

\[
p=\frac{s}{L}=\beta^2z+(1-\beta^2)u,
\qquad
v=\frac{\mathcal E(Lp)}{K_E L},
\qquad
\lVert v\rVert_F\le\lVert p\rVert_F.
\]

With

\[
\alpha=\eta\gamma L
=\frac{505021761888849}{1665672704000000},
\qquad
r=\frac{K_E}{\gamma}
=\frac{251582619905461}{505021761888849},
\]

the displacement and normalized momentum update are

\[
d=W_+-W=-\alpha p-\alpha r v,
\qquad
z_+=\beta z+(1-\beta)u.
\]

Thus the full-matrix residual supply is

\[
J_E=\lVert p\rVert_F^2-\lVert v\rVert_F^2\ge0.
\]

## 2. Smooth nonconvex interpolation and PL supply

Every differentiable one-smooth scalar potential obeys the directed
`(-1,1)` interpolation inequality

\[
I_{ij}=F_i-F_j-\langle u_j,x_i-x_j\rangle_F
-\frac{\lVert u_i-u_j\rVert_F^2
+2\langle x_i-x_j,u_i-u_j\rangle_F
-\lVert x_i-x_j\rVert_F^2}{4}\ge0.
\]

This does not assume convexity. One direct derivation sets
`H(x)=F(x)+||x||^2/2`: one-smoothness makes `H` convex and two-smooth, and the
standard convex smooth interpolation inequality for `H` expands to the
displayed formula.

Use the two actual samples, translated without loss of generality,

\[
(x_1,u_1,F_1)=(0,u,F(W)),\qquad
(x_2,u_2,F_2)=(d,u_+,F(W_+)).
\]

The normalized PL inequality at the next sample is

\[
J_{\rm PL,+}=\lVert u_+\rVert_F^2-2kF(W_+)\ge0.
\]

This pointwise value-gradient supply replaces the strong-convexity
incremental IQC used by p5. PL alone does not make the gradient monotone and
does not justify the p5 distance-to-one-minimizer storage.

## 3. Exact value--momentum LMI

Let

\[
\chi=(z,u,v,u_+),
\]

write `[z_+,u_+]^T=A chi`, `[z,u]^T=B chi`, and let `Q_12,Q_21,Q_E`
be the scalar symmetric matrices for the non-function parts of `I_12,I_21`
and `J_E`. Let `U_+` select `||u_+||^2`.

Use the storage

\[
V(W,z)=
\left\langle
\begin{bmatrix}z\\u\end{bmatrix},
(P\otimes I)
\begin{bmatrix}z\\u\end{bmatrix}
\right\rangle_F+cF(W),
\]

with

\[
P=\frac1{100000}
\begin{bmatrix}
72435&-3185\\
-3185&499
\end{bmatrix},
\qquad
c=\frac{27066}{100000},
\qquad
\tau=\frac{19999}{20000}.
\]

The exact nonnegative multipliers are

\[
\begin{aligned}
\lambda_{21}&=\frac{20753}{100000},\\
\lambda_{12}&=\lambda_{21}+c\tau^2
=\frac{9563258693533}{20000000000000},\\
\lambda_{\rm PL,+}&=\frac{c(1-\tau^2)}{2k}
=\frac{541306467}{4000000000000},\\
\lambda_E&=\frac{5052}{100000}=\frac{1263}{25000}.
\end{aligned}
\]

Their function-value coefficients cancel exactly:

\[
-c\tau^2+\lambda_{12}-\lambda_{21}=0,
\]

\[
c-\lambda_{12}+\lambda_{21}
-2k\lambda_{\rm PL,+}=0.
\]

The remaining exact quadratic matrix is

\[
\mathcal M=A^TPA-\tau^2B^TPB
+\lambda_{12}Q_{12}+\lambda_{21}Q_{21}
+\lambda_{\rm PL,+}U_++\lambda_EQ_E.
\]

It is strictly negative definite. Sylvester's criterion gives the exact
positive leading minors of `P`

\[
\frac{14487}{20000},\qquad
\frac{650021}{250000000},
\]

and the leading minors of `-M` are approximately

\[
1.6567568\!\times10^{-2},\quad
7.8599846\!\times10^{-5},\quad
4.2583205\!\times10^{-7},\quad
3.7443246\!\times10^{-11}.
\]

The committed exact rational determinant replay, rather than these decimal
readability aids or a numerical solver status, is authoritative.

## 4. Dissipation and consequences

The exact function-flow identities yield

\[
\begin{aligned}
V_+-\tau^2V
&+\lambda_{12}I_{12}+\lambda_{21}I_{21}
+\lambda_{\rm PL,+}J_{\rm PL,+}+\lambda_EJ_E\\
&=\langle\chi,(\mathcal M\otimes I)\chi\rangle_F\le0.
\end{aligned}
\]

All added supplies are nonnegative, so

\[
V_{t+1}\le\tau^2V_t.
\]

Because `P` is positive definite, `c>0`, and `F>=0`, iteration gives

\[
f(W_t)-f_\star
\le\frac{L V_0}{c}\,\tau^{2t},
\]

and both `m_t/L=z_t` and `grad f(W_t)/L=u_t` tend to zero at least
geometrically. Moreover `||v_t||<=||p_t||`, so the increments
`W_(t+1)-W_t=-alpha(p_t+r v_t)` are summable. Consequently `W_t` is Cauchy
and converges to some global minimizer, without any uniqueness conclusion.

## 5. Necessary qualifications

The result covers every fixed differentiable globally `10`-smooth objective
with finite infimum satisfying the global PL inequality with constant `1`,
arbitrary finite matrix shape, arbitrary initial momentum, and changing local
Hessian orientations where Hessians exist. It is not an
incremental-contraction theorem.

For example, `f(x,y)=x^2/2` is `10`-smooth and satisfies PL with constant `1`,
but has the entire line `(0,y)` as minimizers. Two zero-momentum trajectories
initialized at different points on that line remain separated, ruling out a
global arbitrary-pair contraction or unique-minimizer claim for the class.

The theorem does not cover a merely local PL condition, stochastic or
time-varying gradients, BF16 arithmetic, weight decay, aspect-ratio scaling,
the additive-epsilon normalizer, exact current normalization, or an unrepaired
upstream Muon implementation.
