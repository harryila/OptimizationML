# Shape-preserving gated-resolvent certificate

This is an artifact proof/result note, not a paper draft. All theorem claims
below are for deterministic exact real arithmetic. The numerical spectrum
study is separately identified as evidence.

## Statement and scope

Fix an arbitrary finite real rectangular matrix space and the deployed
additive-epsilon scale

\[
\epsilon=10^{-7}.
\]

Let

\[
E_{h,\epsilon}(U)=\mathcal H_h\!\left(
  \frac{U}{\lVert U\rVert_F+\epsilon}
\right),
\]

where `h` is obtained by composing exactly five times the odd quintic

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5.
\]

The normalization, coefficients, iteration count, and epsilon placement are
the exact-real counterpart of the formula pinned to KellerJordan/Muon
revision `f98f1cacc0263b04290753e32be8d498c1efc806`. Upstream does not contain
the radial passivator, shunt, resolvent, or gate defined here.

Let `A_epsilon=E_h,epsilon+G_epsilon` be C19's globally monotone radial
repair, and define

\[
B=A_\epsilon+1000I,
\qquad
J=(I+B/1000)^{-1},
\qquad
Y=1000(I-J),
\qquad
X=E_{h,\epsilon}\circ J.
\]

For `q=||S||_F^2`, use the twice-continuously differentiable gate

\[
\theta(q)=
\begin{cases}
0,&q\le 1/4,\\
\bar\theta\,\psi((4q-1)/3),&1/4<q<1,\\
\bar\theta,&q\ge1,
\end{cases}
\qquad
\psi(t)=6t^5-15t^4+10t^3,
\]

and expose to the optimizer

\[
T_{\bar\theta,c}(S)
=\bigl(1-\theta(\lVert S\rVert_F^2)\bigr)\frac{Y(S)}c
 +\theta(\lVert S\rVert_F^2)X(S).
\]

P17 locks two certified points on the stability--fidelity frontier:

| design | `theta_bar` | `c` | `eta` | role |
| --- | ---: | ---: | ---: | --- |
| high fidelity | `3/4` | `4096` | `1/128000` | primary P17 design |
| full step | `1/8` | `8192` | `1/32000` | retains the P14/P15 numerical step |

Both results are global over every fixed finite matrix shape and every
differentiable globally `10`-smooth objective with finite infimum satisfying
the global PL inequality with constant `1`. They are trajectory-to-some-
minimizer results, not arbitrary-pair incremental-contraction results.

## 1. Dimension-uniform pointwise sector

By C22, `S`, `J(S)`, `Y(S)`, and `X(S)` have common singular-vector
subspaces, including at repeated and zero singular values. Write their
nonnegative mode values as `sigma_i`, `x_i`, `y_i`, and `e_i`. With
`r=||x||_2`, `z=r/epsilon`, and
`g(r)=p(z)/r`, the resolvent graph equation is

\[
\sigma_i=[1+\lambda(\mu+g(r))]x_i+\lambda e_i,
\qquad \lambda=10^{-3},\quad\mu=1000.
\]

Every Jordan stage preserves nonnegativity on `[0,1]`: the quadratic factor
in `x^2` has positive leading coefficient and negative discriminant. Hence
`e_i>=0`. For a positive mode put `H_i=e_i/x_i`. C4's exact derivative upper
certificate and `h(0)=0` give

\[
0\le \frac{h(w)}w\le b,
\qquad b=\frac{4848763}{10000}.
\]

C19 gives `p(z)/z>=d_hat(z)`. Its locked majorant also obeys

\[
(1+z)\widehat d(z)\ge U_0,
\qquad
U_0=\frac{6602082433275499863}{41641817600000000}.
\]

On the constant branch this is immediate. On the tail the left-hand side is
a convex combination of `199437/1250` and
`41528474059081/260261360000`, both strictly larger than `U_0`.
Consequently

\[
0\le\frac{e_i}{\sigma_i}\le M_X,
\qquad
M_X=
\frac{20191130443162880000000}{26793221204801899863}
\approx753.5910030685.
\]

This is an analytic full-matrix bound, not a sampled mode search. Zero modes
follow from rank preservation and continuity, and the statement is
basis-independent on repeated singular subspaces.

The same graph gives the needed modal Yosida bound directly, without
interpreting C20's nonsymmetric incremental IQC as a Loewner bound. Put

\[
k_i=\mu+g(r)+H_i\ge\mu=1000.
\]

Then `y_i=k_i x_i`, `sigma_i=(1+lambda k_i)x_i`, and hence

\[
\frac{y_i}{\sigma_i}=\frac{k_i}{1+\lambda k_i}\in[500,1000).
\]

The zero-mode statement follows by continuity. Therefore, for any
`0<=theta(q)<=theta_bar<1`, every singular-mode gain of `T` lies in

\[
m=\frac{(1-\bar\theta)500}{c},
\qquad
M=\frac{(1-\bar\theta)1000}{c}+\bar\theta M_X.
\]

The locked parameters have `M_X>1000/c`, so the displayed upper endpoint is
the interval maximum. With `gamma=(M+m)/2` and `K=(M-m)/2`, summing the mode
inequalities proves the global origin-centered supply

\[
\lVert T(S)-\gamma S\rVert_F\le K\lVert S\rVert_F.
\]

This is deliberately a **pointwise** sector. It is not an incremental sector
or a Jacobian bound. Differentiating the radial gate adds

\[
2\theta'(\lVert S\rVert_F^2)
\bigl(X(S)-Y(S)/c\bigr)\otimes S,
\]

which is not used or silently discarded.

## 2. Exact smooth-PL certificates

Use the pinned exact-real EMA/Nesterov ordering

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\eta T(s_{t+1}).
\end{aligned}
\]

The P6 value--momentum proof needs only the pointwise residual supply above:
it never differentiates `T` or compares `T(S_1)` with `T(S_2)`. The same two
directed `(-1,1)` smooth interpolation inequalities and next-point PL supply
therefore apply.

For the primary high-fidelity design,

\[
m=\frac{125}{4096},
\qquad
M=\frac{10338975171116261305827625}
        {18290839009144763639808},
\]

and the exact certificate uses

\[
\eta=\frac1{128000},
\qquad
\tau=\frac{16777215}{16777216},
\]

\[
P=\begin{bmatrix}
1792/7757&-2143/9279\\
-2143/9279&2168/9389
\end{bmatrix},
\quad c_F=1,
\quad\omega=\frac{1831}{200},
\quad\lambda_R=\frac{110}{9963}.
\]

For the full-step design,

\[
m=\frac{875}{16384},
\qquad
M=\frac{20699161642352990782380125}
        {219490068109737163677696},
\]

and the exact certificate uses

\[
\eta=\frac1{32000},
\qquad
\tau=\frac{16777209}{16777216},
\]

\[
P=\begin{bmatrix}
13151/100000&-6559/50000\\
-6559/50000&6547/50000
\end{bmatrix},
\quad c_F=1,
\quad\omega=\frac{24999}{5000},
\quad\lambda_R=\frac{737}{100000}.
\]

For each design, the committed exact replay verifies positive definiteness of
`P`, strict negative definiteness of the `4 x 4` dissipation LMI, nonnegative
multipliers, and exact objective-value cancellation. Thus

\[
\mathcal V_{t+1}\le\tau^2\mathcal V_t.
\]

It follows that `f(W_t)-f_star`, the gradient, and momentum converge
geometrically. Since `||T(s)||<=M||s||`, the updates are geometrically
summable; finite-dimensional completeness then gives convergence of `W_t` to
some trajectory-dependent global minimizer. PL does not imply uniqueness.

## 3. Why the passive region matters for local incremental behavior

Raw `X(S)=E_h,epsilon(J(S))` does not inherit C20's incremental sector. In the
scalar parameterization `t=u/(epsilon+u)`, its derivative is

\[
\frac{dX}{dS}
=\frac{E'_{h,\epsilon}(u)}
 {1+\lambda\{\mu+E'_{h,\epsilon}(u)+G_\epsilon'(u)\}}.
\]

At the exact rational control `t=63/10000`, the certificate proves

\[
-147000<\frac{dX}{dS}<-146000,
\qquad
\frac{19}{10000}<S<\frac1{500}.
\]

An outward-rounded nearby unsafe Arb band independently encloses strictly
negative derivatives. The two locked gates are exactly zero
throughout this region because `S^2<1/4`, so their exposed interface is the
passive fallback `Y/c` there.

As a negative control, move the gate transition below the unsafe signal and
let the `1/8` gate already be saturated. The exact/interval replay obtains
`-19000<T'(S)<-18000`; the corresponding frozen scalar EMA/Nesterov Jury
margin is strictly negative. This control is a local/incremental
linearization at a non-equilibrium operating signal. It is not claimed to be
a divergent smooth-PL trajectory. In particular, the location of the passive
region is not a premise of the pointwise-sector PL theorem in Section 2,
which uses only `0<=theta<=bar_theta`; this control instead rules out a local
incremental/passivity interpretation of an under-sized gate.

## 4. Fidelity evidence and its logical status

The P16 thresholds remain unchanged:

\[
\chi(T;S)\ge10^{-3},
\qquad
\frac{\chi(T;S)}{\chi(E_{h,\epsilon};S)}\ge0.1,
\]

where `chi` is the relative residual from the best scalar multiple of `S`.
The canonical `diag(3,4)` decisions are enclosed with outward-rounded Arb
arithmetic, including inflation by the exact P15 residual-to-resolvent bound;
they are not decisions made from an unguarded FP64 root.

The declared spectrum-grid and realistic-rank results are deterministic FP64
diagnostics. They test fidelity, amplitude, rank accumulation, and the
computed P15 residual, but they are not global fidelity extrema or rounding
certificates. The global stability theorem is instead carried by the analytic
dimension-uniform pointwise sector in Section 1.

## 5. Qualifications

P17 certifies an exact-resolvent, exact-real, Muon-derived interface. It does
not yet certify:

- P16 solver error after passing through `X` and the gate;
- FP32 or BF16 arithmetic, rounded residual evaluation, or accelerator parity;
- weight decay, aspect-ratio scaling, stochastic gradients, or model-state
  reconstruction;
- literal upstream Muon or neural-network training;
- arbitrary-pair incremental contraction.

The high-fidelity grid is evidence for the locked operating study, not a
matrix-shape-uniform fidelity theorem. The global theorem is smooth-PL
trajectory convergence from the pointwise sector and should be described as
such.
