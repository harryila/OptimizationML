# P14 exact Yosida stability certificate

## Scope

This note certifies an exact implicit continuation of the P13 additive-epsilon
radial repair. The ambient Hilbert space is

\[
\mathcal X=\mathbb R^{m\times n}
\]

with the Frobenius inner product, for arbitrary fixed finite positive `m,n`.
All operator and LMI statements are dimension independent and use exact real
arithmetic.

The theorem does **not** specify an algorithm for evaluating a resolvent. It
does not bound an iterative solve tolerance, runtime, BF16 arithmetic, or
rounding error. Such solve-error ports are deferred to P15. In particular,
P14 is not a theorem for literal upstream Muon.

## P13 base operator

Fix any `epsilon>0` and let

\[
A_\epsilon=E_{h,\epsilon}+G_\epsilon,
\]

where `E_{h,epsilon}` is the exact-real, five-stage additive-epsilon Jordan
map: it normalizes by `M/(||M||_F+epsilon)` and repeats the stage polynomial
with exact coefficients `6889/2000`, `-191/40`, and `4063/2000` exactly five
times. The term `G_epsilon` is P13's radial correction. P13 proves, over every
finite matrix shape,

\[
\langle A_\epsilon(X)-A_\epsilon(Y),X-Y\rangle_F\ge0.
\]

It also proves that `A_epsilon` is continuous and zero preserving. Its
explicit differential stiffness remains of order `1/epsilon`, which is the
obstruction addressed here.

The practical formula provenance remains KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, whose pinned `muon.py` has
SHA-256 `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
That provenance does not imply the upstream implementation contains the P13
repair or the P14 resolvent.

Define

\[
B=A_\epsilon+\mu I,
\qquad \mu=1000.
\]

Then `B(0)=0`, and for all `X,Y`,

\[
\langle B(X)-B(Y),X-Y\rangle_F
\ge \mu\lVert X-Y\rVert_F^2.
\]

Thus `B` is continuous, full-domain, and `mu`-strongly monotone. In finite
dimension it is maximal monotone.

## Exact resolvent and Yosida operator

Set

\[
\lambda=\frac1{1000},
\qquad
J_{\lambda B}=(I+\lambda B)^{-1},
\qquad
Y_{\lambda B}=\frac1\lambda(I-J_{\lambda B}).
\]

The map `I+lambda B` is continuous and `(1+lambda*mu)`-strongly monotone.
Coercivity gives existence of a preimage for every point, while strict
monotonicity gives uniqueness. Hence the resolvent is a globally defined
single-valued map on every finite matrix space. Since `B(0)=0`,

\[
J_{\lambda B}(0)=0,
\qquad
Y_{\lambda B}(0)=0.
\]

For `x_i=u_i+lambda b_i`, where `b_i=B(u_i)`, monotonicity gives

\[
\langle u_1-u_2,x_1-x_2\rangle_F
\ge\lVert u_1-u_2\rVert_F^2.
\]

Thus the resolvent is firmly nonexpansive. Strong monotonicity additionally
gives

\[
\lVert J_{\lambda B}(x_1)-J_{\lambda B}(x_2)\rVert_F
\le\frac1{1+\lambda\mu}\lVert x_1-x_2\rVert_F.
\]

Writing `y_i=Y_{lambda B}(x_i)=b_i`, one obtains

\[
\begin{aligned}
\langle y_1-y_2,x_1-x_2\rangle_F
&\ge\lambda\lVert y_1-y_2\rVert_F^2,\\
\lVert y_1-y_2\rVert_F
&\le\frac1\lambda\lVert x_1-x_2\rVert_F,\\
\langle y_1-y_2,x_1-x_2\rangle_F
&\ge\frac{\mu}{1+\lambda\mu}
  \lVert x_1-x_2\rVert_F^2.
\end{aligned}
\]

These are, respectively, `lambda`-cocoercivity, the Yosida Lipschitz bound,
and inherited strong monotonicity. At the selected parameters,

\[
m_Y=\frac{\mu}{1+\lambda\mu}=500,
\qquad
M_Y=\frac1\lambda=1000.
\]

More precisely, for `Delta s=Delta input`, `Delta j=Delta J`, and
`Delta y=Delta Y`, so that `Delta s=Delta j+lambda*Delta y`, strong
monotonicity pulls back to the exact joint sector identity

\[
\left\langle \Delta y-\frac{\mu}{1+\lambda\mu}\Delta s,
\frac1\lambda\Delta s-\Delta y\right\rangle_F
=\frac{\langle \Delta y,\Delta j\rangle_F
-\mu\lVert\Delta j\rVert_F^2}
{\lambda(1+\lambda\mu)}\ge0.
\]

No differentiability or symmetry of `B` is used. The constants are independent
of `epsilon`; `epsilon=10^-7` is pinned for the reported control comparison.

## Centered incremental residual

For increments

\[
\Delta s=x_1-x_2,
\qquad
\Delta y=Y_{\lambda B}(x_1)-Y_{\lambda B}(x_2),
\]

the joint pulled-back sector identity is exactly

\[
\left\langle\Delta y-m_Y\Delta s,
M_Y\Delta s-\Delta y\right\rangle_F
=d^2\lVert\Delta s\rVert_F^2
-\lVert\Delta y-\gamma\Delta s\rVert_F^2\ge0,
\]

where `gamma=(M_Y+m_Y)/2` and `d=(M_Y-m_Y)/2`. Therefore

\[
\left\lVert
\Delta y-\frac{M_Y+m_Y}{2}\Delta s
\right\rVert_F
\le
\frac{M_Y-m_Y}{2}\lVert\Delta s\rVert_F.
\]

Consequently

\[
Y_{\lambda B}(s)=\gamma s+\mathcal E_Y(s),
\qquad
\gamma=750,
\qquad
\operatorname{Lip}(\mathcal E_Y)\le d=250,
\]

with `E_Y(0)=0`.

This is an incremental IQC and centered residual-norm statement. It is **not**
a Loewner-order bound `500 I <= DY <= 1000 I`: the Yosida map need not be a
gradient map, and its derivative need not be symmetric.

## Pinned EMA/Nesterov loop

Let `f` be differentiable, globally `L=10` smooth, bounded below, and satisfy
the global Polyak--Lojasiewicz inequality with constant `ell_PL=1`:

\[
\frac12\lVert\nabla f(W)\rVert_F^2
\ge f(W)-f^\star.
\]

P14 studies the exact-real implicit loop

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\beta m_t+(1-\beta)g_t,\\
s_{t+1}&=\beta m_{t+1}+(1-\beta)g_t,\\
W_{t+1}&=W_t-\eta Y_{\lambda B}(s_{t+1}),
\end{aligned}
\]

with

\[
\beta=\frac{19}{20},
\qquad
\eta=\frac1{32000}.
\]

The state and signal order therefore match the pinned EMA/Nesterov recurrence;
only the orthogonalizer has been replaced by the exact Yosida operator.

## Normalized lifted model

Use

\[
z_t=\frac{m_t}{L},
\qquad
u_t=\frac{g_t}{L},
\qquad
v_t=\frac{\mathcal E_Y(s_{t+1})}{dL},
\]

and let

\[
p_t=\beta^2z_t+(1-\beta^2)u_t.
\]

Then

\[
\begin{aligned}
z_{t+1}&=\beta z_t+(1-\beta)u_t,\\
W_{t+1}-W_t&=-\alpha(p_t+r v_t),\\
\alpha&=\eta\gamma L=\frac{15}{64},\\
r&=\frac d\gamma=\frac13,
\end{aligned}
\]

and the centered residual supply is

\[
\lVert p_t\rVert_F^2-\lVert v_t\rVert_F^2\ge0.
\]

This reduction is dimension independent and uniform over every `epsilon>0`:
every scalar matrix in the exact
certificate is tensored with the identity on the ambient matrix space.

## Smooth-PL value storage

Define `F_t=(f(W_t)-f^star)/L` and

\[
\mathcal V_t=
\begin{bmatrix}z_t\\u_t\end{bmatrix}^{\!T}
(P\otimes I)
\begin{bmatrix}z_t\\u_t\end{bmatrix}
+c_FF_t,
\]

where

\[
P=\frac1{10^6}
\begin{bmatrix}
674389&-73827\\
-73827&12368
\end{bmatrix},
\qquad
c_F=\frac{313243}{10^6}.
\]

The exact lifted coordinate is

\[
\chi=(z_t,u_t,v_t,u_{t+1}).
\]

As in P6, the proof uses both directed `(-1,1)` smooth interpolation
inequalities between the current and next iterates and the PL supply at the
next iterate. Its nonnegative exact multipliers are generated from

\[
\lambda_{21}=\frac{622414}{10^6},
\qquad
\lambda_{\rm residual}=\frac{27530}{10^6},
\]

together with the exact function-storage cancellation identities. In
particular,

\[
\lambda_{12}=\lambda_{21}+c_Fq_{14},
\qquad
\lambda_{\rm PL}=\frac{c_F(1-q_{14})}{2(\ell_{\rm PL}/L)}.
\]

The locked rate is

\[
\tau=\frac{499}{500},
\qquad
q_{14}=\tau^2=\frac{249001}{250000}<1.
\]

Direct rational expansion gives a symmetric `4 x 4` LMI. Exact leading
principal minors prove `P` positive definite and the LMI strictly negative
definite. The weighted objective-value coefficients equal exactly
`(c_F q_14,-c_F)`, so no untracked function-value term remains.

Therefore

\[
\boxed{
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t+D_{14},
\qquad D_{14}=0.
}
\]

The zero additive term is deliberate: P14 evaluates the mathematical
resolvent exactly. A nonzero solve residual has not been hidden in the
certificate.

It follows that `V_t`, the objective gap, gradient, and momentum decay
geometrically. The update sequence is absolutely summable, so `W_t` converges
to some trajectory-dependent global minimizer. PL does not imply a unique
minimizer, and none is claimed.

## Exact controls

### P13 explicit limit fails

At `epsilon=10^-7`, direct explicit use of the P13 operator has the exact
origin stiffness recorded by the P13 artifact. Its pinned EMA/Nesterov scalar
Jury margin at `eta=1/32000` is

\[
-\frac{1942248049655000871809068607}
       {66626908160000000000000}<0.
\]

Adding `mu I` before taking a resolvent only increases that explicit origin
gain, so the `lambda=0` limit remains unstable. This is a genuine negative
control, not an inference from failed numerical optimization.

### Selected Yosida point passes

For comparison, the selected Yosida upper gain is exactly `M_Y=1000`. Even
the worst scalar curvature `L=10` has positive pinned Jury margin

\[
2(1+\beta)
-(1-\beta)(1+2\beta)\eta L M_Y
=\frac{2467}{640}>0.
\]

The exact `4 x 4` PL certificate is stronger than this scalar check: it covers
the whole nonlinear smooth-PL class and changing matrix directions.

### A finite under-regularized point still fails

Taking `lambda=1/100000` while retaining `mu=1000` leaves the Yosida upper
gain at `100000`. More importantly, applying that finite resolvent to the
actual shifted P13 origin slope gives the exact curvature-`10` Jury margin

\[
-\frac{6764637554766138263886756183}
        {10717453168649723982394861280}<0.
\]

The corresponding upper-endpoint control is also negative:

\[
2(1+\beta)
-(1-\beta)(1+2\beta)\eta(10)(100000)
=-\frac{101}{160}<0.
\]

Thus success is not a consequence of merely renaming the P13 operator as a
resolvent; the amount of regularization matters.

### Sector-boundary witnesses

The sector bounds are uniform over the stated class of monotone base
operators. They cannot be uniformly tightened using only that premise:

- `A=0` makes the Yosida gain exactly
  `mu/(1+lambda*mu)=500`;
- positive linear gains tending to infinity approach the upper gain
  `1/lambda=1000`;
- in a real two-dimensional plane, let `S^T=-S` and `S^T S=I`. For
  `A=1000S`, exact inversion gives

  \[
  Y=600I+200S,
  \qquad
  \lVert Y-750I\rVert=250,
  \]

  which attains the centered residual radius.

These are boundary witnesses for the abstract monotone-operator class, not a
claim that the locked P13 map attains every boundary.

## Reproduction and audit boundary

The canonical result is generated with

```bash
uv run --locked python scripts/certify_yosida_stability.py \
  --output results/summaries/yosida_stability_certificate.json
```

and independently reconstructed with

```bash
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
```

The reconstruction imports only the Python standard library. It rebuilds the
Yosida constants, normalized dynamics, interpolation supplies, storage LMI,
exact determinants, and controls before reading the canonical artifact.

The committed audit packet identifies the remaining human checks. Until it
is signed, the human proof audit is pending. Automated exact replay does not
replace independent mathematical review.
