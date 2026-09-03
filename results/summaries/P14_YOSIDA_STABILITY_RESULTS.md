# P14 Yosida-stability results

## Verdict

P14 proves that an exact resolvent removes P13's explicit differential-
stiffness obstruction without weakening the full-matrix monotonicity premise.
At the locked parameters, the resulting Yosida operator admits an exact
dimension-independent nonlinear PL convergence certificate at the full
`eta=1/32000` step.

The result is exact-real and implicit. It establishes existence and stability
of a mathematical update, not the cost or accuracy of a finite-iteration
resolvent solver. The additive term is exactly `D14=0`; solve errors are
deferred to P15.

| Item | Exact result |
| --- | --- |
| Additive epsilon | every `epsilon>0`; controls pin `1/10000000` |
| Resolvent parameter | `lambda=1/1000` |
| Strong shift | `mu=1000` |
| Yosida strong monotonicity | `m_Y=500` |
| Yosida Lipschitz bound | `M_Y=1000` |
| Centered representation | `gamma=750`, `Lip(E_Y)<=250` |
| Momentum | `beta=19/20` |
| Step size | `eta=1/32000` |
| Objective class | differentiable, globally `10`-smooth, finite infimum, PL constant `1` |
| Storage rate | `tau=499/500` |
| Function/storage rate | `q14=249001/250000<1` |
| Additive term | `D14=0` |

## Full-matrix operator theorem

For every `epsilon>0`, let

\[
A_\epsilon=E_{h,\epsilon}+G_\epsilon
\]

be P13's continuous, globally monotone, zero-preserving exact-real map on an
arbitrary finite Frobenius matrix space. Here `E_(h,epsilon)` normalizes by
`M/(||M||_F+epsilon)` and uses exactly five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`; `G_epsilon` is P13's exact radial
correction. Define

\[
B=A_\epsilon+1000I,
\qquad
J=(I+B/1000)^{-1},
\qquad
Y=1000(I-J).
\]

The formula is traced to pinned KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; upstream does not contain this
radial repair, shunt, or exact resolvent.

The shifted operator is continuous, full-domain, maximal, and
`1000`-strongly monotone. Consequently the resolvent exists uniquely at every
input and fixes zero. Exact Hilbert-space inequalities give

\[
\begin{aligned}
\langle\Delta Y,\Delta s\rangle_F
&\ge\frac1{1000}\lVert\Delta Y\rVert_F^2,\\
\lVert\Delta Y\rVert_F
&\le1000\lVert\Delta s\rVert_F,\\
\langle\Delta Y,\Delta s\rangle_F
&\ge500\lVert\Delta s\rVert_F^2.
\end{aligned}
\]

These results cover the complete matrix space and require no Jacobian
symmetry. In fact, for `x=Delta input`, `u=Delta J`, and `v=Delta Y`, the
authoritative pulled-back sector is

\[
\left\langle v-500x,1000x-v\right\rangle_F
=\frac{\langle v,u\rangle_F-1000\lVert u\rVert_F^2}
{(1/1000)(2)}\ge0.
\]

The joint sector identity rearranges exactly to

\[
\left\langle\Delta Y-500\Delta s,
1000\Delta s-\Delta Y\right\rangle_F
=250^2\lVert\Delta s\rVert_F^2
-\lVert\Delta Y-750\Delta s\rVert_F^2\ge0,
\]

and hence

\[
\lVert\Delta Y-750\Delta s\rVert_F
\le250\lVert\Delta s\rVert_F.
\]

This is a joint incremental sector and centered residual-norm bound, not a
Loewner sector statement for a nonsymmetric derivative. Its constants are
independent of `epsilon`.

## Exact nonlinear PL certificate

The analyzed recurrence is

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{32000}Y(s_{t+1}).
\end{aligned}
\]

For every differentiable, globally `10`-smooth objective with finite infimum
satisfying the global PL inequality with constant `1`, use normalized
momentum and gradient storage

\[
\mathcal V_t=
\begin{bmatrix}m_t/10\\\nabla f(W_t)/10\end{bmatrix}^{\!T}
(P\otimes I)
\begin{bmatrix}m_t/10\\\nabla f(W_t)/10\end{bmatrix}
+\frac{313243}{10^6}\frac{f(W_t)-f^\star}{10},
\]

where

\[
P=\frac1{10^6}
\begin{bmatrix}
674389&-73827\\
-73827&12368
\end{bmatrix}.
\]

The normalized Yosida decomposition has center `750`, residual bound `250`,
dimensionless step `15/64`, and residual ratio `1/3`. The exact
function/interpolation certificate uses reverse directed-interpolation weight
`622414/10^6` and residual-IQC weight `27530/10^6`. The remaining weights are
fixed by exact function-value cancellation at rate

\[
q_{14}=\left(\frac{499}{500}\right)^2
=\frac{249001}{250000}.
\]

Exact Sylvester checks prove the storage positive definite and the complete
`4 x 4` rational LMI strictly negative definite. The result is

\[
\boxed{
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t,
\qquad D_{14}=0.
}
\]

Thus the objective gap, gradient, and momentum converge geometrically. The
updates are absolutely summable, so each trajectory converges to some global
minimizer. The minimizer need not be unique, and arbitrary-pair incremental
contraction is not claimed.

## Exact controls

P13 records a negative pinned EMA/Nesterov Jury margin for explicit use of its
stiff repaired map at `epsilon=1e-7` and `eta=1/32000`:

\[
-\frac{1942248049655000871809068607}
       {66626908160000000000000}<0.
\]

The explicit `lambda=0` shifted operator is at least as stiff at the origin,
so it remains a failing control.

A finite but under-regularized choice also fails. At `lambda=1/100000`, the
actual shifted P13 origin mode at scalar curvature `10` has exact Jury margin

\[
-\frac{6764637554766138263886756183}
        {10717453168649723982394861280}<0.
\]

The Yosida upper gain is `100000`, giving the simpler endpoint margin

\[
-\frac{101}{160}<0.
\]

At the selected Yosida point, the upper gain is only `1000`. At the worst
scalar curvature `10`, its pinned Jury margin is exactly

\[
2(1+\beta)
-(1-\beta)(1+2\beta)\eta(10)(1000)
=\frac{2467}{640}>0.
\]

That scalar pass is only a boundary check. The exact LMI, not the scalar mode,
carries the full nonlinear PL theorem.

The uniform sector bounds also have exact abstract boundary witnesses. The
monotone base `A=0` attains gain `500`; positive scalar gains approach `1000`;
and on a two-dimensional plane with `S^T=-S`, `S^T S=I`, the monotone base
`A=1000S` gives

\[
Y=600I+200S,
\qquad
\lVert Y-750I\rVert=250.
\]

These witnesses show sharpness over the admissible monotone-operator class;
they are not asserted to be realized by the specific P13 map.

## Scope

The result applies to every finite real matrix shape, but only to the proposed
exact-real implicit operator. It does not establish:

- a finite-step or finite-tolerance resolvent algorithm;
- stability under resolvent solve error;
- BF16, FP32, or other rounded evaluation of the implicit map;
- literal upstream Muon parity;
- stochastic-gradient, weight-decay, aspect-scaling, or neural-network
  convergence.

The P13 human audit remains pending, and P14 has its own unsigned audit packet.
P15 is the appropriate place to introduce solve-error ports and a nonzero
ultimate-bound term.

## Reproduce

Generate the canonical exact certificate:

```bash
uv run --locked python scripts/certify_yosida_stability.py \
  --output results/summaries/yosida_stability_certificate.json
```

Independently rebuild it using only the Python standard library:

```bash
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
```

The canonical artifact records the exact operator constants, lifted matrices,
storage and LMI minors, objective-value cancellation, positive and negative
controls, source and prior-artifact hashes, software, hardware, Git SHA, and a
null seed. Numerical sampling is not used to certify the theorem.
