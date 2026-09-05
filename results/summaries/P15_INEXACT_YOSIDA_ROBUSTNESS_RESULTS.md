# P15 inexact-Yosida robustness results

## Verdict

P15 converts P14's exact implicit update into a robustness theorem for an
exact-real inexact-resolvent oracle with a directly checkable graph residual.
At relative residual tolerance `kappa=1/250`, it retains the full P14 step and
rate. A nonzero absolute residual floor produces an explicit, finite
objective neighborhood.

This is a theorem about any oracle output satisfying the residual rule. It is
not yet an algorithm, an iteration-complexity result, a finite-precision
residual check, or a BF16/upstream implementation.

| Item | Exact result |
| --- | --- |
| Matrix domain | every fixed finite real rectangular shape |
| Additive epsilon | every `epsilon>0`; provenance controls pin `1/10000000` |
| Resolvent parameters | `lambda=1/1000`, `mu=1000` |
| Residual | `r=s-u_hat-lambda*B(u_hat)` |
| Approximate output | `Y_hat(s)=(s-u_hat)/lambda` |
| Residual-to-output gain | `500` |
| Stopping rule | `||r||_F <= ||s||_F/250 + rbar` at every oracle call |
| Effective centered radius | `252` |
| Momentum and step | `beta=19/20`, `eta=1/32000` |
| Objective class | differentiable, globally `10`-smooth, finite infimum, global PL constant `1` |
| Storage rate | `q15=249001/250000<1` |
| Absolute-residual coefficient | `C15=5/2` |
| Ultimate storage | `limsup V_t <= (625000/999)*rbar^2` |
| Ultimate objective gap | `limsup(f-f*) <= (6250000000000/312929757)*rbar^2` |

## Exact residual reduction

P15 inherits P13's globally monotone exact-real operator
`A_epsilon=E_(h,epsilon)+G_epsilon` and P14's shift

\[
B=A_\epsilon+1000I,
\qquad J=(I+B/1000)^{-1},
\qquad Y=1000(I-J).
\]

The additive-epsilon Jordan term uses normalization
`M/(||M||_F+epsilon)`, exactly five polynomial stages, and exact
coefficients `6889/2000`, `-191/40`, and `4063/2000`. The practical formula
is traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; upstream contains neither the
radial passivator nor this resolvent design.

Given `s`, the oracle returns `u_hat`, and the exact-real graph residual is

\[
r=s-\widehat u-\frac1{1000}B(\widehat u).
\]

The approximate output is deliberately defined as

\[
\widehat Y(s)=1000(s-\widehat u),
\]

not as `B(u_hat)`. Since `u_hat=J(s-r)`, the candidate and output errors
satisfy

\[
\|\widehat u-J(s)\|_F\le\frac12\|r\|_F,
\qquad
\widehat Y(s)-Y(s)=1000\bigl(J(s)-J(s-r)\bigr),
\qquad
\|\widehat Y(s)-Y(s)\|_F\le500\|r\|_F.
\]

The alternative graph output `B(u_hat)=Y(s-r)` has only the uniform error
bound `1000*||r||_F` and is not covered by this storage certificate.
The gain follows from P14's `Lip(J)<=1/2` and is attained by the abstract
boundary operator `A=0`.

Under

\[
\|r\|_F\le\frac1{250}\|s\|_F+\bar r,
\]

the output error can be split pointwise into a relative term bounded by
`2*||s||_F` and an absolute term bounded by `500*rbar`. Combining the
relative term with P14's centered radius `250` gives

\[
\widehat Y(s)=750s+e_{\rm rel}+e_{\rm abs},
\quad
\|e_{\rm rel}\|_F\le252\|s\|_F,
\quad
\|e_{\rm abs}\|_F\le500\bar r.
\]

This is a pointwise trajectory supply. P15 does not infer that an arbitrary
oracle selection is an incrementally Lipschitz or monotone operator.

## Exact smooth-PL certificate

The pinned recurrence is

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}g_t,\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}g_t,\\
W_{t+1}&=W_t-\frac1{32000}\widehat Y(s_{t+1}).
\end{aligned}
\]

The rule is evaluated on each call pair `(s_(t+1),r_(t+1))`; the generic
`(s,r)` notation above avoids an index shift.

For every differentiable globally `10`-smooth objective with finite infimum
satisfying the global PL inequality with constant `1`, P15 retains P14's
storage

\[
P=10^{-6}\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix},
\qquad c_F=\frac{313243}{10^6},
\]

and its exact directed-interpolation, PL, and residual multipliers. The
relative-error `4 x 4` LMI is strictly negative at centered radius `252`.
Adding the absolute output-error coordinate gives a strictly negative exact
`5 x 5` LMI with physical gain `1/100000`:

\[
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t
+\frac1{100000}\|e_{{\rm abs},t}\|_F^2.
\]

Using `||e_abs,t||_F<=500*rbar` yields

\[
\boxed{
\mathcal V_{t+1}
\le\frac{249001}{250000}\mathcal V_t+\frac52\bar r^2.}
\]

Therefore

\[
\limsup_t\mathcal V_t\le\frac{625000}{999}\bar r^2,
\qquad
\limsup_t(f(W_t)-f^\star)
\le\frac{6250000000000}{312929757}\bar r^2
\approx19972.533324787\bar r^2.
\]

When `rbar=0`, the inexact relative-residual oracle retains the P14 rate:
objective gap, gradient, and momentum converge geometrically, and the
iterates converge to some trajectory-dependent global minimizer. With a
persistent nonzero `rbar`, only the certified ultimate neighborhoods are
claimed; parameter convergence is not.

## Controls

- `kappa=0`, `rbar=0` forces an exact solve and reproduces the P14 fractions
  exactly.
- `kappa=3/500` gives centered radius `253` and fails the unchanged frozen-
  storage `4 x 4` definiteness test. This rejects that sufficient
  certificate; it is not an instability or impossibility theorem.
- In the abstract admissible case `A=0`, `B=1000*I`, choosing `u_hat=s`
  gives `r=-s` and `Y_hat=0`. Thus a `kappa=1`, `rbar=0` rule can stall away
  from stationarity. This is a uniform-class boundary control, not a claim
  about the specific P13 operator.
- At the same abstract boundary, `kappa=2`, `r=-2s`, and `u_hat=3s/2`
  produce `Y_hat=-500s`. For scalar curvature `1`, the pinned EMA/Nesterov
  characteristic has `p(1)=-1/1280<0`, a genuine loose-tolerance instability
  control.
- The same `A=0` boundary attains the factor-`500` residual-to-output gain.

No sampled pass is used as a global certificate.

## Scope

P15 certifies an exact-real stopping criterion, not a way to achieve it. It
does not provide:

- a concrete iterative solver or finite iteration bound;
- rounded evaluation of `B(u_hat)` or the residual;
- BF16/FP32 or accelerator behavior;
- literal upstream-Muon parity;
- weight decay, aspect scaling, stochastic-gradient, or neural-network
  guarantees.

The P13 and P14 human proof audits remain pending, and P15 has its own unsigned
audit packet.

## Reproduce

Generate the exact certificate:

```bash
uv run --locked python scripts/certify_inexact_yosida_robustness.py \
  --output results/summaries/inexact_yosida_robustness_certificate.json
```

Independently reconstruct it using only the Python standard library:

```bash
uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py \
  --require-canonical
```

The canonical JSON, not this prose summary, is authoritative for every exact
fraction and source hash. Automated generation and reconstruction do not
replace the pending independent human proof audit.
