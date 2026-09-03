# P11 certified implementation-margin result

## Result

P11 adds two explicit error interfaces to the frozen P10 proof-reference
optimizer at shape `4096 x 11008`:

\[
\lVert\zeta_t\rVert_F\le a_g\sqrt{V_t}+b_g,
\qquad
\lVert\nu_t\rVert_F\le a_R\sqrt{V_t}+b_R.
\]

Here `zeta_t` is error in the real gradient/model-weight signal before P10's
one final FP32 cast. A model-weight error is covered only through the gradient
error it induces; global `10`-smoothness gives
`||Delta g||_F<=10||Delta W||_F`. The second port is the real difference
between a finite contiguous FP32 deployed output and the P9 proof-reference
output. It is not an extra executed FP32 addition.

For every error realization satisfying those bounds, exact rational
composition with P7 proves

\[
V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).
\]

The target remains the repaired exact max-floor operator with `c=1`, no
additive epsilon, coefficients `(6889/2000,-191/40,4063/2000)`, exactly five
Jordan/Newton--Schulz stages, and
`rho=210177835339081/260261360000`. The loop retains `beta=19/20`,
`eta=1/32000`, FP32 EMA/Nesterov state, the P9 balanced-FP32/two-term-BF16
reference kernel, and the three-word compensated FP32 master. The objective
class remains every differentiable globally `10`-smooth function satisfying
the global PL inequality with constant one, including nonconvex functions and
nonunique minimizers.

## Exact P10 corner

At zero additional error, P11 reproduces every P10 affine envelope and the
three headline fractions exactly:

\[
q_{11}(0,0)=\frac{549700907325}{549755813888},\qquad
D_{11}(0,0)=\frac{2162331}{1099511627776},
\]

\[
\limsup_t(f(W_t)-f_\star)
\le\frac{399957341889}{549755813888}.
\]

This is exact equality of rational certificate fields, not agreement after
decimal rounding.

## Jointly nonzero certified profile

The following four-way error budget passes at the unchanged full step:

\[
a_g=b_g=\frac1{4096},\qquad
a_R=\frac1{128},\qquad b_R=\frac18.
\]

Its exact result is:

| Quantity | Exact value | Decimal |
| --- | ---: | ---: |
| `q11` | `274850515349 / 274877906944` | `0.999900349957898` |
| `D11` | `1254603 / 549755813888` | `2.28210956266e-6` |
| `1-q11-D11` | `53528587 / 549755813888` | `9.73679325398e-5` |
| objective-gap limsup | `930325132219 / 1099511627776` | `0.846125778679` |
| signal bound on `V<=1` | `13872266672489 / 549755813888` | `25.2335060804` |
| P9 reference-output bound | `8965087254758081 / 274877906944` | `32614.7974365` |
| deployed-output bound | `8965123761980097 / 274877906944` | `32614.9302490` |
| rounded-step entry bound | `1120640590271 / 1099511627776` | `1.01921667944` |

Thus `q11<1`, `D11<=1-q11`, the objective neighborhood is finite and below
one, the deployed output remains below `2^15`, and the rounded step remains
below `2`. All FP32 EMA intermediates, the P9 signal guard, and P10's
middle/low master-word guards close. The high-word `2^30` guard is still a
conditional premise rechecked after each update; PL storage does not make it
unconditional.

This profile proves nonzero joint tolerance. It is not a measurement of errors
from a production model or kernel.

## Exact one-axis maxima

The locked acceptance grid is `{k/2^40:k>=0}`. With all other new
coefficients zero, the largest accepted grid values are:

| Coordinate | Largest certified grid value | Decimal | Adjacent rejected value |
| --- | ---: | ---: | ---: |
| `a_g` | `10815225547 / 2^40` | `0.00983639033348` | `10815225548 / 2^40` |
| `a_R` | `513245498810 / 2^40` | `0.466794061876` | `513245498811 / 2^40` |
| `b_g` | `10879487718 / 2^40` | `0.00989483643752` | `10879487719 / 2^40` |
| `b_R` | `13351103462525 / 2^40` | `12.1427578620` | `13351103462526 / 2^40` |

Each accepted endpoint has exact invariance slack `1-q11-D11=0`. At the
adjacent point the slack is exactly `-1/2^40`, and unit-storage forward
invariance is the only failed check. Because all certificate expressions and
guards are monotone in the nonnegative budget, these are exact maxima on the
declared grid. They are not unrestricted-real maxima.

## Joint Pareto slices

The two tables below give exact coordinatewise-maximal slices of the
four-dimensional acceptance set. Values are integer ticks over `2^40`.
Increasing either coordinate in a row by one tick while fixing the other is a
committed negative control and fails the unit-storage invariance check.

### Slope budgets (`b_g=b_R=0`)

| Row | `2^40 a_g` | `2^40 a_R` |
| ---: | ---: | ---: |
| 0 | 0 | 513245498810 |
| 1 | 1351903193 | 508734339015 |
| 2 | 2703806386 | 495772587382 |
| 3 | 4055709580 | 473693173922 |
| 4 | 5407612773 | 441184385742 |
| 5 | 6759515966 | 395798829226 |
| 6 | 8111419160 | 332578695399 |
| 7 | 9463322353 | 238696271394 |
| 8 | 10815225547 | 1295 |

Every accepted slope row has

\[
q_{11}=\frac{1099509465445}{1099511627776},\qquad
D_{11}=\frac{2162331}{1099511627776},\qquad
1-q_{11}-D_{11}=0.
\]

### Intercept budgets (`a_g=a_R=0`)

| Row | `2^40 b_g` | `2^40 b_R` |
| ---: | ---: | ---: |
| 0 | 2622 | 13351103462525 |
| 1 | 1359935964 | 13231702254350 |
| 2 | 2719871929 | 12867815091068 |
| 3 | 4079807894 | 12240945421027 |
| 4 | 5439743859 | 11314491418802 |
| 5 | 6799679823 | 10020390941550 |
| 6 | 8159615788 | 8222316916289 |
| 7 | 9519551753 | 5577370189287 |
| 8 | 10879487718 | 8583 |

Every accepted intercept row has

\[
q_{11}=\frac{549700907325}{549755813888},\qquad
D_{11}=\frac{54906563}{549755813888},\qquad
1-q_{11}-D_{11}=0.
\]

The small nonzero coordinates at two endpoints are genuine outward-rounding
plateaus. The tables are two exact two-dimensional slices, not the complete
four-dimensional Pareto surface.

## Negative controls and logical strength

All four axis controls and both coordinatewise controls beside every frontier
row are exactly one `2^-40` tick outside the accepted point. Each fails the
same sufficient condition, `D11<=1-q11`, while the other arithmetic guards
remain closed. These controls verify the reported certificate boundary. They
do **not** prove that the optimizer is dynamically unstable immediately
outside that boundary.

## Evidence, provenance, and limitations

The canonical exact artifact is
`results/summaries/implementation_margin_certificate.json`. Its standalone
standard-library reconstruction rebuilds the external sensitivities,
augmented envelopes, storage inequality, four axis maxima, two nine-row
frontiers, adjacent negative controls, and guard comparisons without importing
the theorem module.

The frozen P10 commits have different provenance roles:

| Role | Commit |
| --- | --- |
| theorem/source | `2d62b566e5a34895470d4eeb4b76789add826cbe` |
| exact artifact | `6e7ea000692a269cbd3766146eb7423733d18694` |
| diagnostic/final P10 checkpoint | `3246972d44fc6e04c15cf4e205615acbb879df36` |

P10 and P11 human proof review remain pending. Automated exact replay and unit
cross-checks are not human sign-off.

P11 does not cover literal upstream Muon, current-plus-epsilon normalization,
an unrepaired operator, native GPU/tensor-core arithmetic, a neural-network
forward/backward implementation, actual production error magnitudes, aspect
scaling, weight decay, stochastic rounding, FTZ/DAZ, or training convergence.
It controls the P7 storage and function-value/gradient/momentum consequences,
not full-parameter ISS or convergence to a unique minimizer.

## Reproduce

```bash
uv run --locked python scripts/certify_implementation_margin.py \
  --output results/summaries/implementation_margin_certificate.json
uv run --locked python scripts/reconstruct_implementation_margin.py \
  --require-canonical
```

Both commands are exact theorem evidence. No sampled experiment carries the
global margin claim.

