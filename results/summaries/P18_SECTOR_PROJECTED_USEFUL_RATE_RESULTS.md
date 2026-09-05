# P18 sector-projected resolvent: useful-rate exact theorem

## Verdict

P18 is a positive exact-real result. It preserves the spectral direction of
P17's Jordan-after-resolvent branch while replacing P17's very large global
pointwise gain interval by the exact full-matrix sector

```text
[125/1024, 509/512].
```

An exact rational smooth-PL certificate then proves global one-trajectory
convergence at `eta=1/83`. Its certified objective half-life is less than ten
times P14's, so it passes the predeclared useful-rate gate. The initially
requested `eta=1/50` is ruled out for the whole declared sector class by an
exact complex-skew boundary control; that control is not claimed to be an
instability example for the more structured P18 map.

## Locked operator

Retain the P14--P17 exact resolvent `J`, Yosida map `Y=1000(I-J)`, additive
normalizer `epsilon=1/10000000`, five Jordan stages, and the P17 C2 gate
`theta(||S||_F^2)` with ceiling `3/4`. Define

```text
X(S)       = E_h,epsilon(J(S))
alpha(S)   = min(1, <X(S),S>_F / ||X(S)||_F^2)
Z(S)       = alpha(S) X(S)
T(S)       = (1-theta) Y(S)/1024 + theta Z(S),
```

with `Z=0` when `X=0`. The Jordan coefficients are `6889/2000`, `-191/40`,
and `4063/2000`, the stage count is five, `lambda=1/1000`, `mu=1000`, and
`beta=19/20`. The upstream formula remains traced to KellerJordan/Muon
revision `f98f1cacc0263b04290753e32be8d498c1efc806`; upstream contains none of
the repair, resolvent, gate, or ray projection.

## Exact sector projection

The P16 singular-vector theorem and positivity of the locked Jordan stages
give `<X(S),S>_F>=0`. If the projection is inactive, its defining test gives

```text
||Z||_F^2 <= <Z,S>_F.
```

If it is active, the same relation holds with equality after multiplying by
the positive ray scale. Thus `Z` belongs to the origin-centered pointwise
disk sector `[0,1]` on every finite real rectangular matrix space. It keeps
the singular-vector direction of each nonzero `X`; this is a ray projection,
not a modewise clipping rule.

Combining this disk with P14's Yosida sector `[500,1000]`, divisor `1024`, and
`0<=theta<=3/4` gives

```text
m_T     = 125/1024
M_T     = 509/512
center  = 1143/2048
radius  = 893/2048
M_T/m_T = 1018/125 = 8.144.
```

This is a dimension-uniform **pointwise** sector. It is not an incremental
sector, derivative bound, or Lipschitz theorem, and the ray projection has an
active-set kink. The value--momentum proof uses only the pointwise supply.

## Useful-rate smooth-PL certificate

For every differentiable, globally `10`-smooth objective with finite infimum
that satisfies the global PL inequality with constant `1`, the pinned
EMA/Nesterov recurrence at `eta=1/83` satisfies

```text
V_(t+1) <= (999598040401/1000000000000) V_t.
```

The exact storage and multipliers are

```text
P = ((97/125, -151/500), (-151/500, 17/100))
function-value weight             = 1
reverse-interpolation multiplier  = 3459/500
pointwise-residual multiplier     = 9/250.
```

Exact rational Sylvester tests establish strict positivity of the storage
and strict negativity of the `4 x 4` LMI. Consequently objective gap,
gradient, and momentum converge geometrically, and summable updates imply
convergence to some trajectory-dependent global minimizer. The theorem does
not assert minimizer uniqueness or arbitrary-pair contraction.

The exact rate-gate comparison is

```text
(999598040401/10^12)^10 < 249001/250000.
```

The P18 half-life is about `1724.0734` iterations, or `9.9592x` P14's
`173.1135`. A second exact point at `eta=1/120` has rate
`624350169/625000000` and a shorter certified half-life of about `666.31`.
This nonmonotone certificate frontier is why both maximum step and rate are
reported.

## Attempted `eta=1/50` and frontier

The predeclared `eta=1/50` cannot support a theorem for every map in only the
locked sector. For the valid quadratic `f(W)=5||W||_F^2`, the admissible
boundary map

```text
T_0 = (1143/2048) I + (893/2048) Q,
Q^T=-Q,  Q^T Q=I,
```

has exact second Schur--Cohn margin

```text
-3768360579178620269 / 1759218604441600000000000 < 0.
```

The finite exact frontier records passing certificates at steps `1/75`,
`1/83`, `1/90`, `1/95`, `1/120`, and `1/150`. The selected `1/83` point is
the largest recorded step that also passes the ten-times-P14 rate gate. This
does not prove global optimality over unsearched storages or exploit extra
structure of the actual P18 spectral map.

## Fidelity and amplitude diagnostics

The canonical `diag(3,4)` evaluation has, before outward error allowances,

```text
best-scalar departure       about 0.0362830
upstream shaping retention  about 0.518570
output/upstream norm        about 1.44563
output/input norm           about 0.385218.
```

The authoritative certificate uses outward-rounded interval arithmetic for
the frozen P16 departure `>=1/1000` and retention `>=1/10` decisions. The
deterministic FP64 study additionally made fail-closed P16 solver calls and
checked their actual P15 graph residuals.

On the sampled post-exploratory annulus `3/4<=||S||_F<=25`, all `2176/2176`
informative points pass both unchanged fidelity gates. The minimum sampled
departure is about `0.00185931`, and the minimum sampled shaping retention is
about `0.130250`. Only `192/327` informative broad-grid cases pass, because
the fallback near the origin is intentionally radial. No global fidelity
claim is made.

The exact sector also supplies global raw-amplitude and selected-step guards:

```text
1/10 < ||T(S)||_F/||S||_F < 1
1/1000 < (1/83)||T(S)||_F/||S||_F < 1/80,
```

with exact sector endpoints `125/1024`, `509/512` and effective-step
endpoints `125/84992`, `509/42496`. These guards prevent the scale-invariant
fidelity metrics from being passed by a collapsed or exploded output. They
do not establish upstream update-magnitude or learning-rate parity.

## Negative controls

- Removing the projection restores P17's upper pointwise gain of about
  `565.44`; its `eta=1/50` effective upper gain exceeds `11` and fails the
  amplitude/useful-step gates.
- With `K=1/100`, canonical shaping retention is about `0.0388`, and none of
  the `2176` informative annulus cases passes both frozen fidelity gates.
- The exact complex-skew boundary control rejects the requested generic
  sector theorem at `eta=1/50`; it is deliberately not attributed to the
  structured P18 map.

## Evidence classification

| Statement | Evidence | Scope |
| --- | --- | --- |
| Ray-sector projection | Exact algebraic proof | Every finite rectangular shape |
| Global pointwise sector and amplitude bounds | Exact analytic certificate | Exact-real P18 map |
| Smooth-PL convergence and rate gate | Exact rational `4 x 4` LMI | Pinned exact-resolvent EMA/Nesterov recurrence |
| `eta=1/50` obstruction | Exact Schur--Cohn boundary witness | Full declared pointwise-sector class |
| Canonical fidelity gates | Outward-rounded interval certificate | `diag(3,4)` only |
| Annulus/broad-grid/rank behavior | Deterministic residual-checked FP64 study | Sampled, not extremal |

## Reproduction

```bash
uv run --locked python scripts/certify_sector_projected_useful_rate.py \
  --output results/summaries/sector_projected_useful_rate_certificate.json

uv run --locked python scripts/reconstruct_sector_projected_useful_rate.py \
  --canonical results/summaries/sector_projected_useful_rate_certificate.json \
  --require-canonical

uv run --locked python experiments/resolvent/run_p18_sector_projected_study.py \
  --output results/summaries/p18_sector_projected_study.json
```

## Remaining limitations

P18 assumes the exact P14 resolvent and exact real arithmetic. Checking the
P16 graph residual during diagnostics does not itself propagate approximate-
solve error through the nonlinear shape branch, ray projection, and gate.
P18 does not certify FP64/BF16 evaluation, upstream step or amplitude parity,
aspect scaling, weight decay, stochastic gradients, represented model state,
dense accelerator cost, or neural-network training. Independent human proof
review remains pending.
