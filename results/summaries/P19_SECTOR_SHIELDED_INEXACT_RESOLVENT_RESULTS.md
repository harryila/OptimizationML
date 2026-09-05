# P19 sector-shielded inexact-resolvent results

## Verdict

P19 is a positive compositional safety result. It wraps any finite
approximate candidate in the exact P18 pointwise sector before exposing the
update to the optimizer. The shield is the identity on the exact P18 map, so
it preserves P18's intended operator, fidelity diagnostics, two exact
smooth-PL certificates, and certified rates. Solver accuracy is no longer a
premise for stability.

The binary64 implementation is a correctness-first reference: every
successful stored output passes an exact dyadic sector check. It fails closed
when the signal is nonfinite or when no certified binary64 interior output is
representable. This is not yet a scalable BF16/GPU implementation theorem.

## Locked construction

P19 retains P18's exact-real additive normalizer
`U/(||U||_F+epsilon)` with `epsilon=1/10000000`, five Jordan stages with
coefficients `6889/2000`, `-191/40`, and `4063/2000`, P13 radial repair,
`mu=1000`, `lambda=1/1000`, P17 C2 gate with ceiling `3/4`, P18 projection
gain `K=1`, passive divisor `1024`, and `beta=19/20`.

For any finite candidate `C(S)`, define

```text
m      = 125/1024
M      = 509/512
center = 1143/2048
radius = 893/2048

D_S = {U : ||U-center*S||_F <= radius*||S||_F}
T19(S) = projection of C(S) onto D_S.
```

For nonzero `S`, the exact projection is

```text
center*S
  + min(1, radius*||S||_F/||C-center*S||_F)*(C-center*S),
```

with multiplier one at zero displacement. For `S=0`, `D_S={0}` and the
output is exactly zero.

The identity

```text
<U-m*S, M*S-U>_F
  = radius^2*||S||_F^2 - ||U-center*S||_F^2
```

proves that every shielded finite candidate lies in the exact P18
origin-centred pointwise sector on every finite real rectangular matrix
space. This is not an incremental sector.

## Exact P18 identity and nonexpansiveness

The globally certified P18 output already lies in `D_S`; therefore the
shield fixes it exactly. For each fixed `S`, Euclidean projection onto
`D_S` is nonexpansive in its candidate. In particular,

```text
||T19(S)-T18(S)||_F <= ||C(S)-T18(S)||_F.
```

The set changes with `S`, so P19 does not claim joint nonexpansiveness in
`(S,C)` or arbitrary-pair contraction. The P15 graph residual provides
root/Yosida error ingredients, but it is not by itself a final error bound
after P18's nonlinear ray projection. The shield preserves any independently
proved candidate-error bound; it does not manufacture the missing bound.

## Replayed smooth-PL operating points

The P18 value--momentum theorem uses only the pointwise sector supply. Thus
the same theorem holds for every time-varying sequence of finite candidates
after shielding. For every differentiable globally `10`-smooth objective
with finite infimum satisfying the global PL inequality with constant `1`,
the pinned EMA/Nesterov loop has the following exact certificates:

| Role | `eta` | Exact rate `q` | Certified Lyapunov-rate half-life |
| --- | ---: | ---: | ---: |
| maximum step | `1/83` | `999598040401/1000000000000` | about `1724.0734` |
| faster certified rate | `1/120` | `624350169/625000000` | about `666.314` |

Exact rational Sylvester checks replay strict storage positivity and strict
negativity of both `4 x 4` LMIs. The half-life describes the certified
geometric Lyapunov bound, not a promise that every observed objective value
halves after exactly that many steps.

At either operating point, the objective gap, gradient, and momentum
converge geometrically, and the iterates converge to some
trajectory-dependent global minimizer. No unique-minimizer or arbitrary-pair
contraction claim is made.

## Binary64 reference shield

The FP64 reference uses scaled/balanced norm evaluation, a disk radius moved
32 ULPs inward for outside projections, and a mandatory exact rational check
of the stored binary64 result. A safely interior candidate is returned bit
for bit. A nonfinite candidate or a rounded projection that misses the disk
uses the exactly checked interior fallback `S/2`. A nonfinite signal is
rejected without an update.

Not every nonzero binary64 signal admits a representable output in the
strict positive-gain sector. For the least positive subnormal, halving
underflows to zero and the only adjacent representable choices can both lie
outside the disk. The implementation detects this case and raises instead of
returning an uncertified update. Thus the guarantee is precise: **every
successful return** lies in the original P18 sector exactly as stored.

The P18 FP64 ray projection also replaces the old negative-inner-product
exception with positive-part clipping. Tiny rounded negative values now
produce a zero shape branch; the final P19 shield remains the safety
postcondition.

## Fidelity and corruption diagnostics

The deterministic study records:

| Diagnostic | Result |
| --- | ---: |
| sampled annulus calls | `2688` |
| shield inactive and bitwise identity | `2688/2688` |
| informative annulus fidelity passes | `2176/2176` |
| canonical best-scalar departure | about `0.0362830` |
| canonical upstream shaping retained | about `0.518570` |
| computed P15 graph-residual checks | `2690/2690` |
| worst residual / permitted threshold | about `1.356e-11` |

The shield therefore leaves the normal guarded P16/P18 reference-solver path
unchanged on every declared case. These fidelity and annulus results are
sampled diagnostics, not global extrema.

Four deliberately corrupted finite candidates start outside the disk and
pass the exact stored-value postcheck after shielding. Exact rational radial
and tangential controls independently reconstruct the same boundary
behavior. Nonfinite candidate controls fall back to certified interior
updates; zero signals return zero; and nonfinite signals are rejected.

## Evidence classification

| Statement | Evidence | Scope |
| --- | --- | --- |
| Disk/sector equivalence and exact projection | Algebraic proof and rational controls | Every fixed finite real rectangular matrix space |
| Identity on exact P18 | P18 global sector plus metric-projection fixed-point property | Exact-real P18 map |
| Arbitrary finite-candidate safety | Exact projection theorem | Pointwise, including time-varying candidates |
| Both smooth-PL rates | Independent exact rational `4 x 4` LMI reconstruction | Pinned shielded EMA/Nesterov recurrence |
| Successful binary64-return containment | Exact-as-stored dyadic postcheck | Locked NumPy FP64 reference implementation |
| Normal inactivity and fidelity preservation | Deterministic residual-checked study | Declared canonical and sampled annulus cases |

## Reproduction

```bash
uv run --locked python scripts/certify_sector_shielded_inexact_resolvent.py \
  --output results/summaries/sector_shielded_inexact_resolvent_certificate.json

uv run --locked python scripts/reconstruct_sector_shielded_inexact_resolvent.py \
  --canonical results/summaries/sector_shielded_inexact_resolvent_certificate.json \
  --require-canonical

uv run --locked python experiments/resolvent/run_p19_sector_shielded_study.py \
  --output results/summaries/p19_sector_shielded_study.json
```

The standard-library reconstruction imports neither the theorem module nor a
third-party numerical package. Automated replay and tests do not replace the
pending independent human proof audit.

## Remaining limitations

P19 does not give an incremental sector, joint input/candidate
nonexpansiveness, a graph-residual-to-final-candidate fidelity theorem, a
uniform representability result for all binary64 signals, a scalable shield,
BF16 or GPU parity, literal upstream-Muon stability, aspect scaling, weight
decay, represented model-state semantics, stochastic-gradient convergence,
throughput evidence, or neural-network training evidence. The original
unrepaired upstream Muon remains outside the certified theorem.
