# Experiment results

The spectral and quadratic results were generated in float64 on an arm64 CPU;
Section 2 is the separately scoped BF16 deployment check. JSON files are the
source-of-truth manifests; CSV files are flattened views. All grid extrema are
lower-bound witnesses, not continuous-domain upper certificates.

## 1. Canonical exact witness

For the five-step Jordan map on diagonal matrices,

- `A = diag(3, 4)` and `B = diag(5/2, 3)`;
- exact-current-normalized pair gap: `-0.0702543371557703288`;
- exact pair repair threshold: `rho = 0.0562034697246162631`;
- current-plus-`1e-7` pair gap: `-0.07025429353210752`;
- fixed-scale-five pair gap: `+0.3750465803803160`;
- fixed-scale local off-diagonal divided differences before the common
  chain-rule factor `1/5`: `+1.981638806494635...` and
  `+1.315771498952257...`.

The current-normalized local determinant, both finite-pair signs, and all four
fixed-scale `2 x 2` local Jacobian modes at `A` are certified exactly. The
fixed-scale Jacobian is positive definite at that point (and therefore on some
sufficiently small neighborhood by continuity), but no global fixed-scale
monotonicity is claimed. The readable decimals are high-precision evaluations.
See `canonical_witness.json`.

## 2. Backend-specific BF16 deployment witness

The same canonical pair was executed on a macOS 15.5 arm64 CPU with PyTorch
2.13.0 using the literal operation order in pinned KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`: cast once to BF16, orient once,
normalize with retained dimensions and Python-float `eps=1e-7`, then apply five
Jordan stages with `B = b*A + (c*A)@A`.

- returned pair gap: `-7/128 = -0.0546875`;
- returned pair ratio: `-7/160 = -0.04375`;
- interpretation: a violation for this recorded backend, PyTorch build, dtype,
  epsilon, coefficients, iteration count, and operation order only.

The earlier local-shadow value `-3/128` used the real-arithmetic-equivalent but
BF16-inequivalent grouping `c*(A@A)` and is not reported as a deployed-order
result. The committed `bf16_witness.json` records output storage words, every
stage, cast/normalization behavior, an observable accumulation probe, complete
software/hardware data, source hashes, and the upstream revision. It does not
support a universal statement about other BF16 backends.

## 3. Deterministic 2x2 spectral audit

The exact-current unit-spectrum deficit is positive in all 24 audited
polynomial-prefix configurations. The final configured stages are:

For every row, exact rational arithmetic at the unit diagonal spectrum
`(3/5, 4/5)` also verifies unequal composed derivatives. Combined with the
analytic scale obstruction, this establishes an infinite unrestricted deficit
for exact normalization independently of the floating-point grid values.

| Baseline | Stages | Matmuls | Exact-current unit deficit | Fixed-scale unit deficit | Normalization-clean deficit | eps=1 diagonal grid deficit |
|---|---:|---:|---:|---:|---:|---:|
| Jordan quintic | 5 | 15 | 158.675144 | 158.680182 | 0.405923 | 157.958941 |
| Classical cubic | 5 | 10 | 0.015338 | 0 | 0.015338 | 0.007474 |
| Taylor quintic | 5 | 15 | 0.005034 | 0 | 0.005034 | 0.003841 |
| Polar Express repo `71cc` | 5 | 15 | 218.071514 | 218.073003 | 13.567219 | 217.888813 |
| CANS-5x4, delta=0.3 | 4 | 12 | 130.309482 | 3146.191653 | 15.784223 | 123.261935 |

Classical and Taylor are the clean attribution cases on this grid: their
fixed-scale Jacobians are nonnegative while current normalization produces a
strict deficit. Jordan, Polar Express, and CANS also have normalization-clean
witnesses, but their total deficits are dominated by fixed-scale polynomial
nonmonotonicity on parts of the unit-spectrum grid.

Polar Express is pinned to the current repository configuration at commit
`71cc37943d99cae780024c1d198977f2f8795407`. CANS uses the published
degree-five, four-stage, `delta=0.3` coefficient table; no code was copied from
its unlicensed repository.

## 4. Controlled diagonal matrix quadratics

Design: 32 matched 2x2 diagonal SPD quadratics, condition numbers
`{1, 3, 10, 30}`, current-Frobenius-plus-`eps=1` normalization, 750 updates,
73 geometric learning rates in `[1e-5, 3]`, and target-and-hold success defined
as objective ratio at most `1e-4` for every one of the final 50 iterations on
at least 90% of problems.

Each polynomial prefix uses one primary gain-matched comparison and one
algebraically equivalent cross-check:

1. gain-matched repair versus unrepaired, both with the same Jacobian at zero;
2. raw `+rho M` repair versus a gain-only unrepaired control, also with matched
   zero-input gain. This is the same contrast under learning-rate rescaling,
   so it is not independent evidence.

At the primary criterion:

- gain-matched repair versus unrepaired: 16/24 upper endpoints unchanged,
  8/24 repaired-only passing bands, 0 right shifts, and 0 left shifts among
  pairs where both bands existed;
- raw repair versus gain-only control: the same 16 unchanged and 8
  repaired-only pattern;
- raw repair versus unrepaired, which is not gain matched: 11 unchanged,
  5 left shifts, and 8 repaired-only bands;
- the 10 normalization-only classical/Taylor configurations are unchanged in
  both gain-matched comparisons at the learning-rate-grid resolution;
- all repaired-only bands occur in Jordan, Polar Express, or CANS prefixes
  whose fixed-scale maps are already nonmonotone on the audit grid.

Thus this experiment does **not** show that repairing the normalization-only
deficit widens the observed upper target endpoint. It does show repaired-only
finite-horizon target-and-hold bands for several strongly nonmonotone
polynomial prefixes. The corresponding unrepaired trajectories remain bounded
on the sampled learning-rate grid but do not meet the aggregate target within
750 steps. This cannot be attributed to normalization alone, and it is not
evidence that the repair removes divergence.

After normalizing both quantities by the zero-input gain, the complete-case
exploratory association between diagonal deficit and upper target endpoint is
`Spearman r = -0.256` (`n=16`, descriptive `p=0.339`). The eight configurations
whose unrepaired target band is absent are excluded based on the outcome. The
designed configurations are also dependent, so this is neither a confirmatory
test nor a predictor analysis over all 24 configurations.

The repair size is only a sampled-grid correction, not a trajectory-wide
certificate. All 48 repaired configuration summaries observe positive gradient
norms below the audit's lower dimensionless radius of `1e-8` (minimum observed
about `2.22e-162`), and 26 also exceed its upper radius of 1000 (maximum
observed `17100.25`). The analytic check at exactly zero does not certify the
unsampled interval between zero and `1e-8`.

Across nine target/fraction sensitivity criteria, gain-matched repairs range
from 1 to 10 repaired-only bands, 14 to 18 unchanged endpoints, 0 to 2 right
shifts, and 0 to 5 left shifts. This confirms that finite target bands depend
on the operational success definition.

## 5. Horizon check

Jordan step 5 and classical step 5 were repeated at 750 and 3000 iterations
with target ratios `1e-4` and `1e-8`.

- Jordan, target `1e-4`: the unrepaired band is absent at 750 steps but appears
  by 3000 steps with upper endpoint `0.007775`; the gain-matched repaired upper
  endpoint is `0.009263` at both horizons.
- Jordan, target `1e-8`: the unrepaired band is absent at both horizons; the
  gain-matched repaired upper endpoint is `0.003858`.
- Classical: all four interventions share upper endpoint `0.258301` at both
  horizons and both target ratios.

The horizon drift is why these are labeled finite target bands rather than
stability regions. Normalized endpoints above the local linear ceiling of two
are likewise finite-horizon target outcomes; they are not evidence of
asymptotic convergence to zero or of any particular attractor.

## 6. Floored-normalizer full-matrix certificate

For the real-arithmetic five-step Jordan map, define
`F_h,c(M)=H_h(M/max(c, ||M||_F))`, with `c>0`, exact coefficients
`(6889/2000, -191/40, 4063/2000)`, five iterations, and no additive epsilon.

An adaptive dyadic interval proof using outward-rounded Arb balls certifies

- `-159.5496 < h'(s) < 484.8763` for every `s` in `[0,1]`;
- identical 25,370-leaf covers at 160 and 224 bits, maximum dyadic depth 32;
- all diagonal, off-diagonal, repeated/zero, and rectangular-null tangent
  modes lie in the same slope interval by exact secant-average formulas;
- pairwise line integration handles the nondifferentiable floor boundary.

The resulting dimension-uniform full-matrix bracket at `c=1` is

\[
159.549525785 < \delta(F_{h,1})
\le \frac{41528474059081}{260261360000}
=159.5645010810709665084\ldots.
\]

An exact rational finite pair embedded in a rank-one mode inside the floor
proves the displayed strict lower bound; its actual deficit is approximately
`159.54952578566153`. The upper endpoint comes from the projection-envelope
bound. Its gap above the displayed safe lower threshold is `0.0149752961`, or
`0.009386%` relative. It is about `32.91%` of the map's zero-input differential
gain and about `3.04x` smaller than using that full `484.8763` gain as a
generic Lipschitz repair.

Exact rescaling proves `delta(F_h,c)=delta(F_h,1)/c`; therefore
`rho=(41528474059081/260261360000)/c` is a globally sufficient constant
conductance for every finite matrix shape. This is a certified near-minimal
repair, not an exact fixed-shape minimum. Setting
`rho=(41528474059081/260261360000)/c+mu` for `mu>0` yields a
`mu`-strongly monotone map and an `exp(-mu*t)` contraction guarantee for the
explicitly simplified system `dot(M)=-(F_h,c(M)+rho*M-b)`. This does not
establish momentum-Muon training stability.

Related finite-step regularization work by Chang et al. gives global
Frobenius-Lipschitz bounds for an additive denominator regularizer. It does not
provide a monotonicity-deficit certificate for this max-floor architecture:
<https://arxiv.org/abs/2606.01720>.

## 7. Unrun gate

No NanoGPT result is reported. This machine exposes neither CUDA nor an
available MPS device. The `rho` used in the existing quadratic study is only a
finite-grid sampled repair; those results do not retroactively test the new
floored architecture or its global certificate. A matched language-model
sweep remains gated on suitable compute, matched-momentum quadratics, and a
predeclared use of the certified floored design.

## Reproduce

```bash
uv run --locked python scripts/find_counterexample.py \
  --output results/summaries/canonical_witness.json
uv run --locked python scripts/record_bf16_witness.py \
  --output results/summaries/bf16_witness.json
uv run --locked python scripts/certify_floored_repair.py \
  --output results/summaries/floored_repair_certificate.json
uv run --locked python experiments/matrices/run_deficit_audit.py
uv run --locked python experiments/quadratics/run_lr_sweep.py
uv run --locked python experiments/quadratics/run_horizon_check.py
uv run --locked python scripts/make_figures.py
uv run --locked pytest -q
```
