# P17 shape-preserving resolvent: exact theorem and scoped diagnostics

## Verdict

P17 is a positive exact-real result. It restores meaningful spectral shaping
to the P14--P16 resolvent architecture while retaining a global one-trajectory
smooth-PL convergence theorem. Two locked designs pass:

- a primary high-fidelity design at `eta=1/128000`; and
- a secondary full-step design at `eta=1/32000`.

The authoritative result is a dimension-uniform, origin-centered **pointwise**
gain sector plus an exact rational value--momentum LMI. It is not an
incremental sector, a global fidelity lower bound, an FP64/BF16 theorem, or a
claim about literal upstream Muon.

## Locked exact-real interface

For the P14 resolvent `J` and Yosida map `Y=(I-J)/lambda`, P17 exposes

```text
T(S) = (1-theta(||S||_F^2)) Y(S)/c
       + theta(||S||_F^2) E_h,epsilon(J(S)).
```

The gate is the C2 quintic smootherstep in `q=||S||_F^2`: it is zero for
`q<=1/4`, transitions on `(1/4,1)`, and equals its design-specific ceiling for
`q>=1`. The remaining locks are:

- additive epsilon `epsilon=1/10000000`;
- five Jordan stages with coefficients `6889/2000`, `-191/40`, and
  `4063/2000`;
- resolvent parameter `lambda=1/1000`;
- passive shunt `mu=1000`;
- EMA/Nesterov momentum `beta=19/20`; and
- KellerJordan/Muon source revision
  `f98f1cacc0263b04290753e32be8d498c1efc806` for the upstream formula.

Upstream contains neither the P13 repair, the resolvent, nor this gate.

## Authoritative exact certificates

The singular-mode proof gives

```text
M_X = 20191130443162880000000 / 26793221204801899863 < 754
```

for the raw Jordan-after-resolvent gain. Combining this with the Yosida modal
sector `[500,1000]` yields the following global pointwise sectors on every
finite real rectangular matrix space:

| Design | Gate ceiling | Passive divisor | Exact step | Pointwise gain interval | Exact storage rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| High fidelity | `3/4` | `4096` | `1/128000` | `[0.030517578125, 565.254287458]` | `tau=16777215/16777216` |
| Full step | `1/8` | `8192` | `1/32000` | `[0.05340576171875, 94.305686907]` | `tau=16777209/16777216` |

The exact `4 x 4` LMIs prove

```text
V_(t+1) <= tau^2 V_t
```

for every differentiable, globally `10`-smooth objective with finite infimum
that satisfies the global PL inequality with constant `1`. Consequently the
objective gap, gradient, and momentum converge geometrically, and the
summable updates converge to some trajectory-dependent global minimizer.
Neither uniqueness nor arbitrary-pair contraction is claimed.

### Exact/Arb canonical fidelity gate

The separate exact/Arb certificate includes P15 residual inflation on
`S=diag(3,4)` and proves that both designs pass the unchanged P16 gates:

| Design | Best-scalar departure | Upstream shaping retained | FP64 output/input amplitude |
| --- | ---: | ---: | ---: |
| High fidelity | `0.0567757867` | `0.811459784` | `0.246162334` |
| Full step | `0.0203561150` | `0.290936853` | `0.114450881` |

The first two columns are the authoritative Arb-enclosed decisions; the
amplitude column is a matching FP64 diagnostic. The frozen thresholds are
departure at least `0.001` and retention at least `0.1`. The unequal modal
gains are approximately `[0.227131,0.256247]` for the primary design and
`[0.111321,0.116174]` for the full-step design.

## Deterministic FP64 study

The companion study is a falsification and fidelity diagnostic. Every
declared evaluation calls the P16 solver, returns `(S-u_hat)/lambda` for its
Yosida component, and fails closed unless the computed graph residual passes

```text
||r||_F <= ||S||_F/250 + rbar_fp64.
```

The default replay made `3855` certified calls. Its worst computed
residual-to-P15-threshold ratio was `2.352e-9`; the maximum observed Newton
count was `10` and the maximum backtrack count was `8`.

These are computed FP64 residuals, not directed-rounding certificates. For
fidelity reporting the study additionally propagates each residual through
the exact-real bounds

```text
||u_hat-J(S)||_F <= ||r||_F/2,
||Y_hat-Y(S)||_F <= 500 ||r||_F,
Lip(E_h,epsilon) <= 484.8763/epsilon.
```

The operating-annulus diagnostic uses a stricter internal Newton target of
`2^-44 ||S||_F + 2^-50`; the P15 acceptance rule and the mathematical
operator are unchanged.

### Post-exploratory operating annulus

The broad study first exposed the expected loss of shaping near the origin,
where the gate is off. Only after that observation, the following narrower
operating-domain diagnostic was locked:

```text
3/4 <= ||S||_F <= 25.
```

It directly sampled input radii using combined logarithmic and linear grids
and sampled two-mode ratios using a logarithmic grid plus the frozen P16
stress ratios. Of `2688` points, `2176` had upstream best-scalar departure at
least `0.01` and were therefore informative for retention. For the primary
design:

- all `2176/2176` informative point estimates passed both frozen gates;
- all `2176/2176` still passed after the computed-residual allowance;
- minimum sampled departure was `0.004885755`;
- minimum residual-allowance departure lower endpoint was `0.004885534`;
- minimum sampled shaping retention was `0.342260223`;
- minimum residual-allowance retention lower endpoint was `0.342244801`;
- sampled gate weights ranged from `0.259964554` to `0.75`; and
- maximum sampled output/input amplitude was `1.283077795`.

This is a post-exploratory sampled operating-annulus result. It is not a
uniform theorem over the annulus and must not be presented as preregistered or
as evidence near zero.

### Broad two-mode falsification grid

The broad grid parameterized roots over radii from `1e-12` through the P11
guard, mapped them forward through the resolvent graph, and independently
resolved every retained input. It intentionally includes the small-signal
gate-off region:

| Design | Evaluated | Informative | Frozen-gate passes | Pass fraction |
| --- | ---: | ---: | ---: | ---: |
| High fidelity | `418` | `327` | `203` | `0.6208` |
| Full step | `418` | `327` | `199` | `0.6086` |

These failures prohibit a global fidelity claim. They do not contradict the
global pointwise gain sector or PL theorem: fidelity and stability are
different properties, and the fallback branch is deliberately radial near
the origin.

### Complete Transformer spectra

The following synthetic cases evaluate every one of the `min(m,n)` singular
values. They do not allocate the dense matrix or execute its SVD.

| Declared shape | Complete spectrum | High-fidelity amplitude | Full-step amplitude | High-fidelity retention | Full-step retention |
| --- | --- | ---: | ---: | ---: | ---: |
| `768 x 768` | unit flat repeated | `22.6251` | `3.86748` | not informative | not informative |
| `768 x 3072` | half-guard, condition `1e4` | `1.13472` | `0.230122` | `0.97807` | `0.80405` |
| `3072 x 12288` | guard, two repeated blocks | `1.16787` | `0.248440` | `0.95300` | `0.74690` |
| `4096 x 11008` | guard, condition `1e6` | `1.02936` | `0.212627` | `0.97562` | `0.78746` |

The flat repeated spectrum is collinear by symmetry, so its near-zero
best-scalar departure cannot support a retention interpretation.

### Adversarial rank accumulation

A separate grid uses one leading root singular mode and `k-1` modes at ratio
`41/10000`. At `k=11008`, the maximum sampled output/input amplitudes were
`92.3834` for the high-fidelity design and `15.4463` for the full-step design.
Both lie below their authoritative exact pointwise upper gains, but they show
that many individually small modes can accumulate substantial Frobenius
output. The singular-spectrum computation does not establish dense runtime or
memory cost.

## Negative controls

The exact certificate proves that the raw shape branch has negative scalar
derivative on a complete interval near normalized radius `63/10000`. The
locked gate is zero there. If the passive region is instead shrunk to norm
radii `1/2048` and `1/1024`, that witness lies on the shape plateau. The FP64
diagnostic gives derivative `-18345.706`, while the exact/Arb control encloses
it in `(-19000,-18000)` and gives a negative scalar Jury margin.

This rejects an incremental/passivity interpretation of the undersized gate;
it is not a divergent smooth-PL trajectory. A rank-one input is also retained
as a negative fidelity control because one positive singular mode is always
collinear with its input. The global pointwise-sector PL proof uses only
`0<=theta<=bar_theta`, so this negative control does not invalidate that
one-trajectory theorem.

## Evidence classification

| Statement | Evidence | Scope |
| --- | --- | --- |
| Global pointwise gain sectors | Exact rational/analytic certificate | Every finite rectangular shape |
| Global smooth-PL convergence | Exact rational `4 x 4` LMI | Exact-real pinned EMA/Nesterov recurrence |
| Canonical fidelity gates | Exact/Arb enclosure | `diag(3,4)` only |
| Operating-annulus fidelity | Deterministic FP64 grid plus residual allowance | Sampled primary-design points only |
| Broad-grid and rank behavior | Deterministic FP64 falsification | Sampled, not extremal |
| Transformer-shape results | Complete synthetic singular spectra | No dense SVD or runtime benchmark |
| Undersized-gate failure | Exact/Arb scalar control plus FP64 replay | Local derivative/Jury obstruction |

## Reproduction

```bash
uv run --locked python scripts/certify_shape_preserving_resolvent.py \
  --output results/summaries/shape_preserving_resolvent_certificate.json

uv run --locked python scripts/reconstruct_shape_preserving_resolvent.py \
  --canonical results/summaries/shape_preserving_resolvent_certificate.json \
  --require-canonical

uv run --locked python experiments/resolvent/run_p17_shape_preserving_study.py \
  --output results/summaries/p17_shape_preserving_study.json
```

## Remaining limitations

P17 assumes the exact resolvent in its theorem. It does not yet propagate the
P16 approximate-solver error through the shape branch and gate, certify FP64
or BF16 evaluation, establish upstream learning-rate parity, or include
aspect scaling, weight decay, stochastic gradients, model-state
reconstruction, dense accelerator cost, or neural-network training evidence.
Independent human proof review remains pending.
