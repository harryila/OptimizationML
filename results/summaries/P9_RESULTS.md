# P9 scalable mixed-precision result

## Result

P9 replaces P8's fixed `2 x 2`, one-term-BF16 certificate with a
shape-parameterized theorem for a proposed two-term-BF16, balanced-FP32 proof
kernel.  For each audited shape `(r,c)`, every real input with
`max_ij|s_ij|<=2^116` obeys

\[
\lVert\widehat R_{r,c}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c}.
\]

The exact reference uses max-floor normalization with `c=1`, no additive
epsilon, coefficients `(6889/2000,-191/40,4063/2000)`, exactly five Jordan
stages, and repair
`rho=210177835339081/260261360000`.  Shapes are transposed once to
`min(r,c) x max(r,c)`, and the shape domain has `rc<=2^52`.

For the seven locked representative Transformer dimensions,

\[
A_{r,c}=\frac{102465557}{549755813888}
=0.00018638376241142396\ldots.
\]

The largest certified intercept is

\[
B_{4096,11008}=B_{4096,14336}
=\frac{2179083213031}{1099511627776}
=1.9818646369740236\ldots.
\]

The arithmetic contract keeps the normalizer, Horner stage bodies, repair,
and return in FP32.  Every norm and matrix dot uses a deterministic adjacent
balanced FP32 tree.  Each stage boundary is stored as

\[
h=\operatorname{RN}_{b}(Y),\qquad
\ell=\operatorname{RN}_{b}\bigl(\operatorname{fl}_{32}(Y-h)\bigr),
\]

and reconstructed with one FP32 addition.  This two-term BF16 state is a
specified error-feedback design.  It is not an upstream or production kernel.
Its two buffers have the same nominal storage as one FP32 buffer, so the result
does not establish state compression or a throughput advantage.

## P7 consequence

For every differentiable, globally `L=10`-smooth objective satisfying the
global PL inequality with constant one, use P7's otherwise exact-real pinned
EMA/Nesterov loop with `beta=19/20`.  With zero gradient noise, the all-real
adapter instantiates P7's post-operator error port.  At the full P7 step
`eta=1/32000`, every certified representative shape satisfies

\[
V_{t+1}\le q_{r,c}V_t+D_{r,c},
\qquad
q_{r,c}=\frac{137425214491}{137438953472}
=0.9999000357565819\ldots<1.
\]

The corresponding worst-case objective-gap neighborhoods are:

| Shape | `B_(r,c)` | Certified `limsup(f-f*)` |
| --- | ---: | ---: |
| `768 x 768` | `1.057373530` | `0.206682025` |
| `768 x 3072` | `1.100936249` | `0.224063265` |
| `768 x 50257` | `1.188552191` | `0.261145507` |
| `3072 x 12288` | `1.814441125` | `0.608599542` |
| `4096 x 4096` | `1.937451049` | `0.693916766` |
| `4096 x 11008` | `1.981864637` | `0.726095607` |
| `4096 x 14336` | `1.981864637` | `0.726095607` |

The largest neighborhood is exactly

\[
\frac{798350562999}{1099511627776}.
\]

The exact audit includes a forward-invariant `2^116` operator-input guard for
initial storage below `2^232/(1655544025/2600084)`.  That range implication is
for zero gradient noise; noisy signals need a separate guard.

## Obstructions and design consequence

Two negative statements have different logical strength:

- **Executable serial-normalizer obstruction.** On an all-ones input with
  more than `2^24` entries, a serial FP32 sum of unit squares sticks at
  `2^24`.  At `4096 x 11008`, the returned normalized rank-one matrix has
  squared singular value `43/16`, outside the `(5/4)^2` proof tube.  The P9
  kernel therefore uses a balanced norm tree.
- **One-term BF16 proof obstruction.** The generic slope-only boundary
  estimate fits the `5/4` spectral tube only through short rank `71` and fails
  at rank `72`.
  This does not prove executable instability or rule out a sharper one-term
  analysis.  The exact, Sterbenz-free two-term boundary bound raises the
  boundary-only slope gate to `4,656,751` (`4,693,632` under idealized exact
  subtraction/reconstruction).

The full five-stage recurrence, not the boundary gate alone, sets the usable
shape frontier. As one locked negative control, `4608 x 18432` fails only the
stage-five-input tube check. Its exact stage-four envelope is

\[
\frac{10753627185}{8589934592}
=1.25188696954865\ldots>\frac54.
\]

This is a limitation of the current sufficient recurrence, not an observed
failure or an impossibility theorem.

## Native-matmul diagnostic

A deterministic CPU diagnostic compares three otherwise matched boundary
policies on 16 inputs across shapes `2 x 2`, `8 x 16`, `32 x 64`, and
`64 x 128` (48 policy evaluations total).  On the recorded macOS arm64,
PyTorch `2.13.0` run, every output was finite.  The maximum operator errors
against the numerical float64 target were:

| Boundary policy | Maximum Frobenius error |
| --- | ---: |
| One BF16 cast after stage five | `0.0172662955` |
| One-term BF16 at every boundary | `0.8253062732` |
| Compensated high/low BF16 at every boundary | `0.0025095022` |

The compensated policy had no larger error than repeated one-term BF16 in
all `16/16` matched cases.  These observations use PyTorch's optimized native
FP32 matrix multiplication, whose reduction order is not the theorem's
balanced reference graph, and the float64 comparator is not exact arithmetic.
They are implementation diagnostics, not evidence for the global bound or a
claim that compensation always wins.

## Evidence and limitations

The exact per-shape rational recurrence, including outward `2^-40` rounding,
subnormal crumbs, overflow guards, and P7 closure, is authoritative.  It
checks partial-dot and pre-add envelopes, BF16 boundaries, the FP32 repair
shell, and the common rate strictly below one.  Executable small-shape parity
and diagnostics can only falsify the specified implementation.

P9 excludes literal upstream Muon, GPU/BLAS/tensor-core parity, native BF16
matrix multiplication, current-plus-epsilon normalization, an unrepaired
operator, stochastic rounding, FTZ/DAZ, aspect scaling, and weight decay.  Its
outer EMA/Nesterov and parameter shell remain exact real arithmetic; FP32
state ports and compensated or higher-precision master weights are separate
next steps.  No claim is made about neural-network training or kernel
throughput.

## Reproduce

```bash
uv run --locked python scripts/certify_scalable_mixed_precision.py \
  --output results/summaries/scalable_mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py \
  --require-canonical
uv run --locked python \
  experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py \
  --output results/summaries/scalable_mixed_precision_diagnostic.json
```

The exact certificate and independent standard-library reconstruction are
the theorem evidence.  The native-matmul CPU diagnostic is falsification only.
