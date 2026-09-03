# P8 fixed-2x2 mixed-precision result

## Result

P8 certifies one proposed mixed-precision implementation of the repaired
max-floor five-step Jordan operator at fixed shape `2 x 2`.  For every finite
FP32 input `s` with

\[
\max_{i,j}|s_{ij}|\le 2^{116},
\]

the exact-real reference and locked implementation satisfy

\[
\lVert\widehat R(s)-R(s)\rVert_F
\le\frac{11}{100000}\lVert s\rVert_F+\frac{347}{100}.
\]

The reference uses the exact max floor `c=1`, no additive epsilon, exact
coefficients `(6889/2000,-191/40,4063/2000)`, exactly five stages, and
`rho=210177835339081/260261360000`.  The implementation uses a scaled FP32
normalizer, locked FP32 coefficient encodings, a fixed serial Horner stage,
one initial BF16 state plus one BF16 round after each of the five stages, and
FP32 repair/return arithmetic.
Each length-two dot uses two separately rounded FP32 multiplications and one
separately rounded FP32 addition; FMA contraction and native BF16 matmul are
excluded.

For an arbitrary real `2 x 2` input in the same range, let `C32` be entrywise
IEEE binary32 conversion and define
`Rhat_R(s)=Rhat(C32(s))`.  The exact ideal-map Lipschitz bound absorbs the
interface cast and gives

\[
\lVert\widehat R_{\mathbb R}(s)-R(s)\rVert_F
\le\frac1{5000}\lVert s\rVert_F+\frac{347}{100}.
\]

The proof assumes IEEE round-to-nearest, ties-to-even arithmetic, gradual
underflow, and no FTZ/DAZ.  It uses an exact Sturm range proof and a rational
five-stage matrix-norm invariant.  The affine intercept is conservative and
is not claimed minimal.

## P7 consequence

Place this real-input adapter in the otherwise exact-real P7 EMA/Nesterov
loop, set gradient noise to zero, and assume every operator input remains in
the certified magnitude range.  The affine error and exact P7 signal-storage
bound give

\[
V_{t+1}\le q_8V_t+B_V,
\]

where

\[
q_8=\frac{41597186684695561}{41601344000000000}
=0.9999000677645309\ldots<1,
\qquad
B_V=\frac{4936769}{819840000000}.
\]

Let

\[
H_{\rm safe}=\frac{2600084\,2^{232}}{1655544025}
\approx1.08394\times10^{67}.
\]

The exact certificate checks `B_V <= (1-q_8)H_safe`. Therefore
`V_0<=H_safe` inductively gives `V_t<=H_safe` and
`||s_(t+1)||_F<=2^116`, so the finite-input guard holds at every call. This
range invariant is explicitly for zero gradient noise; the noisy signal has
an additional direct disturbance term.

The resulting worst-case objective-gap neighborhood is

\[
\limsup_t(f(W_t)-f_\star)
\le
\frac{462392438350000000}{207695315294468001}
=2.22630172325\ldots.
\]

This is an exact error-to-objective consequence, not an observed training
loss and not a tightness claim.

## Verification

The canonical artifact is `mixed_precision_certificate.json`.  It records the
locked arithmetic contract, exact coefficient encodings, normalization error,
Sturm chain, one-stage recurrence, affine repair shell, every Boolean gate,
and the exact P7 closure.  The standalone reconstruction imports neither the
project package nor numerical libraries and reads the canonical JSON only
after rebuilding its exact algebra.  Focused tests check the certificate,
independent reconstruction, CLI, and bitwise reference-kernel parity.

## Sampled diagnostics

A deterministic 82-case CPU grid combines explicit boundary, subnormal,
rank-one, repeated-singular-value, sign/permutation, and large-magnitude cases
with four seeded families.  Against a float64 evaluation of the ideal formula,
it found zero candidate bound violations and zero nonfinite results.  The
largest observed kernel-bound ratio was `0.32618452101249257`; the largest
all-real-adapter-bound ratio was `0.18142092780684033`.  The largest absolute
error, about `1.53664e30`, occurred at scale `2^116` and is consistent with the
affine repair-coefficient term.

The comparator is float64, not exact arithmetic.  These results are
falsification diagnostics only; they neither prove the global bounds nor
establish tightness or backend performance.  The exact certificate remains
authoritative.

## Scope exclusions

P8 is not literal upstream Muon, an arbitrary-shape or dimension-independent
mixed-precision theorem, an all-BF16-intermediate kernel, or an
accelerator-native BF16 result.  It excludes current-plus-epsilon
normalization, weight decay, aspect scaling, and unspecified rounding modes.

The theorem uses an exact-real outer EMA/Nesterov and parameter shell.  The
repository's FP32 outer-loop helper is only an executable prototype.  FP32
momentum/Nesterov errors require additional internal ports, while FP32
parameter subtraction can stall at large binades.  Consequently this is not a
whole-FP32-optimizer convergence theorem.

P7 remains the submission cutoff and broad robust theorem.  P6 is P7's
zero-disturbance corollary; P8 is a constructive, fixed-shape instantiation of
one P7 disturbance port.

## Reproduce

```bash
uv run --locked python scripts/certify_mixed_precision.py \
  --output results/summaries/mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_mixed_precision.py \
  --require-canonical
uv run --locked python \
  experiments/mixed_precision/run_mixed_precision_falsification.py \
  --output results/summaries/mixed_precision_falsification.json
uv run --locked pytest -q \
  tests/test_mixed_precision.py \
  tests/test_mixed_precision_certificate.py \
  tests/test_mixed_precision_cli.py \
  tests/test_mixed_precision_experiment.py \
  tests/test_mixed_precision_reconstruction.py
```
