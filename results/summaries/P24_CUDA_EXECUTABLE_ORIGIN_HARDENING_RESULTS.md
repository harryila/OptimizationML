# P24 CUDA executable-origin hardening result

## Verdict

P24 **stopped at its first retained A100 baseline diagnostic**, before the
remediation image was built. The native diagnostic reproduced P23's suspected
PyTorch-generated module with the exact locked `2,355` bytes under writable
`/tmp`, but one of its 36 exact checks failed: the fresh process's live Torch
determinism state did not equal the committed runtime lock. The diagnostic
exited `1`, as required.

The offline sanitizer then independently exited `2` without producing a
sanitized artifact. Its frozen recursion transformed mapping values but not
mapping keys; the retained runtime and attestation content contained `/tmp`
as a tmpfs-contract key, which survived redaction and triggered the final
declared-root check. These are two separate pre-acquisition failures.

This is not the requested CUDA repeatability or real-gradient fidelity result.
No remediation image, replacement container, CUDA gradient, Muon candidate,
observer run, shield statistic, or training result was produced.

## Exact retained evidence

The diagnostic was launched from clean static commit
`9003130ca31085f555ff144216a599442cbdf3ca`, tree
`58379372d752b2c17b6db7e1f690f899c2051545`. The frozen contract has SHA-256
`bdb2d3aad7d70d0ed5c6dddcd03642c9384d8b67d6f5ca507987d28e122dd446`.

The first native baseline artifact remains external and is bound here, rather
than copied or summarized as if sanitization had succeeded:

| field | value |
| --- | --- |
| schema | `passive-muon-p24-executable-origin-diagnostic-v1` |
| mode/status | `baseline` / `fails` |
| diagnostic exit | `1` |
| native SHA-256 | `ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2` |
| native byte count | `3,760,729` |
| exact checks | `35` true, `1` false |
| sole false check | `torch_determinism_matches_lock` |

The native artifact observed the exact generated-module SHA-256
`8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8`,
byte count `2,355`, on a writable `tmpfs` mounted at `/tmp`. This establishes
the causal localization sought after P23 for the minimal trigger
`import torch; Parameter; torch.optim.SGD`: it does not reach a model, data,
CUDA forward/backward pass, Muon call, or optimizer step.

## Determinism mismatch

The single false aggregate check comprises these exact live-versus-lock
differences:

| field | live process | committed lock |
| --- | ---: | ---: |
| deterministic algorithms | `false` | `true` |
| deterministic debug mode | `0` | `error` |
| interop threads | `30` | `1` |
| cuDNN deterministic | `false` | `true` |
| FP16 reduced-precision reduction | `true` | `false` |
| BF16 reduced-precision reduction | `true` | `false` |

The other 35 diagnostic checks passed. That does not authorize ignoring the
one failed check or proceeding to an image build: the contract requires the
complete Boolean conjunction.

## Sanitizer stop

No sanitized diagnostic exists. The sanitizer's exact stderr was:

```text
P24 diagnostic sanitization blocked: redacted diagnostic retains a declared native path
```

Its stdout was empty. The stderr has SHA-256
`3939a4478d2bd761c810732b01d9f3af9dd31b09c31efe376eece88ac279034f`
and byte count `88`; both empty streams have the standard SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

The failure is reconstructible from the frozen source without the external
native manifest: mapping values are sanitized recursively while keys are
copied unchanged, and the committed P23 runtime lock contains `/tmp` as a
tmpfs-contract mapping key. The final path scan therefore rejects the output.
Failing closed here is correct; P25 must repair the transformation rather than
mislabel the native file as sanitized.

## Gate status and route

| gate | outcome |
| --- | --- |
| baseline diagnostic | failed exact Torch determinism match |
| baseline sanitization | failed closed; no output |
| remediation image build | not run |
| replacement runtime and remediated diagnostic | not run |
| trace-off A/B and exact repeatability | not run |
| trace-on and observer noninterference | not run |
| candidate observations | `0` |
| fidelity aggregate and training | not run |

The result routes mechanically to
`p25-cuda-diagnostic-determinism-and-redaction`. P25 must configure or
otherwise reconcile the diagnostic's live Torch state with the frozen lock,
sanitize mapping keys as well as values under an exact reviewed rule, and
freeze those corrections before another diagnostic attempt. It must not
weaken P23's fidelity thresholds or rerun P24 until a favorable outcome
appears.

The P18--P21 analytic and finite-precision results are unaffected. P24 proves
neither CUDA repeatability, candidate fidelity, native CUDA shield safety,
training quality, nor stability of unrepaired Muon.

## Machine-readable record

The compact
[`p24_cuda_executable_origin_outcome.json`](p24_cuda_executable_origin_outcome.json)
binds the external native artifact and all retained client streams. It is an
operator-recorded outcome packet, not the absent sanitized native manifest.
Independently replay its static commit/contract relations and source-level
sanitizer diagnosis with:

```bash
python scripts/reconstruct_p24_cuda_executable_origin_outcome.py
```
