# P25 CUDA diagnostic determinism and redaction result

## Verdict

Attempt `20260906-01` is **terminal before `trace_off_a`**. Its first and only
corrected diagnostic genuinely passed all `40/40` exact checks, and the
corrected sanitizer produced the complete committed wrapper. The subsequent
host acquisition preflight failed closed before any gradient process began.

The stop was a host-permission defect, not a diagnostic-content mismatch. The
retained evidence wrapper was owned by `root:root` with mode `0600`; the host
wrapper invoked unprivileged `cmp -s`, which could not read it and emitted the
generic “wrappers differ” stop. The retained and repository wrappers in fact
have the same SHA-256 and byte count. The unprivileged comparison exited `2`;
the privileged comparison of those same bytes exited `0`.

This is not the requested CUDA repeatability or real-gradient fidelity result.
There are zero trace, repeatability, noninterference, or aggregate artifacts,
zero candidate observations, and no shielded training result. The attempt must
not be retried until favorable.

## Exact diagnostic result

The replacement runtime was frozen at commit
`a967f30dae3c8926f9ce5c4783a0923e42bcdb0d`, tree
`8dca3402f27dfe1d0770cc2849fcca24778d76d8`. It used the immutable image
`localhost:5000/p25-runtime@sha256:e1a636b8c113aaa2823bd8e03bc0dee626d91cdc6897154e05a2bda54b178ee6`
on the pinned non-MIG `NVIDIA A100-SXM4-40GB`.

| field | value |
| --- | --- |
| attempt | `20260906-01` |
| diagnostic mode/status | `remediated` / `passes` |
| exact checks | `40` true, `0` false |
| diagnostic exit | `0` |
| native SHA-256 | `e6479d1a02556cee451cbcf1da3f8b10df498bca7835e9db3730a08f886d9184` |
| native byte count | `4,113,445` |
| sanitized SHA-256 | `56983629e935e3e18ac89b894ee84bc56e47f862abfcc36a6a5a15dbb8e5a354` |
| sanitized byte count | `5,097,711` |

The full sanitized manifest is committed as
[`p25_executable_origin_diagnostic.sanitized.json`](p25_executable_origin_diagnostic.sanitized.json).
It independently reconstructs the closed 40-check pass, including the exact
contract and sources, runtime lock and attestation, deterministic Python/Torch
state, live CUDA/GPU state, immutable generated-module origin, and trigger
order. The generated module had the locked 2,355 bytes and was observed on a
read-only effective mount.

This diagnostic remains deliberately narrow: it constructed one CPU FP32
parameter and `torch.optim.SGD`. It did not allocate the model, read FineWeb,
run CUDA forward/backward, call Muon, take an optimizer step, or observe a
candidate.

## Frozen provenance chain

The exact direct-child chain reconstructed from Git is:

| stage | commit | tree | only delta |
| --- | --- | --- | --- |
| source freeze | `5814621979ffedf370f04f3d99d925d8584b640f` | `31cafc8bae9a7311b18940c267a4da52ed6e8068` | P25 source packet |
| preregistration | `e76ab62f92c95e6f0716cf2f1ed38a583cadfe56` | `4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2` | sanitized historical P24 wrapper |
| runtime freeze | `a967f30dae3c8926f9ce5c4783a0923e42bcdb0d` | `8dca3402f27dfe1d0770cc2849fcca24778d76d8` | P25 runtime lock and host attestation |
| diagnostic bridge | `a037b04d0ae1e9c5b6b371ae78293e5e5640d91c` | `ea5941ee873f518f147f51fd15555989c331caef` | committed sanitized P25 diagnostic |

The frozen contract SHA-256 is
`51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2`.
The runtime lock and host attestation hashes are respectively
`95f06d7fcf6731c331811fc35cb48fa89606f22d53c968e5c5d625ca10e117f3`
and
`2dc43b993028e624a0026fe0446f82e506ff51bf11c9bb38dc9fd17a173ab7b2`.

## Terminal acquisition stop

The exact client stderr was:

```text
P25 host orchestration blocked: reviewed and retained diagnostic wrappers differ
```

Including its terminating newline, it is 81 bytes with SHA-256
`3672bba450cc04d67c90d393864596a80e1c9216c91db14b8e7fcf858e586feb`.
The host wrapper exited `1` before the first `p23-trace-off-a` invocation.

Both wrapper copies were 5,097,711 bytes with SHA-256
`56983629e935e3e18ac89b894ee84bc56e47f862abfcc36a6a5a15dbb8e5a354`.
The retained copy was `root:root`, mode `0600`, while the comparison ran as the
unprivileged host user. Thus the generic error text does not establish a
content mismatch; it records a failed readability-dependent comparison. The
unprivileged `cmp` exit was `2`, while the privileged comparison exit was `0`.

The stop occurred before the bridge reconstruction and bridge-verification
commands, so their four expected stdout/stderr logs are absent. The complete
expected acquisition inventory is also empty: no native or sanitized
`trace-off-a`, `trace-off-b`, repeatability, `trace-on`, raw-trace,
noninterference, or aggregate artifact exists; none of the six P23 acquisition
process-stream labels exists; and no P23 sanitizer process ran.

| gate | outcome |
| --- | --- |
| replacement image/runtime | completed |
| corrected diagnostic | passed `40/40` |
| corrected sanitizer | passed; full wrapper committed |
| acquisition host preflight | failed closed on unreadable retained wrapper |
| trace-off A | not started |
| trace-off B / repeatability | not run |
| trace-on / noninterference | not run |
| candidate observations | `0` |
| fidelity aggregate / training | not run |

The machine-readable record enumerates all ten expected native acquisition
artifact names and all six expected acquisition process-stream labels, with
empty observed lists and zero bridge-reconstruction/verification log count.

## Scope and route

Attempt `20260906-01` is terminal. Reusing it or rerunning it until a favorable
outcome is forbidden. A new preregistered attempt must make the retained
wrapper readable to its exact authentication step, or explicitly authorize a
privileged authentication step, without weakening any frozen P23 threshold.
No next branch name is assigned by this packet.

The P18--P21 analytic and finite-precision results and P24's terminal outcome
are unaffected. P25 establishes neither CUDA repeatability, observer
noninterference, candidate fidelity, native CUDA shield safety, nor training
quality.

## Machine-readable record

The compact
[`p25_cuda_diagnostic_outcome.json`](p25_cuda_diagnostic_outcome.json) binds
the repository-authenticated diagnostic and explicitly labels the external
operator-recorded facts. Reconstruct the repository facts and packet's
internal consistency with:

```bash
python scripts/reconstruct_p25_cuda_diagnostic_outcome.py
```
