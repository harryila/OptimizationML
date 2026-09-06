# P23 deterministic-CUDA shadow-trace result

## Verdict

The operator-recorded P23 attempt **blocked before step zero at the
executable-origin provenance gate**. The first and only frozen `trace_off_a`
invocation is recorded as launched against the committed A100 runtime lock and
as exiting with code `2` when its initialized loaded-file closure reported
`_remote_module_non_scriptable.py` on the writable `/tmp` tmpfs. No
trace-off manifest was written, so trace-off B, exact repeatability, trace-on,
observer noninterference, and fidelity aggregation were not run.

This is not the requested real-gradient fidelity result. The operator record
is consistent with the preregistered provenance boundary failing closed rather
than silently accepting an executable origin on a writable mount.

## Locked attempted execution

Operator checks immediately before the attempt recorded clean OptimizationML
head `f89ea1f` and tree
`778661ea834294a4056ec5f36d0f1e2d5fcf9840`. Immediately before acquisition,
the live container ID, init PID, immutable image digest, A100 UUID, and
namespace-entered mountinfo hash all matched the committed runtime lock:

- container `9932b8b28bb3686c184b1e54dfb859bad716772168a710a870024bda384e4954`;
- image digest `sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1`;
- GPU `GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d`, an A100-SXM4-40GB; and
- mountinfo SHA-256
  `b1f509d303abb8ac4552d68c9357f9b9be8caffbe4204f3ac04f6c6c95e27381`.

The retained Docker event records exec ID
`e4dbbf141b6fa9c63a0334f0c6b72a932a8cf8a893e766077b841cb71a8a7728`
dying with exit code `2` at `2026-09-06T08:26:15Z`. The operator-client
transcript reports:

```text
P23 blocked: executable origin lies on a writable mount: tmpfs:tmp1s2eyibl/_remote_module_non_scriptable.py
```

The runner's checked-in ordering and its parameter-count output place the
reported failure after model/optimizer construction but before the initial
state hash, first batch, forward/backward pass, or optimizer step. The operator
record therefore reports zero gradient observations and zero candidates. A
post-failure directory listing found only the already committed runtime lock
and host attestation.

The failed runner did not have a stage-aware failure-manifest path. The exact
operator-client streams and Docker event are transcribed into the compact
outcome record, with the explicit qualification that the streams were not
hash-bound at execution time.

## Causal boundary

P23 retains no rule-7-complete, native causal-localization artifact. The
reported generated-module filename is enough to route the next investigation,
but not enough to attribute the stop to a particular import chain. The
nanoGPT training path, FineWeb batch, Muon candidate arithmetic, shadow
observer, and CUDA determinism were not reached or tested.

The stop does not justify allowlisting writable `/tmp`, weakening the
loaded-file closure, or rerunning until a favorable result appears. P24 must
first reproduce and localize the initialization path with a committed,
hash-bound diagnostic. Any image-level remedy then requires a new digest and
runtime lock.

## Gate status and route

| Gate | Outcome |
| --- | --- |
| trace-off A | blocked during initialized loaded-file closure |
| trace-off B | not run |
| exact baseline repeatability | not run |
| trace-on | mechanically barred |
| observer noninterference | not run |
| 1,152-observation aggregate | not run |

The failure occurs before all four outcomes in P23's preregistered post-run
routing table. The evidence therefore opens
`p24-cuda-executable-origin-hardening`: reproduce and localize the implicated
initialization path, apply an image-resident remedy without relaxing the
general writable-origin rejection, build a new digest-pinned image, refreeze
the runtime, and only then begin a new acquisition.

The P18--P21 mathematical results remain unchanged. P23 establishes neither
CUDA repeatability nor candidate fidelity, observer noninterference, training
quality, native shield correctness, or stability of unrepaired Muon.

## Machine-readable record

The compact operator outcome record is
[`p23_cuda_shadow_trace_outcome.json`](p23_cuda_shadow_trace_outcome.json).
Its static hashes match the historical source tree and frozen runtime
artifacts, while its execution fields and stage-legal absences remain
operator-recorded. It is not a
mathematical certificate, a native acquisition manifest, cryptographic remote
attestation, or causal-localization artifact.

Independently replay its static bindings with:

```bash
uv run --locked python scripts/reconstruct_p23_cuda_shadow_trace_outcome.py
```
