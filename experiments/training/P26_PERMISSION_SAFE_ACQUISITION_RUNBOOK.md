# P26 permission-safe CUDA acquisition runbook

## Evidence status and immutable boundary

This is a **post-runtime, pre-diagnostic, pre-acquisition one-shot runbook**.
It records one prepared and reviewed P26 runtime, but no P26 diagnostic,
sanitized bridge, permission-safe bridge pass, CUDA gradient, repeatability
result, observer noninterference result, candidate-fidelity result, shield
execution, or training result.

P25 attempt `20260906-01` is terminal. It stopped before `trace_off_a` because
an unprivileged host `cmp` could not read the retained `root:root 0600`
sanitized wrapper. Its exact diagnostic result and zero-observation outcome
remain frozen at commit `f055405cc879ba0ac5afe26bc34a7336d5d2efbf`, tree
`a5ee62de68b091719938829e579853b226b7f086`. Never resume, modify, copy into,
or rerun that attempt.

P26 permits exactly one fresh attempt: `20260906-02`. Source-freeze commit
`bc2c84880a7e6f3e762d7a05dfbd5048772b2c69`, tree
`7332be7fdd15a1dff2374c10ae3c4555b7f60fec`, was frozen before that runtime
was prepared. During post-preparation review, two control-order discrepancies
were found:

1. the control-parent check did not yet represent a post-runtime correction
   commit descending directly from the pre-runtime source freeze; and
2. the executable order placed independent P25 reconstruction after the
   image/no-acquisition-state checks instead of the declared
   reconstruction-first order.

Both discrepancies fail closed. No diagnostic, sanitized bridge,
permission-safe bridge, `trace_off_a`, gradient, or candidate observation ran.
The final P26 control commit must be exactly one direct child of `bc2c848...`
and must use contract status
`frozen_post_fresh_runtime_pre_diagnostic_and_pre_acquisition`.

Attempt `20260906-02` remains valid because its runtime was created solely
from the frozen P25 preregistration commit
`e76ab62f92c95e6f0716cf2f1ed38a583cadfe56` and the already-frozen P25
inputs. The corrected P26 control is not part of the image or acquisition
source checkout; it changes only the diagnostic-to-acquisition bridge and the
order of fail-closed host checks. It does not change the image, container,
runtime lock, host attestation, P23 runner, model, data, seed, observer, or a
scientific gate.

At acquisition time, the committed diagnostic wrapper and retained `0600`
wrapper will be read and compared inside the logged root process of the
reviewed attested container. The retained sanitized and native evidence remain
`root:root 0600`; no permission change is part of the correction.

Read and independently reconstruct
`p26_permission_safe_acquisition_contract.json` before any diagnostic or
acquisition work.
The P21 fidelity gates, P22 protocol and erratum, P23 CUDA addendum, P23
runner, observer, seed, model, data, 256-step schedule, 24 capture steps, and
all thresholds remain byte-for-byte authorities. P26 does not change a
scientific gate.

## 1. Freeze the post-runtime correction and stage two separate checkouts

The fresh acquisition uses two clean checkouts:

1. A detached P25 acquisition-source checkout at exactly
   `e76ab62f92c95e6f0716cf2f1ed38a583cadfe56`, tree
   `4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2`. Its frozen P25 script already
   ran `prepare-runtime` once and may now run only `run-diagnostic`; its
   `prepare-runtime` and `run-acquisition` phases are both forbidden from this
   point forward.
2. A separate clean P26 control checkout at the final reviewed post-runtime
   correction commit. That commit must be one direct child of
   `bc2c84880a7e6f3e762d7a05dfbd5048772b2c69`, which is itself one direct
   child of the terminal P25 outcome. It supplies the corrected acquisition
   orchestrator, root-container verifier, contract, and independent
   reconstructor.

Before transfer, run locally:

```bash
python3 scripts/reconstruct_p26_permission_safe_acquisition.py \
  --canonical experiments/training/p26_permission_safe_acquisition_contract.json
uv run --locked pytest -q tests/test_p26_permission_safe_bridge.py
bash -n scripts/run_p26_permission_safe_acquisition.sh
git diff --check
```

Commit the final contract, reconstructor, orchestrator, verifier, tests, and
this runbook once as the post-runtime correction. Do not amend `bc2c848...`.
Transfer the reviewed final control commit by a credential-free mechanism such
as a verified Git bundle. On the A100 host, choose explicit paths and prove
both checkouts are exact and clean:

```bash
set -euo pipefail

export P25_SOURCE_REPO=/secure/p25/source-e76ab62
export P26_CONTROL_REPO=/secure/p26/OptimizationML-control

test "$(git -C "$P25_SOURCE_REPO" rev-parse HEAD)" = \
  e76ab62f92c95e6f0716cf2f1ed38a583cadfe56
test "$(git -C "$P25_SOURCE_REPO" rev-parse HEAD^{tree})" = \
  4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2
test -z "$(git -C "$P25_SOURCE_REPO" status --porcelain=v1 --untracked-files=all)"

export P26_CONTROL_HEAD=REPLACE_WITH_REVIEWED_40_HEX_CONTROL_COMMIT
export P26_CONTROL_TREE=REPLACE_WITH_REVIEWED_40_HEX_CONTROL_TREE
test "$(git -C "$P26_CONTROL_REPO" rev-parse HEAD)" = "$P26_CONTROL_HEAD"
test "$(git -C "$P26_CONTROL_REPO" rev-parse HEAD^{tree})" = "$P26_CONTROL_TREE"
test "$(git -C "$P26_CONTROL_REPO" rev-list --parents -n 1 "$P26_CONTROL_HEAD")" = \
  "$P26_CONTROL_HEAD bc2c84880a7e6f3e762d7a05dfbd5048772b2c69"
test "$(git -C "$P26_CONTROL_REPO" rev-parse bc2c84880a7e6f3e762d7a05dfbd5048772b2c69^{tree})" = \
  7332be7fdd15a1dff2374c10ae3c4555b7f60fec
test -z "$(git -C "$P26_CONTROL_REPO" status --porcelain=v1 --untracked-files=all)"
```

Replace the two P26 placeholders with reviewed full object IDs before running
the diagnostic or acquisition. Do not derive them from an unreviewed live
checkout.

## 2. Authenticate the already-prepared fresh runtime

Do **not** run `prepare-runtime` again. It already ran exactly once for attempt
`20260906-02` from the original pinned nanoGPT, Muon, materialized FineWeb,
and frozen P25 source inputs. Export the unchanged P25 interface for the later
single diagnostic:

```bash
export P25_ATTEMPT_ID=20260906-02
export P25_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P25_NANOGPT_HOST=/secure/p23/nanoGPT
export P25_MUON_HOST=/secure/p23/Muon
export P25_DATA_HOST=/secure/p23/optimizationml-p22-data
export P25_HISTORICAL_WRAPPER="$P25_SOURCE_REPO/results/summaries/p25_historical_p24_native_outcome.json"
export P25_BOOTSTRAP_COMMIT=5814621979ffedf370f04f3d99d925d8584b640f
export P25_BOOTSTRAP_TREE=31cafc8bae9a7311b18940c267a4da52ed6e8068
export P25_PREREG_COMMIT=e76ab62f92c95e6f0716cf2f1ed38a583cadfe56
export P25_EXPECTED_CONTRACT_SHA256=51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2
export P25_EXPECTED_RUNTIME_LOCK_SHA256=04be2154daf5afbe97f7d2278ee7936da769d230663dfc6dfd50f0370a456755
export P25_EXPECTED_HOST_ATTESTATION_SHA256=8caa4d761ca89741517e6292e174acd17948f4d312b8418d0056070d3247e046
export P25_DIAGNOSTIC_HEAD=ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0
export P25_DIAGNOSTIC_TREE=e7cdf25c86360ecdd42f2dc188ec416744321fe1
```

The completed runtime review binds these exact facts:

| Runtime fact | Reviewed value |
| --- | --- |
| Attempt ID | `20260906-02` |
| Image digest | `sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0` |
| Container ID | `e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c` |
| Container init PID | `39227` |
| Container restart count | `0` |
| Build-metadata SHA-256 | `13e0f05189321f86b4d8a41253eed97768bb1774ac7db64ad7c8b25e2f026091` |
| Runtime-lock SHA-256 | `04be2154daf5afbe97f7d2278ee7936da769d230663dfc6dfd50f0370a456755` |
| Host-attestation SHA-256 | `8caa4d761ca89741517e6292e174acd17948f4d312b8418d0056070d3247e046` |
| Remote runtime-review commit | `ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0` |
| Remote runtime-review tree | `e7cdf25c86360ecdd42f2dc188ec416744321fe1` |
| Runtime-review parent | `e76ab62f92c95e6f0716cf2f1ed38a583cadfe56` |

The runtime-review commit added only these two paths in the fresh attempt
checkout:

```text
experiments/training/p25_cuda_runtime_lock.json
experiments/training/p25_host_attestation.json
```

That commit is the value named `P25_DIAGNOSTIC_HEAD` by the frozen P25
interface even though the diagnostic itself has not run. It is a direct child
of `e76ab62...`, and its exact name-status delta contains only those two
additions. The reviewed runtime remains bound to the full A100 UUID, ten bind
mounts, one locked `/tmp` tmpfs, `network=none`, and a read-only root
filesystem. The final P26 orchestrator must recheck every bound fact above,
including live container ID, init PID, restart count, image, lock, and
attestation, before acquisition.

## 3. Run one fresh corrected diagnostic and freeze its bridge

The exact reviewed runtime/commit values are already exported in Section 2.
First confirm that the remote attempt checkout is still the reviewed runtime
commit and that no diagnostic artifact or diagnostic process log exists;
then make the one permitted diagnostic invocation:

```bash
export P25_ATTEMPT_ROOT=/secure/p25/attempt-20260906-02
export P25_HOST_REPO="$P25_ATTEMPT_ROOT/OptimizationML"
export P25_HOST_EVIDENCE="$P25_ATTEMPT_ROOT/evidence"

test "$(git -C "$P25_HOST_REPO" rev-parse HEAD)" = \
  ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0
test "$(git -C "$P25_HOST_REPO" rev-parse HEAD^{tree})" = \
  e7cdf25c86360ecdd42f2dc188ec416744321fe1
test ! -e "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.native.json"
test ! -e "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.sanitized.json"
test ! -e "$P25_HOST_EVIDENCE/p25-diagnostic.stdout.log"
test ! -e "$P25_HOST_EVIDENCE/p25-diagnostic.stderr.log"
test ! -e "$P25_HOST_EVIDENCE/p25-diagnostic-sanitizer.stdout.log"
test ! -e "$P25_HOST_EVIDENCE/p25-diagnostic-sanitizer.stderr.log"

bash "$P25_SOURCE_REPO/scripts/run_p25_cuda_attempt.sh" run-diagnostic
```

This is the sole P26 diagnostic invocation. It must produce fresh external
native and sanitized files, and it must copy the sanitized bytes into the
fresh attempt checkout at:

```text
results/summaries/p25_executable_origin_diagnostic.sanitized.json
```

If the diagnostic or sanitizer fails, attempt `20260906-02` is terminal. Do
not repair or rerun it.

On success, review the native diagnostic, sanitized wrapper, exact hashes,
sizes, modes, and logs. The retained sanitized and native files must remain
`root:root 0600`. Do not issue `chmod`, `chown`, privileged `cmp`, or any other
manual permission workaround. Commit only the repository wrapper above. The
bridge commit must be one direct child of the diagnostic commit, and its exact
name-status delta must be that single added path. Record the full bridge
commit and tree as `P25_BRIDGE_HEAD` and `P25_BRIDGE_TREE`. Do not rerun the
diagnostic after this commit.

## 4. Bind the P26 control plane

Use these reviewed byte bindings for the frozen P26 sources:

```bash
export P26_ATTEMPT_ID=20260906-02
export P26_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P26_EXPECTED_CONTRACT_SHA256=1a3562f90671384c5b90f4e830f3342ba993381c03738cd81685803563c538e1
export P26_EXPECTED_ORCHESTRATOR_SHA256=66dd6889912e0abfc7782e790356222851cc753d237018c6263942d3a7a4290a
export P26_EXPECTED_RECONSTRUCTOR_SHA256=27b1b5f65438f58f8c7733115c4d01f7965a583e9e809e28571770886040b246
export P26_EXPECTED_VERIFIER_SHA256=2225a9822076e0055d8017f8b8549eb9e937434d4789463adb2fcec26ee3756a
export P26_DIAGNOSTIC_HEAD="$P25_DIAGNOSTIC_HEAD"
export P26_DIAGNOSTIC_TREE="$P25_DIAGNOSTIC_TREE"
export P26_BRIDGE_HEAD=REPLACE_WITH_REVIEWED_40_HEX_BRIDGE_COMMIT
export P26_BRIDGE_TREE=REPLACE_WITH_REVIEWED_40_HEX_BRIDGE_TREE
```

The contract and reconstructor hashes above are the reviewed post-runtime v2
bytes. Before acquisition, replace both bridge placeholders with reviewed
values and rerun the independent P26 reconstruction in the clean control
checkout. The final control commit must be one direct child of
`bc2c848...`, and the contract status must be post-runtime,
pre-diagnostic/pre-acquisition—exactly
`frozen_post_fresh_runtime_pre_diagnostic_and_pre_acquisition`. If a source or
contract hash differs, stop and freeze a new protocol; never update an
expected hash merely to match unreviewed bytes.

The P26 orchestrator will independently require:

- the exact clean P26 control commit and tree;
- all four P26 SHA-256 bindings above;
- the exact fresh P25 preregistration → diagnostic → bridge history;
- the retained BuildKit digest and matching live container image;
- no preexisting P23 acquisition artifact or P23 process log; and
- successful independent reconstruction of both the P26 and P25 contracts.

## 5. Execute the permission-safe bridge and unchanged acquisition once

Invoke only the P26 acquisition entry point:

```bash
bash "$P26_CONTROL_REPO/scripts/run_p26_permission_safe_acquisition.sh" \
  run-acquisition
```

Do **not** invoke the frozen P25 `run-acquisition` phase. Do not rerun this
command after any exit, including a bridge or preflight exit.

Before `trace_off_a`, the P26 script rehashes the verifier immediately before
streaming its reviewed bytes to `/opt/p23-venv/bin/python -` inside the fresh
root container. Its no-overwrite stdout/stderr transcript is retained under
the prefix `p26-permission-safe-bridge-verification`. The verifier:

1. rejects missing, symlink, or nonregular wrapper/native inputs;
2. stable-reads the committed and retained wrappers once each;
3. requires repository mode `0644` and retained `root:root 0600`;
4. requires exact equality of the captured wrapper byte strings;
5. strict-parses both captured byte strings independently;
6. requires retained native `root:root 0600` and exact wrapper-bound SHA-256
   and size;
7. reconstructs the native pass semantics through the frozen P25 sanitizer;
   and
8. rehashes every P25 contract source.

The verifier performs no mutation. Failure bars `trace_off_a`.

Only after the verifier passes, the exact frozen P23 sequence runs with fresh,
no-overwrite files:

1. `trace_off_a`;
2. `trace_off_b` only after A succeeds;
3. exact `verify-repeatability`;
4. `trace_on` only after zero exact off-A/off-B mismatches;
5. exact `verify-noninterference`;
6. `aggregate` only after zero off-A/trace-on mismatches; and
7. sanitization of every success artifact, or of the first failure artifact
   before stopping.

Exact equality is mandatory; `allclose` and numeric tolerances are forbidden.
A successful trace-on must contain exactly `48 × 24 = 1,152` actual stored
post-aspect BF16 CUDA candidates, covering all 48 intended Muon matrices and
`84,934,656/84,934,656` elements. The unchanged P21 hard, mild-intervention,
P16/P18 shaping, cosine, correction, amplitude, effective-gain, and annulus
gates apply exactly as frozen. No post-observation tuning is permitted.

## 6. Retain the first outcome and route mechanically

Retain the new build metadata, runtime lock, host attestation, native and
sanitized diagnostic, committed wrapper, bridge transcript, every process
stdout/stderr log, all native and sanitized P23 artifacts, and their hashes.
Record only the first terminal outcome:

| First terminal condition | Required route |
| --- | --- |
| Permission-safe bridge fails | Stop before `trace_off_a`; retain the first bridge transcript and investigate the bridge/runtime binding without reusing the attempt. |
| `trace_off_a` fails | Stop and retain its native and sanitized failure evidence. |
| `trace_off_b` fails | Stop and retain completed A plus B failure evidence. |
| Exact repeatability fails | Bar `trace_on`; localize the first CUDA operation-level difference in a new preregistered branch. |
| Exact noninterference fails | Bar aggregation; redesign the observer in a new preregistered branch. |
| Equality passes but a frozen fidelity gate fails | Record the failure without tuning; preregister a candidate adapter and evaluate it on a held-out seed. |
| Every frozen gate passes | Record the real-gradient shadow-trace result; proceed to a native CUDA sector shield and matched shielded training. |

There is no cleanup, resume, retry-until-favorable, permission workaround, or
same-attempt second diagnostic/acquisition path. CPU CI and contract replay
remain protocol evidence only and cannot be cited as execution on the A100.
