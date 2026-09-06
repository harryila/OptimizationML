# P27 CUDA deleted-mapping localization runbook

## Evidence boundary

This runbook governs one **fresh, no-training localization attempt**. It does
not authorize another P23 scientific acquisition. It may initialize CUDA,
construct GPT-2 small, move that model to the pinned A100, and construct the
pinned optimizer solely to take four process-map snapshots. It may not load
training data, execute forward or backward, evaluate or capture a Muon
candidate, call an optimizer step, apply a sector shield, or update a
parameter.

P26 attempt `20260906-02` is terminal and must not be resumed, inspected as a
live localization target, restarted, or mutated. Its compact outcome is
commit `5429da23ff18888daa2312c530a4587780484d8b`, tree
`a5b42f820beb7b4be0b65d89c51d49c1a5436647`, annotated tag object
`bacad707d3779bfa10957e18cb4c69b1a7f0cbce`, and outcome SHA-256
`9be5727213f1a23f00b05524940d88c2d02461510fb61316b29dacc6ee8ead47`.
That attempt retained only the rejected `/proc/self/maps` line number, not its
bytes. Never guess the mapping from a later process.

Read and independently reconstruct
`p27_cuda_deleted_mapping_localization_contract.json` before any remote
operation. Every source-hash placeholder in that contract must have been
replaced with a reviewed 64-hex hash and every corresponding placeholder flag
must be `false`. The only host entry point is the reviewed
`scripts/run_p27_cuda_deleted_mapping_localization.sh`, whose only accepted
phases are `prepare-runtime` and `run-localization`. It has no cleanup, resume,
or retry phase.

## 1. Freeze P27 sources before creating a container

The first phase is a normal Git source freeze. It includes the contract,
no-training localizer, host orchestrator, narrow P23 failure-sanitizer
correction, separate P27 localization sanitizer, reconstructors, tests, CI
workflow, and this runbook.
Run locally:

```bash
python3 scripts/reconstruct_p26_permission_safe_cuda_acquisition_outcome.py
python3 scripts/reconstruct_p27_cuda_deleted_mapping_localization.py
uv run --locked pytest -q \
  tests/test_p27_cuda_deleted_mapping_localization_contract.py \
  tests/test_p27_cuda_deleted_mapping_localization.py \
  tests/test_p27_cuda_deleted_mapping_localization_orchestration.py \
  tests/test_p27_cuda_deleted_mapping_localization_sanitizer.py \
  tests/test_p27_p26_failure_ingestion.py \
  tests/test_p23_deterministic_cuda_shadow_trace.py
bash -n scripts/run_p27_cuda_deleted_mapping_localization.sh
uv run --locked ruff format --check \
  experiments/training scripts tests
uv run --locked ruff check experiments/training scripts tests
git diff --check
```

Commit and review those bytes before creating the P27 attempt root or
container. Record the full commit and tree as `P27_SOURCE_FREEZE_COMMIT` and
`P27_SOURCE_FREEZE_TREE`. These values are selected after the commit; they are
not self-referential fields inside the JSON contract. Transfer that exact
commit to a separate clean control checkout, for example
`/secure/p27/control-source`, using a credential-free, hash-checked Git bundle.

From that reviewed clean checkout, export the complete source authority for
the host orchestrator. Record the literal values in the operator log before
proceeding:

```bash
export P27_CONTROL_REPO=/secure/p27/control-source
export P27_SOURCE_FREEZE_COMMIT="$(git -C "$P27_CONTROL_REPO" rev-parse HEAD)"
export P27_SOURCE_FREEZE_TREE="$(git -C "$P27_CONTROL_REPO" rev-parse HEAD^{tree})"
export P27_EXPECTED_CONTRACT_SHA256="$(sha256sum "$P27_CONTROL_REPO/experiments/training/p27_cuda_deleted_mapping_localization_contract.json" | cut -d' ' -f1)"
export P27_EXPECTED_ORCHESTRATOR_SHA256="$(sha256sum "$P27_CONTROL_REPO/scripts/run_p27_cuda_deleted_mapping_localization.sh" | cut -d' ' -f1)"
export P27_EXPECTED_RECONSTRUCTOR_SHA256="$(sha256sum "$P27_CONTROL_REPO/scripts/reconstruct_p27_cuda_deleted_mapping_localization.py" | cut -d' ' -f1)"
export P27_EXPECTED_LOCALIZER_SHA256="$(sha256sum "$P27_CONTROL_REPO/experiments/training/run_p27_cuda_deleted_mapping_localization.py" | cut -d' ' -f1)"
export P27_EXPECTED_INGESTER_SHA256="$(sha256sum "$P27_CONTROL_REPO/scripts/ingest_p26_trace_off_a_failure.py" | cut -d' ' -f1)"
export P27_EXPECTED_P23_CORE_SHA256="$(sha256sum "$P27_CONTROL_REPO/experiments/training/p23_deterministic_cuda_shadow_trace.py" | cut -d' ' -f1)"
export P27_EXPECTED_P27_SANITIZER_SHA256="$(sha256sum "$P27_CONTROL_REPO/scripts/sanitize_p27_cuda_deleted_mapping_localization.py" | cut -d' ' -f1)"
```

The orchestrator also checks that the finalized contract carries the same
execution-source hashes; these environment variables cannot select different
scientific authorities.

The repository mounted inside the localization container is **not** this
control checkout. It is a separate clean authority checkout at exactly:

```text
commit 185e444afc0b44ca0a09b1bde49a6b6fa3973355
tree   24f4bdac331a57bd7c1b807747d7c7fba253ee5a
```

That preserves the frozen P25 preflight and scientific authorities. The P27
diagnostic bytes enter through the already-counted writable native-evidence
mount after an exact source/destination hash check. Do not add an eleventh
mount and do not mount the current working repository at
`/workspace/OptimizationML`.

## 2. Prepare one fresh localization runtime without observing mappings

Use these fixed identities:

```bash
export P27_ATTEMPT_ID=20260906-01
export P27_ATTEMPT_ROOT=/secure/p27/attempt-20260906-01
export P27_CONTAINER=p27-localization-20260906-01
export P27_IMAGE=localhost:5000/p25-runtime@sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0
export P27_IMAGE_DIGEST=sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0
export P27_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P27_AUTHORITY_REPO=/secure/p27/authority-185e444
export P27_CONTROL_REPO=/secure/p27/control-source
export P27_NANOGPT_HOST=/secure/p23/nanoGPT
export P27_MUON_HOST=/secure/p23/Muon
export P27_DATA_HOST=/secure/p23/optimizationml-p22-data
export P27_HOST_EVIDENCE=/secure/p27/attempt-20260906-01/evidence
```

Before creating anything, prove that both the attempt root and container name
are unused. Also prove that the terminal P26 container ID
`e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c`
is not the selected runtime. Stop if either fresh target already exists.

The authority checkout must be detached, clean, and exact:

```bash
test "$(git -C "$P27_AUTHORITY_REPO" rev-parse HEAD)" = \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
test "$(git -C "$P27_AUTHORITY_REPO" rev-parse HEAD^{tree})" = \
  24f4bdac331a57bd7c1b807747d7c7fba253ee5a
test -z "$(git -C "$P27_AUTHORITY_REPO" status --porcelain=v1 --untracked-files=all)"
```

Create the new container from the exact retained image digest. Reproduce the
P26 runtime's deterministic environment, `network=none`, read-only rootfs,
locked `/tmp` tmpfs, full non-MIG GPU UUID, and exactly these ten bind-mount
roles:

1. clean authority repository, read-only;
2. pinned nanoGPT, read-only;
3. pinned Muon, read-only;
4. materialized data, read-only (present for runtime parity, never read);
5. preprocessor alias, read-only;
6. new P27 native-evidence directory, writable;
7. image inspection, read-only;
8. running-container inspection, read-only;
9. running mountinfo, read-only; and
10. `nvidia-smi` query, read-only.

Use the exact environment values from the P26 runtime lock: PyTorch
`2.7.0+cu128`, CUDA runtime `12.8`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`PYTHONHASHSEED=1337`, deterministic algorithms, SDPA math backend, TF32 and
reduced-precision reductions disabled, one CPU/inter-op thread, and
`PYTHONDONTWRITEBYTECODE=1`.

Freeze the runtime with the unchanged P23 `freeze-runtime` entry point from
the authority checkout. Write new outputs only to the writable evidence mount
as:

```text
p27_cuda_runtime_lock.json
p27_host_attestation.json
```

The sole authorized entry point for this preparation is:

```bash
bash "$P27_CONTROL_REPO/scripts/run_p27_cuda_deleted_mapping_localization.sh" prepare-runtime
```

Do not execute the P27 localizer in this phase. Review the image identity,
new container ID and init PID, zero restart count, GPU identity, ten mounts,
runtime lock, and host attestation. The new container ID must differ from the
terminal P26 ID.

Copy the two reviewed runtime artifacts into the clean P27 control checkout at
exactly:

```text
experiments/training/p27_cuda_runtime_lock.json
experiments/training/p27_host_attestation.json
```

Commit them together. The runtime-review commit must be one direct child of
`P27_SOURCE_FREEZE_COMMIT`, and its complete name-status delta must be:

```text
A	experiments/training/p27_cuda_runtime_lock.json
A	experiments/training/p27_host_attestation.json
```

Record the reviewed commit and tree as `P27_RUNTIME_REVIEW_COMMIT` and
`P27_RUNTIME_REVIEW_TREE`. No mapping observation, localizer process, model,
optimizer, gradient, or candidate may exist before that commit is reviewed.

Export and independently review the exact phase-two authority:

```bash
export P27_RUNTIME_REVIEW_COMMIT="$(git -C "$P27_CONTROL_REPO" rev-parse HEAD)"
export P27_RUNTIME_REVIEW_TREE="$(git -C "$P27_CONTROL_REPO" rev-parse HEAD^{tree})"
export P27_EXPECTED_RUNTIME_LOCK_SHA256="$(sha256sum "$P27_CONTROL_REPO/experiments/training/p27_cuda_runtime_lock.json" | cut -d' ' -f1)"
export P27_EXPECTED_HOST_ATTESTATION_SHA256="$(sha256sum "$P27_CONTROL_REPO/experiments/training/p27_host_attestation.json" | cut -d' ' -f1)"
```

## 3. Stage hash-bound diagnostic bytes through the existing evidence mount

From the reviewed runtime-control checkout, authenticate:

```text
experiments/training/run_p27_cuda_deleted_mapping_localization.py
```

against the finalized contract SHA-256. Copy those exact bytes once to:

```text
/secure/p27/attempt-20260906-01/evidence/run_p27_cuda_deleted_mapping_localization.py
```

Reject a preexisting destination. Immediately compare the source and
destination SHA-256 and byte count. This file becomes visible inside the
container at:

```text
/workspace/evidence/p23/run_p27_cuda_deleted_mapping_localization.py
```

This uses the existing native-evidence mount; do not add or alter a mount.
The P25 source is supplied separately and read-only from the authority
checkout:

```text
/workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py
SHA-256 6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770
```

Revalidate the live container ID, PID, restart count, image digest, GPU UUID,
mount inventory, runtime lock, host attestation, clean authority repository,
and clean reviewed P27 control commit immediately before execution.

Before any GPU initialization or model allocation, use the narrowly corrected
P23 sanitizer source with exact SHA-256
`ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab`
through the reviewed ingester
`scripts/ingest_p26_trace_off_a_failure.py`, SHA-256
`3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb`.
Its regression test is `tests/test_p27_p26_failure_ingestion.py`, SHA-256
`7846fbbe9c1768e52467e1178075595fdf979d600f2f7891a0aec9feffc43d41`.
Ingest the already-terminal P26 native failure through a one-read root bridge
whose outputs are fresh under the P27 evidence root, never inside the terminal
P26 directory. Require native SHA-256
`b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91`,
66,283 bytes, `root:root`, mode `0600`. The correction may map only the exact
already-validated `/opt/p23-venv/bin/python` to
`python_environment:bin/python`; arbitrary symlink escapes and unknown paths
remain fatal. Never mutate or reinterpret the P26 native bytes. If ingestion
fails, stop before invoking the localizer.

## 4. Run the sole no-training localization invocation

The localizer's exact output schema is
`passive-muon-p27-cuda-deleted-mapping-localization-v1`. Its native output is
new-only:

```text
/workspace/evidence/p23/p27_cuda_deleted_mapping_localization.native.json
```

The host orchestrator expands its `run-localization` phase to exactly this
container invocation after it ingests the P26 failure and revalidates every
authority. Do not invoke this inner command directly:

```bash
sudo docker exec \
  --env CUDA_VISIBLE_DEVICES="$P27_GPU_UUID" \
  --env NVIDIA_VISIBLE_DEVICES="$P27_GPU_UUID" \
  "$P27_CONTAINER" \
  /opt/p23-venv/bin/python \
  /workspace/evidence/p23/run_p27_cuda_deleted_mapping_localization.py \
  --repository /workspace/OptimizationML \
  --nanogpt-root /workspace/inputs/nanoGPT \
  --muon-source /workspace/inputs/muon/muon.py \
  --p25-source /workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py \
  --runtime-lock /workspace/evidence/p23/p27_cuda_runtime_lock.json \
  --host-attestation /workspace/evidence/p23/p27_host_attestation.json \
  --p25-contract /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
  --expected-source-sha256 "$P27_EXPECTED_LOCALIZER_SHA256" \
  --expected-p25-source-sha256 6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770 \
  --expected-p25-contract-sha256 51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2 \
  --expected-runtime-lock-sha256 "$P27_EXPECTED_RUNTIME_LOCK_SHA256" \
  --expected-host-attestation-sha256 "$P27_EXPECTED_HOST_ATTESTATION_SHA256" \
  --expected-repository-head 185e444afc0b44ca0a09b1bde49a6b6fa3973355 \
  --expected-repository-tree 24f4bdac331a57bd7c1b807747d7c7fba253ee5a \
  --output /workspace/evidence/p23/p27_cuda_deleted_mapping_localization.native.json
```

Execute that expansion only through the reviewed one-shot phase:

```bash
bash "$P27_CONTROL_REPO/scripts/run_p27_cuda_deleted_mapping_localization.sh" run-localization
```

The orchestrator captures stdout and stderr to fresh no-overwrite host files.
Any exit, missing output, schema error, unexpected write, or preexisting
artifact makes this sole attempt terminal.

The process must snapshot, in order:

1. `pre_cuda`;
2. `post_cuda_init`;
3. `post_model_move`; and
4. `post_optimizer`.

Every snapshot binds the raw `/proc/self/maps` bytes and records every mapping
with exact line number, address interval, permissions, offset, device/inode,
lossless pathname bytes, deletion flag, classification, `/proc/self/map_files`
probe, and matching `/proc/self/fd` probes. Probe failures are evidence and
must retain exact errno; they are never silently discarded. The process must
rehash `/proc/self/maps` after its probes. Unrelated mapping churn is retained
evidence rather than an automatic failure, but disappearance or identity
change of any deleted target is fatal to a completed diagnostic. A
conclusively identified character-device mapping is classified as such; it is
not collapsed into `deleted_unclassifiable`. The process must
terminate after `post_optimizer` with zero data loads, forward/backward calls,
optimizer steps, parameter updates, gradient observations, and candidate
observations.

Do not run the command again, including after nonreproduction or a tooling
failure.

## 5. Sanitize P27 evidence without weakening provenance

After the sole localizer invocation, sanitize its manifest on the host-side
root bridge with `sudo /usr/bin/python3` and the separately reviewed
P27-specific sanitizer; never run this sanitizer inside the container. Its
source SHA-256 is
`3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4`.
Its focused test SHA-256 is
`a3befd5a6968f7dcb83d9ecfdba2c84e5457d0a83f4cfff0b68aa5d9585ad089`.
The native artifact remains external with owner `root:root` and mode `0600`.
The repository-safe wrapper must bind its SHA-256, byte count, uid,
gid, and mode; preserve all non-path evidence and classifications; replace
every raw-line, pathname, readlink-target, argv, and other absolute-path byte
value with SHA-256 plus byte count and a declared logical label when
applicable; and publish no native absolute-path bytes. Duplicate keys,
nonfinite values, unknown schemas, unknown absolute paths, aliasing, and
overwrite all fail closed.

Independently reconstruct both wrapper/native bindings and the complete P27
outcome before committing any result.

## 6. Route mechanically from the first outcome

Apply the following rows in order and select the first matching route. After
selection, freeze the exact P27 outcome and route. No row authorizes future
scientific acquisition; that always requires a separately reviewed contract.

| precedence | sole P27 outcome | required route |
| --- | --- | --- |
| 1 | sanitization or reconstruction failure | retain native evidence externally and stop |
| 2 | any deleted executable or writable mapping | stop; retain every exact witness; repair the image or loader; do not authorize acquisition |
| 3 | any deleted unclassifiable mapping or deleted-target identity instability | stop; retain every witness and exact failed probe or transition; do not relax rejection |
| 4 | a conclusively identified read-only, nonexecutable deleted-mapping set | retain every exact regular-file, character-device, and other-nonregular witness; preregister a narrow provenance or device-mapping policy before acquisition |
| 5 | no deleted mapping reproduced | record nonreproduction; do not infer safety and do not rerun |

No P21/P23 threshold, seed, model, optimizer, data, capture schedule, equality
rule, or fidelity gate changes in P27. Executable or unclassifiable deleted
mappings remain fatal. A read-only mapping observation is a lower-bound
witness about this exact runtime, not a global permission to ignore deleted
mappings.
