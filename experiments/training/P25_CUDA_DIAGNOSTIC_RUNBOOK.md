# P25 one-shot CUDA diagnostic and acquisition runbook

## Evidence status

This is a **pre-execution runbook**. It records no passing P25 diagnostic and
no CUDA gradient, repeatability, fidelity, shield, or training result. P24 is
terminal and must not be rerun. Commands below create one fresh P25 attempt;
there is intentionally no cleanup or retry command.

Read `p25_cuda_diagnostic_contract.json` and reconstruct it before use. Every
P25 source hash, the contract SHA-256, and the runtime artifacts are review
inputs, never values to repair after observing an outcome.

## 1. Freeze the correction sources

After all four pending P25 execution sources are review-complete, run the
one-shot finalizer. It fills only declared pending hashes and pins the
resulting contract SHA-256 in the independent reconstructor. If an already
finalized source differs, it fails instead of refreshing the hash.

```bash
python scripts/finalize_p25_cuda_diagnostic_contract.py
python scripts/reconstruct_p25_cuda_diagnostic_contract.py
git diff --check
uv run --locked pytest -q tests/test_p25_cuda_diagnostic_contract.py
```

Commit the finalized contract, correction sources, reconstructor, tests,
runbook, workflow, and audit packet as a source-freeze bootstrap. This commit
does not contain the historical wrapper and does not authorize an image build
or diagnostic. Transfer that exact commit to the selected host (for example
with a verified Git bundle) and record its full commit and tree as
`P25_BOOTSTRAP_COMMIT` and `P25_BOOTSTRAP_TREE`.

## 2. Authenticate and wrap the retained P24 bytes

The native manifest contains absolute container paths, and the sanitizer
requires every declared root to exist. Therefore do not run it locally with
fictional `/workspace` roots. Run it exactly once in a fresh offline container
from the old pinned P23 image, with the original destination paths. The fresh
P25 output root is writable, while the one retained P24 native file is
overlay-mounted read-only at its exact old container filename; no historical
P23/P24 evidence directory is writable. This container has no GPU request, no
network, and a read-only root; it runs only the standard-library P25 ingester
and never executes the P24 diagnostic.

```bash
set -euo pipefail
export P25_INGESTION_ID=20260906-01
export P25_BOOTSTRAP_COMMIT=0000000000000000000000000000000000000000
export P25_BOOTSTRAP_TREE=0000000000000000000000000000000000000000
export P25_SOURCE_REPO=/secure/p23/OptimizationML
export P25_P23_IMAGE=localhost:5000/p23-runtime@sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1
export P25_HISTORICAL_CONTAINER="p25-historical-$P25_INGESTION_ID"
export P25_HISTORICAL_ROOT="/secure/p25/historical-$P25_INGESTION_ID"
export P25_HISTORICAL_EVIDENCE="$P25_HISTORICAL_ROOT/evidence"
export P25_HISTORICAL_HOST="$P25_HISTORICAL_EVIDENCE/p25-historical-p24-native-outcome.json"
export P25_HISTORICAL_STDOUT="$P25_HISTORICAL_ROOT/process.stdout.log"
export P25_HISTORICAL_STDERR="$P25_HISTORICAL_ROOT/process.stderr.log"
export P25_HISTORICAL_INSPECT="$P25_HISTORICAL_ROOT/container-inspect.json"

test "$(git -C "$P25_SOURCE_REPO" rev-parse HEAD)" = "$P25_BOOTSTRAP_COMMIT"
test "$(git -C "$P25_SOURCE_REPO" rev-parse HEAD^{tree})" = "$P25_BOOTSTRAP_TREE"
test -z "$(git -C "$P25_SOURCE_REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$P25_HISTORICAL_ROOT"
mkdir -p "$P25_HISTORICAL_EVIDENCE"
if sudo docker container inspect "$P25_HISTORICAL_CONTAINER" >/dev/null 2>&1; then
  echo "historical-ingestion container already exists; rerun forbidden" >&2
  exit 1
fi

historical_status=0
sudo docker run --name "$P25_HISTORICAL_CONTAINER" \
  --pull never \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824 \
  --env CUDA_VISIBLE_DEVICES= \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env PYTHONNOUSERSITE=1 \
  --mount type=bind,src="$P25_SOURCE_REPO",dst=/workspace/OptimizationML,readonly \
  --mount type=bind,src=/secure/p23/nanoGPT,dst=/workspace/inputs/nanoGPT,readonly \
  --mount type=bind,src=/secure/p23/Muon,dst=/workspace/inputs/muon,readonly \
  --mount type=bind,src=/secure/p23/optimizationml-p22-data,dst=/private/tmp/optimizationml-p22-data,readonly \
  --mount type=bind,src="$P25_SOURCE_REPO/experiments/training/materialize_p22_fineweb.py",dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly \
  --mount type=bind,src="$P25_HISTORICAL_EVIDENCE",dst=/workspace/evidence/p23 \
  --mount type=bind,src=/secure/p23/evidence/p24-baseline-executable-origin.native.json,dst=/workspace/evidence/p23/p24-baseline-executable-origin.native.json,readonly \
  --entrypoint /opt/p23-venv/bin/python \
  "$P25_P23_IMAGE" \
  /workspace/OptimizationML/scripts/ingest_p24_native_outcome_for_p25.py \
  --native /workspace/evidence/p23/p24-baseline-executable-origin.native.json \
  --path-root repository=/workspace/OptimizationML \
  --path-root nanogpt=/workspace/inputs/nanoGPT \
  --path-root muon=/workspace/inputs/muon \
  --path-root data=/private/tmp/optimizationml-p22-data \
  --path-root preprocessor_alias=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
  --path-root python_environment=/opt/p23-venv \
  --path-root python_standard_library=/usr/local/lib/python3.12 \
  --path-root temporary=/tmp \
  --path-root native=/workspace/evidence/p23 \
  --output /workspace/evidence/p23/p25-historical-p24-native-outcome.json \
  >"$P25_HISTORICAL_STDOUT" 2>"$P25_HISTORICAL_STDERR" || historical_status=$?
if sudo docker container inspect "$P25_HISTORICAL_CONTAINER" >/dev/null 2>&1; then
  sudo docker inspect "$P25_HISTORICAL_CONTAINER" >"$P25_HISTORICAL_INSPECT"
fi
if (( historical_status != 0 )); then
  echo "historical ingestion failed with exit $historical_status; do not retry" >&2
  exit "$historical_status"
fi

test ! -e "$P25_SOURCE_REPO/results/summaries/p25_historical_p24_native_outcome.json"
sudo install -m 0644 "$P25_HISTORICAL_HOST" \
  "$P25_SOURCE_REPO/results/summaries/p25_historical_p24_native_outcome.json"
sudo chown "$(id -u):$(id -g)" \
  "$P25_SOURCE_REPO/results/summaries/p25_historical_p24_native_outcome.json"
```

Replace the zero bootstrap placeholders with the reviewed commit and tree. Review the
wrapper and retain the original native bytes and offline-container inspection
separately. The wrapper must continue to say P24 failed and must not authorize
an image build. Commit only this exact wrapper on top of the bootstrap and
record the clean resulting commit/tree as `P25_PREREG_COMMIT`. No image build
may precede that final preregistration commit.

## 3. Freeze one replacement runtime

On the selected A100 host, define the exact reviewed inputs. The source repo
must already contain the preregistration commit (for example via a verified
Git bundle). `P25_ATTEMPT_ID` names a new path; if it exists, the script stops.

```bash
set -euo pipefail
export P25_ATTEMPT_ID=20260906-01
export P25_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P25_SOURCE_REPO=/secure/p23/OptimizationML
export P25_NANOGPT_HOST=/secure/p23/nanoGPT
export P25_MUON_HOST=/secure/p23/Muon
export P25_DATA_HOST=/secure/p23/optimizationml-p22-data
export P25_HISTORICAL_WRAPPER="$P25_SOURCE_REPO/results/summaries/p25_historical_p24_native_outcome.json"
export P25_BOOTSTRAP_COMMIT=0000000000000000000000000000000000000000
export P25_BOOTSTRAP_TREE=0000000000000000000000000000000000000000
export P25_PREREG_COMMIT=0000000000000000000000000000000000000000
export P25_EXPECTED_CONTRACT_SHA256=0000000000000000000000000000000000000000000000000000000000000000
bash "$P25_SOURCE_REPO/scripts/run_p25_cuda_attempt.sh" prepare-runtime
```

Replace all zero placeholders with the reviewed full values. The phase first
proves that `P25_PREREG_COMMIT` is one direct child of the bootstrap and adds
only the historical wrapper, which was absent from the bootstrap. It then:

- creates `/secure/p25/attempt-$P25_ATTEMPT_ID` only if absent;
- makes a fresh detached checkout and resets its origin to the credential-free
  public GitHub URL;
- builds the unchanged P24 remediation recipe with `--no-cache` and retains
  BuildKit metadata;
- selects and pulls the immutable repository digest;
- precreates all four host-evidence files;
- launches a read-only-root, network-none container with exactly ten bind
  mounts, the one locked `/tmp` tmpfs, and the pinned full GPU UUID;
- updates the two bound running-inspection files **in place**; and
- freezes fresh `p25_host_attestation.json` and
  `p25_cuda_runtime_lock.json` files without overwriting any destination.

Review the image metadata, image inspection, running-container inspection,
namespace-entered mountinfo, GPU query, lock, and attestation. Commit only the
new P25-named lock and attestation to the fresh attempt checkout. This review
commit must be one direct child of `P25_PREREG_COMMIT`, with exactly two added
paths (`p25_cuda_runtime_lock.json` and `p25_host_attestation.json`) and no
other tree change. This review commit must precede the diagnostic.

## 4. Run the one corrected diagnostic

Set the exact reviewed lock/attestation hashes and diagnostic commit/tree.
The script accepts no baseline mode and refuses preexisting outputs.

```bash
export P25_EXPECTED_RUNTIME_LOCK_SHA256=0000000000000000000000000000000000000000000000000000000000000000
export P25_EXPECTED_HOST_ATTESTATION_SHA256=0000000000000000000000000000000000000000000000000000000000000000
export P25_DIAGNOSTIC_HEAD=0000000000000000000000000000000000000000
export P25_DIAGNOSTIC_TREE=0000000000000000000000000000000000000000
bash "$P25_SOURCE_REPO/scripts/run_p25_cuda_attempt.sh" run-diagnostic
```

Replace all placeholders with reviewed values. The script independently
reconstructs the immutable image reference from the retained BuildKit
metadata and matches it to the live container before continuing. The
diagnostic does a
standard-library preflight, imports Torch without CUDA initialization,
configures and exactly verifies the complete locked deterministic state,
constructs one CPU FP32 parameter and `torch.optim.SGD`, and verifies the
generated source at its immutable read-only image path. It does not allocate a
model, read data, run forward/backward, call Muon, take an optimizer step, or
observe a candidate.

The sanitizer runs even when the diagnostic exits nonzero. Any diagnostic or
sanitizer defect ends the attempt permanently before gradients. On success,
the sanitizer does not trust the runner's Boolean summary: it independently
reconstructs the closed 40-check pass semantics, frozen source/artifact
bindings, lock/attestation and deterministic runtime, process/event/snapshot
order, pinned Torch sources, and immutable generated-module closure. The host
script then copies the sanitized wrapper to
`results/summaries/p25_executable_origin_diagnostic.sanitized.json`. Review
and commit that exact wrapper without changing any execution source, gate, or
contract. This is the diagnostic-to-acquisition bridge; do not rerun the
diagnostic after the commit.

## 5. Run the unchanged P23 acquisition conditionally

Only after the one diagnostic and sanitizer pass and their wrapper is in the
reviewed bridge commit, set that exact commit/tree and invoke:

```bash
export P25_DIAGNOSTIC_HEAD=0000000000000000000000000000000000000000
export P25_DIAGNOSTIC_TREE=0000000000000000000000000000000000000000
export P25_BRIDGE_HEAD=0000000000000000000000000000000000000000
export P25_BRIDGE_TREE=0000000000000000000000000000000000000000
bash "$P25_SOURCE_REPO/scripts/run_p25_cuda_attempt.sh" run-acquisition
```

The bridge must be one direct child of the diagnostic commit and add exactly
one path:
`results/summaries/p25_executable_origin_diagnostic.sanitized.json`. Before
any gradient work, the host wrapper reruns independent contract
reconstruction, rehashes every frozen authority/execution/remediation source,
strictly parses the reviewed wrapper, and rehashes the retained native
diagnostic bytes and byte count against that wrapper. Supplying a different
bridge head/tree cannot authorize changed runners or gates.

The phase executes, with fresh no-overwrite files:

1. `trace_off_a`;
2. `trace_off_b` only after A passes;
3. exact `verify-repeatability`;
4. `trace_on` only after zero exact mismatches;
5. exact `verify-noninterference`;
6. `aggregate` only after zero exact mismatches; and
7. offline sanitization of every native success artifact, or the first native
   failure artifact before stopping.

The acquisition runner and thresholds are the unchanged P23/P21/P22 bytes.
No `allclose`, numeric tolerance, new seed, changed capture schedule, or gate
tuning is permitted. A successful trace-on must contain exactly 1,152 actual
post-aspect CUDA observations over all 48 Muon matrices. Verifier failure,
noninterference failure, or any earlier process error is the terminal outcome;
do not rerun until favorable.

## 6. Record the first terminal outcome

Retain BuildKit metadata, native artifacts, sanitized wrappers, process
stdout/stderr, and hashes. Add a scoped result summary and compact outcome
record only after the sequence stops or completes. Select the next branch from
that evidence. CPU CI validates contracts and fixtures only; it cannot be
cited as remote A100 execution.
