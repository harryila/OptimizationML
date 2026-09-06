#!/usr/bin/env bash
# Hash-bound host orchestration for the one-shot P25 diagnostic attempt.
#
# This script deliberately has no cleanup or retry command.  A failed attempt
# remains terminal and its fresh attempt directory remains intact.

set -euo pipefail

die() {
  echo "P25 host orchestration blocked: $*" >&2
  exit 1
}

require_var() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "required variable is unset: $name"
}

require_sha256() {
  local name="$1"
  require_var "$name"
  [[ "${!name}" =~ ^[0-9a-f]{64}$ ]] || die "$name is not a lowercase SHA-256"
}

require_commit() {
  local name="$1"
  require_var "$name"
  [[ "${!name}" =~ ^[0-9a-f]{40}$ ]] || die "$name is not a full Git object ID"
}

run_logged() {
  local label="$1"
  shift
  local stdout_path="$P25_HOST_EVIDENCE/$label.stdout.log"
  local stderr_path="$P25_HOST_EVIDENCE/$label.stderr.log"
  [[ ! -e "$stdout_path" && ! -e "$stderr_path" ]] ||
    die "process log already exists: $label"
  local status=0
  "$@" >"$stdout_path" 2>"$stderr_path" || status=$?
  return "$status"
}

require_common() {
  require_var P25_ATTEMPT_ID
  [[ "$P25_ATTEMPT_ID" =~ ^[0-9]{8}-[0-9]{2}$ ]] || die "invalid P25_ATTEMPT_ID"
  require_var P25_GPU_UUID
  [[ "$P25_GPU_UUID" == GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d ]] ||
    die "P25_GPU_UUID differs from the preregistered full A100 UUID"
  require_var P25_SOURCE_REPO
  require_var P25_NANOGPT_HOST
  require_var P25_MUON_HOST
  require_var P25_DATA_HOST
  require_var P25_HISTORICAL_WRAPPER
  require_commit P25_BOOTSTRAP_COMMIT
  require_commit P25_BOOTSTRAP_TREE
  require_commit P25_PREREG_COMMIT
  require_sha256 P25_EXPECTED_CONTRACT_SHA256
  P25_ATTEMPT_ROOT="/secure/p25/attempt-$P25_ATTEMPT_ID"
  P25_HOST_REPO="$P25_ATTEMPT_ROOT/OptimizationML"
  P25_HOST_EVIDENCE="$P25_ATTEMPT_ROOT/evidence"
  P25_CONTAINER="p25-acquisition-$P25_ATTEMPT_ID"
  P25_IMAGE_TAG="localhost:5000/p25-runtime:attempt-$P25_ATTEMPT_ID"
  export P25_ATTEMPT_ROOT P25_HOST_REPO P25_HOST_EVIDENCE P25_CONTAINER P25_IMAGE_TAG
}

assert_source_preregistration() {
  [[ -d "$P25_SOURCE_REPO/.git" ]] || die "source repository is not a Git worktree"
  [[ "$(git -C "$P25_SOURCE_REPO" rev-parse HEAD)" == "$P25_PREREG_COMMIT" ]] ||
    die "source repository HEAD differs from P25_PREREG_COMMIT"
  [[ "$(git -C "$P25_SOURCE_REPO" rev-parse "$P25_BOOTSTRAP_COMMIT^{tree}")" == \
      "$P25_BOOTSTRAP_TREE" ]] ||
    die "source-freeze bootstrap commit/tree binding differs"
  [[ "$(git -C "$P25_SOURCE_REPO" rev-parse "$P25_PREREG_COMMIT^")" == \
      "$P25_BOOTSTRAP_COMMIT" ]] ||
    die "preregistration commit must be one direct child of the source-freeze bootstrap"
  local historical_relative=results/summaries/p25_historical_p24_native_outcome.json
  [[ "$(git -C "$P25_SOURCE_REPO" diff --name-status "$P25_BOOTSTRAP_COMMIT" "$P25_PREREG_COMMIT")" == \
      $'A\t'"$historical_relative" ]] ||
    die "preregistration commit must add only the historical P24 wrapper"
  if git -C "$P25_SOURCE_REPO" cat-file -e \
    "$P25_BOOTSTRAP_COMMIT:$historical_relative" 2>/dev/null; then
    die "historical P24 wrapper unexpectedly predates offline ingestion"
  fi
  [[ -z "$(git -C "$P25_SOURCE_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "source repository is dirty"
  [[ "$P25_HISTORICAL_WRAPPER" == "$P25_SOURCE_REPO/$historical_relative" ]] ||
    die "historical P24 wrapper path differs from the frozen repository location"
  [[ -f "$P25_HISTORICAL_WRAPPER" ]] || die "reviewed historical P24 wrapper is absent"
  [[ "$(git -C "$P25_SOURCE_REPO" ls-files --error-unmatch -- results/summaries/p25_historical_p24_native_outcome.json)" == \
      results/summaries/p25_historical_p24_native_outcome.json ]] ||
    die "historical P24 wrapper is not tracked by the preregistration commit"
  [[ "$(python3 -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$P25_SOURCE_REPO/experiments/training/p25_cuda_diagnostic_contract.json")" == "$P25_EXPECTED_CONTRACT_SHA256" ]] ||
    die "P25 contract bytes differ from the reviewed SHA-256"
  python3 "$P25_SOURCE_REPO/scripts/reconstruct_p25_cuda_diagnostic_contract.py" >/dev/null ||
    die "independent P25 contract reconstruction failed"
}

load_attempt_image() {
  local metadata="$P25_ATTEMPT_ROOT/p25-build-metadata.json"
  [[ -f "$metadata" ]] || die "retained BuildKit metadata is absent"
  P25_IMAGE_DIGEST="$(python3 -c 'import json,pathlib,sys; value=json.loads(pathlib.Path(sys.argv[1]).read_text())["containerimage.digest"]; assert isinstance(value,str); print(value)' "$metadata")"
  [[ "$P25_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] || die "invalid retained BuildKit digest"
  P25_IMAGE="${P25_IMAGE_TAG%:*}@$P25_IMAGE_DIGEST"
  export P25_IMAGE P25_IMAGE_DIGEST
}

prepare_runtime() {
  require_common
  assert_source_preregistration
  [[ ! -e "$P25_ATTEMPT_ROOT" ]] || die "attempt root already exists; retries are forbidden"
  if sudo docker container inspect "$P25_CONTAINER" >/dev/null 2>&1; then
    die "container name has prior state"
  fi
  [[ -d "$P25_NANOGPT_HOST/.git" ]] || die "pinned nanoGPT checkout is absent"
  [[ -f "$P25_MUON_HOST/muon.py" ]] || die "pinned Muon source is absent"
  [[ -f "$P25_DATA_HOST/materialized/p22_fineweb_manifest.json" ]] ||
    die "FineWeb materialization manifest is absent"

  mkdir -p "$P25_ATTEMPT_ROOT" "$P25_HOST_EVIDENCE"
  git clone --no-hardlinks --no-checkout "$P25_SOURCE_REPO" "$P25_HOST_REPO"
  git -C "$P25_HOST_REPO" checkout --detach "$P25_PREREG_COMMIT"
  git -C "$P25_HOST_REPO" remote set-url origin \
    https://github.com/harryila/OptimizationML.git
  [[ -z "$(git -C "$P25_HOST_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "fresh attempt checkout is dirty"

  run_logged p25-image-build sudo docker buildx build --no-cache --platform linux/amd64 \
    --file "$P25_HOST_REPO/experiments/training/p24_runtime.Dockerfile" \
    --tag "$P25_IMAGE_TAG" \
    --metadata-file "$P25_ATTEMPT_ROOT/p25-build-metadata.json" \
    --push "$P25_HOST_REPO/experiments/training"
  load_attempt_image
  run_logged p25-image-pull sudo docker pull "$P25_IMAGE"

  sudo docker image inspect "$P25_IMAGE" >"$P25_ATTEMPT_ROOT/p25-image-inspect.json"
  nvidia-smi --id="$P25_GPU_UUID" \
    --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
    --format=csv,noheader,nounits >"$P25_ATTEMPT_ROOT/p25-nvidia-smi.csv"
  : >"$P25_ATTEMPT_ROOT/p25-running-container-inspect.json"
  : >"$P25_ATTEMPT_ROOT/p25-running-mountinfo.txt"

  sudo docker run --detach --name "$P25_CONTAINER" \
    --gpus "device=$P25_GPU_UUID" \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824 \
    --env PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env VIRTUAL_ENV=/opt/p23-venv \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env CUDA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    --env NVIDIA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env PYTHONHASHSEED=1337 \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env P23_REPO=/workspace/OptimizationML \
    --env P23_NANOGPT=/workspace/inputs/nanoGPT \
    --env P23_MUON_ROOT=/workspace/inputs/muon \
    --mount type=bind,src="$P25_HOST_REPO",dst=/workspace/OptimizationML,readonly \
    --mount type=bind,src="$P25_NANOGPT_HOST",dst=/workspace/inputs/nanoGPT,readonly \
    --mount type=bind,src="$P25_MUON_HOST",dst=/workspace/inputs/muon,readonly \
    --mount type=bind,src="$P25_DATA_HOST",dst=/private/tmp/optimizationml-p22-data,readonly \
    --mount type=bind,src="$P25_HOST_REPO/experiments/training/materialize_p22_fineweb.py",dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly \
    --mount type=bind,src="$P25_HOST_EVIDENCE",dst=/workspace/evidence/p23 \
    --mount type=bind,src="$P25_ATTEMPT_ROOT/p25-image-inspect.json",dst=/mounted-host-evidence/image-inspect.json,readonly \
    --mount type=bind,src="$P25_ATTEMPT_ROOT/p25-running-container-inspect.json",dst=/mounted-host-evidence/running-container-inspect.json,readonly \
    --mount type=bind,src="$P25_ATTEMPT_ROOT/p25-running-mountinfo.txt",dst=/mounted-host-evidence/running-mountinfo.txt,readonly \
    --mount type=bind,src="$P25_ATTEMPT_ROOT/p25-nvidia-smi.csv",dst=/mounted-host-evidence/nvidia-smi.csv,readonly \
    --entrypoint /bin/sh \
    "$P25_IMAGE" -c 'while :; do sleep 3600; done' \
    >"$P25_ATTEMPT_ROOT/p25-container-id.txt"

  # Redirection truncates the already bound inode.  Never replace these files
  # with a temporary-file rename after container launch.
  sudo docker inspect "$P25_CONTAINER" >"$P25_ATTEMPT_ROOT/p25-running-container-inspect.json"
  P25_CONTAINER_INIT_PID="$(sudo docker inspect --format '{{.State.Pid}}' "$P25_CONTAINER")"
  [[ "$P25_CONTAINER_INIT_PID" =~ ^[1-9][0-9]*$ ]] || die "invalid container init PID"
  sudo /usr/bin/nsenter --target "$P25_CONTAINER_INIT_PID" \
    --mount --pid --cgroup -- /usr/bin/cat /proc/self/mountinfo \
    >"$P25_ATTEMPT_ROOT/p25-running-mountinfo.txt"

  run_logged p25-freeze-runtime sudo docker exec \
    --env CUDA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    --env NVIDIA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    "$P25_CONTAINER" /opt/p23-venv/bin/python \
    /workspace/OptimizationML/experiments/training/run_p23_deterministic_cuda_shadow_trace.py \
    freeze-runtime \
    --container-image "$P25_IMAGE" \
    --container-repository-digest "$P25_IMAGE_DIGEST" \
    --host-image-inspection /mounted-host-evidence/image-inspect.json \
    --host-running-container-inspection /mounted-host-evidence/running-container-inspect.json \
    --host-running-mountinfo /mounted-host-evidence/running-mountinfo.txt \
    --host-nvidia-smi-query /mounted-host-evidence/nvidia-smi.csv \
    --host-attestation-output /workspace/evidence/p23/p25_host_attestation.json \
    --runtime-lock-output /workspace/evidence/p23/p25_cuda_runtime_lock.json

  [[ ! -e "$P25_HOST_REPO/experiments/training/p25_host_attestation.json" ]] ||
    die "P25 host-attestation destination already exists"
  [[ ! -e "$P25_HOST_REPO/experiments/training/p25_cuda_runtime_lock.json" ]] ||
    die "P25 runtime-lock destination already exists"
  sudo install -m 0644 "$P25_HOST_EVIDENCE/p25_host_attestation.json" \
    "$P25_HOST_REPO/experiments/training/p25_host_attestation.json"
  sudo install -m 0644 "$P25_HOST_EVIDENCE/p25_cuda_runtime_lock.json" \
    "$P25_HOST_REPO/experiments/training/p25_cuda_runtime_lock.json"
  sudo chown "$(id -u):$(id -g)" \
    "$P25_HOST_REPO/experiments/training/p25_host_attestation.json" \
    "$P25_HOST_REPO/experiments/training/p25_cuda_runtime_lock.json"

  echo "P25 runtime frozen. Review image/build/inspection/lock/attestation bytes,"
  echo "then commit only the new P25 lock and attestation before run-diagnostic."
  echo "P25_IMAGE=$P25_IMAGE"
  echo "P25_CONTAINER=$P25_CONTAINER"
  echo "P25_HOST_REPO=$P25_HOST_REPO"
}

run_diagnostic() {
  require_common
  assert_source_preregistration
  require_sha256 P25_EXPECTED_RUNTIME_LOCK_SHA256
  require_sha256 P25_EXPECTED_HOST_ATTESTATION_SHA256
  require_commit P25_DIAGNOSTIC_HEAD
  require_commit P25_DIAGNOSTIC_TREE
  load_attempt_image
  [[ -d "$P25_HOST_REPO/.git" ]] || die "attempt checkout is absent"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse HEAD)" == "$P25_DIAGNOSTIC_HEAD" ]] ||
    die "attempt checkout is not the reviewed diagnostic commit"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse HEAD^{tree})" == "$P25_DIAGNOSTIC_TREE" ]] ||
    die "attempt checkout tree differs from the reviewed diagnostic tree"
  [[ -z "$(git -C "$P25_HOST_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "attempt checkout is dirty"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse "$P25_DIAGNOSTIC_HEAD^")" == \
      "$P25_PREREG_COMMIT" ]] ||
    die "diagnostic commit must be one direct child of the preregistration commit"
  local expected_diagnostic_delta
  expected_diagnostic_delta=$'A\texperiments/training/p25_cuda_runtime_lock.json\nA\texperiments/training/p25_host_attestation.json'
  [[ "$(git -C "$P25_HOST_REPO" diff --name-status "$P25_PREREG_COMMIT" "$P25_DIAGNOSTIC_HEAD")" == \
      "$expected_diagnostic_delta" ]] ||
    die "diagnostic commit must add only the reviewed P25 lock and attestation"
  if git -C "$P25_HOST_REPO" cat-file -e \
    "$P25_PREREG_COMMIT:experiments/training/p25_cuda_runtime_lock.json" 2>/dev/null ||
    git -C "$P25_HOST_REPO" cat-file -e \
      "$P25_PREREG_COMMIT:experiments/training/p25_host_attestation.json" 2>/dev/null; then
    die "P25 lock or attestation unexpectedly predates the diagnostic commit"
  fi
  [[ "$(sudo docker inspect --format '{{.State.Running}}' "$P25_CONTAINER")" == true ]] ||
    die "replacement container is not running"
  [[ "$(sudo docker inspect --format '{{.Config.Image}}' "$P25_CONTAINER")" == "$P25_IMAGE" ]] ||
    die "replacement container image differs from retained BuildKit digest"

  local native=/workspace/evidence/p23/p25-remediated-executable-origin.native.json
  local sanitized=/workspace/evidence/p23/p25-remediated-executable-origin.sanitized.json
  [[ ! -e "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.native.json" ]] ||
    die "native diagnostic output already exists"
  [[ ! -e "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.sanitized.json" ]] ||
    die "sanitized diagnostic output already exists"

  local status=0
  run_logged p25-diagnostic sudo docker exec \
    --env CUDA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    --env NVIDIA_VISIBLE_DEVICES="$P25_GPU_UUID" \
    "$P25_CONTAINER" /opt/p23-venv/bin/python \
    /workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py \
    --mode remediated \
    --repository /workspace/OptimizationML \
    --runtime-lock /workspace/OptimizationML/experiments/training/p25_cuda_runtime_lock.json \
    --host-attestation /workspace/OptimizationML/experiments/training/p25_host_attestation.json \
    --contract /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
    --expected-contract-sha256 "$P25_EXPECTED_CONTRACT_SHA256" \
    --expected-runtime-lock-sha256 "$P25_EXPECTED_RUNTIME_LOCK_SHA256" \
    --expected-host-attestation-sha256 "$P25_EXPECTED_HOST_ATTESTATION_SHA256" \
    --expected-repository-head "$P25_DIAGNOSTIC_HEAD" \
    --expected-repository-tree "$P25_DIAGNOSTIC_TREE" \
    --output "$native" || status=$?

  local native_sha
  native_sha="$(sudo docker exec "$P25_CONTAINER" /opt/p23-venv/bin/python -c 'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' "$native")"
  [[ "$native_sha" =~ ^[0-9a-f]{64}$ ]] || die "invalid native diagnostic SHA-256"
  local sanitizer_status=0
  run_logged p25-diagnostic-sanitizer sudo docker exec \
    "$P25_CONTAINER" /opt/p23-venv/bin/python \
    /workspace/OptimizationML/scripts/sanitize_p25_executable_origin_diagnostic.py \
    --native "$native" \
    --expected-native-sha256 "$native_sha" \
    --path-root repository=/workspace/OptimizationML \
    --path-root nanogpt=/workspace/inputs/nanoGPT \
    --path-root muon=/workspace/inputs/muon \
    --path-root data=/private/tmp/optimizationml-p22-data \
    --path-root preprocessor_alias=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
    --path-root python_environment=/opt/p23-venv \
    --path-root python_standard_library=/usr/local/lib/python3.12 \
    --path-root temporary=/tmp \
    --path-root native=/workspace/evidence/p23 \
    --output "$sanitized" || sanitizer_status=$?
  (( sanitizer_status == 0 )) ||
    die "P25 sanitizer failed with exit $sanitizer_status; this attempt is terminal"
  (( status == 0 )) || die "P25 diagnostic failed with exit $status; this attempt is terminal"

  local repository_wrapper="$P25_HOST_REPO/results/summaries/p25_executable_origin_diagnostic.sanitized.json"
  [[ ! -e "$repository_wrapper" ]] || die "repository diagnostic wrapper already exists"
  sudo install -m 0644 \
    "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.sanitized.json" \
    "$repository_wrapper"
  sudo chown "$(id -u):$(id -g)" "$repository_wrapper"

  echo "P25 diagnostic and sanitizer passed once. Do not rerun them."
  echo "Review and commit results/summaries/p25_executable_origin_diagnostic.sanitized.json."
  echo "Only that reviewed bridge commit may precede the unchanged P23 acquisition."
}

run_acquisition() {
  require_common
  assert_source_preregistration
  require_commit P25_DIAGNOSTIC_HEAD
  require_commit P25_DIAGNOSTIC_TREE
  require_commit P25_BRIDGE_HEAD
  require_commit P25_BRIDGE_TREE
  load_attempt_image
  [[ -d "$P25_HOST_REPO/.git" ]] || die "attempt checkout is absent"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse HEAD)" == "$P25_BRIDGE_HEAD" ]] ||
    die "attempt checkout is not the reviewed diagnostic bridge commit"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse HEAD^{tree})" == "$P25_BRIDGE_TREE" ]] ||
    die "attempt checkout tree differs from the reviewed bridge tree"
  [[ -z "$(git -C "$P25_HOST_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "attempt checkout is dirty"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse "$P25_DIAGNOSTIC_HEAD^{tree}")" == \
      "$P25_DIAGNOSTIC_TREE" ]] ||
    die "reviewed diagnostic commit/tree binding differs"
  [[ "$(git -C "$P25_HOST_REPO" rev-parse "$P25_BRIDGE_HEAD^")" == \
      "$P25_DIAGNOSTIC_HEAD" ]] ||
    die "bridge must be one direct child of the reviewed diagnostic commit"
  local repository_wrapper="$P25_HOST_REPO/results/summaries/p25_executable_origin_diagnostic.sanitized.json"
  local wrapper_relative=results/summaries/p25_executable_origin_diagnostic.sanitized.json
  [[ "$(git -C "$P25_HOST_REPO" diff --name-status "$P25_DIAGNOSTIC_HEAD" "$P25_BRIDGE_HEAD")" == \
      $'A\t'"$wrapper_relative" ]] ||
    die "bridge delta must add only the reviewed sanitized diagnostic wrapper"
  if git -C "$P25_HOST_REPO" cat-file -e \
    "$P25_DIAGNOSTIC_HEAD:$wrapper_relative" 2>/dev/null; then
    die "diagnostic wrapper unexpectedly predates the bridge commit"
  fi
  [[ -f "$repository_wrapper" ]] || die "reviewed P25 diagnostic wrapper is absent"
  cmp -s "$repository_wrapper" \
    "$P25_HOST_EVIDENCE/p25-remediated-executable-origin.sanitized.json" ||
    die "reviewed and retained diagnostic wrappers differ"
  [[ "$(git -C "$P25_HOST_REPO" ls-files --error-unmatch -- results/summaries/p25_executable_origin_diagnostic.sanitized.json)" == \
      results/summaries/p25_executable_origin_diagnostic.sanitized.json ]] ||
    die "diagnostic wrapper is not tracked by the bridge commit"
  local reconstruction_status=0
  run_logged p25-bridge-contract-reconstruction python3 \
    "$P25_HOST_REPO/scripts/reconstruct_p25_cuda_diagnostic_contract.py" \
    --canonical "$P25_HOST_REPO/experiments/training/p25_cuda_diagnostic_contract.json" ||
    reconstruction_status=$?
  (( reconstruction_status == 0 )) ||
    die "P25 contract reconstruction failed at the bridge with exit $reconstruction_status"
  [[ "$(sudo docker inspect --format '{{.State.Running}}' "$P25_CONTAINER")" == true ]] ||
    die "replacement container is not running"
  [[ "$(sudo docker inspect --format '{{.Config.Image}}' "$P25_CONTAINER")" == "$P25_IMAGE" ]] ||
    die "replacement container image differs from retained BuildKit digest"
  local retained_diagnostic_native=/workspace/evidence/p23/p25-remediated-executable-origin.native.json
  local retained_diagnostic_wrapper=/workspace/evidence/p23/p25-remediated-executable-origin.sanitized.json
  local bridge_status=0
  run_logged p25-diagnostic-bridge-verification sudo docker exec \
    "$P25_CONTAINER" /opt/p23-venv/bin/python -c '
import hashlib
import importlib.util
import json
import pathlib
import sys


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON constant: {token}")


wrapper_path = pathlib.Path(sys.argv[1])
native_path = pathlib.Path(sys.argv[2])
contract_path = pathlib.Path(sys.argv[3])
repository = pathlib.Path(sys.argv[4]).resolve(strict=True)
wrapper = json.loads(
    wrapper_path.read_bytes(),
    object_pairs_hook=reject_duplicates,
    parse_constant=reject_nonfinite,
)
if not isinstance(wrapper, dict):
    raise ValueError("diagnostic wrapper root is not an object")
if wrapper.get("schema_version") != "passive-muon-p25-sanitized-executable-origin-diagnostic-v1":
    raise ValueError("diagnostic wrapper schema differs")
if wrapper.get("native_schema_version") != "passive-muon-p25-executable-origin-diagnostic-v1":
    raise ValueError("native diagnostic schema binding differs")
if wrapper.get("mode") != "remediated" or wrapper.get("status") != "passes" or wrapper.get("passes") is not True:
    raise ValueError("diagnostic wrapper does not authorize acquisition")
binding = wrapper.get("native_artifact")
if not isinstance(binding, dict) or set(binding) != {"byte_count", "sha256"}:
    raise ValueError("native artifact binding is malformed")
native = native_path.read_bytes()
if binding.get("byte_count") != len(native):
    raise ValueError("retained native byte count differs from wrapper")
if binding.get("sha256") != hashlib.sha256(native).hexdigest():
    raise ValueError("retained native SHA-256 differs from wrapper")
sanitizer_spec = importlib.util.spec_from_file_location(
    "p25_bridge_sanitizer", pathlib.Path(sys.argv[5])
)
if sanitizer_spec is None or sanitizer_spec.loader is None:
    raise ValueError("cannot load the frozen P25 sanitizer")
sanitizer = importlib.util.module_from_spec(sanitizer_spec)
sanitizer_spec.loader.exec_module(sanitizer)
native_payload = sanitizer.load_json_strict(native)
if not isinstance(native_payload, dict):
    raise ValueError("retained native diagnostic root is not an object")
sanitizer.reject_credentials_and_nonfinite(native_payload)
sanitizer.validate_native_semantics(native_payload)
manifest = wrapper.get("manifest")
if not isinstance(manifest, dict):
    raise ValueError("redacted native manifest is absent")
if manifest.get("schema_version") != "passive-muon-p25-executable-origin-diagnostic-v1":
    raise ValueError("redacted native manifest schema differs")
if manifest.get("mode") != "remediated" or manifest.get("status") != "passes" or manifest.get("passes") is not True:
    raise ValueError("redacted native manifest does not record a pass")
contract = json.loads(
    contract_path.read_bytes(),
    object_pairs_hook=reject_duplicates,
    parse_constant=reject_nonfinite,
)
if not isinstance(contract, dict):
    raise ValueError("P25 contract root is not an object")
if contract.get("schema_version") != "passive-muon-p25-cuda-diagnostic-correction-contract-v1":
    raise ValueError("P25 contract schema differs")
if contract.get("status") != "frozen_pre_remediation_build_and_pre_acquisition":
    raise ValueError("P25 contract status differs")
records = []
for group_name in ("unchanged_authorities", "execution_sources"):
    group = contract.get(group_name)
    if not isinstance(group, dict):
        raise ValueError(f"P25 contract {group_name} is malformed")
    records.extend(group.values())
remediation = contract.get("remediation_design")
if not isinstance(remediation, dict):
    raise ValueError("P25 remediation design is malformed")
records.extend((remediation.get("build_patch"), remediation.get("image_recipe")))
for record in records:
    if not isinstance(record, dict):
        raise ValueError("P25 source record is malformed")
    relative = record.get("path")
    expected = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("P25 source binding is malformed")
    source = (repository / relative).resolve(strict=True)
    if source != repository and repository not in source.parents:
        raise ValueError("P25 source escapes the repository")
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError(f"P25 frozen source differs: {relative}")
' "$retained_diagnostic_wrapper" "$retained_diagnostic_native" \
    /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
    /workspace/OptimizationML \
    /workspace/OptimizationML/scripts/sanitize_p25_executable_origin_diagnostic.py ||
    bridge_status=$?
  (( bridge_status == 0 )) ||
    die "diagnostic wrapper/native bridge verification failed with exit $bridge_status"

  local runner=/workspace/OptimizationML/experiments/training/run_p23_deterministic_cuda_shadow_trace.py
  local repo=/workspace/OptimizationML
  local nanogpt=/workspace/inputs/nanoGPT
  local muon=/workspace/inputs/muon/muon.py
  local data=/private/tmp/optimizationml-p22-data/materialized/p22_fineweb_manifest.json
  local observer=/workspace/OptimizationML/experiments/training/p22_observed_muon.py
  local lock=/workspace/OptimizationML/experiments/training/p25_cuda_runtime_lock.json
  local attestation=/workspace/OptimizationML/experiments/training/p25_host_attestation.json
  local native=/workspace/evidence/p23
  local preprocessor=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py
  local python_environment=/opt/p23-venv
  local -a py=(
    sudo docker exec
    --env "CUDA_VISIBLE_DEVICES=$P25_GPU_UUID"
    --env "NVIDIA_VISIBLE_DEVICES=$P25_GPU_UUID"
    "$P25_CONTAINER" /opt/p23-venv/bin/python
  )
  local -a common=(
    --nanogpt-root "$nanogpt"
    --muon-source "$muon"
    --fineweb-manifest "$data"
    --instrumentation-patch "$observer"
    --runtime-lock "$lock"
    --host-attestation "$attestation"
  )
  local -a roots=(
    --path-root "repository=$repo"
    --path-root "nanogpt=$nanogpt"
    --path-root "muon=/workspace/inputs/muon"
    --path-root "data=/private/tmp/optimizationml-p22-data"
    --path-root "preprocessor_alias=$preprocessor"
    --path-root "python_environment=$python_environment"
    --path-root "native=$native"
  )

  sanitize_one() {
    local artifact="$1"
    local label="$2"
    local sanitize_status=0
    run_logged "p23-sanitize-$label" "${py[@]}" "$runner" sanitize \
      --manifest "$native/$artifact" "${roots[@]}" \
      --output "$native/sanitized-$artifact" || sanitize_status=$?
    (( sanitize_status == 0 )) ||
      die "P23 sanitizer failed for $artifact with exit $sanitize_status"
  }

  sanitize_if_present() {
    local artifact="$1"
    local label="$2"
    if [[ -f "$P25_HOST_EVIDENCE/$artifact" ]]; then
      sanitize_one "$artifact" "$label"
    fi
  }

  local name
  for name in \
    trace-off-a.json trace-off-a-failure.json \
    trace-off-b.json trace-off-b-failure.json \
    repeatability.json trace-on.json trace-on-failure.json raw-trace.json \
    noninterference.json aggregate.json; do
    [[ ! -e "$P25_HOST_EVIDENCE/$name" ]] || die "acquisition artifact already exists: $name"
  done

  local status=0
  run_logged p23-trace-off-a "${py[@]}" "$runner" run \
    --role trace_off_a "${common[@]}" \
    --output "$native/trace-off-a.json" \
    --failure-output "$native/trace-off-a-failure.json" || status=$?
  if (( status != 0 )); then
    sanitize_if_present trace-off-a.json trace-off-a-on-failure
    [[ -f "$P25_HOST_EVIDENCE/trace-off-a-failure.json" ]] ||
      die "trace_off_a failed without its required native failure artifact"
    sanitize_one trace-off-a-failure.json trace-off-a-failure
    die "trace_off_a failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P25_HOST_EVIDENCE/trace-off-a-failure.json" ]] ||
    die "trace_off_a reported success but left a failure artifact"

  status=0
  run_logged p23-trace-off-b "${py[@]}" "$runner" run \
    --role trace_off_b "${common[@]}" \
    --output "$native/trace-off-b.json" \
    --failure-output "$native/trace-off-b-failure.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_if_present trace-off-b.json trace-off-b-on-failure
    [[ -f "$P25_HOST_EVIDENCE/trace-off-b-failure.json" ]] ||
      die "trace_off_b failed without its required native failure artifact"
    sanitize_one trace-off-b-failure.json trace-off-b-failure
    die "trace_off_b failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P25_HOST_EVIDENCE/trace-off-b-failure.json" ]] ||
    die "trace_off_b reported success but left a failure artifact"

  status=0
  run_logged p23-verify-repeatability "${py[@]}" "$runner" \
    verify-repeatability "${common[@]}" \
    --trace-off-a "$native/trace-off-a.json" \
    --trace-off-b "$native/trace-off-b.json" \
    --output "$native/repeatability.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_one trace-off-b.json trace-off-b
    sanitize_if_present repeatability.json repeatability
    die "exact repeatability failed with exit $status; trace_on remains barred"
  fi

  status=0
  run_logged p23-trace-on "${py[@]}" "$runner" run \
    --role trace_on "${common[@]}" \
    --trace-off-a "$native/trace-off-a.json" \
    --trace-off-b "$native/trace-off-b.json" \
    --repeatability-report "$native/repeatability.json" \
    --raw-trace-output "$native/raw-trace.json" \
    --output "$native/trace-on.json" \
    --failure-output "$native/trace-on-failure.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_one trace-off-b.json trace-off-b
    sanitize_one repeatability.json repeatability
    sanitize_if_present trace-on.json trace-on-on-failure
    sanitize_if_present raw-trace.json raw-trace-on-failure
    [[ -f "$P25_HOST_EVIDENCE/trace-on-failure.json" ]] ||
      die "trace_on failed without its required native failure artifact"
    sanitize_one trace-on-failure.json trace-on-failure
    die "trace_on failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P25_HOST_EVIDENCE/trace-on-failure.json" ]] ||
    die "trace_on reported success but left a failure artifact"

  status=0
  run_logged p23-verify-noninterference "${py[@]}" "$runner" \
    verify-noninterference "${common[@]}" \
    --trace-off-a "$native/trace-off-a.json" \
    --trace-off-b "$native/trace-off-b.json" \
    --trace-on "$native/trace-on.json" \
    --output "$native/noninterference.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_one trace-off-b.json trace-off-b
    sanitize_one repeatability.json repeatability
    sanitize_one trace-on.json trace-on
    sanitize_one raw-trace.json raw-trace
    sanitize_if_present noninterference.json noninterference
    die "exact noninterference failed with exit $status; aggregation remains barred"
  fi

  status=0
  run_logged p23-aggregate "${py[@]}" "$runner" aggregate "${common[@]}" \
    --trace-off-a "$native/trace-off-a.json" \
    --trace-off-b "$native/trace-off-b.json" \
    --trace-on "$native/trace-on.json" \
    --repeatability-report "$native/repeatability.json" \
    --noninterference-report "$native/noninterference.json" \
    --output "$native/aggregate.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_one trace-off-b.json trace-off-b
    sanitize_one repeatability.json repeatability
    sanitize_one trace-on.json trace-on
    sanitize_one raw-trace.json raw-trace
    sanitize_one noninterference.json noninterference
    sanitize_if_present aggregate.json aggregate
    die "fidelity aggregation failed with exit $status"
  fi
  sanitize_one trace-off-a.json trace-off-a
  sanitize_one trace-off-b.json trace-off-b
  sanitize_one repeatability.json repeatability
  sanitize_one trace-on.json trace-on
  sanitize_one raw-trace.json raw-trace
  sanitize_one noninterference.json noninterference
  sanitize_one aggregate.json aggregate
  echo "P25/P23 acquisition sequence completed. Retain and review every native, sanitized, and stream artifact before recording an outcome."
}

usage() {
  cat >&2 <<'EOF'
usage: run_p25_cuda_attempt.sh prepare-runtime|run-diagnostic|run-acquisition

The exact required environment is documented in
experiments/training/P25_CUDA_DIAGNOSTIC_RUNBOOK.md.  There is deliberately no
retry or cleanup command and no acquisition shortcut.
EOF
  exit 2
}

[[ $# == 1 ]] || usage
case "$1" in
  prepare-runtime) prepare_runtime ;;
  run-diagnostic) run_diagnostic ;;
  run-acquisition) run_acquisition ;;
  *) usage ;;
esac
