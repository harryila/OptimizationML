#!/usr/bin/env bash
# One-shot P26 acquisition bridge for fresh attempt 20260906-02.
#
# Runtime preparation and the corrected executable-origin diagnostic remain the
# frozen P25 procedure.  This control-plane script changes only the permission
# boundary: it streams a hash-bound verifier into the already-attested root
# container so the committed and retained sanitized wrappers can be compared
# without changing the retained root:root 0600 evidence file.

set -euo pipefail

readonly P26_REQUIRED_ATTEMPT_ID=20260906-02
readonly P26_P25_PREREG_COMMIT=e76ab62f92c95e6f0716cf2f1ed38a583cadfe56
readonly P26_P25_PREREG_TREE=4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2
readonly P26_P25_CONTRACT_SHA256=51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2
readonly P26_TERMINAL_PARENT_COMMIT=f055405cc879ba0ac5afe26bc34a7336d5d2efbf
readonly P26_REQUIRED_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d

die() {
  echo "P26 permission-safe acquisition blocked: $*" >&2
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

sha256_file() {
  python3 -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}

run_logged() {
  local label="$1"
  shift
  local stdout_path="$P26_HOST_EVIDENCE/$label.stdout.log"
  local stderr_path="$P26_HOST_EVIDENCE/$label.stderr.log"
  [[ ! -e "$stdout_path" && ! -e "$stderr_path" ]] ||
    die "process log already exists: $label"
  local status=0
  "$@" >"$stdout_path" 2>"$stderr_path" || status=$?
  return "$status"
}

run_logged_hash_bound_stdin() {
  local label="$1"
  local stdin_path="$2"
  local expected_sha256="$3"
  shift 3
  local stdout_path="$P26_HOST_EVIDENCE/$label.stdout.log"
  local stderr_path="$P26_HOST_EVIDENCE/$label.stderr.log"
  [[ ! -e "$stdout_path" && ! -e "$stderr_path" ]] ||
    die "process log already exists: $label"
  local status=0
  {
    python3 -c \
      'import hashlib,pathlib,sys
raw=pathlib.Path(sys.argv[1]).read_bytes()
observed=hashlib.sha256(raw).hexdigest()
if observed != sys.argv[2]:
    raise SystemExit(f"hash-bound stdin differs: {observed}")
sys.stdout.buffer.write(raw)' \
      "$stdin_path" "$expected_sha256" | "$@"
  } >"$stdout_path" 2>"$stderr_path" || status=$?
  return "$status"
}

require_common() {
  require_var P26_ATTEMPT_ID
  [[ "$P26_ATTEMPT_ID" == "$P26_REQUIRED_ATTEMPT_ID" ]] ||
    die "P26_ATTEMPT_ID must be the preregistered fresh attempt 20260906-02"
  require_var P26_GPU_UUID
  [[ "$P26_GPU_UUID" == "$P26_REQUIRED_GPU_UUID" ]] ||
    die "P26_GPU_UUID differs from the preregistered full A100 UUID"
  require_var P26_CONTROL_REPO
  require_commit P26_CONTROL_HEAD
  require_commit P26_CONTROL_TREE
  require_sha256 P26_EXPECTED_CONTRACT_SHA256
  require_sha256 P26_EXPECTED_ORCHESTRATOR_SHA256
  require_sha256 P26_EXPECTED_RECONSTRUCTOR_SHA256
  require_sha256 P26_EXPECTED_VERIFIER_SHA256
  require_commit P26_DIAGNOSTIC_HEAD
  require_commit P26_DIAGNOSTIC_TREE
  require_commit P26_BRIDGE_HEAD
  require_commit P26_BRIDGE_TREE

  P26_ATTEMPT_ROOT="/secure/p25/attempt-$P26_ATTEMPT_ID"
  P26_HOST_REPO="$P26_ATTEMPT_ROOT/OptimizationML"
  P26_HOST_EVIDENCE="$P26_ATTEMPT_ROOT/evidence"
  P26_CONTAINER="p25-acquisition-$P26_ATTEMPT_ID"
  P26_IMAGE_TAG="localhost:5000/p25-runtime:attempt-$P26_ATTEMPT_ID"
  export P26_ATTEMPT_ROOT P26_HOST_REPO P26_HOST_EVIDENCE P26_CONTAINER P26_IMAGE_TAG
}

validate_control_checkout() {
  [[ -d "$P26_CONTROL_REPO/.git" ]] || die "P26 control checkout is not a Git worktree"
  [[ "$(git -C "$P26_CONTROL_REPO" rev-parse HEAD)" == "$P26_CONTROL_HEAD" ]] ||
    die "P26 control checkout HEAD differs"
  [[ "$(git -C "$P26_CONTROL_REPO" rev-parse HEAD^{tree})" == "$P26_CONTROL_TREE" ]] ||
    die "P26 control checkout tree differs"
  [[ "$(git -C "$P26_CONTROL_REPO" rev-list --parents -n 1 "$P26_CONTROL_HEAD")" == \
      "$P26_CONTROL_HEAD $P26_TERMINAL_PARENT_COMMIT" ]] ||
    die "P26 control commit must be one direct child of the terminal P25 outcome"
  [[ -z "$(git -C "$P26_CONTROL_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "P26 control checkout is dirty"

  local contract="$P26_CONTROL_REPO/experiments/training/p26_permission_safe_acquisition_contract.json"
  local orchestrator="$P26_CONTROL_REPO/scripts/run_p26_permission_safe_acquisition.sh"
  local reconstructor="$P26_CONTROL_REPO/scripts/reconstruct_p26_permission_safe_acquisition.py"
  local verifier="$P26_CONTROL_REPO/scripts/verify_p26_permission_safe_bridge.py"
  [[ -f "$contract" && -f "$orchestrator" && -f "$reconstructor" && -f "$verifier" ]] ||
    die "P26 control sources are incomplete"
  [[ "$(sha256_file "$contract")" == "$P26_EXPECTED_CONTRACT_SHA256" ]] ||
    die "P26 contract bytes differ from the reviewed SHA-256"
  [[ "$(sha256_file "$orchestrator")" == "$P26_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "P26 host orchestrator bytes differ from the reviewed SHA-256"
  [[ "$(sha256_file "$reconstructor")" == "$P26_EXPECTED_RECONSTRUCTOR_SHA256" ]] ||
    die "P26 contract reconstructor bytes differ from the reviewed SHA-256"
  [[ "$(sha256_file "$verifier")" == "$P26_EXPECTED_VERIFIER_SHA256" ]] ||
    die "P26 root-container verifier bytes differ from the reviewed SHA-256"
  [[ "$(git -C "$P26_CONTROL_REPO" ls-files --error-unmatch -- \
    scripts/run_p26_permission_safe_acquisition.sh)" == \
    scripts/run_p26_permission_safe_acquisition.sh ]] ||
    die "P26 host orchestrator is not tracked"
  [[ "$(git -C "$P26_CONTROL_REPO" ls-files --error-unmatch -- \
    scripts/verify_p26_permission_safe_bridge.py)" == \
    scripts/verify_p26_permission_safe_bridge.py ]] ||
    die "P26 root-container verifier is not tracked"
  [[ "$(git -C "$P26_CONTROL_REPO" ls-files --error-unmatch -- \
    scripts/reconstruct_p26_permission_safe_acquisition.py)" == \
    scripts/reconstruct_p26_permission_safe_acquisition.py ]] ||
    die "P26 contract reconstructor is not tracked"

  local reconstruction_status=0
  run_logged p26-contract-reconstruction python3 "$reconstructor" \
    --canonical "$contract" || reconstruction_status=$?
  (( reconstruction_status == 0 )) ||
    die "P26 contract reconstruction failed with exit $reconstruction_status"
}

validate_fresh_attempt_history() {
  [[ -d "$P26_HOST_REPO/.git" ]] || die "fresh attempt checkout is absent"
  [[ "$(git -C "$P26_HOST_REPO" rev-parse "$P26_P25_PREREG_COMMIT^{tree}")" == \
      "$P26_P25_PREREG_TREE" ]] ||
    die "frozen P25 preregistration commit/tree binding differs"
  [[ "$(git -C "$P26_HOST_REPO" rev-parse HEAD)" == "$P26_BRIDGE_HEAD" ]] ||
    die "attempt checkout is not the reviewed fresh bridge commit"
  [[ "$(git -C "$P26_HOST_REPO" rev-parse HEAD^{tree})" == "$P26_BRIDGE_TREE" ]] ||
    die "attempt checkout tree differs from the reviewed fresh bridge tree"
  [[ -z "$(git -C "$P26_HOST_REPO" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "fresh attempt checkout is dirty"
  [[ "$(git -C "$P26_HOST_REPO" rev-parse "$P26_DIAGNOSTIC_HEAD^{tree}")" == \
      "$P26_DIAGNOSTIC_TREE" ]] ||
    die "fresh diagnostic commit/tree binding differs"
  [[ "$(git -C "$P26_HOST_REPO" rev-list --parents -n 1 "$P26_DIAGNOSTIC_HEAD")" == \
      "$P26_DIAGNOSTIC_HEAD $P26_P25_PREREG_COMMIT" ]] ||
    die "fresh diagnostic must be one direct child of frozen P25 preregistration"
  [[ "$(git -C "$P26_HOST_REPO" diff --name-status \
      "$P26_P25_PREREG_COMMIT" "$P26_DIAGNOSTIC_HEAD")" == \
      $'A\texperiments/training/p25_cuda_runtime_lock.json\nA\texperiments/training/p25_host_attestation.json' ]] ||
    die "fresh diagnostic commit must add only the P25 lock and attestation"
  [[ "$(git -C "$P26_HOST_REPO" rev-list --parents -n 1 "$P26_BRIDGE_HEAD")" == \
      "$P26_BRIDGE_HEAD $P26_DIAGNOSTIC_HEAD" ]] ||
    die "fresh bridge must be one direct child of the fresh diagnostic commit"
  local wrapper_relative=results/summaries/p25_executable_origin_diagnostic.sanitized.json
  [[ "$(git -C "$P26_HOST_REPO" diff --name-status \
      "$P26_DIAGNOSTIC_HEAD" "$P26_BRIDGE_HEAD")" == $'A\t'"$wrapper_relative" ]] ||
    die "fresh bridge delta must add only the reviewed sanitized wrapper"
  if git -C "$P26_HOST_REPO" cat-file -e \
    "$P26_DIAGNOSTIC_HEAD:$wrapper_relative" 2>/dev/null; then
    die "fresh diagnostic wrapper unexpectedly predates the bridge commit"
  fi
  [[ "$(git -C "$P26_HOST_REPO" ls-files --error-unmatch -- "$wrapper_relative")" == \
      "$wrapper_relative" ]] || die "fresh diagnostic wrapper is not tracked"
  [[ "$(sha256_file "$P26_HOST_REPO/experiments/training/p25_cuda_diagnostic_contract.json")" == \
      "$P26_P25_CONTRACT_SHA256" ]] || die "frozen P25 contract bytes differ"
}

load_attempt_image() {
  local metadata="$P26_ATTEMPT_ROOT/p25-build-metadata.json"
  [[ -f "$metadata" ]] || die "fresh retained BuildKit metadata is absent"
  P26_IMAGE_DIGEST="$(python3 -c \
    'import json,pathlib,sys; value=json.loads(pathlib.Path(sys.argv[1]).read_text())["containerimage.digest"]; assert isinstance(value,str); print(value)' \
    "$metadata")"
  [[ "$P26_IMAGE_DIGEST" =~ ^sha256:[0-9a-f]{64}$ ]] ||
    die "invalid fresh retained BuildKit digest"
  P26_IMAGE="${P26_IMAGE_TAG%:*}@$P26_IMAGE_DIGEST"
  export P26_IMAGE P26_IMAGE_DIGEST
}

assert_no_p23_acquisition_state() {
  local name
  for name in \
    trace-off-a.json trace-off-a-failure.json \
    trace-off-b.json trace-off-b-failure.json \
    repeatability.json trace-on.json trace-on-failure.json raw-trace.json \
    noninterference.json aggregate.json \
    sanitized-trace-off-a.json sanitized-trace-off-a-failure.json \
    sanitized-trace-off-b.json sanitized-trace-off-b-failure.json \
    sanitized-repeatability.json sanitized-trace-on.json \
    sanitized-trace-on-failure.json sanitized-raw-trace.json \
    sanitized-noninterference.json sanitized-aggregate.json; do
    [[ ! -e "$P26_HOST_EVIDENCE/$name" ]] ||
      die "fresh attempt already contains a P23 acquisition artifact: $name"
  done

  shopt -s nullglob
  local logs=(
    "$P26_HOST_EVIDENCE"/p23-*.stdout.log
    "$P26_HOST_EVIDENCE"/p23-*.stderr.log
  )
  shopt -u nullglob
  ((${#logs[@]} == 0)) || die "fresh attempt already contains P23 acquisition process logs"
}

run_acquisition() {
  require_common
  [[ "$P26_ATTEMPT_ROOT" != /secure/p25/attempt-20260906-01 ]] ||
    die "terminal P25 attempt 20260906-01 must never be reused"
  [[ -d "$P26_HOST_EVIDENCE" ]] || die "fresh attempt evidence directory is absent"
  validate_control_checkout
  validate_fresh_attempt_history
  load_attempt_image
  assert_no_p23_acquisition_state

  local p25_reconstruction_status=0
  run_logged p26-p25-contract-reconstruction python3 \
    "$P26_HOST_REPO/scripts/reconstruct_p25_cuda_diagnostic_contract.py" \
    --canonical "$P26_HOST_REPO/experiments/training/p25_cuda_diagnostic_contract.json" ||
    p25_reconstruction_status=$?
  (( p25_reconstruction_status == 0 )) ||
    die "frozen P25 contract reconstruction failed with exit $p25_reconstruction_status"

  [[ "$(sudo docker inspect --format '{{.State.Running}}' "$P26_CONTAINER")" == true ]] ||
    die "fresh attested container is not running"
  [[ "$(sudo docker inspect --format '{{.Config.Image}}' "$P26_CONTAINER")" == \
      "$P26_IMAGE" ]] || die "fresh container image differs from retained BuildKit digest"

  local verifier="$P26_CONTROL_REPO/scripts/verify_p26_permission_safe_bridge.py"
  local repository_wrapper=/workspace/OptimizationML/results/summaries/p25_executable_origin_diagnostic.sanitized.json
  local retained_wrapper=/workspace/evidence/p23/p25-remediated-executable-origin.sanitized.json
  local retained_native=/workspace/evidence/p23/p25-remediated-executable-origin.native.json
  local bridge_status=0
  run_logged_hash_bound_stdin p26-permission-safe-bridge-verification "$verifier" \
    "$P26_EXPECTED_VERIFIER_SHA256" \
    sudo docker exec -i "$P26_CONTAINER" /opt/p23-venv/bin/python - \
    --repository-wrapper "$repository_wrapper" \
    --retained-wrapper "$retained_wrapper" \
    --retained-native "$retained_native" \
    --contract /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
    --repository /workspace/OptimizationML \
    --sanitizer /workspace/OptimizationML/scripts/sanitize_p25_executable_origin_diagnostic.py ||
    bridge_status=$?
  (( bridge_status == 0 )) ||
    die "root-container diagnostic bridge verification failed with exit $bridge_status"

  # Everything below this line is the unchanged frozen P23 acquisition and
  # sanitization sequence used by P25.  The P26 bridge changes no P23 gate.
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
    --env "CUDA_VISIBLE_DEVICES=$P26_GPU_UUID"
    --env "NVIDIA_VISIBLE_DEVICES=$P26_GPU_UUID"
    "$P26_CONTAINER" /opt/p23-venv/bin/python
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
    if [[ -f "$P26_HOST_EVIDENCE/$artifact" ]]; then
      sanitize_one "$artifact" "$label"
    fi
  }

  local status=0
  run_logged p23-trace-off-a "${py[@]}" "$runner" run \
    --role trace_off_a "${common[@]}" \
    --output "$native/trace-off-a.json" \
    --failure-output "$native/trace-off-a-failure.json" || status=$?
  if (( status != 0 )); then
    sanitize_if_present trace-off-a.json trace-off-a-on-failure
    [[ -f "$P26_HOST_EVIDENCE/trace-off-a-failure.json" ]] ||
      die "trace_off_a failed without its required native failure artifact"
    sanitize_one trace-off-a-failure.json trace-off-a-failure
    die "trace_off_a failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P26_HOST_EVIDENCE/trace-off-a-failure.json" ]] ||
    die "trace_off_a reported success but left a failure artifact"

  status=0
  run_logged p23-trace-off-b "${py[@]}" "$runner" run \
    --role trace_off_b "${common[@]}" \
    --output "$native/trace-off-b.json" \
    --failure-output "$native/trace-off-b-failure.json" || status=$?
  if (( status != 0 )); then
    sanitize_one trace-off-a.json trace-off-a
    sanitize_if_present trace-off-b.json trace-off-b-on-failure
    [[ -f "$P26_HOST_EVIDENCE/trace-off-b-failure.json" ]] ||
      die "trace_off_b failed without its required native failure artifact"
    sanitize_one trace-off-b-failure.json trace-off-b-failure
    die "trace_off_b failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P26_HOST_EVIDENCE/trace-off-b-failure.json" ]] ||
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
    [[ -f "$P26_HOST_EVIDENCE/trace-on-failure.json" ]] ||
      die "trace_on failed without its required native failure artifact"
    sanitize_one trace-on-failure.json trace-on-failure
    die "trace_on failed with exit $status; this attempt is terminal"
  fi
  [[ ! -e "$P26_HOST_EVIDENCE/trace-on-failure.json" ]] ||
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
  echo "P26/P23 acquisition sequence completed. Retain and review every native, sanitized, bridge-transcript, and stream artifact before recording an outcome."
}

usage() {
  cat >&2 <<'EOF'
usage: run_p26_permission_safe_acquisition.sh run-acquisition

This script is a separate, hash-bound P26 control-plane entry point.  Prepare
fresh attempt 20260906-02 through the frozen P25 prepare-runtime and diagnostic
steps, commit its fresh lock/attestation and sanitized bridge, then invoke this
command from the reviewed P26 control checkout.  There is no retry, cleanup,
permission-mutation, or acquisition shortcut.
EOF
  exit 2
}

[[ $# == 1 ]] || usage
case "$1" in
  run-acquisition) run_acquisition ;;
  *) usage ;;
esac
