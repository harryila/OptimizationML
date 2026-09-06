#!/usr/bin/env bash
# Hash-bound, one-shot host orchestration for P28's bundle-complete bridge to
# P27's unchanged, non-training CUDA deleted-mapping localizer. There is
# deliberately no cleanup, retry, or acquisition phase: every created attempt
# directory and every process transcript is retained as evidence.

set -euo pipefail
umask 077

readonly P28_REQUIRED_ATTEMPT_ID=20260906-01
readonly P28_ATTEMPT_ROOT=/secure/p28/attempt-20260906-01
readonly P28_HOST_EVIDENCE="$P28_ATTEMPT_ROOT/evidence"
readonly P28_CONTAINER=p28-localization-20260906-01
readonly P28_TRANSPORT_ROOT=/secure/p28/transport-20260906-01
readonly P28_SOURCE_BUNDLE="$P28_TRANSPORT_ROOT/p28_source.bundle"
readonly P28_RUNTIME_REVIEW_BUNDLE="$P28_TRANSPORT_ROOT/p28_runtime_review.bundle"
readonly P28_SOURCE_RECEIPT="$P28_TRANSPORT_ROOT/p28_source_bundle_receipt.json"
readonly P28_RUNTIME_REVIEW_RECEIPT="$P28_TRANSPORT_ROOT/p28_runtime_review_bundle_receipt.json"
readonly P28_SOURCE_CLOSURE="$P28_TRANSPORT_ROOT/p28_source_closure.git"
readonly P28_RUNTIME_REVIEW_CLOSURE="$P28_TRANSPORT_ROOT/p28_runtime_review_closure.git"
readonly P28_BOOTSTRAP_VERIFIER="$P28_TRANSPORT_ROOT/verify_p28_control_bundle.py"
readonly P28_CONTROL_REPO=/secure/p28/control-source
readonly P28_IMAGE_DIGEST=sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0
readonly P28_IMAGE="localhost:5000/p25-runtime@$P28_IMAGE_DIGEST"
readonly P28_REQUIRED_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
readonly P28_REQUIRED_GPU_NAME='NVIDIA A100-SXM4-40GB'
readonly P28_AUTHORITY_HEAD=185e444afc0b44ca0a09b1bde49a6b6fa3973355
readonly P28_AUTHORITY_TREE=24f4bdac331a57bd7c1b807747d7c7fba253ee5a
readonly P28_TERMINAL_P27_COMMIT=ec63550331925ded158e3f389e294e4d1f12db3a
readonly P28_TERMINAL_P27_TREE=d8efa72fda9ee41fde0b5d15b126aa87397cd48d
readonly P28_TERMINAL_P26_COMMIT=5429da23ff18888daa2312c530a4587780484d8b
readonly P28_TERMINAL_P26_CONTAINER_ID=e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c
readonly P28_P26_NATIVE=/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json
readonly P28_P26_NATIVE_SHA256=b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91
readonly P28_P26_NATIVE_BYTE_COUNT=66283
readonly P28_P25_SOURCE_SHA256=6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770
readonly P28_P25_CONTRACT_SHA256=51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2
readonly P28_P23_RUNNER_SHA256=a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb
readonly P28_NANOGPT_COMMIT=3adf61e154c3fe3fca428ad6bc3818b27a3b8291
readonly P28_NANOGPT_TREE=ca93bcd9b9c9ff32d3016e1e2556644e68bef86a
readonly P28_MUON_COMMIT=f98f1cacc0263b04290753e32be8d498c1efc806
readonly P28_MUON_SOURCE_SHA256=2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d

die() {
  echo "P28 bundle-complete localization bridge blocked: $*" >&2
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

require_absolute_path() {
  local name="$1"
  require_var "$name"
  [[ "${!name}" == /* && "${!name}" != *$'\n'* ]] ||
    die "$name must be one absolute path without a newline"
}

sha256_file() {
  /usr/bin/python3 -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}

root_sha256_file() {
  sudo /usr/bin/python3 -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}

run_logged() {
  local label="$1"
  shift
  local stdout_path="$P28_HOST_EVIDENCE/$label.stdout.log"
  local stderr_path="$P28_HOST_EVIDENCE/$label.stderr.log"
  [[ ! -e "$stdout_path" && ! -L "$stdout_path" && \
     ! -e "$stderr_path" && ! -L "$stderr_path" ]] ||
    die "process log already exists: $label"
  local status=0
  "$@" >"$stdout_path" 2>"$stderr_path" || status=$?
  return "$status"
}

capture_new() {
  local label="$1"
  local output="$2"
  shift 2
  local stderr_path="$P28_HOST_EVIDENCE/$label.stderr.log"
  [[ ! -e "$output" && ! -L "$output" && \
     ! -e "$stderr_path" && ! -L "$stderr_path" ]] ||
    die "capture output already exists: $label"
  local status=0
  (set -o noclobber; "$@" >"$output" 2>"$stderr_path") || status=$?
  return "$status"
}

refresh_bound_file() {
  local label="$1"
  local output="$2"
  shift 2
  local stderr_path="$P28_HOST_EVIDENCE/$label.stderr.log"
  [[ -f "$output" && ! -L "$output" && ! -s "$output" ]] ||
    die "bound evidence placeholder is not one empty regular file: $output"
  [[ ! -e "$stderr_path" && ! -L "$stderr_path" ]] ||
    die "process log already exists: $label"
  local status=0
  "$@" >"$output" 2>"$stderr_path" || status=$?
  (( status == 0 )) || return "$status"
  [[ -s "$output" ]] || die "bound evidence capture is empty: $label"
}

write_new_status() {
  local output="$1"
  local value="$2"
  /usr/bin/python3 - "$output" "$value" <<'PY'
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
if hasattr(os, "O_NOFOLLOW"):
    flags |= os.O_NOFOLLOW
descriptor = os.open(path, flags, 0o600)
try:
    os.write(descriptor, (sys.argv[2] + "\n").encode("ascii"))
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
}

require_source_environment() {
  require_var P28_ATTEMPT_ID
  [[ "$P28_ATTEMPT_ID" == "$P28_REQUIRED_ATTEMPT_ID" ]] ||
    die "P28_ATTEMPT_ID must equal $P28_REQUIRED_ATTEMPT_ID"
  require_var P28_GPU_UUID
  [[ "$P28_GPU_UUID" == "$P28_REQUIRED_GPU_UUID" ]] ||
    die "P28_GPU_UUID differs from the preregistered full A100 UUID"
  require_absolute_path P28_AUTHORITY_REPO
  require_absolute_path P28_NANOGPT_HOST
  require_absolute_path P28_MUON_HOST
  require_absolute_path P28_DATA_HOST
  require_commit P28_SOURCE_FREEZE_COMMIT
  require_commit P28_SOURCE_FREEZE_TREE
  require_sha256 P28_EXPECTED_SOURCE_BUNDLE_SHA256
  require_var P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT
  [[ "$P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" =~ ^[1-9][0-9]*$ ]] ||
    die "P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT is not a positive integer"
  require_sha256 P28_EXPECTED_SOURCE_RECEIPT_SHA256
  require_sha256 P28_EXPECTED_CONTRACT_SHA256
  require_sha256 P28_EXPECTED_ORCHESTRATOR_SHA256
  require_sha256 P28_EXPECTED_RECONSTRUCTOR_SHA256
  require_sha256 P28_EXPECTED_BUNDLE_VERIFIER_SHA256
  require_sha256 P28_EXPECTED_P27_CONTRACT_SHA256
  require_sha256 P28_EXPECTED_P27_RECONSTRUCTOR_SHA256
  require_sha256 P28_EXPECTED_P27_LOCALIZER_SHA256
  require_sha256 P28_EXPECTED_INGESTER_SHA256
  require_sha256 P28_EXPECTED_P23_CORE_SHA256
  require_sha256 P28_EXPECTED_P27_SANITIZER_SHA256
}

require_runtime_environment() {
  require_source_environment
  require_commit P28_RUNTIME_REVIEW_COMMIT
  require_commit P28_RUNTIME_REVIEW_TREE
  require_sha256 P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256
  require_var P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT
  [[ "$P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" =~ ^[1-9][0-9]*$ ]] ||
    die "P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT is not a positive integer"
  require_sha256 P28_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256
  require_sha256 P28_EXPECTED_RUNTIME_LOCK_SHA256
  require_sha256 P28_EXPECTED_HOST_ATTESTATION_SHA256
}

assert_clean_checkout() {
  local repository="$1"
  local head="$2"
  local tree="$3"
  local label="$4"
  [[ -d "$repository/.git" ]] || die "$label is not a Git worktree"
  [[ "$(git -C "$repository" rev-parse HEAD)" == "$head" ]] ||
    die "$label HEAD differs"
  [[ "$(git -C "$repository" rev-parse HEAD^{tree})" == "$tree" ]] ||
    die "$label tree differs"
  [[ -z "$(git -C "$repository" status --porcelain=v1 --untracked-files=all)" ]] ||
    die "$label is dirty"
}

validate_authority_and_inputs() {
  assert_clean_checkout \
    "$P28_AUTHORITY_REPO" "$P28_AUTHORITY_HEAD" "$P28_AUTHORITY_TREE" \
    "P28 authority checkout"
  if git -C "$P28_AUTHORITY_REPO" symbolic-ref -q HEAD >/dev/null; then
    die "P28 authority checkout must be detached"
  fi
  [[ "$(git -C "$P28_AUTHORITY_REPO" rev-parse --is-shallow-repository)" == "false" ]] ||
    die "P28 authority checkout must not be shallow"
  local authority_indirection
  for authority_indirection in \
    "$P28_AUTHORITY_REPO/.git/objects/info/alternates" \
    "$P28_AUTHORITY_REPO/.git/info/grafts"; do
    [[ ! -e "$authority_indirection" && ! -L "$authority_indirection" ]] ||
      die "P28 authority checkout has forbidden Git object indirection: $authority_indirection"
  done
  [[ -z "$(git -C "$P28_AUTHORITY_REPO" for-each-ref \
      --format='%(refname)' refs/replace/)" ]] ||
    die "P28 authority checkout has forbidden replace refs"
  [[ -z "$(git -C "$P28_AUTHORITY_REPO" config --local --get-regexp \
      '^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$' \
      2>/dev/null || true)" ]] ||
    die "P28 authority checkout has forbidden promisor or partial-clone configuration"
  [[ "$(sha256_file "$P28_AUTHORITY_REPO/experiments/training/run_p25_executable_origin_diagnostic.py")" == \
      "$P28_P25_SOURCE_SHA256" ]] || die "authority P25 source bytes differ"
  [[ "$(sha256_file "$P28_AUTHORITY_REPO/experiments/training/p25_cuda_diagnostic_contract.json")" == \
      "$P28_P25_CONTRACT_SHA256" ]] || die "authority P25 contract bytes differ"
  [[ "$(sha256_file "$P28_AUTHORITY_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py")" == \
      "$P28_P23_RUNNER_SHA256" ]] || die "authority P23 runner bytes differ"

  assert_clean_checkout \
    "$P28_NANOGPT_HOST" "$P28_NANOGPT_COMMIT" "$P28_NANOGPT_TREE" \
    "pinned nanoGPT checkout"
  assert_clean_checkout \
    "$P28_MUON_HOST" "$P28_MUON_COMMIT" \
    "$(git -C "$P28_MUON_HOST" rev-parse "$P28_MUON_COMMIT^{tree}")" \
    "pinned Muon checkout"
  [[ "$(sha256_file "$P28_MUON_HOST/muon.py")" == "$P28_MUON_SOURCE_SHA256" ]] ||
    die "pinned Muon source bytes differ"
  [[ -f "$P28_DATA_HOST/materialized/p22_fineweb_manifest.json" ]] ||
    die "FineWeb materialization manifest is absent"
  [[ -f "$P28_AUTHORITY_REPO/experiments/training/materialize_p22_fineweb.py" ]] ||
    die "preprocessor alias source is absent"
}

validate_control_sources() {
  local expected_head="$1"
  local expected_tree="$2"
  assert_clean_checkout \
    "$P28_CONTROL_REPO" "$expected_head" "$expected_tree" "P28 control checkout"

  local contract="$P28_CONTROL_REPO/experiments/training/p28_bundle_complete_localization_bridge_contract.json"
  local orchestrator="$P28_CONTROL_REPO/scripts/run_p28_bundle_complete_localization_bridge.sh"
  local reconstructor="$P28_CONTROL_REPO/scripts/reconstruct_p28_bundle_complete_localization_bridge.py"
  local bundle_verifier="$P28_CONTROL_REPO/scripts/verify_p28_control_bundle.py"
  local p27_contract="$P28_CONTROL_REPO/experiments/training/p27_cuda_deleted_mapping_localization_contract.json"
  local p27_reconstructor="$P28_CONTROL_REPO/scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
  local p27_localizer="$P28_CONTROL_REPO/experiments/training/run_p27_cuda_deleted_mapping_localization.py"
  local ingester="$P28_CONTROL_REPO/scripts/ingest_p26_trace_off_a_failure.py"
  local p23_core="$P28_CONTROL_REPO/experiments/training/p23_deterministic_cuda_shadow_trace.py"
  local p27_sanitizer="$P28_CONTROL_REPO/scripts/sanitize_p27_cuda_deleted_mapping_localization.py"
  local p23_runner="$P28_CONTROL_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
  local executing_orchestrator
  executing_orchestrator="$(/usr/bin/realpath -- "${BASH_SOURCE[0]}")"
  local path
  for path in "$contract" "$orchestrator" "$reconstructor" "$bundle_verifier" \
    "$p27_contract" "$p27_reconstructor" "$p27_localizer" "$ingester" \
    "$p23_core" "$p27_sanitizer" "$p23_runner"; do
    [[ -f "$path" && ! -L "$path" ]] || die "reviewed control source is absent: $path"
  done
  [[ "$(sha256_file "$contract")" == "$P28_EXPECTED_CONTRACT_SHA256" ]] ||
    die "P28 contract bytes differ"
  [[ "$(sha256_file "$orchestrator")" == "$P28_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "P28 host orchestrator bytes differ"
  [[ "$executing_orchestrator" == "$(/usr/bin/realpath -- "$orchestrator")" && \
     ! -L "${BASH_SOURCE[0]}" ]] ||
    die "the executing P28 orchestrator is not the reviewed control source"
  [[ "$(sha256_file "$executing_orchestrator")" == \
      "$P28_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "executing P28 host orchestrator bytes differ"
  [[ "$(sha256_file "$reconstructor")" == "$P28_EXPECTED_RECONSTRUCTOR_SHA256" ]] ||
    die "P28 contract reconstructor bytes differ"
  [[ "$(sha256_file "$bundle_verifier")" == "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P28 bundle verifier bytes differ"
  [[ "$(sha256_file "$p27_contract")" == "$P28_EXPECTED_P27_CONTRACT_SHA256" ]] ||
    die "frozen P27 contract bytes differ"
  [[ "$(sha256_file "$p27_reconstructor")" == "$P28_EXPECTED_P27_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P27 contract reconstructor bytes differ"
  [[ "$(sha256_file "$p27_localizer")" == "$P28_EXPECTED_P27_LOCALIZER_SHA256" ]] ||
    die "frozen P27 localizer bytes differ"
  [[ "$(sha256_file "$ingester")" == "$P28_EXPECTED_INGESTER_SHA256" ]] ||
    die "P26 failure ingester bytes differ"
  [[ "$(sha256_file "$p23_core")" == "$P28_EXPECTED_P23_CORE_SHA256" ]] ||
    die "corrected P23 core bytes differ"
  [[ "$(sha256_file "$p27_sanitizer")" == "$P28_EXPECTED_P27_SANITIZER_SHA256" ]] ||
    die "frozen P27 sanitizer bytes differ"
  [[ "$(sha256_file "$p23_runner")" == "$P28_P23_RUNNER_SHA256" ]] ||
    die "frozen P23 runner bytes differ"

  local relative
  for relative in \
    experiments/training/p28_bundle_complete_localization_bridge_contract.json \
    scripts/run_p28_bundle_complete_localization_bridge.sh \
    scripts/reconstruct_p28_bundle_complete_localization_bridge.py \
    scripts/verify_p28_control_bundle.py \
    experiments/training/p27_cuda_deleted_mapping_localization_contract.json \
    scripts/reconstruct_p27_cuda_deleted_mapping_localization.py \
    experiments/training/run_p27_cuda_deleted_mapping_localization.py \
    scripts/ingest_p26_trace_off_a_failure.py \
    experiments/training/p23_deterministic_cuda_shadow_trace.py \
    scripts/sanitize_p27_cuda_deleted_mapping_localization.py \
    experiments/training/run_p23_deterministic_cuda_shadow_trace.py; do
    [[ "$(git -C "$P28_CONTROL_REPO" ls-files --error-unmatch -- "$relative")" == \
        "$relative" ]] || die "reviewed control source is not tracked: $relative"
  done
}

validate_source_freeze_history() {
  [[ "$(git -C "$P28_CONTROL_REPO" rev-parse \
      "$P28_SOURCE_FREEZE_COMMIT^{tree}")" == "$P28_SOURCE_FREEZE_TREE" ]] ||
    die "P28 source-freeze commit/tree binding differs"
  [[ "$(git -C "$P28_CONTROL_REPO" rev-list --parents -n 1 \
      "$P28_SOURCE_FREEZE_COMMIT")" == \
      "$P28_SOURCE_FREEZE_COMMIT $P28_TERMINAL_P27_COMMIT" ]] ||
    die "P28 source freeze must be one direct child of the terminal P27 outcome"
  [[ "$(git -C "$P28_CONTROL_REPO" rev-parse \
      "$P28_TERMINAL_P27_COMMIT^{tree}")" == "$P28_TERMINAL_P27_TREE" ]] ||
    die "terminal P27 commit/tree binding differs"
  if git -C "$P28_CONTROL_REPO" cat-file -e \
    "$P28_SOURCE_FREEZE_COMMIT:experiments/training/p28_cuda_runtime_lock.json" \
    2>/dev/null || git -C "$P28_CONTROL_REPO" cat-file -e \
    "$P28_SOURCE_FREEZE_COMMIT:experiments/training/p28_host_attestation.json" \
    2>/dev/null; then
    die "P28 runtime artifacts unexpectedly predate runtime preparation"
  fi
}

validate_runtime_review_history() {
  [[ "$(git -C "$P28_CONTROL_REPO" rev-list --parents -n 1 \
      "$P28_RUNTIME_REVIEW_COMMIT")" == \
      "$P28_RUNTIME_REVIEW_COMMIT $P28_SOURCE_FREEZE_COMMIT" ]] ||
    die "P28 runtime review must be one direct child of the source freeze"
  local expected_delta
  expected_delta=$'A\texperiments/training/p28_cuda_runtime_lock.json\nA\texperiments/training/p28_host_attestation.json'
  [[ "$(git -C "$P28_CONTROL_REPO" diff --name-status \
      "$P28_SOURCE_FREEZE_COMMIT" "$P28_RUNTIME_REVIEW_COMMIT")" == \
      "$expected_delta" ]] ||
    die "runtime review must add only the P28 lock and host attestation"
  validate_source_freeze_history
}

run_contract_reconstruction() {
  local label="$1"
  local reconstructor="$P28_CONTROL_REPO/scripts/reconstruct_p28_bundle_complete_localization_bridge.py"
  local contract="$P28_CONTROL_REPO/experiments/training/p28_bundle_complete_localization_bridge_contract.json"
  local status=0
  run_logged "$label" /usr/bin/python3 "$reconstructor" --canonical "$contract" ||
    status=$?
  (( status == 0 )) || die "P28 contract reconstruction failed with exit $status"
}

run_p27_reconstruction_unlogged() {
  /usr/bin/python3 \
    "$P28_CONTROL_REPO/scripts/reconstruct_p27_cuda_deleted_mapping_localization.py" \
    --canonical \
    "$P28_CONTROL_REPO/experiments/training/p27_cuda_deleted_mapping_localization_contract.json" \
    >/dev/null || die "frozen P27 reconstruction did not pass 23/23"
}

run_p28_reconstruction_unlogged() {
  /usr/bin/python3 \
    "$P28_CONTROL_REPO/scripts/reconstruct_p28_bundle_complete_localization_bridge.py" \
    --canonical \
    "$P28_CONTROL_REPO/experiments/training/p28_bundle_complete_localization_bridge_contract.json" \
    >/dev/null || die "P28 bridge reconstruction did not pass"
}

validate_transport_execution_sources() {
  local orchestrator="$P28_CONTROL_REPO/scripts/run_p28_bundle_complete_localization_bridge.sh"
  local verifier="$P28_CONTROL_REPO/scripts/verify_p28_control_bundle.py"
  local executing_orchestrator
  executing_orchestrator="$(/usr/bin/realpath -- "${BASH_SOURCE[0]}")"
  [[ -f "$orchestrator" && ! -L "$orchestrator" && \
     -f "$verifier" && ! -L "$verifier" && \
     -f "$P28_BOOTSTRAP_VERIFIER" && ! -L "$P28_BOOTSTRAP_VERIFIER" ]] ||
    die "P28 transport execution sources are absent or symlinks"
  [[ "$executing_orchestrator" == "$(/usr/bin/realpath -- "$orchestrator")" && \
     ! -L "${BASH_SOURCE[0]}" ]] ||
    die "the executing P28 orchestrator is not the reviewed control source"
  [[ "$(sha256_file "$executing_orchestrator")" == \
      "$P28_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "executing P28 host orchestrator bytes differ before transport replay"
  [[ "$(sha256_file "$verifier")" == "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P28 bundle verifier bytes differ before transport replay"
  [[ "$(sha256_file "$P28_BOOTSTRAP_VERIFIER")" == \
      "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P28 bootstrap bundle verifier bytes differ before transport replay"
  [[ "$(/usr/bin/stat -c '%a' "$P28_BOOTSTRAP_VERIFIER")" == "600" ]] ||
    die "P28 bootstrap bundle verifier mode differs"
}

verify_source_transport_unlogged() {
  validate_transport_execution_sources
  /usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER" \
    --phase source \
    --bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" \
    --transport-root "$P28_TRANSPORT_ROOT" \
    --bundle "$P28_SOURCE_BUNDLE" \
    --expected-bundle-sha256 "$P28_EXPECTED_SOURCE_BUNDLE_SHA256" \
    --expected-bundle-byte-count "$P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
    --closure-repository "$P28_SOURCE_CLOSURE" \
    --control-checkout "$P28_CONTROL_REPO" \
    --attempt-root "$P28_ATTEMPT_ROOT" \
    --expected-p28-commit "$P28_SOURCE_FREEZE_COMMIT" \
    --expected-p28-tree "$P28_SOURCE_FREEZE_TREE" \
    --verify-existing-receipt "$P28_SOURCE_RECEIPT" \
    --expected-receipt-sha256 "$P28_EXPECTED_SOURCE_RECEIPT_SHA256" \
    >/dev/null || die "reviewed P28 source-bundle receipt did not replay exactly"
}

verify_runtime_review_transport_unlogged() {
  validate_transport_execution_sources
  /usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER" \
    --phase runtime-review \
    --bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" \
    --transport-root "$P28_TRANSPORT_ROOT" \
    --bundle "$P28_RUNTIME_REVIEW_BUNDLE" \
    --expected-bundle-sha256 "$P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
    --expected-bundle-byte-count "$P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
    --closure-repository "$P28_RUNTIME_REVIEW_CLOSURE" \
    --control-checkout "$P28_CONTROL_REPO" \
    --attempt-root "$P28_ATTEMPT_ROOT" \
    --expected-p28-commit "$P28_RUNTIME_REVIEW_COMMIT" \
    --expected-p28-tree "$P28_RUNTIME_REVIEW_TREE" \
    --expected-source-commit "$P28_SOURCE_FREEZE_COMMIT" \
    --expected-source-tree "$P28_SOURCE_FREEZE_TREE" \
    --verify-existing-receipt "$P28_RUNTIME_REVIEW_RECEIPT" \
    --expected-receipt-sha256 "$P28_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256" \
    >/dev/null || die "reviewed P28 runtime-review bundle receipt did not replay exactly"
}

prepare_runtime() {
  # Receipt replay is a read-only precondition. This phase permits exactly one
  # subsequent state-changing prepare attempt; an existing attempt root or
  # container is terminal and never cleaned up, resumed, or retried here.
  require_source_environment
  [[ ! -e "$P28_ATTEMPT_ROOT" && ! -L "$P28_ATTEMPT_ROOT" ]] ||
    die "fresh attempt root already exists; cleanup and retries are forbidden"

  # The complete source bundle and its externally reviewed receipt are
  # authenticated first. The frozen P27 reconstruction must then pass 23/23.
  # No attempt-root directory, evidence log, or container may predate these
  # two fail-closed checks.
  verify_source_transport_unlogged
  validate_control_sources "$P28_SOURCE_FREEZE_COMMIT" "$P28_SOURCE_FREEZE_TREE"
  validate_source_freeze_history
  run_p27_reconstruction_unlogged
  run_p28_reconstruction_unlogged
  [[ ! -e "$P28_ATTEMPT_ROOT" && ! -L "$P28_ATTEMPT_ROOT" ]] ||
    die "source replay or reconstruction created the forbidden attempt root"
  if sudo docker container inspect "$P28_CONTAINER" >/dev/null 2>&1; then
    die "fresh container name already has state"
  fi
  validate_authority_and_inputs
  [[ "$(sudo docker image inspect --format '{{.Id}}' "$P28_IMAGE")" == \
      "$P28_IMAGE_DIGEST" ]] || die "reviewed P28 image digest is unavailable"

  sudo /usr/bin/install -d -m 0700 -o "$(id -u)" -g "$(id -g)" \
    "$P28_ATTEMPT_ROOT" "$P28_HOST_EVIDENCE"
  run_contract_reconstruction p28-prepare-contract-reconstruction

  capture_new p28-image-inspection "$P28_ATTEMPT_ROOT/p28-image-inspect.json" \
    sudo docker image inspect "$P28_IMAGE"
  capture_new p28-nvidia-smi "$P28_ATTEMPT_ROOT/p28-nvidia-smi.csv" \
    nvidia-smi --id="$P28_GPU_UUID" \
    --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
    --format=csv,noheader,nounits
  /usr/bin/install -m 0644 /dev/null \
    "$P28_ATTEMPT_ROOT/p28-running-container-inspect.json"
  /usr/bin/install -m 0644 /dev/null \
    "$P28_ATTEMPT_ROOT/p28-running-mountinfo.txt"

  capture_new p28-container-launch "$P28_ATTEMPT_ROOT/p28-container-id.txt" \
    sudo docker run --detach --name "$P28_CONTAINER" \
    --gpus "device=$P28_GPU_UUID" \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824 \
    --env PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env VIRTUAL_ENV=/opt/p23-venv \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env CUDA_VISIBLE_DEVICES="$P28_GPU_UUID" \
    --env NVIDIA_VISIBLE_DEVICES="$P28_GPU_UUID" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env PYTHONHASHSEED=1337 \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env P23_REPO=/workspace/OptimizationML \
    --env P23_NANOGPT=/workspace/inputs/nanoGPT \
    --env P23_MUON_ROOT=/workspace/inputs/muon \
    --mount type=bind,src="$P28_AUTHORITY_REPO",dst=/workspace/OptimizationML,readonly \
    --mount type=bind,src="$P28_NANOGPT_HOST",dst=/workspace/inputs/nanoGPT,readonly \
    --mount type=bind,src="$P28_MUON_HOST",dst=/workspace/inputs/muon,readonly \
    --mount type=bind,src="$P28_DATA_HOST",dst=/private/tmp/optimizationml-p22-data,readonly \
    --mount type=bind,src="$P28_AUTHORITY_REPO/experiments/training/materialize_p22_fineweb.py",dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly \
    --mount type=bind,src="$P28_HOST_EVIDENCE",dst=/workspace/evidence/p23 \
    --mount type=bind,src="$P28_ATTEMPT_ROOT/p28-image-inspect.json",dst=/mounted-host-evidence/image-inspect.json,readonly \
    --mount type=bind,src="$P28_ATTEMPT_ROOT/p28-running-container-inspect.json",dst=/mounted-host-evidence/running-container-inspect.json,readonly \
    --mount type=bind,src="$P28_ATTEMPT_ROOT/p28-running-mountinfo.txt",dst=/mounted-host-evidence/running-mountinfo.txt,readonly \
    --mount type=bind,src="$P28_ATTEMPT_ROOT/p28-nvidia-smi.csv",dst=/mounted-host-evidence/nvidia-smi.csv,readonly \
    --entrypoint /bin/sh \
    "$P28_IMAGE" -c 'while :; do sleep 3600; done'

  local container_id
  container_id="$(tr -d '\n' <"$P28_ATTEMPT_ROOT/p28-container-id.txt")"
  [[ "$container_id" =~ ^[0-9a-f]{64}$ ]] || die "fresh container ID is malformed"
  [[ "$container_id" != "$P28_TERMINAL_P26_CONTAINER_ID" ]] ||
    die "terminal P26 container was reused"
  [[ "$(sudo docker inspect --format '{{.Id}}' "$P28_CONTAINER")" == "$container_id" ]] ||
    die "fresh container ID differs from launch output"

  refresh_bound_file p28-running-container-inspection \
    "$P28_ATTEMPT_ROOT/p28-running-container-inspect.json" \
    sudo docker inspect "$P28_CONTAINER"
  local init_pid
  init_pid="$(sudo docker inspect --format '{{.State.Pid}}' "$P28_CONTAINER")"
  [[ "$init_pid" =~ ^[1-9][0-9]*$ ]] || die "fresh container init PID is malformed"
  refresh_bound_file p28-running-mountinfo \
    "$P28_ATTEMPT_ROOT/p28-running-mountinfo.txt" \
    sudo /usr/bin/nsenter --target "$init_pid" --mount --pid --cgroup -- \
    /usr/bin/cat /proc/self/mountinfo

  local freeze_status=0
  run_logged p28-freeze-runtime sudo docker exec \
    --env "CUDA_VISIBLE_DEVICES=$P28_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P28_GPU_UUID" \
    "$P28_CONTAINER" /opt/p23-venv/bin/python \
    /workspace/OptimizationML/experiments/training/run_p23_deterministic_cuda_shadow_trace.py \
    freeze-runtime \
    --container-image "$P28_IMAGE" \
    --container-repository-digest "$P28_IMAGE_DIGEST" \
    --host-image-inspection /mounted-host-evidence/image-inspect.json \
    --host-running-container-inspection /mounted-host-evidence/running-container-inspect.json \
    --host-running-mountinfo /mounted-host-evidence/running-mountinfo.txt \
    --host-nvidia-smi-query /mounted-host-evidence/nvidia-smi.csv \
    --host-attestation-output /workspace/evidence/p23/p28_host_attestation.json \
    --runtime-lock-output /workspace/evidence/p23/p28_cuda_runtime_lock.json ||
    freeze_status=$?
  (( freeze_status == 0 )) ||
    die "P28 freeze-runtime failed with exit $freeze_status; attempt is retained"
  [[ -f "$P28_HOST_EVIDENCE/p28_cuda_runtime_lock.json" && \
     -f "$P28_HOST_EVIDENCE/p28_host_attestation.json" ]] ||
    die "P28 freeze-runtime did not retain both runtime artifacts"

  echo "P28 runtime frozen; localization has not run."
  echo "container_id=$container_id"
  echo "container_init_pid=$init_pid"
  echo "runtime_lock_sha256=$(root_sha256_file "$P28_HOST_EVIDENCE/p28_cuda_runtime_lock.json")"
  echo "host_attestation_sha256=$(root_sha256_file "$P28_HOST_EVIDENCE/p28_host_attestation.json")"
  echo "Review and commit exactly the two runtime artifacts as one direct child of $P28_SOURCE_FREEZE_COMMIT."
}

stage_source() {
  local source="$1"
  local destination="$2"
  local expected_sha256="$3"
  [[ "$(sha256_file "$source")" == "$expected_sha256" ]] ||
    die "staged source authority hash differs: $source"
  [[ ! -e "$destination" && ! -L "$destination" ]] ||
    die "staged source destination already exists: $destination"
  sudo /usr/bin/install -m 0400 -o root -g root "$source" "$destination"
  [[ "$(root_sha256_file "$destination")" == "$expected_sha256" ]] ||
    die "staged source destination hash differs: $destination"
  [[ "$(/usr/bin/stat -c '%s' "$source")" == \
      "$(sudo /usr/bin/stat -c '%s' "$destination")" ]] ||
    die "staged source byte count differs: $destination"
}

validate_live_runtime() {
  local label="$1"
  run_logged "$label-container-inspect" sudo docker inspect "$P28_CONTAINER"
  run_logged "$label-mountinfo" sudo docker exec "$P28_CONTAINER" \
    /usr/bin/cat /proc/self/mountinfo
  run_logged "$label-nvidia-smi" nvidia-smi --id="$P28_GPU_UUID" \
    --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
    --format=csv,noheader,nounits

  /usr/bin/python3 - \
    "$P28_CONTROL_REPO/experiments/training/p28_cuda_runtime_lock.json" \
    "$P28_CONTROL_REPO/experiments/training/p28_host_attestation.json" \
    "$P28_HOST_EVIDENCE/$label-container-inspect.stdout.log" \
    "$P28_HOST_EVIDENCE/$label-mountinfo.stdout.log" \
    "$P28_HOST_EVIDENCE/$label-nvidia-smi.stdout.log" \
    "$P28_ATTEMPT_ROOT/p28-nvidia-smi.csv" \
    "$P28_ATTEMPT_ROOT/p28-container-id.txt" \
    "$P28_IMAGE" "$P28_IMAGE_DIGEST" "$P28_GPU_UUID" \
    "$P28_AUTHORITY_REPO" "$P28_NANOGPT_HOST" "$P28_MUON_HOST" \
    "$P28_DATA_HOST" "$P28_HOST_EVIDENCE" <<'PY'
import hashlib
import json
import pathlib
import sys

def strict(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError(f"nonfinite JSON constant in {path}: {token}")
    return json.loads(
        pathlib.Path(path).read_bytes(),
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
    )

(lock_path, attestation_path, inspect_path, mountinfo_path, gpu_path,
 initial_gpu_path, container_id_path, image, image_digest, gpu_uuid,
 authority, nanogpt, muon, data, evidence) = sys.argv[1:]
lock = strict(lock_path)
attestation = strict(attestation_path)
inspection = strict(inspect_path)
if not isinstance(inspection, list) or len(inspection) != 1:
    raise SystemExit("live Docker inspection is not one object")
inspection = inspection[0]
container = lock.get("container")
if not isinstance(container, dict):
    raise SystemExit("runtime lock container record is malformed")
container_id = pathlib.Path(container_id_path).read_text().strip()
if container_id == (
    "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c"
):
    raise SystemExit("terminal P26 container ID was reused")
checks = {
    "runtime_schema": lock.get("schema_version") == "passive-muon-p23-cuda-runtime-lock-v3",
    "runtime_status": lock.get("status") == "pinned_for_acquisition",
    "attestation_schema": attestation.get("schema_version") == "passive-muon-p23-host-attestation-v3",
    "attestation_status": attestation.get("status") == "procedurally_host_attested",
    "container_id": inspection.get("Id") == container_id == container.get("container_id"),
    "running": inspection.get("State", {}).get("Running") is True,
    "pid": inspection.get("State", {}).get("Pid") == container.get("container_init_pid"),
    "restart": inspection.get("RestartCount") == 0,
    "image_reference": inspection.get("Config", {}).get("Image") == image == container.get("image"),
    "image_id": inspection.get("Image") == image_digest == container.get("image_id"),
    "repository_digest": container.get("repository_digest") == image_digest,
    "network": inspection.get("HostConfig", {}).get("NetworkMode") == "none",
    "rootfs": inspection.get("HostConfig", {}).get("ReadonlyRootfs") is True,
    "tmpfs": inspection.get("HostConfig", {}).get("Tmpfs", {}).get("/tmp")
        == "rw,noexec,nosuid,nodev,size=1073741824",
    "mountinfo": hashlib.sha256(pathlib.Path(mountinfo_path).read_bytes()).hexdigest()
        == container.get("mountinfo_sha256"),
    "gpu_query": pathlib.Path(gpu_path).read_bytes()
        == pathlib.Path(initial_gpu_path).read_bytes(),
    "gpu_uuid": lock.get("gpu", {}).get("uuid") == gpu_uuid,
    "gpu_name": lock.get("gpu", {}).get("name") == "NVIDIA A100-SXM4-40GB",
    "attestation_container": attestation.get("container", {}).get("container_id") == container_id,
    "attestation_pid": attestation.get("container", {}).get("container_init_pid")
        == container.get("container_init_pid"),
    "attestation_gpu": attestation.get("gpu") == lock.get("gpu"),
    "attestation_hash": hashlib.sha256(pathlib.Path(attestation_path).read_bytes()).hexdigest()
        == container.get("host_attestation_sha256"),
}
required_env = {
    "PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "VIRTUAL_ENV=/opt/p23-venv",
    "PYTHONNOUSERSITE=1",
    "PYTHONDONTWRITEBYTECODE=1",
    f"CUDA_VISIBLE_DEVICES={gpu_uuid}",
    f"NVIDIA_VISIBLE_DEVICES={gpu_uuid}",
    "CUBLAS_WORKSPACE_CONFIG=:4096:8",
    "PYTHONHASHSEED=1337",
    "NVIDIA_TF32_OVERRIDE=0",
    "OMP_NUM_THREADS=1",
    "MKL_NUM_THREADS=1",
    "P23_REPO=/workspace/OptimizationML",
    "P23_NANOGPT=/workspace/inputs/nanoGPT",
    "P23_MUON_ROOT=/workspace/inputs/muon",
}
checks["environment"] = required_env <= set(inspection.get("Config", {}).get("Env", []))
requests = inspection.get("HostConfig", {}).get("DeviceRequests")
checks["device_request"] = (
    isinstance(requests, list)
    and len(requests) == 1
    and requests[0].get("DeviceIDs") == [gpu_uuid]
    and requests[0].get("Capabilities") == [["gpu"]]
)
preprocessor = str(pathlib.Path(authority) / "experiments/training/materialize_p22_fineweb.py")
expected_mounts = {
    "/workspace/OptimizationML": (authority, False),
    "/workspace/inputs/nanoGPT": (nanogpt, False),
    "/workspace/inputs/muon": (muon, False),
    "/private/tmp/optimizationml-p22-data": (data, False),
    "/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py": (preprocessor, False),
    "/workspace/evidence/p23": (evidence, True),
    "/mounted-host-evidence/image-inspect.json": (
        str(pathlib.Path(evidence).parent / "p28-image-inspect.json"), False
    ),
    "/mounted-host-evidence/running-container-inspect.json": (
        str(pathlib.Path(evidence).parent / "p28-running-container-inspect.json"), False
    ),
    "/mounted-host-evidence/running-mountinfo.txt": (
        str(pathlib.Path(evidence).parent / "p28-running-mountinfo.txt"), False
    ),
    "/mounted-host-evidence/nvidia-smi.csv": (
        str(pathlib.Path(evidence).parent / "p28-nvidia-smi.csv"), False
    ),
}
mounts = inspection.get("Mounts")
observed_mounts = {}
if isinstance(mounts, list):
    for record in mounts:
        if isinstance(record, dict):
            observed_mounts[record.get("Destination")] = (
                record.get("Source"), record.get("RW") is True
            )
checks["exact_ten_mounts"] = len(mounts or []) == 10 and observed_mounts == expected_mounts
failed = sorted(name for name, value in checks.items() if not value)
if failed:
    raise SystemExit(f"live P28 runtime validation failed: {failed}")
PY
}

validate_reviewed_runtime_artifacts() {
  local committed_lock="$P28_CONTROL_REPO/experiments/training/p28_cuda_runtime_lock.json"
  local committed_attestation="$P28_CONTROL_REPO/experiments/training/p28_host_attestation.json"
  local native_lock="$P28_HOST_EVIDENCE/p28_cuda_runtime_lock.json"
  local native_attestation="$P28_HOST_EVIDENCE/p28_host_attestation.json"
  [[ -f "$committed_lock" && -f "$committed_attestation" && \
     -f "$native_lock" && -f "$native_attestation" ]] ||
    die "reviewed P28 runtime artifacts are incomplete"
  [[ "$(sha256_file "$committed_lock")" == "$P28_EXPECTED_RUNTIME_LOCK_SHA256" ]] ||
    die "committed P28 runtime lock differs"
  [[ "$(sha256_file "$committed_attestation")" == "$P28_EXPECTED_HOST_ATTESTATION_SHA256" ]] ||
    die "committed P28 host attestation differs"
  [[ "$(root_sha256_file "$native_lock")" == "$P28_EXPECTED_RUNTIME_LOCK_SHA256" ]] ||
    die "retained P28 runtime lock differs"
  [[ "$(root_sha256_file "$native_attestation")" == "$P28_EXPECTED_HOST_ATTESTATION_SHA256" ]] ||
    die "retained P28 host attestation differs"
  sudo /usr/bin/cmp -s "$committed_lock" "$native_lock" ||
    die "committed and retained P28 runtime locks differ"
  sudo /usr/bin/cmp -s "$committed_attestation" "$native_attestation" ||
    die "committed and retained P28 host attestations differ"
}

assert_staged_source() {
  local path="$1"
  local expected_sha256="$2"
  [[ "$(root_sha256_file "$path")" == "$expected_sha256" ]] ||
    die "staged execution source changed: $path"
  [[ "$(sudo /usr/bin/stat -c '%a:%u:%g' "$path")" == "400:0:0" ]] ||
    die "staged execution source mode/owner differs: $path"
}

run_localization() {
  require_runtime_environment
  [[ -d "$P28_HOST_EVIDENCE" ]] || die "fresh P28 evidence directory is absent"
  local invocation_marker="$P28_HOST_EVIDENCE/p28-run-localization.invoked"
  [[ ! -e "$invocation_marker" && ! -L "$invocation_marker" ]] ||
    die "run-localization was already invoked; retry is forbidden"

  # Caller-supplied review identifiers and clean-checkout preconditions are
  # read-only administrative checks.  Burn the one-shot marker only after all
  # of them pass, but before the first retained process log, source staging,
  # one-read P26 ingestion, or CUDA-localization process.
  verify_runtime_review_transport_unlogged
  validate_control_sources "$P28_RUNTIME_REVIEW_COMMIT" "$P28_RUNTIME_REVIEW_TREE"
  validate_runtime_review_history
  run_p27_reconstruction_unlogged
  run_p28_reconstruction_unlogged
  validate_authority_and_inputs
  validate_reviewed_runtime_artifacts
  write_new_status "$invocation_marker" run-localization
  run_contract_reconstruction p28-run-contract-reconstruction
  validate_live_runtime p28-pre-ingestion

  local staged_localizer="$P28_HOST_EVIDENCE/run_p27_cuda_deleted_mapping_localization.py"
  local staged_ingester="$P28_HOST_EVIDENCE/ingest_p26_trace_off_a_failure.py"
  local staged_p23_core="$P28_HOST_EVIDENCE/p23_deterministic_cuda_shadow_trace.py"
  local staged_sanitizer="$P28_HOST_EVIDENCE/sanitize_p27_cuda_deleted_mapping_localization.py"
  stage_source \
    "$P28_CONTROL_REPO/experiments/training/run_p27_cuda_deleted_mapping_localization.py" \
    "$staged_localizer" "$P28_EXPECTED_P27_LOCALIZER_SHA256"
  stage_source "$P28_CONTROL_REPO/scripts/ingest_p26_trace_off_a_failure.py" \
    "$staged_ingester" "$P28_EXPECTED_INGESTER_SHA256"
  stage_source \
    "$P28_CONTROL_REPO/experiments/training/p23_deterministic_cuda_shadow_trace.py" \
    "$staged_p23_core" "$P28_EXPECTED_P23_CORE_SHA256"
  stage_source "$P28_CONTROL_REPO/scripts/sanitize_p27_cuda_deleted_mapping_localization.py" \
    "$staged_sanitizer" "$P28_EXPECTED_P27_SANITIZER_SHA256"

  local p26_native_copy="$P28_HOST_EVIDENCE/p26-trace-off-a-failure.authenticated.json"
  local p26_sanitized="$P28_HOST_EVIDENCE/p26-trace-off-a-failure.sanitized.json"
  local ingestion_status=0
  assert_staged_source "$staged_ingester" "$P28_EXPECTED_INGESTER_SHA256"
  assert_staged_source "$staged_p23_core" "$P28_EXPECTED_P23_CORE_SHA256"
  [[ "$(sha256_file "$P28_CONTROL_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py")" == \
      "$P28_P23_RUNNER_SHA256" ]] || die "P23 ingestion runner changed"
  run_logged p28-p26-failure-ingestion sudo /usr/bin/python3 "$staged_ingester" \
    --source "$P28_P26_NATIVE" \
    --native-copy "$p26_native_copy" \
    --sanitized-output "$p26_sanitized" \
    --output-root "$P28_HOST_EVIDENCE" \
    --p23-core "$staged_p23_core" \
    --p23-runner "$P28_CONTROL_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py" \
    --repository-root "$P28_CONTROL_REPO" \
    --path-root repository=/workspace/OptimizationML \
    --path-root preprocessor_alias=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
    --path-root nanogpt=/workspace/inputs/nanoGPT \
    --path-root muon=/workspace/inputs/muon \
    --path-root data=/private/tmp/optimizationml-p22-data \
    --path-root python_environment=/opt/p23-venv \
    --path-root native=/workspace/evidence/p23 || ingestion_status=$?
  write_new_status "$P28_HOST_EVIDENCE/p28-p26-failure-ingestion.exit-status.txt" \
    "$ingestion_status"
  (( ingestion_status == 0 )) ||
    die "terminal P26 failure ingestion failed with exit $ingestion_status before localization"
  [[ "$(root_sha256_file "$p26_native_copy")" == "$P28_P26_NATIVE_SHA256" ]] ||
    die "authenticated P26 failure copy hash differs"
  [[ "$(sudo /usr/bin/stat -c '%s:%a:%u:%g' "$p26_native_copy")" == \
      "$P28_P26_NATIVE_BYTE_COUNT:600:0:0" ]] ||
    die "authenticated P26 failure copy size/mode/owner differs"

  validate_control_sources "$P28_RUNTIME_REVIEW_COMMIT" "$P28_RUNTIME_REVIEW_TREE"
  validate_authority_and_inputs
  validate_reviewed_runtime_artifacts
  validate_live_runtime p28-pre-localization
  assert_staged_source "$staged_localizer" "$P28_EXPECTED_P27_LOCALIZER_SHA256"
  assert_staged_source "$staged_ingester" "$P28_EXPECTED_INGESTER_SHA256"
  assert_staged_source "$staged_p23_core" "$P28_EXPECTED_P23_CORE_SHA256"
  assert_staged_source "$staged_sanitizer" "$P28_EXPECTED_P27_SANITIZER_SHA256"

  local native="$P28_HOST_EVIDENCE/p28_cuda_deleted_mapping_localization.native.json"
  local sanitized="$P28_HOST_EVIDENCE/p28_cuda_deleted_mapping_localization.sanitized.json"
  [[ ! -e "$native" && ! -L "$native" && ! -e "$sanitized" && ! -L "$sanitized" ]] ||
    die "P28 localization output already exists"
  local localization_status=0
  run_logged p28-localization sudo docker exec \
    --env "CUDA_VISIBLE_DEVICES=$P28_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P28_GPU_UUID" \
    "$P28_CONTAINER" /opt/p23-venv/bin/python \
    /workspace/evidence/p23/run_p27_cuda_deleted_mapping_localization.py \
    --repository /workspace/OptimizationML \
    --nanogpt-root /workspace/inputs/nanoGPT \
    --muon-source /workspace/inputs/muon/muon.py \
    --p25-source /workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py \
    --runtime-lock /workspace/evidence/p23/p28_cuda_runtime_lock.json \
    --host-attestation /workspace/evidence/p23/p28_host_attestation.json \
    --p25-contract /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
    --expected-source-sha256 "$P28_EXPECTED_P27_LOCALIZER_SHA256" \
    --expected-p25-source-sha256 "$P28_P25_SOURCE_SHA256" \
    --expected-p25-contract-sha256 "$P28_P25_CONTRACT_SHA256" \
    --expected-runtime-lock-sha256 "$P28_EXPECTED_RUNTIME_LOCK_SHA256" \
    --expected-host-attestation-sha256 "$P28_EXPECTED_HOST_ATTESTATION_SHA256" \
    --expected-repository-head "$P28_AUTHORITY_HEAD" \
    --expected-repository-tree "$P28_AUTHORITY_TREE" \
    --output /workspace/evidence/p23/p28_cuda_deleted_mapping_localization.native.json ||
    localization_status=$?
  write_new_status "$P28_HOST_EVIDENCE/p28-localization.exit-status.txt" \
    "$localization_status"

  if [[ ! -e "$native" && ! -L "$native" ]]; then
    die "P28 localization retained no native artifact (exit $localization_status); no rerun is allowed"
  fi
  [[ "$(sudo /usr/bin/stat -c '%a:%u:%g' "$native")" == "600:0:0" ]] ||
    die "P28 native localization artifact mode/owner differs"
  local native_sha256 native_byte_count
  native_sha256="$(root_sha256_file "$native")"
  native_byte_count="$(sudo /usr/bin/stat -c '%s' "$native")"
  [[ "$native_sha256" =~ ^[0-9a-f]{64}$ && "$native_byte_count" =~ ^[1-9][0-9]*$ ]] ||
    die "P28 native localization binding is malformed"

  local sanitizer_status=0
  assert_staged_source "$staged_sanitizer" "$P28_EXPECTED_P27_SANITIZER_SHA256"
  assert_staged_source "$staged_localizer" "$P28_EXPECTED_P27_LOCALIZER_SHA256"
  run_logged p28-localization-sanitizer sudo /usr/bin/python3 "$staged_sanitizer" \
    --native "$native" \
    --output "$sanitized" \
    --expected-native-sha256 "$native_sha256" \
    --expected-native-byte-count "$native_byte_count" \
    --output-root "$P28_HOST_EVIDENCE" \
    --runner-source "$staged_localizer" \
    --repository-root "$P28_CONTROL_REPO" \
    --path-root repository=/workspace/OptimizationML \
    --path-root nanogpt=/workspace/inputs/nanoGPT \
    --path-root muon=/workspace/inputs/muon \
    --path-root data=/private/tmp/optimizationml-p22-data \
    --path-root preprocessor_alias=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
    --path-root native=/workspace/evidence/p23 \
    --path-root python_environment=/opt/p23-venv \
    --path-root usr=/usr \
    --path-root opt=/opt \
    --path-root dev=/dev \
    --path-root proc=/proc \
    --path-root tmp=/tmp \
    --path-root workspace=/workspace \
    --path-root private=/private \
    --path-root etc=/etc \
    --path-root var=/var \
    --path-root run=/run \
    --path-root sys=/sys \
    --path-root root_home=/root \
    --path-root home=/home \
    --path-root secure=/secure || sanitizer_status=$?
  write_new_status "$P28_HOST_EVIDENCE/p28-localization-sanitizer.exit-status.txt" \
    "$sanitizer_status"
  (( sanitizer_status == 0 )) ||
    die "P28 localization sanitizer failed with exit $sanitizer_status; native evidence is retained"
  [[ -f "$sanitized" ]] || die "P28 sanitizer retained no wrapper"
  (( localization_status == 0 )) ||
    die "P28 localization completed with exit $localization_status; sanitized failure evidence is retained"

  echo "P28 sole localization invocation and sanitizer completed. Do not rerun."
  echo "native_sha256=$native_sha256"
  echo "native_byte_count=$native_byte_count"
  echo "sanitized_sha256=$(root_sha256_file "$sanitized")"
}

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_p28_bundle_complete_localization_bridge.sh prepare-runtime|run-localization

There is deliberately no cleanup, retry, acquisition, forward, backward,
optimizer-step, candidate-evaluation, or training phase.
EOF
  exit 2
}

[[ $# == 1 ]] || usage
case "$1" in
  prepare-runtime) prepare_runtime ;;
  run-localization) run_localization ;;
  *) usage ;;
esac
