#!/usr/bin/env bash
# This is a disposable Linux engineering rehearsal, not a scientific P30 run.

set -Eeuo pipefail
umask 077

readonly P29_COMMIT=8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf
readonly P29_TAG_OBJECT=cf3c60b1d5c004b9e601a6afbf630358f80f1d79
readonly P28_COMMIT=7274367f2b05fb3c9ed9f876a59be647703bbdc2
readonly P28_TAG_OBJECT=f3c81a2f14f853f369015a4f81b6695b52eec74e
readonly P27_COMMIT=ec63550331925ded158e3f389e294e4d1f12db3a
readonly P27_TAG_OBJECT=c88b98eae9be2458abde45b05d3dccfe09c0c7ed
readonly P26_CHECKPOINT_OBJECT=bacad707d3779bfa10957e18cb4c69b1a7f0cbce
readonly P26_SOURCE_OBJECT=f35a7dca8f6bc39e9748e79b6712fab4a203396b

readonly P30_NAMESPACE_ROOT=/secure/p30
readonly P30_TRANSPORT_ROOT=/secure/p30/transport-20260906-03
readonly P30_ATTEMPT_ROOT=/secure/p30/attempt-20260906-03
readonly P30_EXECUTION_ROOT=/secure/p30/execution-20260906-03
readonly P30_LEDGER_ROOT=/var/lib/optimizationml-p30-20260906-03
readonly P30_CONTROL_REPO="$P30_TRANSPORT_ROOT/control-source"
readonly P30_SOURCE_BUNDLE="$P30_TRANSPORT_ROOT/p30_source.bundle"
readonly P30_SOURCE_CLOSURE="$P30_TRANSPORT_ROOT/p30_source_closure.git"
readonly P30_SOURCE_RECEIPT="$P30_TRANSPORT_ROOT/p30_source_bundle_receipt.json"
readonly P30_BOOTSTRAP_VERIFIER="$P30_TRANSPORT_ROOT/verify_p30_control_bundle.py"
readonly P30_SEALED_ORCHESTRATOR="$P30_LEDGER_ROOT/run_p30_umask_bound_control_seal.sh"

readonly P30_BRANCH_REF=refs/heads/p30-umask-bound-control-seal
readonly P29_BRANCH_REF=refs/heads/p29-permission-safe-bundle-localization-bridge
readonly P29_TAG_REF=refs/tags/p29-orchestrator-source-mode-diagnostic
readonly P28_BRANCH_REF=refs/heads/p28-bundle-complete-localization-bridge
readonly P28_TAG_REF=refs/tags/p28-control-parent-permission-diagnostic
readonly P27_BRANCH_REF=refs/heads/p27-cuda-deleted-mapping-localization
readonly P27_TAG_REF=refs/tags/p27-cuda-deleted-mapping-localization-diagnostic
readonly P26_CHECKPOINT_REF=refs/tags/p26-permission-safe-acquisition-checkpoint
readonly P26_SOURCE_REF=refs/tags/p26-attempt02-acquisition-source

readonly SCRIPT_ROOT="$({ cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P; })"
readonly SOURCE_COMMIT="${P30_REHEARSAL_SOURCE_COMMIT:-}"
readonly LOG_ROOT="${P30_REHEARSAL_LOG_ROOT:-}"
readonly STAGE_ROOT="${P30_REHEARSAL_STAGE_ROOT:-}"

SOURCE_TREE=
SOURCE_BUNDLE_SHA256=
SOURCE_BUNDLE_BYTE_COUNT=
VERIFIER_SHA256=
ORCHESTRATOR_SHA256=
RECEIPT_SHA256=
BARE_SOURCE=
LOCAL_SOURCE_BUNDLE=
LOCAL_VERIFIER=

die() {
  printf 'P30 UID handoff rehearsal failed: %s\n' "$*" >&2
  return 1
}

clean_git() {
  /usr/bin/env -i \
    HOME=/nonexistent \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    GIT_ATTR_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_TERMINAL_PROMPT=0 \
    /usr/bin/git "$@"
}

require_absolute_outside_scientific_roots() {
  local path="$1"
  local label="$2"
  [[ "$path" == /* ]] || die "$label must be absolute"
  case "$path/" in
    /secure/* | /var/lib/optimizationml-p30-* | "$SCRIPT_ROOT"/*)
      die "$label must remain outside the repository and frozen P30 roots"
      ;;
  esac
}

write_manifest() {
  local overall_status="$1"
  local manifest_status=0
  trap - EXIT
  set +e
  if [[ -d "$LOG_ROOT" ]]; then
    /usr/bin/python3 -I -S - \
      "$LOG_ROOT" "$overall_status" "$SOURCE_COMMIT" "$SOURCE_TREE" \
      "${GITHUB_RUN_ID:-not_available}" "${GITHUB_RUN_ATTEMPT:-not_available}" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

root = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
records = []
for path in sorted(root.iterdir()):
    if path.name == "rehearsal-manifest.json" or not path.is_file() or path.is_symlink():
        continue
    if not path.name.endswith((".stdout.log", ".stderr.log", ".exit-status.txt")):
        continue
    raw = path.read_bytes()
    records.append(
        {
            "name": path.name,
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "mode_octal": format(stat.S_IMODE(path.stat().st_mode), "04o"),
        }
    )
expected_labels = [
    "00-preflight",
    "01-bundle-create",
    "02-namespace-provision",
    "03-stage-source-bundle",
    "04-stage-bootstrap-verifier",
    "05-source-verifier-create",
    "06-source-receipt-review",
    "07-source-verifier-replay",
    "08-root-orchestrator-seal",
    "09-root-seal-verification",
    "10-final-evidence-boundary",
]
expected_names = {
    f"{label}.{suffix}"
    for label in expected_labels
    for suffix in ("stdout.log", "stderr.log", "exit-status.txt")
}
record_by_name = {record["name"]: record for record in records}
attempt_root_created = os.path.lexists("/secure/p30/attempt-20260906-03")
execution_root_created = os.path.lexists("/secure/p30/execution-20260906-03")
if status == 0:
    if set(record_by_name) != expected_names:
        raise SystemExit("successful rehearsal stream inventory differs")
    if any(record["mode_octal"] != "0600" for record in records):
        raise SystemExit("successful rehearsal stream mode differs")
    for label in expected_labels:
        if (root / f"{label}.exit-status.txt").read_bytes() != b"0\n":
            raise SystemExit(f"successful rehearsal command failed: {label}")
    if attempt_root_created or execution_root_created:
        raise SystemExit("successful rehearsal created a forbidden attempt or execution root")
payload = {
    "schema_version": "passive-muon-p30-uid1000-root-engineering-rehearsal-v1",
    "classification": "engineering_rehearsal_not_scientific_acquisition",
    "scientific_acquisition": False,
    "gpu_used": False,
    "cuda_initialized": False,
    "candidate_observations": 0,
    "training_steps": 0,
    "frozen_path_strings_mirrored_on_disposable_runner": True,
    "attempt_root_created": attempt_root_created,
    "execution_root_created": execution_root_created,
    "real_host_attempt_identity_consumed": False,
    "overall_exit_status": status,
    "source_commit": sys.argv[3] or None,
    "source_tree": sys.argv[4] or None,
    "github_run_id": sys.argv[5],
    "github_run_attempt": sys.argv[6],
    "command_streams": records,
}
target = root / "rehearsal-manifest.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
    stream.write("\n")
os.chmod(target, 0o600)
PY
    manifest_status=$?
    if [[ "$manifest_status" -eq 0 ]]; then
      [[ -f "$LOG_ROOT/rehearsal-manifest.json" &&
        ! -L "$LOG_ROOT/rehearsal-manifest.json" ]]
      manifest_status=$?
    fi
  else
    manifest_status=1
  fi
  if [[ "$overall_status" -eq 0 && "$manifest_status" -ne 0 ]]; then
    printf 'P30 UID handoff rehearsal failed: manifest finalization failed\n' >&2
    exit 125
  fi
  exit "$overall_status"
}

finalize_logged_failure() {
  local status="$1"
  local stdout_path="$2"
  local stderr_path="$3"
  local status_path="$4"
  trap - ERR
  set +e
  exec 8>&-
  exec 9>&-
  ( set -o noclobber; printf '%s\n' "$status" >"$status_path" )
  if [[ "$?" -ne 0 ]]; then
    status=125
  fi
  chmod 0600 "$stdout_path" "$stderr_path" "$status_path"
  if [[ "$?" -ne 0 ]]; then
    status=125
  fi
  exit "$status"
}

run_logged() {
  local label="$1"
  shift
  [[ "$label" =~ ^[a-z0-9-]+$ ]] || die "invalid log label: $label"
  local stdout_path="$LOG_ROOT/$label.stdout.log"
  local stderr_path="$LOG_ROOT/$label.stderr.log"
  local status_path="$LOG_ROOT/$label.exit-status.txt"
  [[ ! -e "$stdout_path" && ! -L "$stdout_path" ]] || die "log already exists: $stdout_path"
  [[ ! -e "$stderr_path" && ! -L "$stderr_path" ]] || die "log already exists: $stderr_path"
  [[ ! -e "$status_path" && ! -L "$status_path" ]] || die "log already exists: $status_path"
  set -o noclobber
  exec 8>"$stdout_path"
  exec 9>"$stderr_path"
  set +o noclobber
  trap 'finalize_logged_failure "$?" "$stdout_path" "$stderr_path" "$status_path"' ERR
  "$@" >&8 2>&9
  trap - ERR
  exec 8>&-
  exec 9>&-
  ( set -o noclobber; printf '0\n' >"$status_path" ) || return 125
  chmod 0600 "$stdout_path" "$stderr_path" "$status_path"
}

as_uid1000() {
  /usr/bin/sudo -n /usr/bin/setpriv \
    --reuid=1000 --regid=1000 --clear-groups --no-new-privs \
    /usr/bin/env -i \
    HOME=/nonexistent \
    LANG=C \
    LC_ALL=C \
    PATH=/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONSAFEPATH=1 \
    "$@"
}

preflight() {
  [[ "${GITHUB_ACTIONS:-}" == true ]] ||
    die "this script runs only on a disposable GitHub Actions Linux runner"
  [[ "$(uname -s)" == Linux ]] || die "Linux is required"
  [[ -x /usr/bin/python3 && -x /usr/bin/git && -x /usr/bin/setpriv ]] ||
    die "required absolute Linux tools are unavailable"
  /usr/bin/sudo -n /usr/bin/true
  local controller_uid
  controller_uid="$(/usr/bin/id -u)"
  [[ "$controller_uid" != 0 && "$controller_uid" != 1000 ]] ||
    die "the client controller must be distinct from root and UID 1000"
  [[ "$(/usr/bin/getent passwd 1000 | /usr/bin/cut -d: -f3)" == 1000 ]] ||
    die "numeric UID 1000 is unavailable"
  [[ "$(/usr/bin/getent group 1000 | /usr/bin/cut -d: -f3)" == 1000 ]] ||
    die "numeric GID 1000 is unavailable"
  [[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "source commit is malformed"
  [[ -z "$(clean_git -C "$SCRIPT_ROOT" config --local --name-only --list | /usr/bin/grep -Ei '^include(if)?\.' || true)" ]] ||
    die "the hosted checkout contains local Git include directives"
  [[ "$(clean_git -C "$SCRIPT_ROOT" cat-file -t "$SOURCE_COMMIT")" == commit ]] ||
    die "source commit is unavailable"
  local history
  history="$(clean_git -C "$SCRIPT_ROOT" rev-list --parents -n 1 "$SOURCE_COMMIT")"
  [[ "$history" == "$SOURCE_COMMIT $P29_COMMIT" ]] ||
    die "rehearsed P30 source is not one direct child of terminal P29"
  SOURCE_TREE="$(clean_git -C "$SCRIPT_ROOT" rev-parse "$SOURCE_COMMIT^{tree}")"
  [[ "$SOURCE_TREE" =~ ^[0-9a-f]{40}$ ]] || die "source tree is malformed"
  [[ ! -e "$P30_NAMESPACE_ROOT" && ! -L "$P30_NAMESPACE_ROOT" ]] ||
    die "$P30_NAMESPACE_ROOT must be absent on the disposable runner"
  [[ ! -e "$P30_LEDGER_ROOT" && ! -L "$P30_LEDGER_ROOT" ]] ||
    die "$P30_LEDGER_ROOT must be absent on the disposable runner"
  printf 'classification=engineering_rehearsal_not_scientific_acquisition\n'
  printf 'frozen_path_strings_mirrored_on_disposable_runner=true\n'
  printf 'attempt_root_created=false\nreal_host_attempt_identity_consumed=false\n'
  printf 'source_commit=%s\nsource_tree=%s\n' "$SOURCE_COMMIT" "$SOURCE_TREE"
  /usr/bin/uname -a
  /usr/bin/python3 --version
  /usr/bin/git --version
  /usr/bin/id
  as_uid1000 /usr/bin/id
  /usr/bin/sudo -n /usr/bin/id
}

build_closed_source_bundle() {
  BARE_SOURCE="$STAGE_ROOT/source.git"
  LOCAL_SOURCE_BUNDLE="$STAGE_ROOT/p30_source.bundle"
  LOCAL_VERIFIER="$STAGE_ROOT/verify_p30_control_bundle.py"
  clean_git -c protocol.file.allow=always clone --mirror "$SCRIPT_ROOT" "$BARE_SOURCE"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P30_BRANCH_REF" "$SOURCE_COMMIT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P29_BRANCH_REF" "$P29_COMMIT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P29_TAG_REF" "$P29_TAG_OBJECT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P28_BRANCH_REF" "$P28_COMMIT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P28_TAG_REF" "$P28_TAG_OBJECT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P27_BRANCH_REF" "$P27_COMMIT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P27_TAG_REF" "$P27_TAG_OBJECT"
  clean_git --git-dir="$BARE_SOURCE" update-ref \
    "$P26_CHECKPOINT_REF" "$P26_CHECKPOINT_OBJECT"
  clean_git --git-dir="$BARE_SOURCE" update-ref "$P26_SOURCE_REF" "$P26_SOURCE_OBJECT"
  clean_git --git-dir="$BARE_SOURCE" bundle create "$LOCAL_SOURCE_BUNDLE" \
    "$P30_BRANCH_REF" \
    "$P29_BRANCH_REF" \
    "$P29_TAG_REF" \
    "$P28_BRANCH_REF" \
    "$P28_TAG_REF" \
    "$P27_BRANCH_REF" \
    "$P27_TAG_REF" \
    "$P26_CHECKPOINT_REF" \
    "$P26_SOURCE_REF"
  [[ "$(clean_git bundle list-heads "$LOCAL_SOURCE_BUNDLE" | /usr/bin/wc -l | tr -d ' ')" == 9 ]] ||
    die "source bundle does not advertise exactly nine refs"
  clean_git -C "$SCRIPT_ROOT" bundle verify "$LOCAL_SOURCE_BUNDLE"
  clean_git -C "$SCRIPT_ROOT" show "$SOURCE_COMMIT:scripts/verify_p30_control_bundle.py" \
    >"$LOCAL_VERIFIER"
  chmod 0600 "$LOCAL_SOURCE_BUNDLE" "$LOCAL_VERIFIER"
  SOURCE_BUNDLE_SHA256="$(/usr/bin/sha256sum "$LOCAL_SOURCE_BUNDLE" | /usr/bin/cut -d' ' -f1)"
  SOURCE_BUNDLE_BYTE_COUNT="$(/usr/bin/stat -c '%s' "$LOCAL_SOURCE_BUNDLE")"
  VERIFIER_SHA256="$(/usr/bin/sha256sum "$LOCAL_VERIFIER" | /usr/bin/cut -d' ' -f1)"
  ORCHESTRATOR_SHA256="$({
    clean_git -C "$SCRIPT_ROOT" show \
      "$SOURCE_COMMIT:scripts/run_p30_umask_bound_control_seal.sh" |
      /usr/bin/sha256sum
  } | /usr/bin/cut -d' ' -f1)"
  printf 'bundle_sha256=%s\nbundle_byte_count=%s\n' \
    "$SOURCE_BUNDLE_SHA256" "$SOURCE_BUNDLE_BYTE_COUNT"
  printf 'verifier_sha256=%s\norchestrator_sha256=%s\n' \
    "$VERIFIER_SHA256" "$ORCHESTRATOR_SHA256"
  clean_git bundle list-heads "$LOCAL_SOURCE_BUNDLE"
}

provision_namespace() {
  if [[ -e /secure || -L /secure ]]; then
    [[ ! -L /secure ]] || die "/secure is a symlink"
    [[ "$(/usr/bin/sudo -n /usr/bin/stat -c '%u:%g:%a' /secure)" == 0:0:755 ]] ||
      die "existing /secure authority differs"
  else
    /usr/bin/sudo -n /usr/bin/install -d -o 0 -g 0 -m 0755 /secure
  fi
  /usr/bin/sudo -n /usr/bin/install -d -o 0 -g 0 -m 0755 "$P30_NAMESPACE_ROOT"
  /usr/bin/sudo -n /usr/bin/install -d -o 1000 -g 1000 -m 0700 "$P30_TRANSPORT_ROOT"
  /usr/bin/sudo -n /usr/bin/install -d -o 0 -g 0 -m 0700 "$P30_LEDGER_ROOT"
  /usr/bin/sudo -n /usr/bin/install -d -o 0 -g 0 -m 0700 "$P30_LEDGER_ROOT/deferred"
  [[ "$(/usr/bin/sudo -n /usr/bin/stat -c '%u:%g:%a' "$P30_NAMESPACE_ROOT")" == 0:0:755 ]]
  [[ "$(/usr/bin/sudo -n /usr/bin/stat -c '%u:%g:%a' "$P30_TRANSPORT_ROOT")" == 1000:1000:700 ]]
  [[ "$(/usr/bin/sudo -n /usr/bin/stat -c '%u:%g:%a' "$P30_LEDGER_ROOT")" == 0:0:700 ]]
  [[ "$(/usr/bin/sudo -n /usr/bin/find "$P30_NAMESPACE_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\n')" == transport-20260906-03 ]]
}

copy_o_excl_as_uid1000() {
  local source="$1"
  local destination="$2"
  as_uid1000 /usr/bin/python3 -I -S -c \
    'import os,shutil,sys
p=sys.argv[1]
flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW
descriptor=os.open(p,flags,0o600)
try:
    with os.fdopen(descriptor,"wb",closefd=False) as stream:
        shutil.copyfileobj(sys.stdin.buffer,stream)
        stream.flush()
        os.fsync(stream.fileno())
finally:
    os.close(descriptor)' \
    "$destination" <"$source"
  [[ "$(/usr/bin/sudo -n /usr/bin/stat -c '%u:%g:%a:%h' "$destination")" == 1000:1000:600:1 ]]
}

stage_bundle() {
  copy_o_excl_as_uid1000 "$LOCAL_SOURCE_BUNDLE" "$P30_SOURCE_BUNDLE"
}

stage_verifier() {
  copy_o_excl_as_uid1000 "$LOCAL_VERIFIER" "$P30_BOOTSTRAP_VERIFIER"
}

verifier_common_arguments() {
  printf '%s\0' \
    --phase source \
    --transport-root "$P30_TRANSPORT_ROOT" \
    --bundle "$P30_SOURCE_BUNDLE" \
    --expected-bundle-sha256 "$SOURCE_BUNDLE_SHA256" \
    --expected-bundle-byte-count "$SOURCE_BUNDLE_BYTE_COUNT" \
    --closure-repository "$P30_SOURCE_CLOSURE" \
    --control-checkout "$P30_CONTROL_REPO" \
    --attempt-root "$P30_ATTEMPT_ROOT" \
    --execution-root "$P30_EXECUTION_ROOT" \
    --expected-p30-commit "$SOURCE_COMMIT" \
    --expected-p30-tree "$SOURCE_TREE" \
    --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$VERIFIER_SHA256" \
    --expected-orchestrator-sha256 "$ORCHESTRATOR_SHA256"
}

run_source_verifier_create() {
  local -a arguments=()
  while IFS= read -r -d '' argument; do
    arguments+=("$argument")
  done < <(verifier_common_arguments)
  as_uid1000 /usr/bin/python3 -I -S "$P30_BOOTSTRAP_VERIFIER" \
    "${arguments[@]}" --receipt-output "$P30_SOURCE_RECEIPT"
}

review_source_receipt() {
  as_uid1000 /usr/bin/python3 -I -S - \
    "$P30_SOURCE_RECEIPT" "$SOURCE_COMMIT" "$SOURCE_TREE" \
    "$VERIFIER_SHA256" "$ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import json
import os
import stat
import sys

path = sys.argv[1]
raw = open(path, "rb").read()
def pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise SystemExit("duplicate receipt key")
        result[key] = value
    return result
def nonfinite(value):
    raise SystemExit(f"nonfinite receipt value: {value}")
receipt = json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
if raw != (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode():
    raise SystemExit("receipt is not canonical JSON")
info = os.stat(path, follow_symlinks=False)
if not stat.S_ISREG(info.st_mode) or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode), info.st_nlink) != (1000, 1000, 0o600, 1):
    raise SystemExit("receipt authority differs")
if receipt.get("phase") != "source" or receipt.get("status") != "source_bundle_and_permission_layout_verified_before_attempt_root_creation":
    raise SystemExit("receipt phase or status differs")
history = receipt.get("history", {})
if history.get("source_commit") != sys.argv[2] or history.get("source_tree") != sys.argv[3]:
    raise SystemExit("receipt source identity differs")
reviewed = receipt.get("reviewed_orchestrator_source", {})
if reviewed.get("sha256") != sys.argv[5] or reviewed.get("physical_mode_octal") != "0600":
    raise SystemExit("receipt orchestrator binding differs")
barrier = receipt.get("reconstruction_publication_barrier", {})
required = {
    "p30_contract": 15,
    "p29_contract": 15,
    "p29_terminal_outcome": 11,
    "p28_contract": 12,
    "p28_terminal_outcome": 10,
    "p27_contract": 23,
}
for label, count in required.items():
    record = barrier.get(label, {})
    if record.get("check_count") != count or record.get("true_check_count") != count or not record.get("internally_consistent"):
        raise SystemExit(f"real reconstruction barrier failed: {label}")
if barrier.get("p30_contract", {}).get("reconstructor_sha256") == "0" * 64:
    raise SystemExit("top-level P30 reconstruction was stubbed")
if receipt.get("verifier", {}).get("source_sha256") != sys.argv[4]:
    raise SystemExit("receipt verifier binding differs")
print(hashlib.sha256(raw).hexdigest())
PY
}

run_source_verifier_replay() {
  local -a arguments=()
  while IFS= read -r -d '' argument; do
    arguments+=("$argument")
  done < <(verifier_common_arguments)
  as_uid1000 /usr/bin/python3 -I -S "$P30_BOOTSTRAP_VERIFIER" \
    "${arguments[@]}" \
    --verify-existing-receipt "$P30_SOURCE_RECEIPT" \
    --expected-receipt-sha256 "$RECEIPT_SHA256"
}

seal_reviewed_orchestrator_as_root() {
  local seal_program
  seal_program="$(
    clean_git -C "$SCRIPT_ROOT" show \
      "$SOURCE_COMMIT:experiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md" | \
      /usr/bin/python3 -I -S -c \
      'import pathlib,sys
raw=sys.stdin.read()
begin="# P30_REVIEWED_ORCHESTRATOR_SEAL_BEGIN\n"
end="# P30_REVIEWED_ORCHESTRATOR_SEAL_END\n"
if raw.count(begin) != 1 or raw.count(end) != 1: raise SystemExit("seal markers differ")
sys.stdout.write(raw.split(begin,1)[1].split(end,1)[0])'
  )"
  [[ -n "$seal_program" ]] || die "marked seal program is empty"
  /usr/bin/sudo -n /usr/bin/env -i \
    PATH=/usr/bin:/bin \
    /usr/bin/python3 -I -S -c "$seal_program" \
    "$P30_CONTROL_REPO/scripts/run_p30_umask_bound_control_seal.sh" \
    "$P30_LEDGER_ROOT" "$ORCHESTRATOR_SHA256" 1000 1000 0 0
}

verify_root_seal() {
  /usr/bin/sudo -n /usr/bin/python3 -I -S - \
    "$P30_CONTROL_REPO/scripts/run_p30_umask_bound_control_seal.sh" \
    "$P30_SEALED_ORCHESTRATOR" "$P30_SOURCE_RECEIPT" \
    "$ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

source = pathlib.Path(sys.argv[1])
sealed = pathlib.Path(sys.argv[2])
receipt_path = pathlib.Path(sys.argv[3])
expected = sys.argv[4]
source_raw = source.read_bytes()
sealed_raw = sealed.read_bytes()
if source_raw != sealed_raw or hashlib.sha256(sealed_raw).hexdigest() != expected:
    raise SystemExit("sealed orchestrator bytes differ")
source_info = source.stat(follow_symlinks=False)
sealed_info = sealed.stat(follow_symlinks=False)
if (source_info.st_uid, source_info.st_gid, stat.S_IMODE(source_info.st_mode), source_info.st_nlink) != (1000, 1000, 0o600, 1):
    raise SystemExit("verifier-created source authority differs")
if (sealed_info.st_uid, sealed_info.st_gid, stat.S_IMODE(sealed_info.st_mode), sealed_info.st_nlink) != (0, 0, 0o555, 1):
    raise SystemExit("root-sealed destination authority differs")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
reviewed = receipt["reviewed_orchestrator_source"]
if reviewed["sha256"] != expected or reviewed["physical_mode_octal"] != "0600":
    raise SystemExit("receipt-to-seal binding differs")
if sorted(path.name for path in sealed.parent.iterdir()) != [
    "deferred",
    "run_p30_umask_bound_control_seal.sh",
]:
    raise SystemExit("root ledger inventory differs after seal")
print("uid1000_to_root_seal_handoff=passed")
PY
}

verify_final_evidence_boundary() {
  [[ ! -e "$P30_ATTEMPT_ROOT" && ! -L "$P30_ATTEMPT_ROOT" ]] ||
    die "engineering rehearsal created the forbidden P30 attempt root"
  [[ ! -e "$P30_EXECUTION_ROOT" && ! -L "$P30_EXECUTION_ROOT" ]] ||
    die "engineering rehearsal created the forbidden P30 execution root"
  [[ "$({
    /usr/bin/sudo -n /usr/bin/find "$P30_NAMESPACE_ROOT" \
      -mindepth 1 -maxdepth 1 -printf '%f\n'
  })" == transport-20260906-03 ]] || die "final namespace inventory differs"
  [[ "$({
    /usr/bin/sudo -n /usr/bin/find "$P30_LEDGER_ROOT" \
      -mindepth 1 -maxdepth 1 -printf '%f\n' | /usr/bin/sort
  })" == $'deferred\nrun_p30_umask_bound_control_seal.sh' ]] ||
    die "final root ledger inventory differs"
  printf 'attempt_root_created=false\n'
  printf 'execution_root_created=false\n'
  printf 'candidate_observations=0\ntraining_steps=0\n'
}

main() {
  require_absolute_outside_scientific_roots "$LOG_ROOT" P30_REHEARSAL_LOG_ROOT
  require_absolute_outside_scientific_roots "$STAGE_ROOT" P30_REHEARSAL_STAGE_ROOT
  [[ "$LOG_ROOT" != "$STAGE_ROOT" ]] || die "log and stage roots must differ"
  mkdir "$LOG_ROOT"
  mkdir "$STAGE_ROOT"
  chmod 0700 "$LOG_ROOT" "$STAGE_ROOT"
  trap 'write_manifest "$?"' EXIT

  run_logged 00-preflight preflight
  run_logged 01-bundle-create build_closed_source_bundle
  run_logged 02-namespace-provision provision_namespace
  run_logged 03-stage-source-bundle stage_bundle
  run_logged 04-stage-bootstrap-verifier stage_verifier
  run_logged 05-source-verifier-create run_source_verifier_create
  run_logged 06-source-receipt-review review_source_receipt
  RECEIPT_SHA256="$(/usr/bin/head -n 1 "$LOG_ROOT/06-source-receipt-review.stdout.log")"
  [[ "$RECEIPT_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "reviewed receipt digest is malformed"
  run_logged 07-source-verifier-replay run_source_verifier_replay
  run_logged 08-root-orchestrator-seal seal_reviewed_orchestrator_as_root
  run_logged 09-root-seal-verification verify_root_seal
  run_logged 10-final-evidence-boundary verify_final_evidence_boundary
  printf 'P30 engineering rehearsal passed; no GPU or scientific acquisition ran.\n'
}

main "$@"
