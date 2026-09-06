#!/bin/bash -p
# Hash-bound, one-shot host orchestration for P30's bundle-complete bridge to
# P27's unchanged, non-training CUDA deleted-mapping localizer. There is
# deliberately no cleanup, retry, or acquisition phase: every created attempt
# directory and every state-bearing process transcript is retained as evidence.

set -euo pipefail
umask 077
if [[ $EUID -ne 0 || ! -o privileged ]]; then
  echo "P30 sealed orchestrator requires a direct root invocation through /bin/bash -p" >&2
  exit 1
fi

# All host trust operations use only root-managed executables and an isolated
# Python import environment.  In particular, reconstructed scripts may spawn a
# child ``python3`` or ``git`` process, so the inherited environment must be
# safe as well as each outer command line.
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
for p30_loader_variable in $(compgen -A variable); do
  case "$p30_loader_variable" in
    LD_*|DYLD_*) unset "$p30_loader_variable" ;;
  esac
done
unset p30_loader_variable

# Git identity must come only from each reviewed checkout. Remove every ambient
# Git control variable before the first Git invocation and disable system/global
# configuration; repository-local include directives are rejected below.
for p30_git_variable in $(compgen -A variable GIT_ || true); do
  unset "$p30_git_variable"
done
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_CONFIG_SYSTEM=/dev/null
export GIT_ATTR_NOSYSTEM=1
export GIT_NO_REPLACE_OBJECTS=1
export GIT_OPTIONAL_LOCKS=0
export GIT_TERMINAL_PROMPT=0
unset p30_git_variable
for p30_docker_variable in $(compgen -A variable DOCKER_ || true); do
  unset "$p30_docker_variable"
done
unset p30_docker_variable

readonly P30_REQUIRED_ATTEMPT_ID=20260906-03
readonly P30_NAMESPACE_ROOT=/secure/p30
readonly P30_ATTEMPT_ROOT=/secure/p30/attempt-20260906-03
readonly P30_HOST_EVIDENCE="$P30_ATTEMPT_ROOT/evidence"
readonly P30_EXECUTION_ROOT=/secure/p30/execution-20260906-03
readonly P30_EXECUTION_MANIFEST="$P30_EXECUTION_ROOT/snapshot-manifest.json"
readonly P30_EXECUTION_DATA="$P30_EXECUTION_ROOT/data"
readonly P30_EXECUTION_TOOLS="$P30_EXECUTION_DATA/.p30-tools"
readonly P30_CANONICAL_NANOGPT_HOST=/secure/p23/nanoGPT
readonly P30_CANONICAL_MUON_HOST=/secure/p23/Muon
readonly P30_CANONICAL_DATA_HOST=/secure/p23/optimizationml-p22-data
readonly P30_CONTAINER=p30-localization-20260906-03
readonly P30_TRANSPORT_ROOT=/secure/p30/transport-20260906-03
readonly P30_SOURCE_BUNDLE="$P30_TRANSPORT_ROOT/p30_source.bundle"
readonly P30_RUNTIME_REVIEW_BUNDLE="$P30_TRANSPORT_ROOT/p30_runtime_review.bundle"
readonly P30_SOURCE_RECEIPT="$P30_TRANSPORT_ROOT/p30_source_bundle_receipt.json"
readonly P30_RUNTIME_REVIEW_RECEIPT="$P30_TRANSPORT_ROOT/p30_runtime_review_bundle_receipt.json"
readonly P30_SOURCE_CLOSURE="$P30_TRANSPORT_ROOT/p30_source_closure.git"
readonly P30_RUNTIME_REVIEW_CLOSURE="$P30_TRANSPORT_ROOT/p30_runtime_review_closure.git"
readonly P30_BOOTSTRAP_VERIFIER="$P30_TRANSPORT_ROOT/verify_p30_control_bundle.py"
readonly P30_CONTROL_REPO="$P30_TRANSPORT_ROOT/control-source"
readonly P30_LEDGER_ROOT=/var/lib/optimizationml-p30-20260906-03
readonly P30_DEFERRED_ROOT="$P30_LEDGER_ROOT/deferred"
readonly P30_SEALED_ORCHESTRATOR="$P30_LEDGER_ROOT/run_p30_umask_bound_control_seal.sh"
readonly P30_PREPARE_INVOCATION="$P30_LEDGER_ROOT/prepare-runtime.invoked"
readonly P30_LOCALIZATION_INVOCATION="$P30_LEDGER_ROOT/run-localization.invoked"
readonly P30_LOCALIZATION_AUTHORIZATION="$P30_LEDGER_ROOT/localization.authorized"
readonly P30_LOCALIZATION_EXIT_STATUS="$P30_LEDGER_ROOT/run-localization.exit-status.txt"
readonly P30_IMAGE_DIGEST=sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0
readonly P30_IMAGE="localhost:5000/p25-runtime@$P30_IMAGE_DIGEST"
readonly P30_REQUIRED_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
readonly P30_REQUIRED_GPU_NAME='NVIDIA A100-SXM4-40GB'
readonly P30_AUTHORITY_HEAD=185e444afc0b44ca0a09b1bde49a6b6fa3973355
readonly P30_AUTHORITY_TREE=24f4bdac331a57bd7c1b807747d7c7fba253ee5a
readonly P30_TERMINAL_P29_COMMIT=8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf
readonly P30_TERMINAL_P29_TREE=6ee56f8c218eab1d25047f6f1046b97f7529d996
readonly P30_TERMINAL_P28_COMMIT=7274367f2b05fb3c9ed9f876a59be647703bbdc2
readonly P30_TERMINAL_P28_TREE=8d971e176e664ea47d8f769ae54f8cf349ceb3d5
readonly P30_TERMINAL_P27_COMMIT=ec63550331925ded158e3f389e294e4d1f12db3a
readonly P30_TERMINAL_P27_TREE=d8efa72fda9ee41fde0b5d15b126aa87397cd48d
readonly P30_TERMINAL_P26_COMMIT=5429da23ff18888daa2312c530a4587780484d8b
readonly P30_TERMINAL_P26_CONTAINER_ID=e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c
readonly P30_P26_NATIVE=/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json
readonly P30_P26_NATIVE_SHA256=b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91
readonly P30_P26_NATIVE_BYTE_COUNT=66283
readonly P30_P25_SOURCE_SHA256=6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770
readonly P30_P25_CONTRACT_SHA256=51e1fc22f685cc904fcdfff2775cff4293fdaa67a6e048072ee4dc49777f34e2
readonly P30_P23_RUNNER_SHA256=a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb
readonly P30_NANOGPT_COMMIT=3adf61e154c3fe3fca428ad6bc3818b27a3b8291
readonly P30_NANOGPT_TREE=ca93bcd9b9c9ff32d3016e1e2556644e68bef86a
readonly P30_MUON_COMMIT=f98f1cacc0263b04290753e32be8d498c1efc806
readonly P30_MUON_TREE=4ea5cd8ab6ebd56a18536f06453619efcd636da0
readonly P30_MUON_SOURCE_SHA256=2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d
readonly P30_FINEWEB_MANIFEST_SHA256=1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d
readonly P30_FINEWEB_MANIFEST_BYTE_COUNT=4210

die() {
  echo "P30 bundle-complete localization bridge blocked: $*" >&2
  exit 1
}

p30_git() {
  /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/nonexistent \
    GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_SYSTEM=/dev/null \
    GIT_ATTR_NOSYSTEM=1 \
    GIT_NO_REPLACE_OBJECTS=1 \
    GIT_OPTIONAL_LOCKS=0 \
    GIT_TERMINAL_PROMPT=0 \
    /usr/bin/setpriv \
    --reuid=1000 --regid=1000 --clear-groups --no-new-privs \
    /usr/bin/git \
    -c core.hooksPath=/dev/null \
    -c core.fsmonitor=false \
    -c core.attributesFile=/dev/null \
    -c core.excludesFile=/dev/null \
    -c protocol.file.allow=always \
    "$@"
}

p30_docker() {
  sudo /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    HOME=/nonexistent \
    /usr/bin/docker \
    --host unix:///var/run/docker.sock \
    --config /nonexistent/p30-docker-config \
    "$@"
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
  [[ "${!name}" == /* && "${!name}" != *$'\n'* && "${!name}" != *,* ]] ||
    die "$name must be one absolute path without a newline or comma"
}

sha256_file() {
  /usr/bin/python3 -I -S -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}

root_sha256_file() {
  sudo /usr/bin/python3 -I -S -c \
    'import hashlib,pathlib,sys; print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())' \
    "$1"
}

seal_retained_file() {
  local path="$1"
  /usr/bin/python3 -I -S - \
    "$path" "$P30_HOST_EVIDENCE" "$P30_DEFERRED_ROOT" <<'PY'
import os
import pathlib
import stat
import sys
import hashlib

path, evidence, deferred = map(pathlib.PurePosixPath, sys.argv[1:])
expected_parent = {
    evidence: ((0, 0, 0o555), 0o444),
    deferred: ((0, 0, 0o700), 0o400),
    evidence.parent: ((0, 0, 0o555), 0o444),
}.get(path.parent)
if expected_parent is None or path.name in {"", ".", ".."}:
    raise SystemExit("P30 retained file is outside a sealed output root")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
parent_fd = os.open("/", directory_flags)
descriptor = None
try:
    for component in path.parent.parts[1:]:
        child = os.open(component, directory_flags, dir_fd=parent_fd)
        os.close(parent_fd)
        parent_fd = child
    parent_info = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or (parent_info.st_uid, parent_info.st_gid, stat.S_IMODE(parent_info.st_mode))
        != expected_parent[0]
    ):
        raise SystemExit("P30 retained-file parent authority differs")
    descriptor = os.open(
        path.name,
        os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid) != (0, 0)
        or stat.S_IMODE(before.st_mode) not in {0o400, 0o444, 0o600, 0o644}
    ):
        raise SystemExit("P30 retained-file pre-seal authority differs")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    for target, label in ((parent_fd, "parent"), (descriptor, "file")):
        acl_names = {
            value.decode() if isinstance(value, bytes) else value
            for value in os.listxattr(target)
        }
        if acl_names & forbidden_acl:
            raise SystemExit(f"P30 retained-file {label} has a forbidden ACL")
    os.fsync(descriptor)
    os.fchmod(descriptor, expected_parent[1])
    os.fsync(descriptor)
    def digest_fd(fd):
        os.lseek(fd, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)

    digest_before = digest_fd(descriptor)
    after = os.fstat(descriptor)
    by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if fields(after) != fields(by_name):
        raise SystemExit("P30 retained-file path identity differs after sealing")
    if (
        not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode))
        != (0, 0, expected_parent[1])
    ):
        raise SystemExit("P30 retained-file final authority differs")
    digest_after = digest_fd(descriptor)
    if digest_after != digest_before:
        raise SystemExit("P30 retained-file bytes changed while sealing")
    os.fsync(parent_fd)
finally:
    if descriptor is not None:
        os.close(descriptor)
    os.close(parent_fd)
PY
}

seal_evidence_inventory() {
  /usr/bin/python3 -I -S - "$P30_HOST_EVIDENCE" <<'PY'
import os
import pathlib
import stat
import sys

evidence = pathlib.PurePosixPath(sys.argv[1])
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open("/", directory_flags)
try:
    for component in evidence.parts[1:]:
        child = os.open(component, directory_flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    root_info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(root_info.st_mode)
        or (root_info.st_uid, root_info.st_gid, stat.S_IMODE(root_info.st_mode))
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 evidence root authority differs during sealing")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    failures = []
    for name in sorted(os.listdir(descriptor)):
        file_descriptor = None
        try:
            if name in {"", ".", ".."} or "/" in name or "\x00" in name:
                raise RuntimeError("P30 evidence inventory contains an unsafe name")
            file_descriptor = os.open(
                name,
                os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            before = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_uid, before.st_gid) != (0, 0)
                or stat.S_IMODE(before.st_mode) not in {0o400, 0o444, 0o600, 0o644}
            ):
                raise RuntimeError(f"P30 evidence child authority differs: {name}")
            acl_names = {
                value.decode() if isinstance(value, bytes) else value
                for value in os.listxattr(file_descriptor)
            }
            if acl_names & forbidden_acl:
                raise RuntimeError(f"P30 evidence child has a forbidden ACL: {name}")
            os.fsync(file_descriptor)
            os.fchmod(file_descriptor, 0o444)
            os.fsync(file_descriptor)
            after = os.fstat(file_descriptor)
            by_name = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            fields = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fields(after) != fields(by_name) or (
                after.st_uid,
                after.st_gid,
                stat.S_IMODE(after.st_mode),
            ) != (0, 0, 0o444):
                raise RuntimeError(f"P30 evidence child did not seal exactly: {name}")
        except Exception as error:
            failures.append(f"{name}: {error}")
        finally:
            if file_descriptor is not None:
                try:
                    os.close(file_descriptor)
                except OSError as error:
                    failures.append(f"{name}: close failed: {error}")
    os.fsync(descriptor)
    if failures:
        raise SystemExit(
            "P30 evidence inventory did not seal completely:\n" + "\n".join(failures)
        )
finally:
    os.close(descriptor)
PY
}

run_logged() {
  local label="$1"
  shift
  local stdout_path="$P30_HOST_EVIDENCE/$label.stdout.log"
  local stderr_path="$P30_HOST_EVIDENCE/$label.stderr.log"
  local status_path="$P30_HOST_EVIDENCE/$label.exit-status.txt"
  [[ ! -e "$stdout_path" && ! -L "$stdout_path" && \
     ! -e "$stderr_path" && ! -L "$stderr_path" && \
     ! -e "$status_path" && ! -L "$status_path" ]] ||
    die "process log already exists: $label"
  local status had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  (
    set -euo pipefail
    set -o noclobber
    "$@" >"$stdout_path" 2>"$stderr_path"
  )
  status=$?
  if (( had_errexit )); then set -e; else set +e; fi
  # Finalization is deliberately non-short-circuiting.  Once either shell
  # redirection has created a transcript, every closed artifact must be made
  # immutable even when the status O_EXCL write (or an earlier seal) fails.
  # A finalization failure has the reserved terminal status 125; otherwise the
  # producer's original status is preserved.
  local finalization_failed=0
  set +e
  write_new_status "$status_path" "$status"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stdout_path"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stderr_path"
  (( $? == 0 )) || finalization_failed=1
  seal_evidence_inventory
  (( $? == 0 )) || finalization_failed=1
  if (( had_errexit )); then set -e; else set +e; fi
  (( finalization_failed == 0 )) || return 125
  return "$status"
}

create_admission_log_placeholders() {
  local stdout_path="$1"
  local stderr_path="$2"
  /usr/bin/python3 -I -S - \
    "$P30_DEFERRED_ROOT" "$stdout_path" "$stderr_path" <<'PY'
import os
import pathlib
import stat
import sys

journal, stdout_path, stderr_path = map(pathlib.PurePosixPath, sys.argv[1:])
if (
    stdout_path.parent != journal
    or stderr_path.parent != journal
    or stdout_path.name == stderr_path.name
):
    raise SystemExit("P30 admission-log placeholder paths differ")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open("/", flags)
created = []
try:
    for component in journal.parts[1:]:
        child = os.open(component, flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
        != (0, 0, 0o700)
        or os.listdir(descriptor)
    ):
        raise SystemExit("P30 admission journal is not exactly empty and root-owned")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    if {
        value.decode() if isinstance(value, bytes) else value
        for value in os.listxattr(descriptor)
    } & forbidden_acl:
        raise SystemExit("P30 admission journal has a forbidden ACL")
    for path in (stdout_path, stderr_path):
        if path.name in {"", ".", ".."} or "/" in path.name or "\x00" in path.name:
            raise SystemExit("P30 admission-log placeholder name is unsafe")
        file_descriptor = os.open(
            path.name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=descriptor,
        )
        created.append(file_descriptor)
        os.fchmod(file_descriptor, 0o600)
        os.fsync(file_descriptor)
        file_info = os.fstat(file_descriptor)
        by_name = os.stat(path.name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(file_info.st_mode)
            or file_info.st_nlink != 1
            or (file_info.st_uid, file_info.st_gid, stat.S_IMODE(file_info.st_mode))
            != (0, 0, 0o600)
            or (file_info.st_dev, file_info.st_ino)
            != (by_name.st_dev, by_name.st_ino)
        ):
            raise SystemExit("P30 admission-log placeholder authority differs")
    os.fsync(descriptor)
finally:
    failed = sys.exc_info()[0] is not None
    for file_descriptor in created:
        try:
            if failed:
                os.fchmod(file_descriptor, 0o400)
                os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
    os.close(descriptor)
PY
}

run_root_journal_logged() {
  # This logger is the only retained boundary available after the irreversible
  # prepare token is burned but before an attempt/evidence directory exists.
  # Its transcripts remain in the root-only ledger journal and are never
  # published into the later container-visible evidence mount.
  local label="$1"
  shift
  [[ "$label" == "p30-prepare-pre-attempt-admission" ]] ||
    die "root-journal logging is restricted to the preregistered admission boundary"
  local prefix="$P30_DEFERRED_ROOT/.deferred-$label"
  local stdout_path="$prefix.stdout.log"
  local stderr_path="$prefix.stderr.log"
  local status_path="$prefix.exit-status.txt"
  local path
  for path in "$stdout_path" "$stderr_path" "$status_path"; do
    [[ ! -e "$path" && ! -L "$path" ]] ||
      die "root-journal admission artifact already exists: $path"
  done
  create_admission_log_placeholders "$stdout_path" "$stderr_path" || return 125
  local stdout_binding= stderr_binding= binding_failed=0
  local status had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  stdout_binding="$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$stdout_path")"
  (( $? == 0 )) || binding_failed=1
  stderr_binding="$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$stderr_path")"
  (( $? == 0 )) || binding_failed=1
  [[ "$stdout_binding" == *":600:0:0:1" && \
     "$stderr_binding" == *":600:0:0:1" ]] || binding_failed=1
  if (( binding_failed == 0 )); then
    (
      set -euo pipefail
      "$@"
    ) >|"$stdout_path" 2>|"$stderr_path"
    status=$?
  else
    status=125
  fi
  local stdout_after= stderr_after=
  stdout_after="$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$stdout_path")"
  (( $? == 0 )) || binding_failed=1
  stderr_after="$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$stderr_path")"
  (( $? == 0 )) || binding_failed=1
  [[ "$stdout_after" == "$stdout_binding" && \
     "$stderr_after" == "$stderr_binding" ]] || binding_failed=1
  (( binding_failed == 0 )) || status=125
  if (( had_errexit )); then set -e; else set +e; fi

  local finalization_failed=0
  set +e
  write_new_status "$status_path" "$status"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stdout_path"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stderr_path"
  (( $? == 0 )) || finalization_failed=1
  if [[ -f "$status_path" && ! -L "$status_path" ]]; then
    seal_retained_file "$status_path"
    (( $? == 0 )) || finalization_failed=1
  else
    finalization_failed=1
  fi
  validate_deferred_journal after-admission "$status"
  (( $? == 0 )) || finalization_failed=1
  if (( had_errexit )); then set -e; else set +e; fi
  (( finalization_failed == 0 )) || return 125
  return "$status"
}

run_logged_deferred_evidence() {
  # Runtime-receipt replay must observe the byte-exact frozen prepare evidence
  # inventory. Its own shell-opened logs therefore live in the root-only
  # deferred journal
  # while the verifier runs, and are copied O_EXCL into evidence only after the
  # verifier exits. On failure both copies remain the terminal transcript and
  # no localization marker can be created.
  local label="$1"
  shift
  [[ "$label" == "p30-prepare-attempt-layout" || \
     "$label" == "p30-pre-marker-runtime-receipt-replay" ]] ||
    die "deferred evidence logging is restricted to preregistered boundaries"
  local external_prefix="$P30_DEFERRED_ROOT/.deferred-$label"
  local external_stdout="$external_prefix.stdout.log"
  local external_stderr="$external_prefix.stderr.log"
  local external_status="$external_prefix.exit-status.txt"
  local publication_stdout="$external_prefix.publication.stdout.log"
  local publication_stderr="$external_prefix.publication.stderr.log"
  local publication_status_path="$external_prefix.publication.exit-status.txt"
  local publication_failure="$external_prefix.publication-failure.exit-status.txt"
  local evidence_stdout="$P30_HOST_EVIDENCE/$label.stdout.log"
  local evidence_stderr="$P30_HOST_EVIDENCE/$label.stderr.log"
  local evidence_status="$P30_HOST_EVIDENCE/$label.exit-status.txt"
  local path
  for path in \
    "$external_stdout" "$external_stderr" "$external_status" \
    "$publication_stdout" "$publication_stderr" "$publication_status_path" \
    "$publication_failure" \
    "$evidence_stdout" "$evidence_stderr" "$evidence_status"; do
    [[ ! -e "$path" && ! -L "$path" ]] ||
      die "deferred replay evidence already exists: $path"
  done
  local status had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  (
    set -euo pipefail
    set -o noclobber
    "$@" >"$external_stdout" 2>"$external_stderr"
  )
  status=$?
  if (( had_errexit )); then set -e; else set +e; fi
  local external_finalization_failed=0
  set +e
  write_new_status "$external_status" "$status"
  (( $? == 0 )) || external_finalization_failed=1
  seal_retained_file "$external_stdout"
  (( $? == 0 )) || external_finalization_failed=1
  seal_retained_file "$external_stderr"
  (( $? == 0 )) || external_finalization_failed=1
  if [[ -f "$external_status" && ! -L "$external_status" ]]; then
    seal_retained_file "$external_status"
    (( $? == 0 )) || external_finalization_failed=1
  else
    external_finalization_failed=1
  fi
  if (( had_errexit )); then set -e; else set +e; fi
  (( external_finalization_failed == 0 )) || return 125

  local reviewed_receipt=- reviewed_receipt_sha256=-
  if [[ "$label" == "p30-pre-marker-runtime-receipt-replay" ]]; then
    reviewed_receipt="$P30_RUNTIME_REVIEW_RECEIPT"
    # Producer-side runtime validation deliberately runs inside this retained
    # logger. Keep finalization nounset-safe so a missing reviewed digest is
    # recorded as terminal evidence rather than aborting the logger itself.
    reviewed_receipt_sha256="${P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256:-}"
  fi
  local publication_status=0
  local publication_had_noclobber=0
  [[ -o noclobber ]] && publication_had_noclobber=1
  set +e
  set -o noclobber
  /usr/bin/python3 -I -S - \
    "$P30_DEFERRED_ROOT" "$P30_TRANSPORT_ROOT" "$P30_HOST_EVIDENCE" \
    "$P30_EXECUTION_ROOT" "$label" \
    "$reviewed_receipt" "$reviewed_receipt_sha256" \
    ".deferred-$label.stdout.log" "$label.stdout.log" \
    ".deferred-$label.stderr.log" "$label.stderr.log" \
    ".deferred-$label.exit-status.txt" "$label.exit-status.txt" \
    >"$publication_stdout" 2>"$publication_stderr" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

journal = pathlib.PurePosixPath(sys.argv[1])
transport = pathlib.PurePosixPath(sys.argv[2])
evidence = pathlib.PurePosixPath(sys.argv[3])
execution = pathlib.PurePosixPath(sys.argv[4])
label = sys.argv[5]
receipt = pathlib.PurePosixPath(sys.argv[6]) if sys.argv[6] != "-" else None
expected_receipt_digest = sys.argv[7]
pairs = list(zip(sys.argv[8::2], sys.argv[9::2], strict=True))
required = []
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
    required.append(value)
directory_flags = os.O_RDONLY
for value in required:
    directory_flags |= value
source_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
destination_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW


def open_componentwise(path: pathlib.PurePosixPath) -> int:
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def mount_id(descriptor: int) -> int:
    for line in pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split(":", 1)[1])
    raise SystemExit("P30 deferred-log descriptor has no mount ID")


def require_no_acl(descriptor: int, name: str) -> None:
    names = {
        value.decode() if isinstance(value, bytes) else value
        for value in os.listxattr(descriptor)
    }
    if names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit(f"P30 deferred-log {name} has a forbidden ACL")


def stable_regular_bytes(
    parent_fd: int,
    name: str,
    expected: tuple[int, int, int],
) -> bytes:
    descriptor = os.open(name, source_flags, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode)) != expected
        ):
            raise SystemExit("P30 deferred-log reviewed receipt authority differs")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if fields(before) != fields(after) or fields(after) != fields(by_name):
            raise SystemExit("P30 deferred-log reviewed receipt changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


journal_fd = open_componentwise(journal)
transport_fd = open_componentwise(transport)
attempt_fd = open_componentwise(evidence.parent)
evidence_fd = open_componentwise(evidence)
execution_fd = open_componentwise(execution)
try:
    for descriptor, expected, label in (
        (journal_fd, (0, 0, 0o700), "journal"),
        (transport_fd, (1000, 1000, 0o700), "transport"),
        (evidence_fd, (0, 0, 0o555), "evidence"),
    ):
        value = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(value.st_mode)
            or (value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode)) != expected
        ):
            raise SystemExit(f"P30 deferred-log {label} authority differs")
        require_no_acl(descriptor, label)
    attempt_info = os.fstat(attempt_fd)
    if (
        not stat.S_ISDIR(attempt_info.st_mode)
        or (attempt_info.st_uid, attempt_info.st_gid, stat.S_IMODE(attempt_info.st_mode))
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 deferred-log attempt authority differs")
    require_no_acl(attempt_fd, "attempt")
    execution_info = os.fstat(execution_fd)
    if (
        not stat.S_ISDIR(execution_info.st_mode)
        or (
            execution_info.st_uid,
            execution_info.st_gid,
            stat.S_IMODE(execution_info.st_mode),
        )
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 deferred-log execution authority differs")
    require_no_acl(execution_fd, "execution")
    evidence_info = os.fstat(evidence_fd)
    evidence_by_name = os.stat(evidence.name, dir_fd=attempt_fd, follow_symlinks=False)
    if fields(evidence_info) != fields(evidence_by_name):
        raise SystemExit("P30 deferred-log evidence path identity differs")
    if (
        len(
            {
                os.fstat(fd).st_dev
                for fd in (transport_fd, attempt_fd, evidence_fd, execution_fd)
            }
        )
        != 1
        or len(
            {mount_id(fd) for fd in (transport_fd, attempt_fd, evidence_fd, execution_fd)}
        )
        != 1
    ):
        raise SystemExit("P30 deferred-log paths cross a device or mount boundary")
    if receipt is not None:
        if receipt.parent != transport or receipt.name in {"", ".", ".."}:
            raise SystemExit("P30 deferred-log reviewed receipt path differs")
        receipt_raw = stable_regular_bytes(transport_fd, receipt.name, (1000, 1000, 0o600))
        if hashlib.sha256(receipt_raw).hexdigest() != expected_receipt_digest:
            raise SystemExit("P30 deferred-log reviewed receipt digest differs")
        try:
            receipt_payload = json.loads(receipt_raw)
            guarded = receipt_payload["permission_safe_namespace"][
                "guarded_child_directories"
            ]
            expected_attempt = guarded["attempt_root"]
            expected_evidence = guarded["attempt_evidence"]
            expected_execution = guarded["execution_root"]
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit("P30 deferred-log reviewed receipt is malformed") from exc
        for current, expected, name in (
            (attempt_info, expected_attempt, "attempt"),
            (evidence_info, expected_evidence, "evidence"),
            (execution_info, expected_execution, "execution"),
        ):
            observed = {
                "device": current.st_dev,
                "inode": current.st_ino,
                "uid": current.st_uid,
                "gid": current.st_gid,
                "mode_octal": format(stat.S_IMODE(current.st_mode), "04o"),
                "mount_id": mount_id(
                    attempt_fd
                    if name == "attempt"
                    else (evidence_fd if name == "evidence" else execution_fd)
                ),
                "path": str(
                    evidence.parent
                    if name == "attempt"
                    else (evidence if name == "evidence" else execution)
                ),
            }
            for key, value in observed.items():
                if expected.get(key) != value:
                    raise SystemExit(
                        f"P30 deferred-log reviewed {name} identity differs at {key}"
                    )
    for source_name, destination_name in pairs:
        if "/" in source_name or "/" in destination_name:
            raise SystemExit("P30 deferred-log filename is not one direct child")
        source_fd = os.open(source_name, source_flags, dir_fd=journal_fd)
        destination_fd = None
        try:
            before = os.fstat(source_fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
                != (0, 0, 0o400)
            ):
                raise SystemExit("P30 deferred-log source authority differs")
            destination_fd = os.open(
                destination_name,
                destination_flags,
                0o600,
                dir_fd=evidence_fd,
            )
            source_digest = hashlib.sha256()
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                source_digest.update(chunk)
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    if written <= 0:
                        raise SystemExit("P30 deferred-log copy made no progress")
                    view = view[written:]
            os.fsync(destination_fd)
            after = os.fstat(source_fd)
            source_by_name = os.stat(source_name, dir_fd=journal_fd, follow_symlinks=False)
            destination = os.fstat(destination_fd)
            if fields(before) != fields(after) or fields(after) != fields(source_by_name):
                raise SystemExit("P30 deferred-log source changed during publication")
            if (
                not stat.S_ISREG(destination.st_mode)
                or destination.st_nlink != 1
                or (
                    destination.st_uid,
                    destination.st_gid,
                    stat.S_IMODE(destination.st_mode),
                )
                != (0, 0, 0o600)
                or destination.st_size != after.st_size
            ):
                raise SystemExit("P30 deferred-log destination authority differs")
            os.lseek(destination_fd, 0, os.SEEK_SET)
            destination_digest = hashlib.sha256()
            while True:
                chunk = os.read(destination_fd, 1024 * 1024)
                if not chunk:
                    break
                destination_digest.update(chunk)
            if destination_digest.digest() != source_digest.digest():
                raise SystemExit("P30 deferred-log destination bytes differ")
            os.fchmod(destination_fd, 0o444)
            os.fsync(destination_fd)
            destination_after = os.fstat(destination_fd)
            destination_by_name = os.stat(
                destination_name,
                dir_fd=evidence_fd,
                follow_symlinks=False,
            )
            if (
                (destination_after.st_uid, destination_after.st_gid,
                 stat.S_IMODE(destination_after.st_mode)) != (0, 0, 0o444)
                or fields(destination_after) != fields(destination_by_name)
            ):
                raise SystemExit("P30 deferred-log destination identity changed")
            require_no_acl(destination_fd, "destination")
        finally:
            if destination_fd is not None:
                try:
                    os.fchmod(destination_fd, 0o444)
                    os.fsync(destination_fd)
                finally:
                    os.close(destination_fd)
            os.close(source_fd)
    os.fsync(evidence_fd)
finally:
    os.close(execution_fd)
    os.close(evidence_fd)
    os.close(attempt_fd)
    os.close(transport_fd)
    os.close(journal_fd)
PY
  publication_status=$?
  if (( publication_had_noclobber == 0 )); then
    set +o noclobber
  fi
  local publication_finalization_failed=0
  set +e
  write_new_status "$publication_status_path" "$publication_status"
  (( $? == 0 )) || publication_finalization_failed=1
  seal_retained_file "$publication_stdout"
  (( $? == 0 )) || publication_finalization_failed=1
  seal_retained_file "$publication_stderr"
  (( $? == 0 )) || publication_finalization_failed=1
  if [[ -f "$publication_status_path" && ! -L "$publication_status_path" ]]; then
    seal_retained_file "$publication_status_path"
    (( $? == 0 )) || publication_finalization_failed=1
  else
    publication_finalization_failed=1
  fi
  if (( publication_finalization_failed != 0 )); then
    write_new_status "$publication_failure" 125
    if (( $? == 0 )); then
      seal_retained_file "$publication_failure"
    fi
    if (( had_errexit )); then set -e; else set +e; fi
    return 125
  fi
  if (( publication_status != 0 )); then
    write_new_status "$publication_failure" "$publication_status"
    local failure_status_rc=$?
    if (( failure_status_rc == 0 )); then
      seal_retained_file "$publication_failure"
      failure_status_rc=$?
    fi
    if (( had_errexit )); then set -e; else set +e; fi
    (( failure_status_rc == 0 )) || return 125
    return "$publication_status"
  fi
  if [[ -s "$publication_stdout" || -s "$publication_stderr" ]]; then
    write_new_status "$publication_failure" 125
    local nonempty_failure_status_rc=$?
    if (( nonempty_failure_status_rc == 0 )); then
      seal_retained_file "$publication_failure"
      nonempty_failure_status_rc=$?
    fi
    if (( had_errexit )); then set -e; else set +e; fi
    (( nonempty_failure_status_rc == 0 )) || return 125
    return 125
  fi
  if (( had_errexit )); then set -e; else set +e; fi
  return "$status"
}

capture_new() {
  local label="$1"
  local output="$2"
  shift 2
  local stderr_path="$P30_HOST_EVIDENCE/$label.stderr.log"
  local status_path="$P30_HOST_EVIDENCE/$label.exit-status.txt"
  [[ ! -e "$output" && ! -L "$output" && \
     ! -e "$stderr_path" && ! -L "$stderr_path" && \
     ! -e "$status_path" && ! -L "$status_path" ]] ||
    die "capture output already exists: $label"
  local status had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  (
    set -euo pipefail
    set -o noclobber
    "$@" >"$output" 2>"$stderr_path"
  )
  status=$?
  if (( had_errexit )); then set -e; else set +e; fi
  local finalization_failed=0
  set +e
  write_new_status "$status_path" "$status"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$output"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stderr_path"
  (( $? == 0 )) || finalization_failed=1
  seal_evidence_inventory
  (( $? == 0 )) || finalization_failed=1
  if (( had_errexit )); then set -e; else set +e; fi
  (( finalization_failed == 0 )) || return 125
  return "$status"
}

refresh_bound_file() {
  local label="$1"
  local output="$2"
  shift 2
  local stderr_path="$P30_HOST_EVIDENCE/$label.stderr.log"
  local status_path="$P30_HOST_EVIDENCE/$label.exit-status.txt"
  [[ -f "$output" && ! -L "$output" && ! -s "$output" ]] ||
    die "bound evidence placeholder is not one empty regular file: $output"
  [[ ! -e "$stderr_path" && ! -L "$stderr_path" && \
     ! -e "$status_path" && ! -L "$status_path" ]] ||
    die "process log already exists: $label"
  local output_binding
  output_binding="$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$output")"
  [[ "$output_binding" == *":444:0:0:1" ]] ||
    die "bound evidence placeholder authority differs: $output"
  local status had_errexit=0
  [[ $- == *e* ]] && had_errexit=1
  set +e
  (
    set -euo pipefail
    set -o noclobber
    {
      "$@" >|"$output"
      [[ "$(/usr/bin/stat -c '%d:%i:%a:%u:%g:%h' "$output")" == \
          "$output_binding" ]] || {
        echo "P30 bound evidence output identity changed: $output" >&2
        exit 1
      }
      [[ -s "$output" ]] || {
        echo "P30 bound evidence capture is empty: $label" >&2
        exit 1
      }
    } 2>"$stderr_path"
  )
  status=$?
  if (( had_errexit )); then set -e; else set +e; fi
  local finalization_failed=0
  set +e
  write_new_status "$status_path" "$status"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$output"
  (( $? == 0 )) || finalization_failed=1
  seal_retained_file "$stderr_path"
  (( $? == 0 )) || finalization_failed=1
  seal_evidence_inventory
  (( $? == 0 )) || finalization_failed=1
  if (( had_errexit )); then set -e; else set +e; fi
  (( finalization_failed == 0 )) || return 125
  return "$status"
}

write_new_status() {
  local output="$1"
  local value="$2"
  /usr/bin/python3 -I -S - \
    "$output" "$value" "$P30_HOST_EVIDENCE" "$P30_DEFERRED_ROOT" <<'PY'
import os
import pathlib
import stat
import sys

path = pathlib.PurePosixPath(sys.argv[1])
payload = (sys.argv[2] + "\n").encode("ascii")
evidence = pathlib.PurePosixPath(sys.argv[3])
deferred = pathlib.PurePosixPath(sys.argv[4])
if not path.is_absolute() or path.name in {"", ".", ".."}:
    raise SystemExit("P30 status path is not one absolute direct-child path")
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    flag = getattr(os, name, None)
    if not isinstance(flag, int) or flag == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
parent = path.parent
parent_fd = os.open("/", directory_flags)
descriptor = None
final_mode = None
try:
    for component in parent.parts[1:]:
        child = os.open(component, directory_flags, dir_fd=parent_fd)
        os.close(parent_fd)
        parent_fd = child
    parent_info = os.fstat(parent_fd)
    expected_parent = {
        evidence: ((0, 0, 0o555), 0o444),
        deferred: ((0, 0, 0o700), 0o400),
    }.get(parent)
    if expected_parent is not None:
        final_mode = expected_parent[1]
    if (
        expected_parent is None
        or not stat.S_ISDIR(parent_info.st_mode)
        or (parent_info.st_uid, parent_info.st_gid, stat.S_IMODE(parent_info.st_mode))
        != expected_parent[0]
    ):
        raise SystemExit("P30 status parent authority differs")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(parent_fd)
    }
    if acl_names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit("P30 status parent has a forbidden ACL")
    descriptor = os.open(
        path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=parent_fd,
    )
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise SystemExit("P30 status write made no progress")
        view = view[written:]
    os.fsync(descriptor)
    created = os.fstat(descriptor)
    if (
        not stat.S_ISREG(created.st_mode)
        or created.st_nlink != 1
        or (created.st_uid, created.st_gid, stat.S_IMODE(created.st_mode))
        != (0, 0, 0o600)
        or created.st_size != len(payload)
    ):
        raise SystemExit("P30 status authority or size differs")
    os.lseek(descriptor, 0, os.SEEK_SET)
    observed = bytearray()
    while len(observed) < len(payload):
        chunk = os.read(descriptor, len(payload) - len(observed))
        if not chunk:
            break
        observed.extend(chunk)
    if bytes(observed) != payload or os.read(descriptor, 1) != b"":
        raise SystemExit("P30 status stored bytes differ")
    os.fchmod(descriptor, expected_parent[1])
    os.fsync(descriptor)
    created = os.fstat(descriptor)
    by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    if (
        stat.S_ISLNK(by_name.st_mode)
        or (by_name.st_dev, by_name.st_ino) != (created.st_dev, created.st_ino)
        or (created.st_uid, created.st_gid, stat.S_IMODE(created.st_mode))
        != (0, 0, expected_parent[1])
    ):
        raise SystemExit("P30 status path identity differs")
    os.fsync(parent_fd)
finally:
    if descriptor is not None:
        try:
            if final_mode is not None:
                os.fchmod(descriptor, final_mode)
                os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.close(parent_fd)
PY
}

validate_deferred_journal() {
  local stage="$1"
  local admission_status="${2:-0}"
  /usr/bin/python3 -I -S - \
    "$P30_DEFERRED_ROOT" "$stage" "$admission_status" <<'PY'
import os
import pathlib
import stat
import sys

journal = pathlib.PurePosixPath(sys.argv[1])
stage = sys.argv[2]
try:
    admission_status = int(sys.argv[3])
except ValueError as exc:
    raise SystemExit("P30 admission status is malformed") from exc
if not 0 <= admission_status <= 255 or sys.argv[3] != str(admission_status):
    raise SystemExit("P30 admission status is noncanonical")
labels = {
    "empty": (),
    "after-admission": (),
    "after-prepare": ("p30-prepare-attempt-layout",),
    "after-runtime": (
        "p30-prepare-attempt-layout",
        "p30-pre-marker-runtime-receipt-replay",
    ),
}.get(stage)
if labels is None:
    raise SystemExit("P30 deferred-journal stage is malformed")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open("/", directory_flags)
try:
    for component in journal.parts[1:]:
        child = os.open(component, directory_flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, 0, 0o700)
    ):
        raise SystemExit("P30 deferred-journal authority differs")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    if {
        value.decode() if isinstance(value, bytes) else value
        for value in os.listxattr(descriptor)
    } & forbidden_acl:
        raise SystemExit("P30 deferred journal has a forbidden ACL")
    expected = []
    if stage != "empty":
        expected.extend(
            (
                ".deferred-p30-prepare-pre-attempt-admission.stdout.log",
                ".deferred-p30-prepare-pre-attempt-admission.stderr.log",
                ".deferred-p30-prepare-pre-attempt-admission.exit-status.txt",
            )
        )
    for label in labels:
        prefix = f".deferred-{label}"
        expected.extend(
            (
                f"{prefix}.stdout.log",
                f"{prefix}.stderr.log",
                f"{prefix}.exit-status.txt",
                f"{prefix}.publication.stdout.log",
                f"{prefix}.publication.stderr.log",
                f"{prefix}.publication.exit-status.txt",
            )
        )
    if sorted(os.listdir(descriptor)) != sorted(expected):
        raise SystemExit("P30 deferred-journal inventory differs")
    for name in sorted(expected):
        file_descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            before = os.fstat(file_descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
                != (0, 0, 0o400)
            ):
                raise SystemExit(f"P30 deferred-journal file authority differs: {name}")
            if {
                value.decode() if isinstance(value, bytes) else value
                for value in os.listxattr(file_descriptor)
            } & forbidden_acl:
                raise SystemExit(f"P30 deferred-journal file has an ACL: {name}")
            raw = bytearray()
            while True:
                chunk = os.read(file_descriptor, 1024 * 1024)
                if not chunk:
                    break
                raw.extend(chunk)
            after = os.fstat(file_descriptor)
            by_name = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            fields = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fields(before) != fields(after) or fields(after) != fields(by_name):
                raise SystemExit(f"P30 deferred-journal file changed: {name}")
            expected_status = b"0\n"
            if (
                stage == "after-admission"
                and name
                == ".deferred-p30-prepare-pre-attempt-admission.exit-status.txt"
            ):
                expected_status = f"{admission_status}\n".encode("ascii")
            if name.endswith(".exit-status.txt") and bytes(raw) != expected_status:
                raise SystemExit(f"P30 deferred status differs: {name}")
            if ".publication." in name and name.endswith(("stdout.log", "stderr.log")):
                if raw:
                    raise SystemExit(f"P30 publication transcript is not empty: {name}")
        finally:
            os.close(file_descriptor)
finally:
    os.close(descriptor)
PY
}

validate_localization_ledger() {
  local prepare_state="$1"
  local invocation_state="$2"
  local authorization_state="$3"
  local status_state="${4:-absent}"
  sudo /usr/bin/python3 -I -S - \
    "$P30_LEDGER_ROOT" "$P30_DEFERRED_ROOT" "$P30_SEALED_ORCHESTRATOR" \
    "$P30_PREPARE_INVOCATION" "$P30_LOCALIZATION_INVOCATION" \
    "$P30_LOCALIZATION_AUTHORIZATION" "$P30_LOCALIZATION_EXIT_STATUS" \
    "$prepare_state" "$invocation_state" "$authorization_state" "$status_state" \
    "$P30_EXPECTED_ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

ledger = pathlib.PurePosixPath(sys.argv[1])
deferred = pathlib.PurePosixPath(sys.argv[2])
sealed_orchestrator = pathlib.PurePosixPath(sys.argv[3])
prepare = pathlib.PurePosixPath(sys.argv[4])
invocation = pathlib.PurePosixPath(sys.argv[5])
authorization = pathlib.PurePosixPath(sys.argv[6])
status_path = pathlib.PurePosixPath(sys.argv[7])
prepare_state, invocation_state, authorization_state, status_state = sys.argv[8:12]
expected_orchestrator_sha256 = sys.argv[12]
if ledger != pathlib.PurePosixPath("/var/lib/optimizationml-p30-20260906-03"):
    raise SystemExit("P30 localization ledger path differs")
if (
    deferred != ledger / "deferred"
    or sealed_orchestrator != ledger / "run_p30_umask_bound_control_seal.sh"
    or prepare != ledger / "prepare-runtime.invoked"
    or invocation != ledger / "run-localization.invoked"
    or authorization != ledger / "localization.authorized"
    or status_path != ledger / "run-localization.exit-status.txt"
):
    raise SystemExit("P30 localization ledger child path differs")
if any(value not in {"absent", "present"} for value in (
    prepare_state,
    invocation_state,
    authorization_state,
    status_state,
)):
    raise SystemExit("P30 localization ledger expectation is malformed")
if invocation_state == "present" and prepare_state != "present":
    raise SystemExit("P30 localization invocation cannot precede prepare admission")
if authorization_state == "present" and invocation_state != "present":
    raise SystemExit("P30 localization authorization cannot precede invocation")
if status_state == "present" and invocation_state != "present":
    raise SystemExit("P30 localization status cannot precede invocation")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open("/", flags)
try:
    for component in ledger.parts[1:]:
        child = os.open(component, flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, 0, 0o700)
    ):
        raise SystemExit("P30 localization ledger authority differs")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(descriptor)
    }
    if acl_names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit("P30 localization ledger has a forbidden ACL")
    expected_names = [deferred.name, sealed_orchestrator.name]
    if prepare_state == "present":
        expected_names.append(prepare.name)
    if invocation_state == "present":
        expected_names.append(invocation.name)
    if authorization_state == "present":
        expected_names.append(authorization.name)
    if status_state == "present":
        expected_names.append(status_path.name)
    if sorted(os.listdir(descriptor)) != sorted(expected_names):
        raise SystemExit("P30 localization ledger inventory differs")
    deferred_info = os.stat(deferred.name, dir_fd=descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(deferred_info.st_mode)
        or (deferred_info.st_uid, deferred_info.st_gid, stat.S_IMODE(deferred_info.st_mode))
        != (0, 0, 0o700)
    ):
        raise SystemExit("P30 deferred-journal authority differs")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    deferred_fd = os.open(
        deferred.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=descriptor,
    )
    try:
        if {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(deferred_fd)
        } & forbidden_acl:
            raise SystemExit("P30 deferred journal has a forbidden ACL")
    finally:
        os.close(deferred_fd)
    orchestrator_info = os.stat(
        sealed_orchestrator.name, dir_fd=descriptor, follow_symlinks=False
    )
    if (
        not stat.S_ISREG(orchestrator_info.st_mode)
        or orchestrator_info.st_nlink != 1
        or (
            orchestrator_info.st_uid,
            orchestrator_info.st_gid,
            stat.S_IMODE(orchestrator_info.st_mode),
        ) != (0, 0, 0o555)
    ):
        raise SystemExit("P30 sealed-orchestrator authority differs")
    orchestrator_fd = os.open(
        sealed_orchestrator.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=descriptor,
    )
    try:
        before = os.fstat(orchestrator_fd)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(orchestrator_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(orchestrator_fd)
        by_name = os.stat(
            sealed_orchestrator.name,
            dir_fd=descriptor,
            follow_symlinks=False,
        )
        fields = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fields(before) != fields(after) or fields(after) != fields(by_name):
            raise SystemExit("P30 sealed orchestrator changed during read")
        if digest.hexdigest() != expected_orchestrator_sha256:
            raise SystemExit("P30 sealed orchestrator digest differs")
        if {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(orchestrator_fd)
        } & forbidden_acl:
            raise SystemExit("P30 sealed orchestrator has a forbidden ACL")
    finally:
        os.close(orchestrator_fd)
    expected_tokens = []
    if prepare_state == "present":
        expected_tokens.append((prepare.name, b"prepare-runtime\n", False))
    if invocation_state == "present":
        expected_tokens.append((invocation.name, b"run-localization\n", False))
    if authorization_state == "present":
        expected_tokens.append((authorization.name, b"localization-authorized\n", False))
    if status_state == "present":
        expected_tokens.append((status_path.name, None, True))
    for name, expected_payload, is_status in expected_tokens:
        token_info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(token_info.st_mode)
            or token_info.st_nlink != 1
            or (token_info.st_uid, token_info.st_gid, stat.S_IMODE(token_info.st_mode))
            != (0, 0, 0o400)
        ):
            raise SystemExit(f"P30 localization ledger file authority differs: {name}")
        token_fd = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptor,
        )
        try:
            before = os.fstat(token_fd)
            payload = os.read(token_fd, 4096)
            if os.read(token_fd, 1) != b"":
                raise SystemExit("P30 localization ledger file is unexpectedly large")
            after = os.fstat(token_fd)
            by_name = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            fields = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fields(before) != fields(after) or fields(after) != fields(by_name):
                raise SystemExit("P30 localization ledger file changed during read")
            if is_status:
                try:
                    parsed = int(payload.strip())
                except ValueError as exc:
                    raise SystemExit("P30 localization status payload is malformed") from exc
                if payload != f"{parsed}\n".encode("ascii") or not 0 <= parsed <= 255:
                    raise SystemExit("P30 localization status payload differs")
            elif payload != expected_payload:
                raise SystemExit(f"P30 localization token payload differs: {name}")
        finally:
            os.close(token_fd)
    mount_points = {
        line.split()[4].replace("\\040", " ").replace("\\011", "\t")
        for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
        if len(line.split()) >= 5
    }
    if str(ledger) in mount_points:
        raise SystemExit("P30 localization ledger is an unexpected mount point")
finally:
    os.close(descriptor)
PY
}

create_localization_token() {
  local kind="$1"
  local token payload expected_before expected_after
  case "$kind" in
    prepare)
      token="$P30_PREPARE_INVOCATION"
      payload=prepare-runtime
      expected_before=$'deferred\nrun_p30_umask_bound_control_seal.sh'
      expected_after=$'deferred\nprepare-runtime.invoked\nrun_p30_umask_bound_control_seal.sh'
      ;;
    invocation)
      token="$P30_LOCALIZATION_INVOCATION"
      payload=run-localization
      expected_before=$'deferred\nprepare-runtime.invoked\nrun_p30_umask_bound_control_seal.sh'
      expected_after=$'deferred\nprepare-runtime.invoked\nrun-localization.invoked\nrun_p30_umask_bound_control_seal.sh'
      ;;
    authorization)
      token="$P30_LOCALIZATION_AUTHORIZATION"
      payload=localization-authorized
      expected_before=$'deferred\nprepare-runtime.invoked\nrun-localization.invoked\nrun_p30_umask_bound_control_seal.sh'
      expected_after=$'deferred\nlocalization.authorized\nprepare-runtime.invoked\nrun-localization.invoked\nrun_p30_umask_bound_control_seal.sh'
      ;;
    *) die "P30 localization-token kind is malformed" ;;
  esac
  sudo /usr/bin/python3 -I -S - \
    "$P30_LEDGER_ROOT" "$token" "$payload" "$expected_before" "$expected_after" \
    "$P30_EXPECTED_ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

ledger = pathlib.PurePosixPath(sys.argv[1])
token = pathlib.PurePosixPath(sys.argv[2])
payload = (sys.argv[3] + "\n").encode("ascii")
expected_before = sorted(filter(None, sys.argv[4].splitlines()))
expected_after = sorted(filter(None, sys.argv[5].splitlines()))
expected_orchestrator_sha256 = sys.argv[6]
if (
    len(expected_orchestrator_sha256) != 64
    or any(character not in "0123456789abcdef" for character in expected_orchestrator_sha256)
):
    raise SystemExit("P30 expected sealed-orchestrator digest is malformed")
if token.parent != ledger or token.name not in {
    "prepare-runtime.invoked",
    "run-localization.invoked",
    "localization.authorized",
}:
    raise SystemExit("P30 localization token path differs")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
ledger_fd = os.open("/", flags)
token_fd = None
deferred_fd = None
try:
    for component in ledger.parts[1:]:
        child = os.open(component, flags, dir_fd=ledger_fd)
        os.close(ledger_fd)
        ledger_fd = child
    ledger_info = os.fstat(ledger_fd)
    if (
        not stat.S_ISDIR(ledger_info.st_mode)
        or (ledger_info.st_uid, ledger_info.st_gid, stat.S_IMODE(ledger_info.st_mode))
        != (0, 0, 0o700)
    ):
        raise SystemExit("P30 localization ledger authority differs")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(ledger_fd)
    }
    if acl_names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit("P30 localization ledger has a forbidden ACL")
    if sorted(os.listdir(ledger_fd)) != expected_before:
        raise SystemExit("P30 localization ledger pre-inventory differs")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    deferred_fd = os.open(
        "deferred",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=ledger_fd,
    )
    deferred_before = os.fstat(deferred_fd)
    deferred_by_name_before = os.stat(
        "deferred", dir_fd=ledger_fd, follow_symlinks=False
    )
    deferred_fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        deferred_fields(deferred_before) != deferred_fields(deferred_by_name_before)
        or not stat.S_ISDIR(deferred_before.st_mode)
        or (
            deferred_before.st_uid,
            deferred_before.st_gid,
            stat.S_IMODE(deferred_before.st_mode),
        )
        != (0, 0, 0o700)
        or {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(deferred_fd)
        }
        & forbidden_acl
    ):
        raise SystemExit("P30 deferred journal authority differs before token creation")
    orchestrator_fd = os.open(
        "run_p30_umask_bound_control_seal.sh",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=ledger_fd,
    )
    try:
        before = os.fstat(orchestrator_fd)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(orchestrator_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(orchestrator_fd)
        by_name = os.stat(
            "run_p30_umask_bound_control_seal.sh",
            dir_fd=ledger_fd,
            follow_symlinks=False,
        )
        fields = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            fields(before) != fields(after)
            or fields(after) != fields(by_name)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode))
            != (0, 0, 0o555)
            or digest.hexdigest() != expected_orchestrator_sha256
            or {
                name.decode() if isinstance(name, bytes) else name
                for name in os.listxattr(orchestrator_fd)
            }
            & forbidden_acl
        ):
            raise SystemExit("P30 sealed orchestrator differs before token creation")
    finally:
        os.close(orchestrator_fd)
    token_fd = os.open(
        token.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
        dir_fd=ledger_fd,
    )
    view = memoryview(payload)
    while view:
        written = os.write(token_fd, view)
        if written <= 0:
            raise SystemExit("P30 localization token write made no progress")
        view = view[written:]
    os.fchmod(token_fd, 0o400)
    os.fsync(token_fd)
    created = os.fstat(token_fd)
    by_name = os.stat(token.name, dir_fd=ledger_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(created.st_mode)
        or created.st_nlink != 1
        or (created.st_uid, created.st_gid, stat.S_IMODE(created.st_mode))
        != (0, 0, 0o400)
        or created.st_size != len(payload)
        or (created.st_dev, created.st_ino) != (by_name.st_dev, by_name.st_ino)
    ):
        raise SystemExit("P30 localization token stored authority differs")
    if sorted(os.listdir(ledger_fd)) != expected_after:
        raise SystemExit("P30 localization ledger post-inventory differs")
    deferred_after = os.fstat(deferred_fd)
    deferred_by_name_after = os.stat(
        "deferred", dir_fd=ledger_fd, follow_symlinks=False
    )
    if (
        deferred_fields(deferred_before) != deferred_fields(deferred_after)
        or deferred_fields(deferred_after) != deferred_fields(deferred_by_name_after)
        or {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(deferred_fd)
        }
        & forbidden_acl
    ):
        raise SystemExit("P30 deferred journal changed during token creation")
    os.fsync(ledger_fd)
finally:
    if token_fd is not None:
        os.close(token_fd)
    if deferred_fd is not None:
        os.close(deferred_fd)
    os.close(ledger_fd)
PY
}

burn_localization_invocation() {
  create_localization_token invocation || return
}

burn_prepare_invocation() {
  create_localization_token prepare || return
}

authorize_localization() {
  create_localization_token authorization || return
}

write_localization_exit_status() {
  local status="$1"
  [[ "$status" =~ ^[0-9]+$ && "$status" -le 255 ]] ||
    die "P30 localization exit status is malformed"
  sudo /usr/bin/python3 -I -S - \
    "$P30_LEDGER_ROOT" "$P30_LOCALIZATION_EXIT_STATUS" "$status" \
    "$P30_EXPECTED_ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

ledger = pathlib.PurePosixPath(sys.argv[1])
path = pathlib.PurePosixPath(sys.argv[2])
payload = (sys.argv[3] + "\n").encode("ascii")
expected_orchestrator_sha256 = sys.argv[4]
if (
    len(expected_orchestrator_sha256) != 64
    or any(character not in "0123456789abcdef" for character in expected_orchestrator_sha256)
):
    raise SystemExit("P30 expected sealed-orchestrator digest is malformed")
if path != ledger / "run-localization.exit-status.txt":
    raise SystemExit("P30 localization status path differs")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
ledger_fd = os.open("/", flags)
file_fd = None
deferred_fd = None
try:
    for component in ledger.parts[1:]:
        child = os.open(component, flags, dir_fd=ledger_fd)
        os.close(ledger_fd)
        ledger_fd = child
    info = os.fstat(ledger_fd)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)) != (0, 0, 0o700)
    ):
        raise SystemExit("P30 localization ledger authority differs")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(ledger_fd)
    }
    if acl_names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit("P30 localization ledger has a forbidden ACL")
    names = sorted(os.listdir(ledger_fd))
    authorization_state = (
        "present" if "localization.authorized" in names else "absent"
    )
    allowed = [
        "deferred",
        "prepare-runtime.invoked",
        "run-localization.invoked",
        "run_p30_umask_bound_control_seal.sh",
    ]
    if authorization_state == "present":
        allowed.append("localization.authorized")
    if names != sorted(allowed):
        raise SystemExit("P30 localization ledger pre-status inventory differs")

    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    deferred_fd = os.open(
        "deferred",
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=ledger_fd,
    )
    deferred_before = os.fstat(deferred_fd)
    deferred_by_name_before = os.stat(
        "deferred", dir_fd=ledger_fd, follow_symlinks=False
    )
    deferred_fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        deferred_fields(deferred_before) != deferred_fields(deferred_by_name_before)
        or not stat.S_ISDIR(deferred_before.st_mode)
        or (
            deferred_before.st_uid,
            deferred_before.st_gid,
            stat.S_IMODE(deferred_before.st_mode),
        )
        != (0, 0, 0o700)
        or {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(deferred_fd)
        }
        & forbidden_acl
    ):
        raise SystemExit("P30 deferred journal differs before terminal status")

    orchestrator_fd = os.open(
        "run_p30_umask_bound_control_seal.sh",
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=ledger_fd,
    )
    try:
        before = os.fstat(orchestrator_fd)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(orchestrator_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(orchestrator_fd)
        by_name = os.stat(
            "run_p30_umask_bound_control_seal.sh",
            dir_fd=ledger_fd,
            follow_symlinks=False,
        )
        fields = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if (
            fields(before) != fields(after)
            or fields(after) != fields(by_name)
            or not stat.S_ISREG(after.st_mode)
            or after.st_nlink != 1
            or (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode))
            != (0, 0, 0o555)
            or digest.hexdigest() != expected_orchestrator_sha256
            or {
                name.decode() if isinstance(name, bytes) else name
                for name in os.listxattr(orchestrator_fd)
            }
            & forbidden_acl
        ):
            raise SystemExit("P30 sealed orchestrator differs before terminal status")
    finally:
        os.close(orchestrator_fd)

    def require_token(name, expected):
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=ledger_fd,
        )
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
                != (0, 0, 0o400)
            ):
                raise SystemExit(f"P30 localization token authority differs: {name}")
            observed = os.read(descriptor, len(expected) + 1)
            after = os.fstat(descriptor)
            by_name = os.stat(name, dir_fd=ledger_fd, follow_symlinks=False)
            fields = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fields(before) != fields(after) or fields(after) != fields(by_name):
                raise SystemExit(f"P30 localization token changed: {name}")
            if observed != expected:
                raise SystemExit(f"P30 localization token bytes differ: {name}")
        finally:
            os.close(descriptor)

    require_token("prepare-runtime.invoked", b"prepare-runtime\n")
    require_token("run-localization.invoked", b"run-localization\n")
    if authorization_state == "present":
        require_token("localization.authorized", b"localization-authorized\n")
    file_fd = os.open(
        path.name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o400,
        dir_fd=ledger_fd,
    )
    view = memoryview(payload)
    while view:
        written = os.write(file_fd, view)
        if written <= 0:
            raise SystemExit("P30 localization status write made no progress")
        view = view[written:]
    os.fchmod(file_fd, 0o400)
    os.fsync(file_fd)
    created = os.fstat(file_fd)
    by_name = os.stat(path.name, dir_fd=ledger_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(created.st_mode)
        or created.st_nlink != 1
        or (created.st_uid, created.st_gid, stat.S_IMODE(created.st_mode))
        != (0, 0, 0o400)
        or created.st_size != len(payload)
        or (created.st_dev, created.st_ino) != (by_name.st_dev, by_name.st_ino)
    ):
        raise SystemExit("P30 localization status authority differs")
    os.lseek(file_fd, 0, os.SEEK_SET)
    observed = os.read(file_fd, len(payload) + 1)
    if observed != payload:
        raise SystemExit("P30 localization status stored bytes differ")
    expected_after = sorted([*allowed, path.name])
    if sorted(os.listdir(ledger_fd)) != expected_after:
        raise SystemExit("P30 localization ledger post-status inventory differs")
    deferred_after = os.fstat(deferred_fd)
    deferred_by_name_after = os.stat(
        "deferred", dir_fd=ledger_fd, follow_symlinks=False
    )
    if (
        deferred_fields(deferred_before) != deferred_fields(deferred_after)
        or deferred_fields(deferred_after) != deferred_fields(deferred_by_name_after)
        or {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(deferred_fd)
        }
        & forbidden_acl
    ):
        raise SystemExit("P30 deferred journal changed during terminal status")
    os.fsync(ledger_fd)
finally:
    if file_fd is not None:
        os.close(file_fd)
    if deferred_fd is not None:
        os.close(deferred_fd)
    os.close(ledger_fd)
PY
}

validate_root_evidence_file() {
  local path="$1"
  local expected_sha256="$2"
  local expected_byte_count="$3"
  local expected_mode="$4"
  local expected_uid="$5"
  local expected_gid="$6"
  sudo /usr/bin/python3 -I -S - \
    "$P30_HOST_EVIDENCE" "$path" "$expected_sha256" \
    "$expected_byte_count" "$expected_mode" "$expected_uid" "$expected_gid" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

(
    evidence_text,
    path_text,
    expected_digest,
    expected_size_text,
    expected_mode_text,
    expected_uid_text,
    expected_gid_text,
) = sys.argv[1:]
evidence = pathlib.PurePosixPath(evidence_text)
path = pathlib.PurePosixPath(path_text)
if path.parent != evidence or path.name in {"", ".", ".."}:
    raise SystemExit("P30 retained file is not a direct evidence child")

for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def open_componentwise(target: pathlib.PurePosixPath) -> int:
    descriptor = os.open("/", directory_flags)
    try:
        for component in target.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


evidence_fd = open_componentwise(evidence)
file_fd = None
try:
    evidence_info = os.fstat(evidence_fd)
    if (
        not stat.S_ISDIR(evidence_info.st_mode)
        or (
            evidence_info.st_uid,
            evidence_info.st_gid,
            stat.S_IMODE(evidence_info.st_mode),
        )
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 evidence-directory authority differs")
    file_fd = os.open(path.name, file_flags, dir_fd=evidence_fd)
    before = os.fstat(file_fd)
    expected_mode = int(expected_mode_text, 8)
    expected_uid = int(expected_uid_text)
    expected_gid = int(expected_gid_text)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (expected_uid, expected_gid, expected_mode)
    ):
        raise SystemExit("P30 retained-file type, link count, owner, or mode differs")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(file_fd)
    }
    if acl_names & {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }:
        raise SystemExit("P30 retained file has a forbidden ACL")
    digest = hashlib.sha256()
    while True:
        chunk = os.read(file_fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    after = os.fstat(file_fd)
    by_name = os.stat(path.name, dir_fd=evidence_fd, follow_symlinks=False)
    if (
        fields(before) != fields(after)
        or stat.S_ISLNK(by_name.st_mode)
        or fields(by_name) != fields(after)
    ):
        raise SystemExit("P30 retained-file identity or metadata changed during read")
    actual_digest = digest.hexdigest()
    if expected_digest != "-" and actual_digest != expected_digest:
        raise SystemExit("P30 retained-file SHA-256 differs")
    if expected_size_text != "-" and after.st_size != int(expected_size_text):
        raise SystemExit("P30 retained-file byte count differs")
    mount_points = {
        line.split()[4].replace("\\040", " ").replace("\\011", "\t")
        for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
        if len(line.split()) >= 5
    }
    if str(path) in mount_points:
        raise SystemExit("P30 retained file is an unexpected mount point")
    print(
        f"{actual_digest}:{after.st_size}:{after.st_dev}:{after.st_ino}:"
        f"{after.st_uid}:{after.st_gid}:{stat.S_IMODE(after.st_mode):04o}"
    )
finally:
    if file_fd is not None:
        os.close(file_fd)
    os.close(evidence_fd)
PY
}

read_sealed_evidence_line() {
  local path="$1"
  local pattern="$2"
  /usr/bin/python3 -I -S - "$P30_HOST_EVIDENCE" "$path" "$pattern" <<'PY'
import os
import pathlib
import re
import stat
import sys

evidence = pathlib.PurePosixPath(sys.argv[1])
path = pathlib.PurePosixPath(sys.argv[2])
pattern = sys.argv[3]
if path.parent != evidence or path.name in {"", ".", ".."}:
    raise SystemExit("P30 sealed-line path differs")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
parent_fd = os.open("/", flags)
descriptor = None
try:
    for component in evidence.parts[1:]:
        child = os.open(component, flags, dir_fd=parent_fd)
        os.close(parent_fd)
        parent_fd = child
    descriptor = os.open(
        path.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_fd,
    )
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (0, 0, 0o444)
    ):
        raise SystemExit("P30 sealed-line authority differs")
    raw = os.read(descriptor, 4096)
    if os.read(descriptor, 1) != b"":
        raise SystemExit("P30 sealed-line input is unexpectedly large")
    after = os.fstat(descriptor)
    by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P30 sealed-line input changed during read")
    try:
        value = raw.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as exc:
        raise SystemExit("P30 sealed-line input is not ASCII") from exc
    if raw != (value + "\n").encode("ascii") or re.fullmatch(pattern, value) is None:
        raise SystemExit("P30 sealed-line payload differs")
    print(value)
finally:
    if descriptor is not None:
        os.close(descriptor)
    os.close(parent_fd)
PY
}

require_source_environment() {
  [[ "$(/usr/bin/id -u)" == "0" && "$(/usr/bin/id -g)" == "0" ]] ||
    die "P30 host orchestration must run as root from the sealed orchestrator"
  require_var P30_ATTEMPT_ID
  [[ "$P30_ATTEMPT_ID" == "$P30_REQUIRED_ATTEMPT_ID" ]] ||
    die "P30_ATTEMPT_ID must equal $P30_REQUIRED_ATTEMPT_ID"
  require_var P30_GPU_UUID
  [[ "$P30_GPU_UUID" == "$P30_REQUIRED_GPU_UUID" ]] ||
    die "P30_GPU_UUID differs from the preregistered full A100 UUID"
  require_absolute_path P30_AUTHORITY_REPO
  [[ "$P30_AUTHORITY_REPO" == "$P30_TRANSPORT_ROOT/authority-185e444" ]] ||
    die "P30_AUTHORITY_REPO differs from the transport-contained authority path"
  require_absolute_path P30_NANOGPT_HOST
  require_absolute_path P30_MUON_HOST
  require_absolute_path P30_DATA_HOST
  [[ "$P30_NANOGPT_HOST" == "$P30_CANONICAL_NANOGPT_HOST" ]] ||
    die "P30_NANOGPT_HOST differs from the frozen canonical source"
  [[ "$P30_MUON_HOST" == "$P30_CANONICAL_MUON_HOST" ]] ||
    die "P30_MUON_HOST differs from the frozen canonical source"
  [[ "$P30_DATA_HOST" == "$P30_CANONICAL_DATA_HOST" ]] ||
    die "P30_DATA_HOST differs from the frozen canonical source"
  require_commit P30_SOURCE_FREEZE_COMMIT
  require_commit P30_SOURCE_FREEZE_TREE
  require_sha256 P30_EXPECTED_SOURCE_BUNDLE_SHA256
  require_var P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT
  [[ "$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" =~ ^[1-9][0-9]*$ ]] ||
    die "P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT is not a positive integer"
  require_sha256 P30_EXPECTED_SOURCE_RECEIPT_SHA256
  require_sha256 P30_EXPECTED_CONTRACT_SHA256
  require_sha256 P30_EXPECTED_ORCHESTRATOR_SHA256
  require_sha256 P30_EXPECTED_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_BUNDLE_VERIFIER_SHA256
  require_sha256 P30_EXPECTED_P29_CONTRACT_SHA256
  require_sha256 P30_EXPECTED_P29_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_P29_OUTCOME_SHA256
  require_sha256 P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_P28_CONTRACT_SHA256
  require_sha256 P30_EXPECTED_P28_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_P28_OUTCOME_SHA256
  require_sha256 P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_P27_CONTRACT_SHA256
  require_sha256 P30_EXPECTED_P27_RECONSTRUCTOR_SHA256
  require_sha256 P30_EXPECTED_P27_LOCALIZER_SHA256
  require_sha256 P30_EXPECTED_INGESTER_SHA256
  require_sha256 P30_EXPECTED_P23_CORE_SHA256
  require_sha256 P30_EXPECTED_P27_SANITIZER_SHA256
}

require_runtime_environment() {
  require_source_environment
  require_commit P30_RUNTIME_REVIEW_COMMIT
  require_commit P30_RUNTIME_REVIEW_TREE
  require_sha256 P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256
  require_var P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT
  [[ "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" =~ ^[1-9][0-9]*$ ]] ||
    die "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT is not a positive integer"
  require_sha256 P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256
  require_sha256 P30_EXPECTED_RUNTIME_LOCK_SHA256
  require_sha256 P30_EXPECTED_HOST_ATTESTATION_SHA256
  require_var P30_EXPECTED_CONTAINER_ID
  [[ "$P30_EXPECTED_CONTAINER_ID" =~ ^[0-9a-f]{64}$ ]] ||
    die "P30_EXPECTED_CONTAINER_ID is not one full container ID"
}

assert_clean_checkout() {
  local repository="$1"
  local head="$2"
  local tree="$3"
  local label="$4"
  [[ -d "$repository" && ! -L "$repository" && \
     -d "$repository/.git" && ! -L "$repository/.git" ]] ||
    die "$label is not a plain Git worktree"
  local forbidden_git_path
  for forbidden_git_path in \
    "$repository/.git/commondir" \
    "$repository/.git/objects/info/alternates" \
    "$repository/.git/objects/info/http-alternates" \
    "$repository/.git/info/grafts"; do
    [[ ! -e "$forbidden_git_path" && ! -L "$forbidden_git_path" ]] ||
      die "$label has forbidden Git indirection: $forbidden_git_path"
  done
  local config_output config_status
  set +e
  config_output="$(p30_git --git-dir="$repository/.git" config --local --no-includes \
      --name-only --list 2>/dev/null)"
  config_status=$?
  set -e
  (( config_status == 0 )) ||
    die "$label Git config inspection failed"
  local config_key normalized_key
  while IFS= read -r config_key; do
    normalized_key="${config_key,,}"
    case "$normalized_key" in
      include.path|includeif.*.path|core.worktree|core.fsmonitor|core.hookspath|\
      core.sshcommand|core.attributesfile|core.editor|core.pager|credential.helper|\
      diff.external|diff.*.command|filter.*.clean|filter.*.smudge|filter.*.process|\
      merge.*.driver|pager.*|interactive.difffilter|extensions.worktreeconfig|\
      extensions.partialclone|remote.*.promisor|remote.*.partialclonefilter)
        die "$label has forbidden local Git configuration: $config_key"
        ;;
    esac
  done <<<"$config_output"
  [[ "$(/usr/bin/realpath -- "$repository")" == "$repository" ]] ||
    die "$label lexical and resolved worktree paths differ"
  [[ "$(/usr/bin/realpath -- "$repository/.git")" == "$repository/.git" ]] ||
    die "$label lexical and resolved Git-directory paths differ"
  [[ "$(p30_git --git-dir="$repository/.git" rev-parse --is-shallow-repository)" == \
      "false" ]] || die "$label checkout is shallow"
  local replace_refs
  replace_refs="$(p30_git --git-dir="$repository/.git" for-each-ref \
      --format='%(refname)' refs/replace/)" ||
    die "$label replace-ref inspection failed"
  [[ -z "$replace_refs" ]] || die "$label checkout has replacement refs"
  local actual_head actual_tree actual_top_level actual_git_dir status_output
  actual_top_level="$(p30_git -C "$repository" rev-parse --show-toplevel)" ||
    die "$label worktree root could not be resolved"
  [[ "$actual_top_level" == "$repository" ]] ||
    die "$label Git top-level differs from the mounted worktree"
  actual_git_dir="$(p30_git -C "$repository" rev-parse --absolute-git-dir)" ||
    die "$label Git directory could not be resolved"
  [[ "$actual_git_dir" == "$repository/.git" ]] ||
    die "$label Git directory differs from the mounted worktree"
  actual_head="$(p30_git -C "$repository" rev-parse HEAD)" ||
    die "$label HEAD could not be resolved"
  [[ "$actual_head" == "$head" ]] ||
    die "$label HEAD differs"
  actual_tree="$(p30_git -C "$repository" rev-parse HEAD^{tree})" ||
    die "$label tree could not be resolved"
  [[ "$actual_tree" == "$tree" ]] ||
    die "$label tree differs"
  status_output="$(p30_git -C "$repository" status --porcelain=v1 \
      --untracked-files=all --ignored=matching)" ||
    die "$label cleanliness inspection failed"
  [[ -z "$status_output" ]] ||
    die "$label contains a tracked change, untracked file, or ignored file"
}

validate_namespace_layout() {
  /usr/bin/python3 -I -S - \
    "$P30_NAMESPACE_ROOT" "$P30_TRANSPORT_ROOT" "$P30_CONTROL_REPO" \
    "$P30_AUTHORITY_REPO" "$P30_ATTEMPT_ROOT" "$P30_EXECUTION_ROOT" <<'PY'
import os
import pathlib
import stat
import sys

namespace, transport, control, authority, attempt, execution = map(
    pathlib.PurePosixPath, sys.argv[1:]
)
secure_parent = namespace.parent
if secure_parent != pathlib.PurePosixPath("/secure"):
    raise SystemExit("P30 secure-parent path differs")
if namespace != pathlib.PurePosixPath("/secure/p30"):
    raise SystemExit("P30 namespace path differs")
if transport != namespace / "transport-20260906-03":
    raise SystemExit("P30 transport path differs")
if control != transport / "control-source":
    raise SystemExit("P30 control checkout is not transport-contained")
if authority != transport / "authority-185e444":
    raise SystemExit("P30 authority checkout is not transport-contained")
if attempt != namespace / "attempt-20260906-03":
    raise SystemExit("P30 attempt-root path differs")
if execution != namespace / "execution-20260906-03":
    raise SystemExit("P30 execution-root path differs")

flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
flags |= getattr(os, "O_NOFOLLOW", 0)


def mount_id(descriptor: int) -> int:
    for line in pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split(":", 1)[1])
    raise SystemExit("P30 directory descriptor has no mount ID")


def bind_directory(path: pathlib.PurePosixPath) -> tuple[int, int, int, int, int, int]:
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        value = os.fstat(descriptor)
        if not stat.S_ISDIR(value.st_mode):
            raise SystemExit(f"P30 path is not a directory: {path}")
        acl_names = {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(descriptor)
        }
        forbidden = {
            "system.posix_acl_access",
            "system.posix_acl_default",
            "system.nfs4_acl",
            "system.richacl",
        }
        if acl_names & forbidden:
            raise SystemExit(f"P30 path has an ACL: {path}")
        return (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            value.st_gid,
            stat.S_IMODE(value.st_mode),
            mount_id(descriptor),
        )
    finally:
        os.close(descriptor)


first_secure_parent = bind_directory(secure_parent)
first_namespace = bind_directory(namespace)
first_transport = bind_directory(transport)
if first_secure_parent[2:5] != (0, 0, 0o755):
    raise SystemExit("P30 secure-parent owner or mode differs")
if first_namespace[2:5] != (0, 0, 0o755):
    raise SystemExit("P30 namespace owner or mode differs")
if first_transport[2:5] != (1000, 1000, 0o700):
    raise SystemExit("P30 transport owner or mode differs")
if len({first_secure_parent[0], first_namespace[0], first_transport[0]}) != 1:
    raise SystemExit("P30 secure parent, namespace, and transport are on different devices")
if len({first_secure_parent[5], first_namespace[5], first_transport[5]}) != 1:
    raise SystemExit("P30 secure parent, namespace, and transport cross a mount-ID boundary")

namespace_fd = os.open(str(namespace), flags)
try:
    namespace_children = sorted(os.listdir(namespace_fd))
finally:
    os.close(namespace_fd)
allowed_inventories = [
    ["transport-20260906-03"],
    [
        "attempt-20260906-03",
        "execution-20260906-03",
        "transport-20260906-03",
    ],
]
if namespace_children not in allowed_inventories:
    raise SystemExit(
        "P30 namespace child inventory differs from the sole-capability layout: "
        f"{namespace_children!r}"
    )

mount_points = set()
for raw_line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines():
    fields = raw_line.split()
    if len(fields) >= 5:
        mount_points.add(fields[4].replace("\\040", " ").replace("\\011", "\t"))
if any(str(path) in mount_points for path in (secure_parent, namespace, transport)):
    raise SystemExit("P30 secure parent, namespace, or transport is an unexpected mount point")

if bind_directory(secure_parent) != first_secure_parent:
    raise SystemExit("P30 secure-parent identity changed during validation")
if bind_directory(namespace) != first_namespace:
    raise SystemExit("P30 namespace identity changed during validation")
if bind_directory(transport) != first_transport:
    raise SystemExit("P30 transport identity changed during validation")
PY
}

validate_fresh_attempt_layout() {
  /usr/bin/python3 -I -S - \
    "$P30_NAMESPACE_ROOT" "$P30_TRANSPORT_ROOT" \
    "$P30_ATTEMPT_ROOT" "$P30_HOST_EVIDENCE" "$P30_EXECUTION_ROOT" <<'PY'
import os
import pathlib
import stat
import sys

namespace, transport, attempt, evidence, execution = map(
    pathlib.PurePosixPath, sys.argv[1:]
)
secure_parent = namespace.parent
if secure_parent != pathlib.PurePosixPath("/secure"):
    raise SystemExit("P30 runtime secure-parent layout differs")
if transport != namespace / "transport-20260906-03":
    raise SystemExit("P30 transport layout differs")
if attempt != namespace / "attempt-20260906-03" or evidence != attempt / "evidence":
    raise SystemExit("P30 attempt layout differs")
if execution != namespace / "execution-20260906-03":
    raise SystemExit("P30 execution snapshot layout differs")

required_flags = []
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
    required_flags.append(value)
flags = os.O_RDONLY
for value in required_flags:
    flags |= value


def mount_id(descriptor: int) -> int:
    for line in pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split(":", 1)[1])
    raise SystemExit("P30 attempt descriptor has no mount ID")


def require_authority(
    descriptor: int,
    path: pathlib.PurePosixPath,
    expected: tuple[int, int, int],
) -> tuple[int, int, int, int, int, int]:
    value = os.fstat(descriptor)
    actual = (value.st_uid, value.st_gid, stat.S_IMODE(value.st_mode))
    if not stat.S_ISDIR(value.st_mode) or actual != expected:
        raise SystemExit(f"P30 attempt component authority differs: {path}")
    acl_names = {
        name.decode() if isinstance(name, bytes) else name
        for name in os.listxattr(descriptor)
    }
    forbidden = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    if acl_names & forbidden:
        raise SystemExit(f"P30 attempt component has an ACL: {path}")
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
        mount_id(descriptor),
    )


secure_parent_fd = os.open(str(secure_parent), flags)
namespace_fd = os.open("p30", flags, dir_fd=secure_parent_fd)
try:
    namespace_children = sorted(os.listdir(namespace_fd))
    if namespace_children != [
        "attempt-20260906-03",
        "execution-20260906-03",
        "transport-20260906-03",
    ]:
        raise SystemExit(f"P30 runtime namespace inventory differs: {namespace_children!r}")
    transport_fd = os.open("transport-20260906-03", flags, dir_fd=namespace_fd)
    try:
        attempt_fd = os.open("attempt-20260906-03", flags, dir_fd=namespace_fd)
        try:
            execution_fd = os.open("execution-20260906-03", flags, dir_fd=namespace_fd)
            try:
                evidence_fd = os.open("evidence", flags, dir_fd=attempt_fd)
                try:
                    records = {
                        "secure_parent": require_authority(
                            secure_parent_fd, secure_parent, (0, 0, 0o755)
                        ),
                        "namespace": require_authority(
                            namespace_fd, namespace, (0, 0, 0o755)
                        ),
                        "transport": require_authority(
                            transport_fd, transport, (1000, 1000, 0o700)
                        ),
                        "attempt": require_authority(
                            attempt_fd, attempt, (0, 0, 0o555)
                        ),
                        "evidence": require_authority(
                            evidence_fd, evidence, (0, 0, 0o555)
                        ),
                        "execution": require_authority(
                            execution_fd, execution, (0, 0, 0o555)
                        ),
                    }
                    devices = {record[0] for record in records.values()}
                    mounts = {record[5] for record in records.values()}
                    if len(devices) != 1 or len(mounts) != 1:
                        raise SystemExit(
                            "P30 runtime layout crosses a device or mount-ID boundary"
                        )
                    for path, descriptor, record in (
                        (secure_parent, secure_parent_fd, records["secure_parent"]),
                        (namespace, namespace_fd, records["namespace"]),
                        (transport, transport_fd, records["transport"]),
                        (attempt, attempt_fd, records["attempt"]),
                        (evidence, evidence_fd, records["evidence"]),
                        (execution, execution_fd, records["execution"]),
                    ):
                        by_name = os.lstat(path)
                        if stat.S_ISLNK(by_name.st_mode) or (
                            by_name.st_dev,
                            by_name.st_ino,
                        ) != record[:2]:
                            raise SystemExit(
                                f"P30 runtime component path identity differs: {path}"
                            )
                        descriptor_info = os.fstat(descriptor)
                        if (descriptor_info.st_dev, descriptor_info.st_ino) != record[:2]:
                            raise SystemExit(
                                f"P30 runtime component descriptor identity differs: {path}"
                            )
                    execution_children = sorted(os.listdir(execution_fd))
                    if execution_children != [
                        "data",
                        "muon",
                        "nanogpt",
                        "repository",
                        "snapshot-manifest.json",
                    ]:
                        raise SystemExit(
                            f"P30 execution top-level inventory differs: {execution_children!r}"
                        )
                    manifest_info = os.stat(
                        "snapshot-manifest.json",
                        dir_fd=execution_fd,
                        follow_symlinks=False,
                    )
                    if (
                        not stat.S_ISREG(manifest_info.st_mode)
                        or manifest_info.st_nlink != 1
                        or (
                            manifest_info.st_uid,
                            manifest_info.st_gid,
                            stat.S_IMODE(manifest_info.st_mode),
                        )
                        != (0, 0, 0o444)
                    ):
                        raise SystemExit("P30 execution manifest authority differs")
                finally:
                    os.close(evidence_fd)
            finally:
                os.close(execution_fd)
        finally:
            os.close(attempt_fd)
    finally:
        os.close(transport_fd)
finally:
    os.close(namespace_fd)
    os.close(secure_parent_fd)

mount_points = {
    line.split()[4].replace("\\040", " ").replace("\\011", "\t")
    for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
    if len(line.split()) >= 5
}
if any(
    str(path) in mount_points
    for path in (secure_parent, namespace, transport, attempt, evidence, execution)
):
    raise SystemExit("P30 runtime component is an unexpected mount point")
print(
    "|".join(
        f"{label}:{record[0]}:{record[1]}:{record[2]}:{record[3]}:"
        f"{record[4]:04o}:{record[5]}"
        for label, record in records.items()
    )
)
PY
}

validate_expected_attempt_layout() {
  local expected_binding="$1"
  local actual_binding
  actual_binding="$(validate_fresh_attempt_layout)"
  [[ "$actual_binding" == "$expected_binding" ]] ||
    die "P30 attempt or evidence directory identity changed"
  echo "$actual_binding"
}

validate_attempt_layout_against_log() {
  local binding_log="$1"
  local expected_binding
  expected_binding="$(/usr/bin/python3 -I -S - \
    "$P30_HOST_EVIDENCE" "$binding_log" <<'PY'
import os
import pathlib
import re
import stat
import sys

evidence = pathlib.PurePosixPath(sys.argv[1])
path = pathlib.PurePosixPath(sys.argv[2])
if path.parent != evidence or path.name in {"", ".", ".."}:
    raise SystemExit("P30 attempt-binding log is not a direct evidence child")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
evidence_fd = os.open("/", flags)
file_fd = None
try:
    for component in evidence.parts[1:]:
        child = os.open(component, flags, dir_fd=evidence_fd)
        os.close(evidence_fd)
        evidence_fd = child
    evidence_info = os.fstat(evidence_fd)
    if (
        not stat.S_ISDIR(evidence_info.st_mode)
        or (evidence_info.st_uid, evidence_info.st_gid, stat.S_IMODE(evidence_info.st_mode))
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 attempt-binding evidence authority differs")
    file_fd = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=evidence_fd)
    before = os.fstat(file_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (0, 0, 0o444)
    ):
        raise SystemExit("P30 attempt-binding log authority differs")
    chunks = []
    while True:
        chunk = os.read(file_fd, 4096)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(file_fd)
    by_name = os.stat(path.name, dir_fd=evidence_fd, follow_symlinks=False)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P30 attempt-binding log changed during read")
    raw = b"".join(chunks)
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise SystemExit("P30 attempt-binding log is not ASCII") from exc
    pattern = (
        r"secure_parent:\d+:\d+:0:0:0755:\d+\|"
        r"namespace:\d+:\d+:0:0:0755:\d+\|"
        r"transport:\d+:\d+:1000:1000:0700:\d+\|"
        r"attempt:\d+:\d+:0:0:0555:\d+\|"
        r"evidence:\d+:\d+:0:0:0555:\d+\|"
        r"execution:\d+:\d+:0:0:0555:\d+\n"
    )
    if re.fullmatch(pattern, text) is None:
        raise SystemExit("P30 attempt-binding log payload is malformed")
    sys.stdout.write(text[:-1])
finally:
    if file_fd is not None:
        os.close(file_fd)
    os.close(evidence_fd)
PY
  )" || die "P30 retained attempt binding could not be authenticated"
  validate_expected_attempt_layout "$expected_binding"
}

validate_runtime_namespace_against_receipt() {
  /usr/bin/python3 -I -S - \
    "$P30_RUNTIME_REVIEW_RECEIPT" "$P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256" \
    "$P30_NAMESPACE_ROOT" "$P30_TRANSPORT_ROOT" "$P30_ATTEMPT_ROOT" \
    "$P30_HOST_EVIDENCE" "$P30_EXECUTION_ROOT" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

(
    receipt_text,
    expected_receipt_sha256,
    namespace_text,
    transport_text,
    attempt_text,
    evidence_text,
    execution_text,
) = sys.argv[1:]
receipt = pathlib.PurePosixPath(receipt_text)
namespace = pathlib.PurePosixPath(namespace_text)
transport = pathlib.PurePosixPath(transport_text)
attempt = pathlib.PurePosixPath(attempt_text)
evidence = pathlib.PurePosixPath(evidence_text)
execution = pathlib.PurePosixPath(execution_text)
secure_parent = namespace.parent
if receipt != transport / "p30_runtime_review_bundle_receipt.json":
    raise SystemExit("P30 reviewed runtime receipt path differs")
expected_paths = {
    "secure_parent": secure_parent,
    "namespace": namespace,
    "transport": transport,
    "attempt_root": attempt,
    "attempt_evidence": evidence,
    "execution_root": execution,
}
required_flags = []
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
    required_flags.append(value)
directory_flags = os.O_RDONLY
for value in required_flags:
    directory_flags |= value
file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def reject_duplicates(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(token):
    raise ValueError(f"nonfinite JSON constant: {token}")


def fields(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def open_directory(path):
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def mount_id(descriptor):
    for line in pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
        if line.startswith("mnt_id:"):
            return int(line.split(":", 1)[1])
    raise SystemExit("P30 receipt-bound descriptor lacks a mount ID")


transport_fd = open_directory(transport)
receipt_fd = None
try:
    receipt_fd = os.open(receipt.name, file_flags, dir_fd=transport_fd)
    before = os.fstat(receipt_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (1000, 1000, 0o600)
    ):
        raise SystemExit("P30 reviewed runtime receipt authority differs")
    raw = bytearray()
    while True:
        chunk = os.read(receipt_fd, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
    after = os.fstat(receipt_fd)
    by_name = os.stat(receipt.name, dir_fd=transport_fd, follow_symlinks=False)
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P30 reviewed runtime receipt changed during read")
finally:
    if receipt_fd is not None:
        os.close(receipt_fd)
    os.close(transport_fd)
if hashlib.sha256(raw).hexdigest() != expected_receipt_sha256:
    raise SystemExit("P30 reviewed runtime receipt digest differs")
try:
    payload = json.loads(
        bytes(raw),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    permission = payload["permission_safe_namespace"]
    guarded = permission["guarded_child_directories"]
except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
    raise SystemExit("P30 reviewed runtime receipt is malformed") from exc
top_records = {
    "secure_parent": permission.get("secure_parent"),
    "namespace": permission.get("namespace"),
    "transport": permission.get("transport"),
}
records = {
    **top_records,
    "attempt_root": guarded.get("attempt_root"),
    "attempt_evidence": guarded.get("attempt_evidence"),
    "execution_root": guarded.get("execution_root"),
}
expected_authority = {
    "secure_parent": (0, 0, "0755"),
    "namespace": (0, 0, "0755"),
    "transport": (1000, 1000, "0700"),
    "attempt_root": (0, 0, "0555"),
    "attempt_evidence": (0, 0, "0555"),
    "execution_root": (0, 0, "0555"),
}
descriptors = {}
try:
    for label, path in expected_paths.items():
        record = records.get(label)
        if not isinstance(record, dict) or record.get("path") != str(path):
            raise SystemExit(f"P30 reviewed {label} receipt record differs")
        descriptor = open_directory(path)
        descriptors[label] = descriptor
        info = os.fstat(descriptor)
        uid, gid, mode = expected_authority[label]
        observed = {
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "mode_octal": format(stat.S_IMODE(info.st_mode), "04o"),
            "mount_id": mount_id(descriptor),
            "path": str(path),
        }
        if (info.st_uid, info.st_gid, observed["mode_octal"]) != (uid, gid, mode):
            raise SystemExit(f"P30 current {label} authority differs")
        for key, value in observed.items():
            if record.get(key) != value:
                raise SystemExit(f"P30 current {label} differs from receipt at {key}")
        acl_names = {
            item.decode() if isinstance(item, bytes) else item
            for item in os.listxattr(descriptor)
        }
        if acl_names & {
            "system.posix_acl_access",
            "system.posix_acl_default",
            "system.nfs4_acl",
            "system.richacl",
        }:
            raise SystemExit(f"P30 current {label} has a forbidden ACL")
    if len({os.fstat(value).st_dev for value in descriptors.values()}) != 1:
        raise SystemExit("P30 receipt-bound namespace crosses devices")
    if len({mount_id(value) for value in descriptors.values()}) != 1:
        raise SystemExit("P30 receipt-bound namespace crosses mount IDs")
finally:
    for descriptor in descriptors.values():
        os.close(descriptor)
print("runtime_receipt_namespace_binding=true")
PY
}

validate_post_freeze_attempt_state() {
  local binding_log="$1"
  local actual_binding
  actual_binding="$(validate_attempt_layout_against_log "$binding_log")" ||
    die "P30 post-freeze attempt binding validation failed"
  printf 'attempt_binding=%s\n' "$actual_binding"
  echo -n "runtime_lock_binding="
  validate_root_evidence_file \
    "$P30_HOST_EVIDENCE/p30_cuda_runtime_lock.json" - - 0444 0 0
  echo -n "host_attestation_binding="
  validate_root_evidence_file \
    "$P30_HOST_EVIDENCE/p30_host_attestation.json" - - 0444 0 0
}

transport_checkout_layout_binding() {
  local label="$1"
  local checkout="$2"
  local basename="$3"
  /usr/bin/python3 -I -S - \
    "$P30_TRANSPORT_ROOT" "$checkout" "$basename" "$label" <<'PY'
import os
import pathlib
import stat
import sys

transport = pathlib.PurePosixPath(sys.argv[1])
checkout = pathlib.PurePosixPath(sys.argv[2])
basename = sys.argv[3]
label = sys.argv[4]
if basename not in {"control-source", "authority-185e444"}:
    raise SystemExit("P30 checkout basename is not preregistered")
if checkout != transport / basename:
    raise SystemExit(f"P30 {label} path differs")
flags = os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0)
flags |= getattr(os, "O_NOFOLLOW", 0)
transport_fd = os.open(str(transport), flags)
try:
    checkout_fd = os.open(basename, flags, dir_fd=transport_fd)
    try:
        transport_info = os.fstat(transport_fd)
        checkout_info = os.fstat(checkout_fd)
        if not stat.S_ISDIR(checkout_info.st_mode):
            raise SystemExit(f"P30 {label} is not a directory")
        if (
            checkout_info.st_uid,
            checkout_info.st_gid,
            stat.S_IMODE(checkout_info.st_mode),
        ) != (1000, 1000, 0o700):
            raise SystemExit(f"P30 {label} owner or mode differs")
        if transport_info.st_dev != checkout_info.st_dev:
            raise SystemExit(f"P30 {label} crosses a device boundary")
        def mount_id(descriptor):
            for line in pathlib.Path(f"/proc/self/fdinfo/{descriptor}").read_text().splitlines():
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1])
            raise SystemExit(f"P30 {label} descriptor has no mount ID")
        if mount_id(transport_fd) != mount_id(checkout_fd):
            raise SystemExit(f"P30 {label} crosses a mount-ID boundary")
        acl_names = {
            name.decode() if isinstance(name, bytes) else name
            for name in os.listxattr(checkout_fd)
        }
        if acl_names & {
            "system.posix_acl_access",
            "system.posix_acl_default",
            "system.nfs4_acl",
            "system.richacl",
        }:
            raise SystemExit(f"P30 {label} has an ACL")
        mount_points = {
            line.split()[4].replace("\\040", " ").replace("\\011", "\t")
            for line in pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
            if len(line.split()) >= 5
        }
        if str(checkout) in mount_points:
            raise SystemExit(f"P30 {label} is an unexpected mount point")
        by_name = os.stat(basename, dir_fd=transport_fd, follow_symlinks=False)
        if stat.S_ISLNK(by_name.st_mode) or (
            by_name.st_dev,
            by_name.st_ino,
        ) != (checkout_info.st_dev, checkout_info.st_ino):
            raise SystemExit(f"P30 {label} path identity differs")
        print(
            f"{checkout_info.st_dev}:{checkout_info.st_ino}:"
            f"{checkout_info.st_uid}:{checkout_info.st_gid}:"
            f"{stat.S_IMODE(checkout_info.st_mode):04o}:{mount_id(checkout_fd)}"
        )
    finally:
        os.close(checkout_fd)
finally:
    os.close(transport_fd)
PY
}

authority_layout_binding() {
  transport_checkout_layout_binding \
    authority "$P30_AUTHORITY_REPO" authority-185e444
}

control_layout_binding() {
  transport_checkout_layout_binding \
    control "$P30_CONTROL_REPO" control-source
}

validate_authority_and_inputs() {
  local authority_binding_before
  authority_binding_before="$(authority_layout_binding)" ||
    die "P30 authority layout could not be bound"
  assert_clean_checkout \
    "$P30_AUTHORITY_REPO" "$P30_AUTHORITY_HEAD" "$P30_AUTHORITY_TREE" \
    "P30 authority checkout"
  local symbolic_status
  set +e
  p30_git -C "$P30_AUTHORITY_REPO" symbolic-ref -q HEAD >/dev/null 2>&1
  symbolic_status=$?
  set -e
  (( symbolic_status == 1 )) || {
    (( symbolic_status == 0 )) && die "P30 authority checkout must be detached"
    die "P30 authority symbolic-ref inspection failed"
  }
  local shallow_status
  shallow_status="$(p30_git -C "$P30_AUTHORITY_REPO" rev-parse --is-shallow-repository)" ||
    die "P30 authority shallow-state inspection failed"
  [[ "$shallow_status" == "false" ]] ||
    die "P30 authority checkout must not be shallow"
  local authority_indirection
  for authority_indirection in \
    "$P30_AUTHORITY_REPO/.git/objects/info/alternates" \
    "$P30_AUTHORITY_REPO/.git/info/grafts"; do
    [[ ! -e "$authority_indirection" && ! -L "$authority_indirection" ]] ||
      die "P30 authority checkout has forbidden Git object indirection: $authority_indirection"
  done
  local replace_refs
  replace_refs="$(p30_git -C "$P30_AUTHORITY_REPO" for-each-ref \
      --format='%(refname)' refs/replace/)" ||
    die "P30 authority replace-ref inspection failed"
  [[ -z "$replace_refs" ]] ||
    die "P30 authority checkout has forbidden replace refs"
  local partial_config partial_status
  set +e
  partial_config="$(p30_git -C "$P30_AUTHORITY_REPO" config --local --get-regexp \
      '^(extensions\.partialclone|remote\..*\.(promisor|partialclonefilter))$' \
      2>/dev/null)"
  partial_status=$?
  set -e
  (( partial_status == 0 || partial_status == 1 )) ||
    die "P30 authority partial-clone config inspection failed"
  [[ -z "$partial_config" ]] ||
    die "P30 authority checkout has forbidden promisor or partial-clone configuration"
  [[ "$(sha256_file "$P30_AUTHORITY_REPO/experiments/training/run_p25_executable_origin_diagnostic.py")" == \
      "$P30_P25_SOURCE_SHA256" ]] || die "authority P25 source bytes differ"
  [[ "$(sha256_file "$P30_AUTHORITY_REPO/experiments/training/p25_cuda_diagnostic_contract.json")" == \
      "$P30_P25_CONTRACT_SHA256" ]] || die "authority P25 contract bytes differ"
  [[ "$(sha256_file "$P30_AUTHORITY_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py")" == \
      "$P30_P23_RUNNER_SHA256" ]] || die "authority P23 runner bytes differ"

  assert_clean_checkout \
    "$P30_NANOGPT_HOST" "$P30_NANOGPT_COMMIT" "$P30_NANOGPT_TREE" \
    "pinned nanoGPT checkout"
  assert_clean_checkout \
    "$P30_MUON_HOST" "$P30_MUON_COMMIT" "$P30_MUON_TREE" \
    "pinned Muon checkout"
  [[ "$(sha256_file "$P30_MUON_HOST/muon.py")" == "$P30_MUON_SOURCE_SHA256" ]] ||
    die "pinned Muon source bytes differ"
  [[ -f "$P30_DATA_HOST/materialized/p22_fineweb_manifest.json" ]] ||
    die "FineWeb materialization manifest is absent"
  [[ -f "$P30_AUTHORITY_REPO/experiments/training/materialize_p22_fineweb.py" ]] ||
    die "preprocessor alias source is absent"
  local authority_binding_after
  authority_binding_after="$(authority_layout_binding)" ||
    die "P30 authority layout could not be rebound"
  [[ "$authority_binding_after" == "$authority_binding_before" ]] ||
    die "P30 authority directory identity changed during validation"
}

validate_control_sources() {
  local expected_head="$1"
  local expected_tree="$2"
  local control_binding_before control_binding_after
  control_binding_before="$(control_layout_binding)" ||
    die "P30 control layout could not be bound"
  assert_clean_checkout \
    "$P30_CONTROL_REPO" "$expected_head" "$expected_tree" "P30 control checkout"

  local contract="$P30_CONTROL_REPO/experiments/training/p30_umask_bound_control_seal_contract.json"
  local orchestrator="$P30_CONTROL_REPO/scripts/run_p30_umask_bound_control_seal.sh"
  local reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p30_umask_bound_control_seal.py"
  local bundle_verifier="$P30_CONTROL_REPO/scripts/verify_p30_control_bundle.py"
  local p29_contract="$P30_CONTROL_REPO/experiments/training/p29_permission_safe_bundle_localization_bridge_contract.json"
  local p29_reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py"
  local p29_outcome="$P30_CONTROL_REPO/results/summaries/p29_permission_safe_bundle_localization_bridge_outcome.json"
  local p29_outcome_reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py"
  local p28_contract="$P30_CONTROL_REPO/experiments/training/p28_bundle_complete_localization_bridge_contract.json"
  local p28_reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p28_bundle_complete_localization_bridge.py"
  local p28_outcome="$P30_CONTROL_REPO/results/summaries/p28_bundle_complete_localization_bridge_outcome.json"
  local p28_outcome_reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py"
  local p27_contract="$P30_CONTROL_REPO/experiments/training/p27_cuda_deleted_mapping_localization_contract.json"
  local p27_reconstructor="$P30_CONTROL_REPO/scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
  local p27_localizer="$P30_CONTROL_REPO/experiments/training/run_p27_cuda_deleted_mapping_localization.py"
  local ingester="$P30_CONTROL_REPO/scripts/ingest_p26_trace_off_a_failure.py"
  local p23_core="$P30_CONTROL_REPO/experiments/training/p23_deterministic_cuda_shadow_trace.py"
  local p27_sanitizer="$P30_CONTROL_REPO/scripts/sanitize_p27_cuda_deleted_mapping_localization.py"
  local p23_runner="$P30_CONTROL_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
  local executing_orchestrator
  executing_orchestrator="$(/usr/bin/realpath -- "${BASH_SOURCE[0]}")"
  local path
  for path in "$contract" "$orchestrator" "$reconstructor" "$bundle_verifier" \
    "$p29_contract" "$p29_reconstructor" "$p29_outcome" \
    "$p29_outcome_reconstructor" \
    "$p28_contract" "$p28_reconstructor" "$p28_outcome" \
    "$p28_outcome_reconstructor" \
    "$p27_contract" "$p27_reconstructor" "$p27_localizer" "$ingester" \
    "$p23_core" "$p27_sanitizer" "$p23_runner"; do
    [[ -f "$path" && ! -L "$path" ]] || die "reviewed control source is absent: $path"
  done
  [[ "$(sha256_file "$contract")" == "$P30_EXPECTED_CONTRACT_SHA256" ]] ||
    die "P30 contract bytes differ"
  [[ "$(sha256_file "$orchestrator")" == "$P30_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "P30 host orchestrator bytes differ"
  [[ "$executing_orchestrator" == "$P30_SEALED_ORCHESTRATOR" && \
     ! -L "${BASH_SOURCE[0]}" ]] ||
    die "the executing P30 orchestrator is not the root-sealed source"
  [[ "$(sha256_file "$executing_orchestrator")" == \
      "$P30_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "executing P30 host orchestrator bytes differ"
  [[ "$(sha256_file "$reconstructor")" == "$P30_EXPECTED_RECONSTRUCTOR_SHA256" ]] ||
    die "P30 contract reconstructor bytes differ"
  [[ "$(sha256_file "$bundle_verifier")" == "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P30 bundle verifier bytes differ"
  [[ "$(sha256_file "$p29_contract")" == "$P30_EXPECTED_P29_CONTRACT_SHA256" ]] ||
    die "frozen P29 contract bytes differ"
  [[ "$(sha256_file "$p29_reconstructor")" == "$P30_EXPECTED_P29_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P29 contract reconstructor bytes differ"
  [[ "$(sha256_file "$p29_outcome")" == "$P30_EXPECTED_P29_OUTCOME_SHA256" ]] ||
    die "frozen P29 outcome bytes differ"
  [[ "$(sha256_file "$p29_outcome_reconstructor")" == \
      "$P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P29 outcome reconstructor bytes differ"
  [[ "$(sha256_file "$p28_contract")" == "$P30_EXPECTED_P28_CONTRACT_SHA256" ]] ||
    die "frozen P28 contract bytes differ"
  [[ "$(sha256_file "$p28_reconstructor")" == "$P30_EXPECTED_P28_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P28 contract reconstructor bytes differ"
  [[ "$(sha256_file "$p28_outcome")" == "$P30_EXPECTED_P28_OUTCOME_SHA256" ]] ||
    die "frozen P28 outcome bytes differ"
  [[ "$(sha256_file "$p28_outcome_reconstructor")" == \
      "$P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P28 outcome reconstructor bytes differ"
  [[ "$(sha256_file "$p27_contract")" == "$P30_EXPECTED_P27_CONTRACT_SHA256" ]] ||
    die "frozen P27 contract bytes differ"
  [[ "$(sha256_file "$p27_reconstructor")" == "$P30_EXPECTED_P27_RECONSTRUCTOR_SHA256" ]] ||
    die "frozen P27 contract reconstructor bytes differ"
  [[ "$(sha256_file "$p27_localizer")" == "$P30_EXPECTED_P27_LOCALIZER_SHA256" ]] ||
    die "frozen P27 localizer bytes differ"
  [[ "$(sha256_file "$ingester")" == "$P30_EXPECTED_INGESTER_SHA256" ]] ||
    die "P26 failure ingester bytes differ"
  [[ "$(sha256_file "$p23_core")" == "$P30_EXPECTED_P23_CORE_SHA256" ]] ||
    die "corrected P23 core bytes differ"
  [[ "$(sha256_file "$p27_sanitizer")" == "$P30_EXPECTED_P27_SANITIZER_SHA256" ]] ||
    die "frozen P27 sanitizer bytes differ"
  [[ "$(sha256_file "$p23_runner")" == "$P30_P23_RUNNER_SHA256" ]] ||
    die "frozen P23 runner bytes differ"

  local relative
  for relative in \
    experiments/training/p30_umask_bound_control_seal_contract.json \
    scripts/run_p30_umask_bound_control_seal.sh \
    scripts/reconstruct_p30_umask_bound_control_seal.py \
    scripts/verify_p30_control_bundle.py \
    experiments/training/p29_permission_safe_bundle_localization_bridge_contract.json \
    scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py \
    results/summaries/p29_permission_safe_bundle_localization_bridge_outcome.json \
    scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py \
    experiments/training/p28_bundle_complete_localization_bridge_contract.json \
    scripts/reconstruct_p28_bundle_complete_localization_bridge.py \
    results/summaries/p28_bundle_complete_localization_bridge_outcome.json \
    scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py \
    experiments/training/p27_cuda_deleted_mapping_localization_contract.json \
    scripts/reconstruct_p27_cuda_deleted_mapping_localization.py \
    experiments/training/run_p27_cuda_deleted_mapping_localization.py \
    scripts/ingest_p26_trace_off_a_failure.py \
    experiments/training/p23_deterministic_cuda_shadow_trace.py \
    scripts/sanitize_p27_cuda_deleted_mapping_localization.py \
    experiments/training/run_p23_deterministic_cuda_shadow_trace.py; do
    [[ "$(p30_git -C "$P30_CONTROL_REPO" ls-files --error-unmatch -- "$relative")" == \
        "$relative" ]] || die "reviewed control source is not tracked: $relative"
  done
  control_binding_after="$(control_layout_binding)" ||
    die "P30 control layout could not be rebound"
  [[ "$control_binding_after" == "$control_binding_before" ]] ||
    die "P30 control directory identity changed during validation"
}

validate_source_freeze_history() {
  [[ "$(p30_git -C "$P30_CONTROL_REPO" rev-parse \
      "$P30_SOURCE_FREEZE_COMMIT^{tree}")" == "$P30_SOURCE_FREEZE_TREE" ]] ||
    die "P30 source-freeze commit/tree binding differs"
  [[ "$(p30_git -C "$P30_CONTROL_REPO" rev-list --parents -n 1 \
      "$P30_SOURCE_FREEZE_COMMIT")" == \
      "$P30_SOURCE_FREEZE_COMMIT $P30_TERMINAL_P29_COMMIT" ]] ||
    die "P30 source freeze must be one direct child of the terminal P29 outcome"
  [[ "$(p30_git -C "$P30_CONTROL_REPO" rev-parse \
      "$P30_TERMINAL_P29_COMMIT^{tree}")" == "$P30_TERMINAL_P29_TREE" ]] ||
    die "terminal P29 commit/tree binding differs"
  [[ "$(p30_git -C "$P30_CONTROL_REPO" rev-parse \
      "$P30_TERMINAL_P28_COMMIT^{tree}")" == "$P30_TERMINAL_P28_TREE" ]] ||
    die "terminal P28 commit/tree binding differs"
  if p30_git -C "$P30_CONTROL_REPO" cat-file -e \
    "$P30_SOURCE_FREEZE_COMMIT:experiments/training/p30_cuda_runtime_lock.json" \
    2>/dev/null || p30_git -C "$P30_CONTROL_REPO" cat-file -e \
    "$P30_SOURCE_FREEZE_COMMIT:experiments/training/p30_host_attestation.json" \
    2>/dev/null; then
    die "P30 runtime artifacts unexpectedly predate runtime preparation"
  fi
}

validate_runtime_review_history() {
  [[ "$(p30_git -C "$P30_CONTROL_REPO" rev-list --parents -n 1 \
      "$P30_RUNTIME_REVIEW_COMMIT")" == \
      "$P30_RUNTIME_REVIEW_COMMIT $P30_SOURCE_FREEZE_COMMIT" ]] ||
    die "P30 runtime review must be one direct child of the source freeze"
  local expected_delta
  expected_delta=$'A\texperiments/training/p30_cuda_runtime_lock.json\nA\texperiments/training/p30_host_attestation.json'
  [[ "$(p30_git -C "$P30_CONTROL_REPO" diff --name-status \
      "$P30_SOURCE_FREEZE_COMMIT" "$P30_RUNTIME_REVIEW_COMMIT")" == \
      "$expected_delta" ]] ||
    die "runtime review must add only the P30 lock and host attestation"
  validate_source_freeze_history
}

validate_transport_execution_sources() {
  local orchestrator="$P30_CONTROL_REPO/scripts/run_p30_umask_bound_control_seal.sh"
  local verifier="$P30_CONTROL_REPO/scripts/verify_p30_control_bundle.py"
  local executing_orchestrator
  executing_orchestrator="$(/usr/bin/realpath -- "${BASH_SOURCE[0]}")"
  [[ -f "$orchestrator" && ! -L "$orchestrator" && \
     -f "$verifier" && ! -L "$verifier" && \
     -f "$P30_BOOTSTRAP_VERIFIER" && ! -L "$P30_BOOTSTRAP_VERIFIER" ]] ||
    die "P30 transport execution sources are absent or symlinks"
  [[ "$executing_orchestrator" == "$P30_SEALED_ORCHESTRATOR" && \
     ! -L "${BASH_SOURCE[0]}" ]] ||
    die "the executing P30 orchestrator is not the root-sealed source"
  [[ "$(sha256_file "$executing_orchestrator")" == \
      "$P30_EXPECTED_ORCHESTRATOR_SHA256" ]] ||
    die "executing P30 host orchestrator bytes differ before transport replay"
  [[ "$(sha256_file "$verifier")" == "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P30 bundle verifier bytes differ before transport replay"
  [[ "$(sha256_file "$P30_BOOTSTRAP_VERIFIER")" == \
      "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" ]] ||
    die "P30 bootstrap bundle verifier bytes differ before transport replay"
  [[ "$(/usr/bin/stat -c '%a' "$P30_BOOTSTRAP_VERIFIER")" == "600" ]] ||
    die "P30 bootstrap bundle verifier mode differs"
}

run_bootstrap_verifier() {
  # Do not ask Python to reopen a same-UID transport path as executable source.
  # This trusted inline launcher binds the exact direct-child inode O_NOFOLLOW,
  # authenticates and compiles those held bytes, keeps the descriptor open for
  # the whole verifier run, then rejects any by-name identity transition.
  /usr/bin/python3 -I -S - \
    "$P30_TRANSPORT_ROOT" "$P30_BOOTSTRAP_VERIFIER" \
    "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" "$@" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

transport = pathlib.PurePosixPath(sys.argv[1])
source = pathlib.PurePosixPath(sys.argv[2])
expected_digest = sys.argv[3]
arguments = sys.argv[4:]
if source != transport / "verify_p30_control_bundle.py":
    raise SystemExit("P30 bootstrap verifier is not the canonical direct child")
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW


def open_componentwise(path: pathlib.PurePosixPath) -> int:
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


transport_fd = open_componentwise(transport)
source_fd = None
exit_code = 0
try:
    transport_info = os.fstat(transport_fd)
    if (
        not stat.S_ISDIR(transport_info.st_mode)
        or (
            transport_info.st_uid,
            transport_info.st_gid,
            stat.S_IMODE(transport_info.st_mode),
        )
        != (1000, 1000, 0o700)
    ):
        raise SystemExit("P30 transport authority differs at bootstrap execution")
    source_fd = os.open(source.name, file_flags, dir_fd=transport_fd)
    before = os.fstat(source_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (1000, 1000, 0o600)
    ):
        raise SystemExit("P30 bootstrap verifier authority differs")
    raw = bytearray()
    digest = hashlib.sha256()
    while True:
        chunk = os.read(source_fd, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
        digest.update(chunk)
    if digest.hexdigest() != expected_digest:
        raise SystemExit("P30 bootstrap verifier digest differs")
    after_read = os.fstat(source_fd)
    by_name = os.stat(source.name, dir_fd=transport_fd, follow_symlinks=False)
    if fields(before) != fields(after_read) or fields(after_read) != fields(by_name):
        raise SystemExit("P30 bootstrap verifier changed during authenticated read")
    code = compile(bytes(raw), str(source), "exec", dont_inherit=True)
    namespace = {
        "__builtins__": __builtins__,
        "__cached__": None,
        "__doc__": None,
        "__file__": str(source),
        "__loader__": None,
        "__name__": "__main__",
        "__package__": None,
        "__spec__": None,
    }
    if os.geteuid() != 0 or os.getegid() != 0:
        raise SystemExit("P30 bootstrap launcher did not start with root authority")
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    if os.geteuid() != 1000 or os.getegid() != 1000 or os.getgroups():
        raise SystemExit("P30 bootstrap verifier privilege drop failed")
    original_argv = sys.argv
    sys.argv = [str(source), *arguments]
    try:
        exec(code, namespace, namespace)
    except SystemExit as exc:
        if exc.code is None:
            exit_code = 0
        elif isinstance(exc.code, int):
            exit_code = exc.code
        else:
            print(exc.code, file=sys.stderr)
            exit_code = 1
    finally:
        sys.argv = original_argv
    after_exec = os.fstat(source_fd)
    by_name_after = os.stat(source.name, dir_fd=transport_fd, follow_symlinks=False)
    if fields(after_read) != fields(after_exec) or fields(after_exec) != fields(by_name_after):
        raise SystemExit("P30 bootstrap verifier identity changed during execution")
finally:
    if source_fd is not None:
        os.close(source_fd)
    os.close(transport_fd)
raise SystemExit(exit_code)
PY
}

verify_source_transport_unlogged() {
  validate_transport_execution_sources
  run_bootstrap_verifier \
    --phase source \
    --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
    --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
    --transport-root "$P30_TRANSPORT_ROOT" \
    --bundle "$P30_SOURCE_BUNDLE" \
    --expected-bundle-sha256 "$P30_EXPECTED_SOURCE_BUNDLE_SHA256" \
    --expected-bundle-byte-count "$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
    --closure-repository "$P30_SOURCE_CLOSURE" \
    --control-checkout "$P30_CONTROL_REPO" \
    --attempt-root "$P30_ATTEMPT_ROOT" \
    --execution-root "$P30_EXECUTION_ROOT" \
    --expected-p30-commit "$P30_SOURCE_FREEZE_COMMIT" \
    --expected-p30-tree "$P30_SOURCE_FREEZE_TREE" \
    --verify-existing-receipt "$P30_SOURCE_RECEIPT" \
    --expected-receipt-sha256 "$P30_EXPECTED_SOURCE_RECEIPT_SHA256" \
    >/dev/null || die "reviewed P30 source-bundle receipt did not replay exactly"
}

verify_runtime_review_transport_unlogged() {
  require_runtime_environment
  validate_transport_execution_sources
  run_bootstrap_verifier \
    --phase runtime-review \
    --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
    --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
    --transport-root "$P30_TRANSPORT_ROOT" \
    --bundle "$P30_RUNTIME_REVIEW_BUNDLE" \
    --expected-bundle-sha256 "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
    --expected-bundle-byte-count "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
    --closure-repository "$P30_RUNTIME_REVIEW_CLOSURE" \
    --control-checkout "$P30_CONTROL_REPO" \
    --attempt-root "$P30_ATTEMPT_ROOT" \
    --execution-root "$P30_EXECUTION_ROOT" \
    --expected-p30-commit "$P30_RUNTIME_REVIEW_COMMIT" \
    --expected-p30-tree "$P30_RUNTIME_REVIEW_TREE" \
    --expected-source-commit "$P30_SOURCE_FREEZE_COMMIT" \
    --expected-source-tree "$P30_SOURCE_FREEZE_TREE" \
    --verify-existing-receipt "$P30_RUNTIME_REVIEW_RECEIPT" \
    --expected-receipt-sha256 "$P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256" \
    >/dev/null || die "reviewed P30 runtime-review bundle receipt did not replay exactly"
}

launch_fresh_container() {
  local container_id inspected_id
  container_id="$(p30_docker run "$@")" || return
  container_id="${container_id//$'\n'/}"
  [[ "$container_id" =~ ^[0-9a-f]{64}$ ]] ||
    die "fresh container ID is malformed"
  [[ "$container_id" != "$P30_TERMINAL_P26_CONTAINER_ID" ]] ||
    die "terminal P26 container was reused"
  inspected_id="$(p30_docker inspect --format '{{.Id}}' "$container_id")" ||
    die "fresh container ID could not be inspected"
  [[ "$inspected_id" == "$container_id" ]] ||
    die "fresh container ID differs from launch output"
  printf '%s\n' "$container_id"
}

read_fresh_container_id() {
  /usr/bin/python3 -I -S - \
    "$P30_ATTEMPT_ROOT" "$P30_ATTEMPT_ROOT/p30-container-id.txt" \
    "$P30_TERMINAL_P26_CONTAINER_ID" <<'PY'
import os
import pathlib
import stat
import sys

attempt = pathlib.PurePosixPath(sys.argv[1])
path = pathlib.PurePosixPath(sys.argv[2])
forbidden = sys.argv[3]
if path != attempt / "p30-container-id.txt":
    raise SystemExit("P30 container-ID path differs")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
attempt_fd = os.open("/", flags)
file_fd = None
try:
    for component in attempt.parts[1:]:
        child = os.open(component, flags, dir_fd=attempt_fd)
        os.close(attempt_fd)
        attempt_fd = child
    attempt_info = os.fstat(attempt_fd)
    if (
        not stat.S_ISDIR(attempt_info.st_mode)
        or (attempt_info.st_uid, attempt_info.st_gid, stat.S_IMODE(attempt_info.st_mode))
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 container-ID parent authority differs")
    file_fd = os.open(
        path.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=attempt_fd,
    )
    before = os.fstat(file_fd)
    raw = os.read(file_fd, 4096)
    if os.read(file_fd, 1) != b"":
        raise SystemExit("P30 container-ID file is unexpectedly large")
    after = os.fstat(file_fd)
    by_name = os.stat(path.name, dir_fd=attempt_fd, follow_symlinks=False)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if (
        fields(before) != fields(after)
        or fields(after) != fields(by_name)
        or not stat.S_ISREG(after.st_mode)
        or after.st_nlink != 1
        or (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode))
        != (0, 0, 0o444)
    ):
        raise SystemExit("P30 container-ID file identity or authority differs")
    try:
        value = raw.decode("ascii").removesuffix("\n")
    except UnicodeDecodeError as exc:
        raise SystemExit("P30 container ID is not ASCII") from exc
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == forbidden
        or raw != (value + "\n").encode("ascii")
    ):
        raise SystemExit("P30 container ID is malformed or forbidden")
    print(value)
finally:
    if file_fd is not None:
        os.close(file_fd)
    os.close(attempt_fd)
PY
}

freeze_runtime_from_bound_container() {
  local expected_id
  expected_id="$(read_fresh_container_id)" ||
    die "fresh container-ID evidence could not be authenticated for freeze-runtime"
  p30_docker exec \
    --env "CUDA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    "$expected_id" /opt/p23-venv/bin/python \
    /workspace/OptimizationML/experiments/training/run_p23_deterministic_cuda_shadow_trace.py \
    freeze-runtime \
    --container-image "$P30_IMAGE" \
    --container-repository-digest "$P30_IMAGE_DIGEST" \
    --host-image-inspection /mounted-host-evidence/image-inspect.json \
    --host-running-container-inspection /mounted-host-evidence/running-container-inspect.json \
    --host-running-mountinfo /mounted-host-evidence/running-mountinfo.txt \
    --host-nvidia-smi-query /mounted-host-evidence/nvidia-smi.csv \
    --host-attestation-output /workspace/evidence/p23/p30_host_attestation.json \
    --runtime-lock-output /workspace/evidence/p23/p30_cuda_runtime_lock.json
}

capture_running_container_inspection() {
  local expected_id inspection parsed_id parsed_pid
  expected_id="$(read_fresh_container_id)" ||
    die "fresh container-ID evidence could not be read"
  [[ "$expected_id" =~ ^[0-9a-f]{64}$ ]] ||
    die "fresh container-ID evidence is malformed"
  inspection="$(p30_docker inspect "$expected_id")" ||
    die "fresh container inspection failed"
  IFS=: read -r parsed_id parsed_pid <<<"$(
    /usr/bin/python3 -I -S -c \
      'import json,sys; p=json.loads(sys.argv[1]); assert isinstance(p,list) and len(p)==1; r=p[0]; print("{}:{}".format(r["Id"],r["State"]["Pid"]))' \
      "$inspection"
  )"
  [[ "$parsed_id" == "$expected_id" && "$parsed_pid" =~ ^[1-9][0-9]*$ ]] ||
    die "fresh container inspection identity or init PID differs"
  printf '%s\n' "$inspection"
}

capture_running_mountinfo() {
  local init_pid current_id current_pid expected_id
  IFS=: read -r expected_id init_pid <<<"$(
    /usr/bin/python3 -I -S - \
      "$P30_ATTEMPT_ROOT/p30-running-container-inspect.json" <<'PY'
import json
import os
import stat
import sys

path = sys.argv[1]
descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
try:
    before = os.fstat(descriptor)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (0, 0, 0o444)
    ):
        raise SystemExit("P30 running-inspection file authority differs")
    raw = bytearray()
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
    after = os.fstat(descriptor)
    by_name = os.lstat(path)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P30 running-inspection file changed during read")
    payload = json.loads(bytes(raw))
    if not isinstance(payload, list) or len(payload) != 1:
        raise SystemExit("P30 running-inspection payload differs")
    record = payload[0]
    container_id = record.get("Id")
    state = record.get("State")
    pid = state.get("Pid") if isinstance(state, dict) else None
    if (
        not isinstance(container_id, str)
        or len(container_id) != 64
        or any(character not in "0123456789abcdef" for character in container_id)
        or not isinstance(pid, int)
        or pid <= 0
    ):
        raise SystemExit("P30 running-inspection ID or init PID is malformed")
    print(f"{container_id}:{pid}")
finally:
    os.close(descriptor)
PY
  )"
  current_id="$(p30_docker inspect --format '{{.Id}}' "$expected_id")" ||
    die "fresh container ID reinspection failed"
  current_pid="$(p30_docker inspect --format '{{.State.Pid}}' "$expected_id")" ||
    die "fresh container PID reinspection failed"
  [[ "$current_id" == "$expected_id" && "$current_pid" == "$init_pid" ]] ||
    die "fresh container identity changed before mount capture"
  sudo /usr/bin/nsenter --target "$init_pid" --mount --pid --cgroup -- \
    /usr/bin/cat /proc/self/mountinfo
}

create_fresh_attempt_layout() {
  sudo /usr/bin/python3 -I -S - \
    "$P30_NAMESPACE_ROOT" "$P30_TRANSPORT_ROOT" \
    "$P30_ATTEMPT_ROOT" "$P30_HOST_EVIDENCE" \
    "$P30_EXECUTION_ROOT" "$P30_AUTHORITY_REPO" \
    "$P30_NANOGPT_HOST" "$P30_MUON_HOST" "$P30_DATA_HOST" \
    "$P30_CONTROL_REPO" "$P30_P26_NATIVE" \
    "$P30_AUTHORITY_HEAD" "$P30_AUTHORITY_TREE" \
    "$P30_NANOGPT_COMMIT" "$P30_NANOGPT_TREE" \
    "$P30_MUON_COMMIT" "$P30_MUON_TREE" \
    "$P30_FINEWEB_MANIFEST_SHA256" "$P30_FINEWEB_MANIFEST_BYTE_COUNT" \
    "$P30_EXPECTED_P27_LOCALIZER_SHA256" "$P30_EXPECTED_INGESTER_SHA256" \
    "$P30_EXPECTED_P23_CORE_SHA256" "$P30_EXPECTED_P27_SANITIZER_SHA256" \
    "$P30_P26_NATIVE_SHA256" "$P30_P26_NATIVE_BYTE_COUNT" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys

(
    namespace_text,
    transport_text,
    attempt_text,
    evidence_text,
    execution_text,
    authority_text,
    nanogpt_text,
    muon_text,
    data_text,
    control_text,
    p26_native_text,
    authority_head,
    authority_tree,
    nanogpt_head,
    nanogpt_tree,
    muon_head,
    muon_tree,
    fineweb_manifest_sha256,
    fineweb_manifest_size,
    localizer_sha256,
    ingester_sha256,
    p23_core_sha256,
    sanitizer_sha256,
    p26_native_sha256,
    p26_native_size,
) = sys.argv[1:]
namespace = pathlib.PurePosixPath(namespace_text)
transport = pathlib.PurePosixPath(transport_text)
attempt = pathlib.PurePosixPath(attempt_text)
evidence = pathlib.PurePosixPath(evidence_text)
execution = pathlib.PurePosixPath(execution_text)
authority = pathlib.PurePosixPath(authority_text)
nanogpt = pathlib.PurePosixPath(nanogpt_text)
muon = pathlib.PurePosixPath(muon_text)
data = pathlib.PurePosixPath(data_text)
control = pathlib.PurePosixPath(control_text)
p26_native = pathlib.PurePosixPath(p26_native_text)
if transport != namespace / "transport-20260906-03":
    raise SystemExit("P30 transport layout differs at attempt creation")
if attempt != namespace / "attempt-20260906-03" or evidence != attempt / "evidence":
    raise SystemExit("P30 attempt layout differs at creation")
if execution != namespace / "execution-20260906-03":
    raise SystemExit("P30 execution snapshot layout differs at creation")
if authority != transport / "authority-185e444" or control != transport / "control-source":
    raise SystemExit("P30 reviewed source checkout layout differs at snapshot creation")
for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
    value = getattr(os, name, None)
    if not isinstance(value, int) or value == 0:
        raise SystemExit(f"platform lacks mandatory {name}")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
file_read_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
file_create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
for value in (
    authority_head,
    authority_tree,
    nanogpt_head,
    nanogpt_tree,
    muon_head,
    muon_tree,
):
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit("P30 snapshot Git identity is malformed")
for value in (
    fineweb_manifest_sha256,
    localizer_sha256,
    ingester_sha256,
    p23_core_sha256,
    sanitizer_sha256,
    p26_native_sha256,
):
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit("P30 snapshot input digest is malformed")

ACL_NAMES = {
    "system.posix_acl_access",
    "system.posix_acl_default",
    "system.nfs4_acl",
    "system.richacl",
}
CANONICAL_CONFIG = (
    b"[core]\n"
    b"\trepositoryformatversion = 0\n"
    b"\tfilemode = true\n"
    b"\tbare = false\n"
    b"\tlogallrefupdates = true\n"
)
mount_points = {
    raw.split()[4].replace("\\040", " ").replace("\\011", "\t")
    for raw in pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
    if len(raw.split()) >= 5
}
entries = []


def open_componentwise(path: pathlib.PurePosixPath) -> int:
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def stable_fields(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def require_no_acl(descriptor: int, label: str) -> None:
    names = {
        value.decode() if isinstance(value, bytes) else value
        for value in os.listxattr(descriptor)
    }
    if names & ACL_NAMES:
        raise SystemExit(f"P30 snapshot source has a forbidden ACL: {label}")


def require_not_mount(path: pathlib.PurePosixPath) -> None:
    if str(path) in mount_points:
        raise SystemExit(f"P30 snapshot source crosses a mount point: {path}")


def copy_regular(
    source_parent_fd: int,
    destination_parent_fd: int,
    source_name: str,
    destination_name: str,
    source_path: pathlib.PurePosixPath,
    relative: pathlib.PurePosixPath,
    override: bytes | None = None,
    mode_override: int | None = None,
) -> tuple[str, int]:
    source_fd = os.open(source_name, file_read_flags, dir_fd=source_parent_fd)
    destination_fd = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise SystemExit(f"P30 snapshot input is not one regular file: {source_path}")
        require_no_acl(source_fd, str(source_path))
        digest = hashlib.sha256()
        raw = bytearray()
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            raw.extend(chunk)
        after_read = os.fstat(source_fd)
        by_name = os.stat(source_name, dir_fd=source_parent_fd, follow_symlinks=False)
        if (
            stable_fields(before) != stable_fields(after_read)
            or stable_fields(after_read) != stable_fields(by_name)
        ):
            raise SystemExit(f"P30 snapshot input changed during copy: {source_path}")
        source_digest = digest.hexdigest()
        payload = override if override is not None else bytes(raw)
        mode = (
            mode_override
            if mode_override is not None
            else (0o555 if stat.S_IMODE(before.st_mode) & 0o111 else 0o444)
        )
        if mode not in {0o444, 0o555}:
            raise SystemExit("P30 snapshot stored-file mode is not preregistered")
        destination_fd = os.open(
            destination_name,
            file_create_flags,
            0o600,
            dir_fd=destination_parent_fd,
        )
        output_digest = hashlib.sha256()
        output_size = 0
        for offset in range(0, len(payload), 1024 * 1024):
            chunk = payload[offset : offset + 1024 * 1024]
            output_digest.update(chunk)
            output_size += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise SystemExit("P30 snapshot write made no progress")
                view = view[written:]
        os.fchmod(destination_fd, mode)
        os.fsync(destination_fd)
        created = os.fstat(destination_fd)
        after_copy = os.fstat(source_fd)
        source_by_name_after_copy = os.stat(
            source_name,
            dir_fd=source_parent_fd,
            follow_symlinks=False,
        )
        if (
            stable_fields(after_read) != stable_fields(after_copy)
            or stable_fields(after_copy) != stable_fields(source_by_name_after_copy)
        ):
            raise SystemExit(f"P30 snapshot input changed after authenticated read: {source_path}")
        if override is None and (
            output_digest.hexdigest() != source_digest or output_size != before.st_size
        ):
            raise SystemExit(f"P30 snapshot output differs from authenticated input: {source_path}")
        created_by_name = os.stat(
            destination_name,
            dir_fd=destination_parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or (created.st_uid, created.st_gid, stat.S_IMODE(created.st_mode)) != (0, 0, mode)
            or created.st_size != output_size
            or stable_fields(created) != stable_fields(created_by_name)
        ):
            raise SystemExit(f"P30 snapshot stored-file authority differs: {relative}")
        entries.append(
            {
                "relative_path": str(relative),
                "kind": "file",
                "mode_octal": f"{mode:04o}",
                "uid": 0,
                "gid": 0,
                "byte_count": output_size,
                "sha256": output_digest.hexdigest(),
            }
        )
        return source_digest, before.st_size
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def copy_tree(
    source_fd: int,
    destination_fd: int,
    source_path: pathlib.PurePosixPath,
    relative: pathlib.PurePosixPath,
    git_head: str | None = None,
    finalize_root: bool = True,
) -> None:
    require_not_mount(source_path)
    require_no_acl(source_fd, str(source_path))
    before = os.fstat(source_fd)
    names_before = sorted(os.listdir(source_fd))
    for name in names_before:
        if name in {"", ".", ".."} or "/" in name or "\x00" in name:
            raise SystemExit("P30 snapshot input name is unsafe")
        child_source_path = source_path / name
        child_relative = relative / name
        child_info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        if stat.S_ISLNK(child_info.st_mode):
            raise SystemExit(f"P30 snapshot input contains a symlink: {child_source_path}")
        if stat.S_ISDIR(child_info.st_mode):
            if len(child_relative.parts) >= 2 and child_relative.parts[-2:] in {
                (".git", "hooks"),
                (".git", "logs"),
            }:
                continue
            require_not_mount(child_source_path)
            os.mkdir(name, 0o700, dir_fd=destination_fd)
            child_source_fd = os.open(name, flags, dir_fd=source_fd)
            child_destination_fd = os.open(name, flags, dir_fd=destination_fd)
            try:
                copy_tree(
                    child_source_fd,
                    child_destination_fd,
                    child_source_path,
                    child_relative,
                    git_head=git_head,
                )
            finally:
                os.close(child_destination_fd)
                os.close(child_source_fd)
        elif stat.S_ISREG(child_info.st_mode):
            override = None
            if len(child_relative.parts) >= 2 and child_relative.parts[-2:] == (".git", "config"):
                override = CANONICAL_CONFIG
            elif (
                git_head is not None
                and len(child_relative.parts) >= 2
                and child_relative.parts[-2:] == (".git", "HEAD")
            ):
                override = (git_head + "\n").encode("ascii")
            copy_regular(
                source_fd,
                destination_fd,
                name,
                name,
                child_source_path,
                child_relative,
                override=override,
            )
        else:
            raise SystemExit(f"P30 snapshot input contains a special file: {child_source_path}")
    names_after = sorted(os.listdir(source_fd))
    after = os.fstat(source_fd)
    if names_after != names_before or stable_fields(before) != stable_fields(after):
        raise SystemExit(f"P30 snapshot directory changed during copy: {source_path}")
    if finalize_root:
        os.fchmod(destination_fd, 0o555)
        os.fsync(destination_fd)
        stored = os.fstat(destination_fd)
        if (stored.st_uid, stored.st_gid, stat.S_IMODE(stored.st_mode)) != (0, 0, 0o555):
            raise SystemExit(f"P30 snapshot directory authority differs: {relative}")
        entries.append(
            {
                "relative_path": str(relative),
                "kind": "directory",
                "mode_octal": "0555",
                "uid": 0,
                "gid": 0,
            }
        )


def copy_external_file(
    source: pathlib.PurePosixPath,
    destination_parent_fd: int,
    destination_name: str,
    relative: pathlib.PurePosixPath,
    expected_digest: str,
    expected_size: int | None,
    stored_mode: int,
) -> tuple[str, int]:
    parent_fd = open_componentwise(source.parent)
    try:
        actual_digest, actual_size = copy_regular(
            parent_fd,
            destination_parent_fd,
            source.name,
            destination_name,
            source,
            relative,
            mode_override=stored_mode,
        )
        if actual_digest != expected_digest or (
            expected_size is not None and actual_size != expected_size
        ):
            raise SystemExit(f"P30 snapshot adjunct source binding differs: {source}")
        return actual_digest, actual_size
    finally:
        os.close(parent_fd)


def git_value(repository: pathlib.PurePosixPath, *arguments: str) -> str:
    command = [
        "/usr/bin/git",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.fsmonitor=false",
        "-c", "core.attributesFile=/dev/null",
        "-c", "core.excludesFile=/dev/null",
        "-C", str(repository),
        *arguments,
    ]
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": "/nonexistent",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_SYSTEM": "/dev/null",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        },
        text=True,
    )
    return completed.stdout.rstrip("\n")


secure_parent_fd = open_componentwise(namespace.parent)
namespace_fd = open_componentwise(namespace)
attempt_fd = None
evidence_fd = None
execution_fd = None
try:
    secure_parent_info = os.fstat(secure_parent_fd)
    namespace_info = os.fstat(namespace_fd)
    if (
        not stat.S_ISDIR(secure_parent_info.st_mode)
        or (
            secure_parent_info.st_uid,
            secure_parent_info.st_gid,
            stat.S_IMODE(secure_parent_info.st_mode),
        )
        != (0, 0, 0o755)
    ):
        raise SystemExit("P30 secure-parent authority differs at attempt creation")
    if (
        not stat.S_ISDIR(namespace_info.st_mode)
        or (
            namespace_info.st_uid,
            namespace_info.st_gid,
            stat.S_IMODE(namespace_info.st_mode),
        )
        != (0, 0, 0o755)
    ):
        raise SystemExit("P30 namespace authority differs at attempt creation")
    children = sorted(os.listdir(namespace_fd))
    if children != ["transport-20260906-03"]:
        raise SystemExit(f"P30 pre-attempt namespace inventory differs: {children!r}")
    # Attempt creation is the first irreversible state. A snapshot failure
    # leaves this root behind, mechanically consuming the sole attempt.
    os.mkdir("attempt-20260906-03", 0o700, dir_fd=namespace_fd)
    os.fsync(namespace_fd)
    attempt_fd = os.open("attempt-20260906-03", flags, dir_fd=namespace_fd)
    os.mkdir("evidence", 0o700, dir_fd=attempt_fd)
    os.fsync(attempt_fd)
    evidence_fd = os.open("evidence", flags, dir_fd=attempt_fd)
    os.fchown(evidence_fd, 0, 0)
    os.fchmod(evidence_fd, 0o555)
    os.fsync(evidence_fd)
    for placeholder_name in (
        "p30-image-inspect.json",
        "p30-nvidia-smi.csv",
        "p30-container-id.txt",
        "p30-running-container-inspect.json",
        "p30-running-mountinfo.txt",
    ):
        placeholder_fd = os.open(
            placeholder_name,
            file_create_flags,
            0o600,
            dir_fd=attempt_fd,
        )
        try:
            os.fchown(placeholder_fd, 0, 0)
            os.fchmod(placeholder_fd, 0o444)
            os.fsync(placeholder_fd)
            placeholder_info = os.fstat(placeholder_fd)
            if (
                not stat.S_ISREG(placeholder_info.st_mode)
                or placeholder_info.st_nlink != 1
                or (
                    placeholder_info.st_uid,
                    placeholder_info.st_gid,
                    stat.S_IMODE(placeholder_info.st_mode),
                )
                != (0, 0, 0o444)
                or placeholder_info.st_size != 0
            ):
                raise SystemExit("P30 bound placeholder authority differs")
        finally:
            os.close(placeholder_fd)

    os.mkdir("execution-20260906-03", 0o700, dir_fd=namespace_fd)
    os.fsync(namespace_fd)
    execution_fd = os.open("execution-20260906-03", flags, dir_fd=namespace_fd)
    source_specs = (
        ("repository", authority, authority_head, authority_tree),
        ("nanogpt", nanogpt, nanogpt_head, nanogpt_tree),
        ("muon", muon, muon_head, muon_tree),
        ("data", data, None, None),
    )
    source_authorities = {}
    for name, source, expected_head, expected_tree in source_specs:
        os.mkdir(name, 0o700, dir_fd=execution_fd)
        source_fd = open_componentwise(source)
        destination_fd = os.open(name, flags, dir_fd=execution_fd)
        try:
            copy_tree(
                source_fd,
                destination_fd,
                source,
                pathlib.PurePosixPath(name),
                git_head=expected_head,
                finalize_root=name != "data",
            )
        finally:
            os.close(destination_fd)
            os.close(source_fd)
        if expected_head is not None and expected_tree is not None:
            snapshot_repository = execution / name
            forbidden_git_paths = (
                snapshot_repository / ".git/commondir",
                snapshot_repository / ".git/shallow",
                snapshot_repository / ".git/objects/info/alternates",
                snapshot_repository / ".git/objects/info/http-alternates",
                snapshot_repository / ".git/info/grafts",
            )
            if any(pathlib.Path(path).exists() or pathlib.Path(path).is_symlink() for path in forbidden_git_paths):
                raise SystemExit(f"P30 snapshot {name} contains Git object indirection")
            if git_value(snapshot_repository, "rev-parse", "HEAD") != expected_head:
                raise SystemExit(f"P30 snapshot {name} HEAD differs")
            if git_value(snapshot_repository, "rev-parse", "HEAD^{tree}") != expected_tree:
                raise SystemExit(f"P30 snapshot {name} tree differs")
            if git_value(
                snapshot_repository,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            ):
                raise SystemExit(f"P30 snapshot {name} worktree is not exact")
            if git_value(
                snapshot_repository,
                "for-each-ref",
                "--format=%(refname)",
                "refs/replace/",
            ):
                raise SystemExit(f"P30 snapshot {name} contains replacement refs")
            config_lines = git_value(
                snapshot_repository,
                "config",
                "--local",
                "--no-includes",
                "--list",
            ).splitlines()
            config = {}
            for line in config_lines:
                key, separator, value = line.partition("=")
                if not separator or key.lower() in config:
                    raise SystemExit(f"P30 snapshot {name} Git config is malformed")
                config[key.lower()] = value
            if config != {
                "core.repositoryformatversion": "0",
                "core.filemode": "true",
                "core.bare": "false",
                "core.logallrefupdates": "true",
            }:
                raise SystemExit(f"P30 snapshot {name} Git config differs")
            git_value(snapshot_repository, "fsck", "--full", "--strict", "--no-reflogs")
            source_authorities[name] = {"commit": expected_head, "tree": expected_tree}

    data_fd = os.open("data", flags, dir_fd=execution_fd)
    tools_fd = None
    try:
        if ".p30-tools" in os.listdir(data_fd):
            raise SystemExit("P30 data source collides with the reserved adjunct")
        os.mkdir(".p30-tools", 0o700, dir_fd=data_fd)
        tools_fd = os.open(".p30-tools", flags, dir_fd=data_fd)
        helper_specs = (
            (
                control / "experiments/training/run_p27_cuda_deleted_mapping_localization.py",
                "run_p27_cuda_deleted_mapping_localization.py",
                localizer_sha256,
                None,
                0o555,
            ),
            (
                control / "scripts/ingest_p26_trace_off_a_failure.py",
                "ingest_p26_trace_off_a_failure.py",
                ingester_sha256,
                None,
                0o555,
            ),
            (
                control / "experiments/training/p23_deterministic_cuda_shadow_trace.py",
                "p23_deterministic_cuda_shadow_trace.py",
                p23_core_sha256,
                None,
                0o444,
            ),
            (
                control / "scripts/sanitize_p27_cuda_deleted_mapping_localization.py",
                "sanitize_p27_cuda_deleted_mapping_localization.py",
                sanitizer_sha256,
                None,
                0o555,
            ),
            (
                p26_native,
                "p26-trace-off-a-failure.authenticated.json",
                p26_native_sha256,
                int(p26_native_size),
                0o444,
            ),
        )
        helper_authorities = {}
        for source, name, expected_digest, expected_size, stored_mode in helper_specs:
            actual_digest, actual_size = copy_external_file(
                source,
                tools_fd,
                name,
                pathlib.PurePosixPath("data/.p30-tools") / name,
                expected_digest,
                expected_size,
                stored_mode,
            )
            helper_authorities[name] = {
                "sha256": actual_digest,
                "byte_count": actual_size,
            }
        os.fchmod(tools_fd, 0o555)
        os.fsync(tools_fd)
        entries.append(
            {
                "relative_path": "data/.p30-tools",
                "kind": "directory",
                "mode_octal": "0555",
                "uid": 0,
                "gid": 0,
            }
        )
        os.fchmod(data_fd, 0o555)
        os.fsync(data_fd)
        entries.append(
            {
                "relative_path": "data",
                "kind": "directory",
                "mode_octal": "0555",
                "uid": 0,
                "gid": 0,
            }
        )
    finally:
        if tools_fd is not None:
            os.close(tools_fd)
        os.close(data_fd)

    fineweb_entry = next(
        (
            entry
            for entry in entries
            if entry["relative_path"] == "data/materialized/p22_fineweb_manifest.json"
        ),
        None,
    )
    if (
        fineweb_entry is None
        or fineweb_entry.get("sha256") != fineweb_manifest_sha256
        or fineweb_entry.get("byte_count") != int(fineweb_manifest_size)
    ):
        raise SystemExit("P30 snapshot FineWeb manifest binding differs")
    source_authorities["data"] = {
        "original_manifest_relative_path": "materialized/p22_fineweb_manifest.json",
        "original_manifest_sha256": fineweb_manifest_sha256,
        "original_manifest_byte_count": int(fineweb_manifest_size),
    }
    source_authorities["helpers"] = {
        name: record["sha256"]
        for name, record in helper_authorities.items()
        if name != "p26-trace-off-a-failure.authenticated.json"
    }
    source_authorities["p26_authenticated_input"] = helper_authorities[
        "p26-trace-off-a-failure.authenticated.json"
    ]
    manifest_payload = {
        "schema_version": "passive-muon-p30-immutable-execution-snapshot-v1",
        "source_authorities": source_authorities,
        "entries": sorted(entries, key=lambda entry: entry["relative_path"]),
    }
    manifest_raw = (
        json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    manifest_fd = os.open(
        "snapshot-manifest.json",
        file_create_flags,
        0o600,
        dir_fd=execution_fd,
    )
    try:
        view = memoryview(manifest_raw)
        while view:
            written = os.write(manifest_fd, view)
            if written <= 0:
                raise SystemExit("P30 snapshot manifest write made no progress")
            view = view[written:]
        os.fchmod(manifest_fd, 0o444)
        os.fsync(manifest_fd)
    finally:
        os.close(manifest_fd)
    os.fchmod(execution_fd, 0o555)
    os.fsync(execution_fd)
    os.fchmod(attempt_fd, 0o555)
    os.fsync(attempt_fd)
    os.fsync(namespace_fd)
finally:
    if execution_fd is not None:
        os.close(execution_fd)
    if evidence_fd is not None:
        os.close(evidence_fd)
    if attempt_fd is not None:
        os.close(attempt_fd)
    os.close(namespace_fd)
    os.close(secure_parent_fd)
PY
  validate_fresh_attempt_layout
}

prepare_runtime_admission() {
  # Receipt replay is a read-only precondition. This phase permits exactly one
  # subsequent state-changing prepare attempt; an existing attempt root or
  # container is terminal and never cleaned up, resumed, or retried here.
  require_source_environment
  validate_localization_ledger present absent absent absent
  [[ ! -e "$P30_ATTEMPT_ROOT" && ! -L "$P30_ATTEMPT_ROOT" && \
     ! -e "$P30_EXECUTION_ROOT" && ! -L "$P30_EXECUTION_ROOT" ]] ||
    die "fresh attempt or execution root already exists; cleanup and retries are forbidden"

  # The complete source bundle and its externally reviewed receipt are
  # authenticated first. The frozen P28 outcome and contract, P27 contract,
  # and current P30 contract must all reconstruct before state creation.
  # No attempt-root directory, evidence log, or container may predate these
  # two fail-closed checks.
  verify_source_transport_unlogged
  validate_namespace_layout
  validate_control_sources "$P30_SOURCE_FREEZE_COMMIT" "$P30_SOURCE_FREEZE_TREE"
  validate_source_freeze_history
  validate_localization_ledger present absent absent absent
  [[ ! -e "$P30_ATTEMPT_ROOT" && ! -L "$P30_ATTEMPT_ROOT" && \
     ! -e "$P30_EXECUTION_ROOT" && ! -L "$P30_EXECUTION_ROOT" ]] ||
    die "source replay created a forbidden attempt or execution root"
  if p30_docker container inspect "$P30_CONTAINER" >/dev/null 2>&1; then
    die "fresh container name already has state"
  fi
  validate_authority_and_inputs
  validate_namespace_layout
  [[ "$(p30_docker image inspect --format '{{.Id}}' "$P30_IMAGE")" == \
      "$P30_IMAGE_DIGEST" ]] || die "reviewed P30 image digest is unavailable"
}

prepare_runtime() {
  run_root_journal_logged p30-prepare-pre-attempt-admission \
    prepare_runtime_admission
  run_logged_deferred_evidence p30-prepare-attempt-layout \
    create_fresh_attempt_layout
  run_logged p30-prepare-execution-snapshot-verification \
    validate_prepare_snapshot_boundary

  refresh_bound_file p30-image-inspection "$P30_ATTEMPT_ROOT/p30-image-inspect.json" \
    p30_docker image inspect "$P30_IMAGE"
  refresh_bound_file p30-nvidia-smi "$P30_ATTEMPT_ROOT/p30-nvidia-smi.csv" \
    nvidia-smi --id="$P30_GPU_UUID" \
    --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
    --format=csv,noheader,nounits
  refresh_bound_file p30-container-launch "$P30_ATTEMPT_ROOT/p30-container-id.txt" \
    launch_fresh_container --detach --name "$P30_CONTAINER" \
    --gpus "device=$P30_GPU_UUID" \
    --user 0:0 \
    --userns host \
    --cap-drop ALL \
    --cap-add DAC_OVERRIDE \
    --network none \
    --read-only \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824 \
    --env PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    --env VIRTUAL_ENV=/opt/p23-venv \
    --env PYTHONNOUSERSITE=1 \
    --env PYTHONDONTWRITEBYTECODE=1 \
    --env PYTHONSAFEPATH=1 \
    --env GIT_OPTIONAL_LOCKS=0 \
    --env CUDA_VISIBLE_DEVICES="$P30_GPU_UUID" \
    --env NVIDIA_VISIBLE_DEVICES="$P30_GPU_UUID" \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    --env PYTHONHASHSEED=1337 \
    --env NVIDIA_TF32_OVERRIDE=0 \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env P23_REPO=/workspace/OptimizationML \
    --env P23_NANOGPT=/workspace/inputs/nanoGPT \
    --env P23_MUON_ROOT=/workspace/inputs/muon \
    --mount type=bind,src="$P30_EXECUTION_ROOT/repository",dst=/workspace/OptimizationML,readonly \
    --mount type=bind,src="$P30_EXECUTION_ROOT/nanogpt",dst=/workspace/inputs/nanoGPT,readonly \
    --mount type=bind,src="$P30_EXECUTION_ROOT/muon",dst=/workspace/inputs/muon,readonly \
    --mount type=bind,src="$P30_EXECUTION_DATA",dst=/private/tmp/optimizationml-p22-data,readonly \
    --mount type=bind,src="$P30_EXECUTION_ROOT/repository/experiments/training/materialize_p22_fineweb.py",dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly \
    --mount type=bind,src="$P30_HOST_EVIDENCE",dst=/workspace/evidence/p23 \
    --mount type=bind,src="$P30_ATTEMPT_ROOT/p30-image-inspect.json",dst=/mounted-host-evidence/image-inspect.json,readonly \
    --mount type=bind,src="$P30_ATTEMPT_ROOT/p30-running-container-inspect.json",dst=/mounted-host-evidence/running-container-inspect.json,readonly \
    --mount type=bind,src="$P30_ATTEMPT_ROOT/p30-running-mountinfo.txt",dst=/mounted-host-evidence/running-mountinfo.txt,readonly \
    --mount type=bind,src="$P30_ATTEMPT_ROOT/p30-nvidia-smi.csv",dst=/mounted-host-evidence/nvidia-smi.csv,readonly \
    --entrypoint /bin/sh \
    "$P30_IMAGE" -c 'while :; do sleep 3600; done'

  run_logged p30-fresh-container-id-validation read_fresh_container_id

  refresh_bound_file p30-running-container-inspection \
    "$P30_ATTEMPT_ROOT/p30-running-container-inspect.json" \
    capture_running_container_inspection
  refresh_bound_file p30-running-mountinfo \
    "$P30_ATTEMPT_ROOT/p30-running-mountinfo.txt" \
    capture_running_mountinfo

  local freeze_status=0
  set +e
  run_logged p30-freeze-runtime freeze_runtime_from_bound_container
  freeze_status=$?
  set -e
  (( freeze_status == 0 )) ||
    die "P30 freeze-runtime failed with exit $freeze_status; attempt is retained"
  run_logged p30-post-freeze-attempt-layout \
    validate_post_freeze_attempt_state \
      "$P30_HOST_EVIDENCE/p30-prepare-attempt-layout.stdout.log"

  echo "P30 runtime frozen; localization has not run."
  echo "Container identity and both artifact bindings are retained in the logged prepare evidence."
  echo "Review and commit exactly the two runtime artifacts as one direct child of $P30_SOURCE_FREEZE_COMMIT."
}

validate_live_runtime() {
  local label="$1"
  run_logged "$label-container-inspect" p30_docker inspect "$P30_EXPECTED_CONTAINER_ID"
  run_logged "$label-mountinfo" p30_docker exec "$P30_EXPECTED_CONTAINER_ID" \
    /usr/bin/cat /proc/self/mountinfo
  run_logged "$label-nvidia-smi" nvidia-smi --id="$P30_GPU_UUID" \
    --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
    --format=csv,noheader,nounits

  run_logged "$label-runtime-binding-validation" /usr/bin/python3 -I -S - \
    "$P30_CONTROL_REPO/experiments/training/p30_cuda_runtime_lock.json" \
    "$P30_CONTROL_REPO/experiments/training/p30_host_attestation.json" \
    "$P30_HOST_EVIDENCE/$label-container-inspect.stdout.log" \
    "$P30_HOST_EVIDENCE/$label-mountinfo.stdout.log" \
    "$P30_HOST_EVIDENCE/$label-nvidia-smi.stdout.log" \
    "$P30_ATTEMPT_ROOT/p30-nvidia-smi.csv" \
    "$P30_ATTEMPT_ROOT/p30-container-id.txt" \
    "$P30_IMAGE" "$P30_IMAGE_DIGEST" "$P30_GPU_UUID" \
    "$P30_EXECUTION_ROOT/repository" "$P30_EXECUTION_ROOT/nanogpt" \
    "$P30_EXECUTION_ROOT/muon" "$P30_EXECUTION_DATA" \
    "$P30_HOST_EVIDENCE" "$P30_EXPECTED_CONTAINER_ID" \
    "$P30_EXPECTED_RUNTIME_LOCK_SHA256" "$P30_EXPECTED_HOST_ATTESTATION_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
import sys

def stable_bytes(path_text, uid, gid, mode, expected_sha256=None):
    path = pathlib.PurePosixPath(path_text)
    directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    parent_fd = os.open("/", directory_flags)
    descriptor = None
    try:
        for component in path.parent.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = child
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
            != (uid, gid, mode)
        ):
            raise SystemExit(f"live P30 input authority differs: {path}")
        acl_names = {
            item.decode() if isinstance(item, bytes) else item
            for item in os.listxattr(descriptor)
        }
        if acl_names & {
            "system.posix_acl_access",
            "system.posix_acl_default",
            "system.nfs4_acl",
            "system.richacl",
        }:
            raise SystemExit(f"live P30 input has a forbidden ACL: {path}")
        raw = bytearray()
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            raw.extend(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        fields = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_uid,
            value.st_gid,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fields(before) != fields(after) or fields(after) != fields(by_name):
            raise SystemExit(f"live P30 input changed during read: {path}")
        if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
            raise SystemExit(f"live P30 input digest differs: {path}")
        return bytes(raw)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def strict(raw, label):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result
    def nonfinite(token):
        raise ValueError(f"nonfinite JSON constant in {label}: {token}")
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=nonfinite,
    )

(lock_path, attestation_path, inspect_path, mountinfo_path, gpu_path,
 initial_gpu_path, container_id_path, image, image_digest, gpu_uuid,
 authority, nanogpt, muon, data, evidence, expected_container_id,
 expected_lock_sha256, expected_attestation_sha256) = sys.argv[1:]
lock_raw = stable_bytes(lock_path, 1000, 1000, 0o644, expected_lock_sha256)
attestation_raw = stable_bytes(
    attestation_path, 1000, 1000, 0o644, expected_attestation_sha256
)
inspect_raw = stable_bytes(inspect_path, 0, 0, 0o444)
mountinfo_raw = stable_bytes(mountinfo_path, 0, 0, 0o444)
gpu_raw = stable_bytes(gpu_path, 0, 0, 0o444)
initial_gpu_raw = stable_bytes(initial_gpu_path, 0, 0, 0o444)
container_id_raw = stable_bytes(container_id_path, 0, 0, 0o444)
lock = strict(lock_raw, lock_path)
attestation = strict(attestation_raw, attestation_path)
inspection = strict(inspect_raw, inspect_path)
if not isinstance(inspection, list) or len(inspection) != 1:
    raise SystemExit("live Docker inspection is not one object")
inspection = inspection[0]
container = lock.get("container")
if not isinstance(container, dict):
    raise SystemExit("runtime lock container record is malformed")
try:
    container_id = container_id_raw.decode("ascii").removesuffix("\n")
except UnicodeDecodeError as exc:
    raise SystemExit("live P30 container-ID file is not ASCII") from exc
if container_id_raw != (container_id + "\n").encode("ascii"):
    raise SystemExit("live P30 container-ID bytes differ")
if container_id == (
    "e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c"
):
    raise SystemExit("terminal P26 container ID was reused")
checks = {
    "runtime_schema": lock.get("schema_version") == "passive-muon-p23-cuda-runtime-lock-v3",
    "runtime_status": lock.get("status") == "pinned_for_acquisition",
    "attestation_schema": attestation.get("schema_version") == "passive-muon-p23-host-attestation-v3",
    "attestation_status": attestation.get("status") == "procedurally_host_attested",
    "container_id": inspection.get("Id") == container_id
        == expected_container_id == container.get("container_id"),
    "running": inspection.get("State", {}).get("Running") is True,
    "pid": inspection.get("State", {}).get("Pid") == container.get("container_init_pid"),
    "restart": inspection.get("RestartCount") == 0,
    "image_reference": inspection.get("Config", {}).get("Image") == image == container.get("image"),
    "image_id": inspection.get("Image") == image_digest == container.get("image_id"),
    "repository_digest": container.get("repository_digest") == image_digest,
    "network": inspection.get("HostConfig", {}).get("NetworkMode") == "none",
    "container_user": inspection.get("Config", {}).get("User") == "0:0",
    "userns": inspection.get("HostConfig", {}).get("UsernsMode") == "host",
    "cap_drop": inspection.get("HostConfig", {}).get("CapDrop") == ["ALL"],
    "cap_add": inspection.get("HostConfig", {}).get("CapAdd") == ["DAC_OVERRIDE"],
    "rootfs": inspection.get("HostConfig", {}).get("ReadonlyRootfs") is True,
    "tmpfs": inspection.get("HostConfig", {}).get("Tmpfs", {}).get("/tmp")
        == "rw,noexec,nosuid,nodev,size=1073741824",
    "mountinfo": hashlib.sha256(mountinfo_raw).hexdigest()
        == container.get("mountinfo_sha256"),
    "gpu_query": gpu_raw == initial_gpu_raw,
    "gpu_uuid": lock.get("gpu", {}).get("uuid") == gpu_uuid,
    "gpu_name": lock.get("gpu", {}).get("name") == "NVIDIA A100-SXM4-40GB",
    "attestation_container": attestation.get("container", {}).get("container_id") == container_id,
    "attestation_pid": attestation.get("container", {}).get("container_init_pid")
        == container.get("container_init_pid"),
    "attestation_gpu": attestation.get("gpu") == lock.get("gpu"),
    "attestation_hash": hashlib.sha256(attestation_raw).hexdigest()
        == container.get("host_attestation_sha256"),
}
required_env = {
    "PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "VIRTUAL_ENV=/opt/p23-venv",
    "PYTHONNOUSERSITE=1",
    "PYTHONDONTWRITEBYTECODE=1",
    "PYTHONSAFEPATH=1",
    "GIT_OPTIONAL_LOCKS=0",
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
        str(pathlib.Path(evidence).parent / "p30-image-inspect.json"), False
    ),
    "/mounted-host-evidence/running-container-inspect.json": (
        str(pathlib.Path(evidence).parent / "p30-running-container-inspect.json"), False
    ),
    "/mounted-host-evidence/running-mountinfo.txt": (
        str(pathlib.Path(evidence).parent / "p30-running-mountinfo.txt"), False
    ),
    "/mounted-host-evidence/nvidia-smi.csv": (
        str(pathlib.Path(evidence).parent / "p30-nvidia-smi.csv"), False
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
    raise SystemExit(f"live P30 runtime validation failed: {failed}")
PY
}

validate_reviewed_runtime_artifacts() {
  local committed_lock="$P30_CONTROL_REPO/experiments/training/p30_cuda_runtime_lock.json"
  local committed_attestation="$P30_CONTROL_REPO/experiments/training/p30_host_attestation.json"
  local native_lock="$P30_HOST_EVIDENCE/p30_cuda_runtime_lock.json"
  local native_attestation="$P30_HOST_EVIDENCE/p30_host_attestation.json"
  [[ -f "$committed_lock" && ! -L "$committed_lock" && \
     -f "$committed_attestation" && ! -L "$committed_attestation" ]] ||
    die "reviewed P30 runtime artifacts are incomplete"
  [[ "$(sha256_file "$committed_lock")" == "$P30_EXPECTED_RUNTIME_LOCK_SHA256" ]] ||
    die "committed P30 runtime lock differs"
  [[ "$(sha256_file "$committed_attestation")" == "$P30_EXPECTED_HOST_ATTESTATION_SHA256" ]] ||
    die "committed P30 host attestation differs"
  validate_root_evidence_file \
    "$native_lock" "$P30_EXPECTED_RUNTIME_LOCK_SHA256" - 0444 0 0
  validate_root_evidence_file \
    "$native_attestation" "$P30_EXPECTED_HOST_ATTESTATION_SHA256" - 0444 0 0
}

require_evidence_outputs_absent() {
  /usr/bin/python3 -I -S - "$P30_HOST_EVIDENCE" "$@" <<'PY'
import os
import pathlib
import stat
import sys

evidence = pathlib.PurePosixPath(sys.argv[1])
names = sys.argv[2:]
if not names or any(not name or "/" in name or name in {".", ".."} for name in names):
    raise SystemExit("P30 absent-output name inventory is malformed")
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
descriptor = os.open("/", flags)
try:
    for component in evidence.parts[1:]:
        child = os.open(component, flags, dir_fd=descriptor)
        os.close(descriptor)
        descriptor = child
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 absent-output evidence authority differs")
    for name in names:
        try:
            os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise SystemExit(f"P30 one-shot output already exists: {name}")
finally:
    os.close(descriptor)
PY
}

validate_p26_ingestion_artifacts() {
  sudo /usr/bin/python3 -I -S - \
    "$P30_HOST_EVIDENCE" \
    "$P30_P26_NATIVE_SHA256" "$P30_P26_NATIVE_BYTE_COUNT" <<'PY'
import hashlib
import json
import os
import pathlib
import stat
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


evidence = pathlib.PurePosixPath(sys.argv[1])
expected_digest = sys.argv[2]
expected_size = int(sys.argv[3])
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
read_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
evidence_fd = os.open("/", directory_flags)
try:
    for component in evidence.parts[1:]:
        child = os.open(component, directory_flags, dir_fd=evidence_fd)
        os.close(evidence_fd)
        evidence_fd = child
    evidence_info = os.fstat(evidence_fd)
    if (
        not stat.S_ISDIR(evidence_info.st_mode)
        or (
            evidence_info.st_uid,
            evidence_info.st_gid,
            stat.S_IMODE(evidence_info.st_mode),
        )
        != (0, 0, 0o555)
    ):
        raise SystemExit("P30 P26-ingestion evidence authority differs")

    def read_bound(name, digest_expected=None, size_expected=None):
        descriptor = os.open(name, read_flags, dir_fd=evidence_fd)
        try:
            before = os.fstat(descriptor)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
                != (0, 0, 0o444)
            ):
                raise SystemExit(f"P30 retained P26 file authority differs: {name}")
            if {
                value.decode() if isinstance(value, bytes) else value
                for value in os.listxattr(descriptor)
            } & {
                "system.posix_acl_access",
                "system.posix_acl_default",
                "system.nfs4_acl",
                "system.richacl",
            }:
                raise SystemExit(f"P30 retained P26 file has an ACL: {name}")
            raw = bytearray()
            digest = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                raw.extend(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            by_name = os.stat(name, dir_fd=evidence_fd, follow_symlinks=False)
            fields = lambda value: (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_uid,
                value.st_gid,
                value.st_nlink,
                value.st_size,
                value.st_mtime_ns,
                value.st_ctime_ns,
            )
            if fields(before) != fields(after) or fields(after) != fields(by_name):
                raise SystemExit(f"P30 retained P26 file changed during read: {name}")
            actual_digest = digest.hexdigest()
            if digest_expected is not None and actual_digest != digest_expected:
                raise SystemExit(f"P30 retained P26 file digest differs: {name}")
            if size_expected is not None and after.st_size != size_expected:
                raise SystemExit(f"P30 retained P26 file size differs: {name}")
            return bytes(raw), actual_digest, after
        finally:
            os.close(descriptor)

    _native_raw, native_digest, native_info = read_bound(
        "p26-trace-off-a-failure.authenticated.json",
        expected_digest,
        expected_size,
    )
    sanitized_raw, sanitized_digest, sanitized_info = read_bound(
        "p26-trace-off-a-failure.sanitized.json"
    )
finally:
    os.close(evidence_fd)
payload = json.loads(
    sanitized_raw,
    object_pairs_hook=reject_duplicates,
    parse_constant=reject_nonfinite,
)
if (
    not isinstance(payload, dict)
    or payload.get("schema_version")
    != "passive-muon-p23-sanitized-complete-manifest-v1"
    or payload.get("native_artifact_sha256") != expected_digest
    or payload.get("native_artifact_byte_count") != int(expected_size)
    or not isinstance(payload.get("p27_ingestion"), dict)
    or payload["p27_ingestion"].get("schema_version")
    != "passive-muon-p27-p26-failure-ingestion-v1"
):
    raise SystemExit("P30 sanitized P26 ingestion wrapper differs")
print(
    f"native_copy_binding={native_digest}:{native_info.st_size}:"
    f"{native_info.st_dev}:{native_info.st_ino}:0:0:0444"
)
print(
    f"sanitized_copy_binding={sanitized_digest}:{sanitized_info.st_size}:"
    f"{sanitized_info.st_dev}:{sanitized_info.st_ino}:0:0:0444"
)
print("p26_sanitized_schema_and_native_binding=true")
PY
}

run_p26_ingestion_once() {
  local native_basename="$1"
  local sanitized_basename="$2"
  shift 2
  require_evidence_outputs_absent "$native_basename" "$sanitized_basename"
  "$@"
}

run_localizer_once() {
  local native_basename="$1"
  local sanitized_basename="$2"
  shift 2
  require_evidence_outputs_absent "$native_basename" "$sanitized_basename"
  p30_docker exec "$@"
}

sanitize_localization_artifact() {
  local native="$1"
  local sanitized="$2"
  local native_container="$3"
  local sanitized_container="$4"
  shift 4
  local binding native_sha256 native_byte_count remainder
  binding="$(validate_root_evidence_file "$native" - - 0444 0 0)" ||
    die "P30 native localization artifact could not be bound for sanitization"
  IFS=: read -r native_sha256 native_byte_count remainder <<<"$binding"
  [[ "$native_sha256" =~ ^[0-9a-f]{64}$ && \
     "$native_byte_count" =~ ^[1-9][0-9]*$ && -n "$remainder" ]] ||
    die "P30 native localization binding is malformed"
  printf 'native_binding=%s\n' "$binding"
  p30_docker exec \
    --env "CUDA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    "$P30_EXPECTED_CONTAINER_ID" /opt/p23-venv/bin/python \
    /private/tmp/optimizationml-p22-data/.p30-tools/sanitize_p27_cuda_deleted_mapping_localization.py \
    --native "$native_container" \
    --output "$sanitized_container" \
    --expected-native-sha256 "$native_sha256" \
    --expected-native-byte-count "$native_byte_count" \
    "$@"
  echo -n "sanitized_binding="
  seal_retained_file "$sanitized" || die "P30 sanitized artifact could not be sealed"
  validate_root_evidence_file "$sanitized" - - 0444 0 0
}

validate_localization_entry_state() {
  require_runtime_environment
  validate_localization_ledger present present absent absent
  validate_deferred_journal after-runtime
  [[ -d "$P30_HOST_EVIDENCE" && ! -L "$P30_HOST_EVIDENCE" ]] ||
    die "fresh P30 evidence directory is absent or a symlink"
  validate_fresh_attempt_layout
}

validate_prepare_snapshot_boundary() {
  validate_deferred_journal after-prepare
  run_bootstrap_verifier \
    --verify-execution-snapshot-only \
    --transport-root "$P30_TRANSPORT_ROOT" \
    --control-checkout "$P30_CONTROL_REPO" \
    --execution-root "$P30_EXECUTION_ROOT" \
    --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
    --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
    --expected-source-commit "$P30_SOURCE_FREEZE_COMMIT" \
    --expected-source-tree "$P30_SOURCE_FREEZE_TREE"
}

validate_after_runtime_receipt_boundary() {
  validate_deferred_journal after-runtime
  validate_runtime_namespace_against_receipt
}

run_localization() {
  # The root-owned invocation token was burned before this function was
  # entered. After this function returns, the dispatcher makes one O_EXCL
  # attempt to retain its exact exit status, including failures before evidence
  # logging becomes available. A failed status transaction is itself terminal
  # and leaves the invocation explicitly incomplete and non-rerunnable.
  # Receipt replay must be the first operation: it requires the byte-exact
  # frozen 27-file prepare evidence inventory. Its own transcripts remain in
  # the root journal until that verifier exits, and are published only after
  # the exact-inventory check has completed.
  run_logged_deferred_evidence p30-pre-marker-runtime-receipt-replay \
    verify_runtime_review_transport_unlogged
  run_logged p30-localization-entry-preflight validate_localization_entry_state
  run_logged p30-pre-marker-receipt-namespace-binding \
    validate_after_runtime_receipt_boundary
  run_logged p30-pre-marker-control-sources \
    validate_control_sources "$P30_RUNTIME_REVIEW_COMMIT" "$P30_RUNTIME_REVIEW_TREE"
  run_logged p30-pre-marker-runtime-history validate_runtime_review_history
  run_logged p30-pre-marker-authority-inputs validate_authority_and_inputs
  run_logged p30-pre-marker-runtime-artifacts validate_reviewed_runtime_artifacts
  run_logged p30-pre-marker-namespace-revalidation validate_namespace_layout
  run_logged p30-pre-marker-snapshot-layout validate_fresh_attempt_layout
  validate_live_runtime p30-pre-ingestion

  local p26_native_copy="$P30_HOST_EVIDENCE/p26-trace-off-a-failure.authenticated.json"
  local p26_sanitized="$P30_HOST_EVIDENCE/p26-trace-off-a-failure.sanitized.json"
  local ingestion_status=0
  set +e
  run_logged p30-p26-failure-ingestion run_p26_ingestion_once \
    p26-trace-off-a-failure.authenticated.json \
    p26-trace-off-a-failure.sanitized.json \
    p30_docker exec \
    --env "CUDA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    "$P30_EXPECTED_CONTAINER_ID" /opt/p23-venv/bin/python \
    /private/tmp/optimizationml-p22-data/.p30-tools/ingest_p26_trace_off_a_failure.py \
    --source /private/tmp/optimizationml-p22-data/.p30-tools/p26-trace-off-a-failure.authenticated.json \
    --native-copy /workspace/evidence/p23/p26-trace-off-a-failure.authenticated.json \
    --sanitized-output /workspace/evidence/p23/p26-trace-off-a-failure.sanitized.json \
    --output-root /workspace/evidence/p23 \
    --p23-core /private/tmp/optimizationml-p22-data/.p30-tools/p23_deterministic_cuda_shadow_trace.py \
    --p23-runner /workspace/OptimizationML/experiments/training/run_p23_deterministic_cuda_shadow_trace.py \
    --repository-root /workspace/OptimizationML \
    --path-root repository=/workspace/OptimizationML \
    --path-root preprocessor_alias=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
    --path-root nanogpt=/workspace/inputs/nanoGPT \
    --path-root muon=/workspace/inputs/muon \
    --path-root data=/private/tmp/optimizationml-p22-data \
    --path-root python_environment=/opt/p23-venv \
    --path-root native=/workspace/evidence/p23
  ingestion_status=$?
  set -e
  run_logged p30-post-ingestion-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt
  (( ingestion_status == 0 )) ||
    die "terminal P26 failure ingestion failed with exit $ingestion_status before localization"
  run_logged p30-post-ingestion-artifacts validate_p26_ingestion_artifacts

  run_logged p30-pre-localization-control-sources \
    validate_control_sources "$P30_RUNTIME_REVIEW_COMMIT" "$P30_RUNTIME_REVIEW_TREE"
  run_logged p30-pre-localization-authority-inputs validate_authority_and_inputs
  run_logged p30-pre-localization-runtime-artifacts validate_reviewed_runtime_artifacts
  run_logged p30-pre-localization-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt
  validate_live_runtime p30-pre-localization

  # Only a completely reviewed prelocalization state receives the second,
  # root-owned authorization capability. The host validates all three ledger
  # tokens before dispatch; the containerized P27 localizer receives only the
  # full frozen container ID, not the root-only ledger.
  authorize_localization
  run_logged p30-post-authorization-ledger-validation \
    validate_localization_ledger present present present absent
  run_logged p30-post-authorization-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt

  local native="$P30_HOST_EVIDENCE/p30_cuda_deleted_mapping_localization.native.json"
  local sanitized="$P30_HOST_EVIDENCE/p30_cuda_deleted_mapping_localization.sanitized.json"
  local localization_status=0
  set +e
  run_logged p30-localization run_localizer_once \
    p30_cuda_deleted_mapping_localization.native.json \
    p30_cuda_deleted_mapping_localization.sanitized.json \
    --env "CUDA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    --env "NVIDIA_VISIBLE_DEVICES=$P30_GPU_UUID" \
    "$P30_EXPECTED_CONTAINER_ID" /opt/p23-venv/bin/python \
    /private/tmp/optimizationml-p22-data/.p30-tools/run_p27_cuda_deleted_mapping_localization.py \
    --repository /workspace/OptimizationML \
    --nanogpt-root /workspace/inputs/nanoGPT \
    --muon-source /workspace/inputs/muon/muon.py \
    --p25-source /workspace/OptimizationML/experiments/training/run_p25_executable_origin_diagnostic.py \
    --runtime-lock /workspace/evidence/p23/p30_cuda_runtime_lock.json \
    --host-attestation /workspace/evidence/p23/p30_host_attestation.json \
    --p25-contract /workspace/OptimizationML/experiments/training/p25_cuda_diagnostic_contract.json \
    --expected-source-sha256 "$P30_EXPECTED_P27_LOCALIZER_SHA256" \
    --expected-p25-source-sha256 "$P30_P25_SOURCE_SHA256" \
    --expected-p25-contract-sha256 "$P30_P25_CONTRACT_SHA256" \
    --expected-runtime-lock-sha256 "$P30_EXPECTED_RUNTIME_LOCK_SHA256" \
    --expected-host-attestation-sha256 "$P30_EXPECTED_HOST_ATTESTATION_SHA256" \
    --expected-repository-head "$P30_AUTHORITY_HEAD" \
    --expected-repository-tree "$P30_AUTHORITY_TREE" \
    --output /workspace/evidence/p23/p30_cuda_deleted_mapping_localization.native.json
  localization_status=$?
  set -e
  run_logged p30-post-localizer-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt
  run_logged p30-post-localizer-ledger-validation \
    validate_localization_ledger present present present absent

  local native_binding_status=0
  set +e
  run_logged p30-native-localization-binding \
    validate_root_evidence_file "$native" - - 0444 0 0
  native_binding_status=$?
  set -e
  (( native_binding_status == 0 )) ||
    die "P30 localization retained no valid native artifact (localizer exit $localization_status, binding exit $native_binding_status); no rerun is allowed"

  local sanitizer_status=0
  set +e
  run_logged p30-localization-sanitizer sanitize_localization_artifact \
    "$native" "$sanitized" \
    /workspace/evidence/p23/p30_cuda_deleted_mapping_localization.native.json \
    /workspace/evidence/p23/p30_cuda_deleted_mapping_localization.sanitized.json \
    --output-root /workspace/evidence/p23 \
    --runner-source /private/tmp/optimizationml-p22-data/.p30-tools/run_p27_cuda_deleted_mapping_localization.py \
    --repository-root /workspace/OptimizationML \
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
    --path-root secure=/secure
  sanitizer_status=$?
  set -e
  run_logged p30-post-sanitizer-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt
  run_logged p30-post-sanitizer-ledger-validation \
    validate_localization_ledger present present present absent
  (( sanitizer_status == 0 )) ||
    die "P30 localization sanitizer failed with exit $sanitizer_status; native evidence is retained"
  (( localization_status == 0 )) ||
    die "P30 localization completed with exit $localization_status; sanitized failure evidence is retained"
  run_logged p30-post-localization-receipt-namespace-binding \
    validate_runtime_namespace_against_receipt

  echo "P30 sole localization invocation and sanitizer completed. Do not rerun."
  echo "Native and sanitized artifact bindings are retained in the logged evidence."
}

usage() {
  cat >&2 <<'EOF'
usage: scripts/run_p30_umask_bound_control_seal.sh prepare-runtime|run-localization

There is deliberately no cleanup, retry, acquisition, forward, backward,
optimizer-step, candidate-evaluation, or training phase.
EOF
  exit 2
}

[[ $# == 1 ]] || usage
case "$1" in
  prepare-runtime)
    burn_prepare_invocation ||
      die "prepare-runtime was already invoked or its admission token could not be retained"
    prepare_runtime
    ;;
  run-localization)
    # This root-owned O_EXCL token is the first state transition after phase
    # dispatch and permanently forbids a favorable-result retry. The entire
    # run executes in a subshell so the parent can attempt one root-owned,
    # O_EXCL terminal-status transaction even when an early logger fails.
    # Token creation is one self-validating O_EXCL transaction. Only the
    # process for which that transaction returns success owns the right to
    # close the invocation with a terminal status; a replay that merely sees
    # an existing token must not write the first process's status.
    burn_localization_invocation ||
      die "run-localization was already invoked or its invocation token could not be retained"
    localization_exit_status=0
    set +e
    (
      set -e
      run_localization
    )
    localization_exit_status=$?
    set -e
    write_localization_exit_status "$localization_exit_status" ||
      die "P30 terminal localization status could not be retained"
    exit "$localization_exit_status"
    ;;
  *) usage ;;
esac
