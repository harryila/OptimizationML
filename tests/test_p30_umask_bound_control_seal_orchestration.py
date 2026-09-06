from __future__ import annotations

import hashlib
import os
import re
import shlex
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p30_umask_bound_control_seal.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function(source: str, name: str, following: str) -> str:
    start = source.index(f"{name}() {{")
    body_start = source.index("\n", start) + 1
    next_function = re.search(r"^[a-zA-Z_][a-zA-Z0-9_]*\(\) \{$", source[body_start:], re.M)
    end = len(source) if next_function is None else body_start + next_function.start()
    return source[start:end]


def _dispatch_branch(source: str, phase: str) -> str:
    dispatch = source[source.rindex('case "$1" in') :]
    start = dispatch.index(f"  {phase})") + len(f"  {phase})")
    end = dispatch.index("\n    ;;", start)
    return dispatch[start:end]


def _heredoc_python(function_source: str) -> str:
    start = function_source.index("<<'PY'\n") + len("<<'PY'\n")
    end = function_source.index("\nPY\n", start)
    return function_source[start:end]


def _run_logger_status_fault(
    tmp_path: Path, logger_name: str, status_fault: str
) -> tuple[subprocess.CompletedProcess[str], list[str], Path]:
    source = _source()
    following = {
        "run_logged": "run_logged_deferred_evidence",
        "run_root_journal_logged": "run_logged_deferred_evidence",
        "run_logged_deferred_evidence": "capture_new",
        "capture_new": "refresh_bound_file",
        "refresh_bound_file": "write_new_status",
    }[logger_name]
    definition = _function(source, logger_name, following)
    if logger_name in {"refresh_bound_file", "run_root_journal_logged"}:
        stat_command = "/usr/bin/stat -c '%d:%i:%a:%u:%g:%h'"
        expected_stat_count = 4 if logger_name == "run_root_journal_logged" else 2
        assert definition.count(stat_command) == expected_stat_count
        definition = definition.replace(stat_command, "logger_stat")

    evidence = tmp_path / "evidence"
    deferred = tmp_path / "deferred"
    evidence.mkdir()
    deferred.mkdir()
    trace = tmp_path / "trace.log"
    producer_done = tmp_path / "producer.done"
    label = {
        "run_logged_deferred_evidence": "p30-prepare-attempt-layout",
        "run_root_journal_logged": "p30-prepare-pre-attempt-admission",
    }.get(logger_name, "fault-injection")
    retained_mode_command = "stat -f '%Lp'" if sys.platform == "darwin" else "stat -c '%a'"
    if logger_name == "run_logged":
        output = evidence / f"{label}.stdout.log"
        stderr = evidence / f"{label}.stderr.log"
        status = evidence / f"{label}.exit-status.txt"
        invocation = f"run_logged {label} producer"
    elif logger_name in {
        "run_logged_deferred_evidence",
        "run_root_journal_logged",
    }:
        prefix = deferred / f".deferred-{label}"
        output = Path(f"{prefix}.stdout.log")
        stderr = Path(f"{prefix}.stderr.log")
        status = Path(f"{prefix}.exit-status.txt")
        invocation = f"{logger_name} {label} producer"
    else:
        output = evidence / "bound-output.json"
        stderr = evidence / f"{label}.stderr.log"
        status = evidence / f"{label}.exit-status.txt"
        if logger_name == "refresh_bound_file":
            output.touch()
            invocation = f"refresh_bound_file {label} {shlex.quote(str(output))} producer"
        else:
            invocation = f"capture_new {label} {shlex.quote(str(output))} producer"

    harness = f"""
set -uo pipefail
P30_HOST_EVIDENCE={shlex.quote(str(evidence))}
P30_DEFERRED_ROOT={shlex.quote(str(deferred))}
TRACE={shlex.quote(str(trace))}
PRODUCER_DONE={shlex.quote(str(producer_done))}
EXPECTED_STDOUT={shlex.quote(str(output))}
EXPECTED_STDERR={shlex.quote(str(stderr))}
EXPECTED_STATUS={shlex.quote(str(status))}
STATUS_FAULT={shlex.quote(status_fault)}
LOGGER_STAT_MODE={"600" if logger_name == "run_root_journal_logged" else "444"}
LOGGER_NAME={shlex.quote(logger_name)}

{definition}

die() {{
  printf 'die:%s\n' "$*" >>"$TRACE"
  return 99
}}

logger_stat() {{
  printf '1:2:%s:0:0:1\n' "$LOGGER_STAT_MODE"
}}

create_admission_log_placeholders() {{
  : >"$1"
  : >"$2"
  chmod 600 "$1" "$2"
  printf 'admission-placeholders-created\n' >>"$TRACE"
  return 0
}}

producer() {{
  printf 'producer stdout\n'
  printf 'producer stderr\n' >&2
  if [[ "$STATUS_FAULT" == o_excl ]]; then
    printf 'collision\n' >"$EXPECTED_STATUS"
  fi
  printf 'done\n' >"$PRODUCER_DONE"
  return 37
}}

write_new_status() {{
  if [[ -f "$PRODUCER_DONE" && -s "$EXPECTED_STDOUT" && -s "$EXPECTED_STDERR" ]]; then
    printf 'status-after-producer:yes\n' >>"$TRACE"
  else
    printf 'status-after-producer:no\n' >>"$TRACE"
  fi
  printf 'status-value:%s\n' "$2" >>"$TRACE"
  if [[ "$STATUS_FAULT" == partial_write ]]; then
    printf 'partial\n' >"$1"
    return 73
  fi
  if [[ "$STATUS_FAULT" == o_excl ]]; then
    return 73
  fi
  printf '%s\n' "$2" >"$1"
  return 0
}}

seal_retained_file() {{
  printf 'seal:%s\n' "$1" >>"$TRACE"
  if [[ "$1" == "$P30_DEFERRED_ROOT/"* ]]; then
    chmod 400 "$1"
  else
    chmod 444 "$1"
  fi
  return 0
}}

seal_evidence_inventory() {{
  if [[ -f "$EXPECTED_STATUS" ]]; then
    printf 'inventory-status:present\n' >>"$TRACE"
    chmod 444 "$EXPECTED_STATUS"
  else
    printf 'inventory-status:absent\n' >>"$TRACE"
  fi
  return 0
}}

validate_deferred_journal() {{
  printf 'journal-validation:%s:%s\n' "$1" "$2" >>"$TRACE"
  if [[ "$LOGGER_NAME" == run_root_journal_logged ]]; then
    [[ "$1" == after-admission ]] || return 81
    [[ "$(<"$EXPECTED_STATUS")" == "$2" ]] || return 82
    local retained
    for retained in "$EXPECTED_STDOUT" "$EXPECTED_STDERR" "$EXPECTED_STATUS"; do
      [[ "$({retained_mode_command} "$retained")" == 400 ]] || return 83
    done
  fi
  return 0
}}

set +e
{invocation}
logger_status=$?
printf 'return:%s\n' "$logger_status" >>"$TRACE"
exit 0
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    trace_lines = trace.read_text(encoding="utf-8").splitlines() if trace.exists() else []
    return result, trace_lines, status


def _run_token_creator(
    tmp_path: Path, mutation: str, *, initialize: bool = True
) -> tuple[subprocess.CompletedProcess[str], Path]:
    if initialize:
        tmp_path.mkdir(parents=True, exist_ok=True)
    function_source = _function(
        _source(), "create_localization_token", "burn_localization_invocation"
    )
    python_source = _heredoc_python(function_source)
    uid = os.getuid()
    gid = os.getgid()
    for mode in (0o700, 0o555, 0o400):
        python_source = python_source.replace(f"(0, 0, 0o{mode:o})", f"({uid}, {gid}, 0o{mode:o})")
    acl_fd = {
        "deferred_acl": "deferred_fd",
        "orchestrator_acl": "orchestrator_fd",
    }.get(mutation)
    listxattr_body = "[]"
    if acl_fd is not None:
        listxattr_body = (
            '["system.posix_acl_access"] '
            f'if "{acl_fd}" in globals() and _descriptor == {acl_fd} else []'
        )
    python_source = python_source.replace(
        "import sys\n\nledger =",
        f"import sys\n\nos.listxattr = lambda _descriptor: {listxattr_body}\n\nledger =",
    )
    if mutation == "orchestrator_unstable":
        marker = "        after = os.fstat(orchestrator_fd)\n"
        assert python_source.count(marker) == 1
        python_source = python_source.replace(
            marker,
            marker
            + "        os.utime(\n"
            + '            "run_p30_umask_bound_control_seal.sh",\n'
            + "            ns=(after.st_atime_ns, after.st_mtime_ns + 1),\n"
            + "            dir_fd=ledger_fd,\n"
            + "            follow_symlinks=False,\n"
            + "        )\n",
        )
    if mutation == "post_create_failure":
        marker = "    deferred_after = os.fstat(deferred_fd)\n"
        assert python_source.count(marker) == 1
        python_source = python_source.replace(
            marker,
            '    raise SystemExit("injected post-create validation failure")\n' + marker,
        )

    ledger = tmp_path / "ledger"
    deferred = ledger / "deferred"
    orchestrator = ledger / "run_p30_umask_bound_control_seal.sh"
    orchestrator_raw = b"#!/bin/bash\nexit 0\n"
    token = ledger / "prepare-runtime.invoked"
    if initialize:
        ledger.mkdir(mode=0o700)
        ledger.chmod(0o700)
        if mutation == "deferred_symlink":
            deferred_target = tmp_path / "deferred-target"
            deferred_target.mkdir(mode=0o700)
            deferred.symlink_to(deferred_target, target_is_directory=True)
        else:
            deferred.mkdir(mode=0o700)
            deferred.chmod(0o755 if mutation == "deferred_mode" else 0o700)

        if mutation == "orchestrator_symlink":
            target = tmp_path / "sealed-orchestrator-target"
            target.write_bytes(orchestrator_raw)
            target.chmod(0o555)
            orchestrator.symlink_to(target)
        else:
            orchestrator.write_bytes(orchestrator_raw)
            orchestrator.chmod(0o444 if mutation == "orchestrator_mode" else 0o555)

        if mutation == "preexisting_token":
            token.write_bytes(b"prepare-runtime\n")
            token.chmod(0o400)
    else:
        assert ledger.is_dir() and deferred.is_dir() and orchestrator.is_file()
    expected_digest = hashlib.sha256(orchestrator_raw).hexdigest()
    if mutation == "orchestrator_digest":
        expected_digest = "0" * 64
    expected_before = "deferred\nrun_p30_umask_bound_control_seal.sh"
    expected_after = "deferred\nprepare-runtime.invoked\nrun_p30_umask_bound_control_seal.sh"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            python_source,
            str(ledger),
            str(token),
            "prepare-runtime",
            expected_before,
            expected_after,
            expected_digest,
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, token


def test_shell_syntax_and_only_two_public_phases() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=ROOT, check=True)
    source = _source()
    case = source[source.rindex('case "$1" in') :]
    assert "prepare-runtime)" in case
    assert "burn_prepare_invocation" in case
    assert "prepare_runtime" in case
    assert "run-localization)" in case
    assert "burn_localization_invocation" in case
    assert "run_localization" in case
    assert "cleanup)" not in case
    assert "retry)" not in case
    assert "run-acquisition)" not in case
    assert "compgen -A variable GIT_" in source
    assert "export GIT_CONFIG_NOSYSTEM=1" in source
    assert "export GIT_CONFIG_GLOBAL=/dev/null" in source


def test_unknown_phase_fails_before_any_remote_command(tmp_path: Path) -> None:
    environment = {"PATH": os.environ["PATH"]}
    result = subprocess.run(
        ["bash", str(SCRIPT), "not-a-phase"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 1
    assert "requires a direct root invocation" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_attempt_image_gpu_and_authority_are_fixed_and_env_checked() -> None:
    source = _source()
    assert "readonly P30_NAMESPACE_ROOT=/secure/p30" in source
    assert "readonly P30_ATTEMPT_ROOT=/secure/p30/attempt-20260906-03" in source
    assert "readonly P30_CONTAINER=p30-localization-20260906-03" in source
    assert "readonly P30_TRANSPORT_ROOT=/secure/p30/transport-20260906-03" in source
    assert (
        'readonly P30_BOOTSTRAP_VERIFIER="$P30_TRANSPORT_ROOT/verify_p30_control_bundle.py"'
        in source
    )
    assert 'readonly P30_CONTROL_REPO="$P30_TRANSPORT_ROOT/control-source"' in source
    assert "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0" in source
    assert "185e444afc0b44ca0a09b1bde49a6b6fa3973355" in source
    assert "24f4bdac331a57bd7c1b807747d7c7fba253ee5a" in source
    required = _function(source, "require_source_environment", "require_runtime_environment")
    assert "require_var P30_ATTEMPT_ID" in required
    assert '"$P30_ATTEMPT_ID" == "$P30_REQUIRED_ATTEMPT_ID"' in required
    assert "require_var P30_GPU_UUID" in required
    assert '"$P30_GPU_UUID" == "$P30_REQUIRED_GPU_UUID"' in required
    assert '"$P30_AUTHORITY_REPO" == "$P30_TRANSPORT_ROOT/authority-185e444"' in required


def test_source_bundle_receipt_and_all_reconstructions_precede_attempt_write() -> None:
    source = _source()
    admission = _function(source, "prepare_runtime_admission", "prepare_runtime")
    prepare = _function(source, "prepare_runtime", "validate_live_runtime")
    source_replay = admission.index("verify_source_transport_unlogged")
    namespace_check = admission.index("validate_namespace_layout")
    authority_check = admission.index("validate_authority_and_inputs")
    admission_log = prepare.index("run_root_journal_logged p30-prepare-pre-attempt-admission")
    attempt_write = prepare.index("run_logged_deferred_evidence p30-prepare-attempt-layout")
    container_write = prepare.index("refresh_bound_file p30-container-launch")
    assert source_replay < namespace_check < authority_check
    assert admission_log < attempt_write < container_write
    assert admission.count("verify_source_transport_unlogged") == 1
    assert "run_p27_reconstruction_unlogged" not in admission
    assert "run_p28_reconstructions_unlogged" not in admission
    assert "run_p30_reconstruction_unlogged" not in admission
    assert "create_fresh_attempt_layout" not in admission
    assert "capture_new " not in admission

    replay = _function(
        source, "verify_source_transport_unlogged", "verify_runtime_review_transport_unlogged"
    )
    assert "--phase source" in replay
    assert "run_bootstrap_verifier" in replay
    assert '--bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER"' in replay
    assert '--expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256"' in replay
    assert '"$P30_SOURCE_BUNDLE"' in replay
    assert '"$P30_SOURCE_CLOSURE"' in replay
    assert '"$P30_SOURCE_RECEIPT"' in replay
    assert "--verify-existing-receipt" in replay
    assert "--receipt-output" not in replay
    assert '"$P30_EXPECTED_SOURCE_RECEIPT_SHA256"' in replay


def test_runtime_review_receipt_replay_precedes_localization_marker() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    deferred_replay = run.index(
        "run_logged_deferred_evidence p30-pre-marker-runtime-receipt-replay"
    )
    runtime_replay = run.index("verify_runtime_review_transport_unlogged")
    entry_preflight = run.index("run_logged p30-localization-entry-preflight")
    ingestion = run.index("run_logged p30-p26-failure-ingestion")
    localization = run.index("run_logged p30-localization run_localizer_once")
    assert deferred_replay < runtime_replay < entry_preflight < ingestion < localization
    assert "run_logged " not in run[:deferred_replay]
    entry_validator = _function(
        source, "validate_localization_entry_state", "validate_prepare_snapshot_boundary"
    )
    assert "validate_deferred_journal after-runtime" in entry_validator
    assert "validate_deferred_journal after-prepare" not in entry_validator
    dispatch = _dispatch_branch(source, "run-localization")
    assert dispatch.index("burn_localization_invocation") < dispatch.index("run_localization")
    assert "run_p27_reconstruction_unlogged" not in run
    assert "run_p28_reconstructions_unlogged" not in run
    assert run.index("validate_reviewed_runtime_artifacts") < ingestion

    replay = _function(source, "verify_runtime_review_transport_unlogged", "prepare_runtime")
    assert (
        replay.index("require_runtime_environment")
        < replay.index("validate_transport_execution_sources")
        < replay.index("run_bootstrap_verifier")
    )
    assert "--phase runtime-review" in replay
    assert "run_bootstrap_verifier" in replay
    assert '--bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER"' in replay
    assert '--expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256"' in replay
    assert '"$P30_RUNTIME_REVIEW_BUNDLE"' in replay
    assert '"$P30_RUNTIME_REVIEW_CLOSURE"' in replay
    assert '"$P30_RUNTIME_REVIEW_RECEIPT"' in replay
    assert "--expected-source-commit" in replay
    assert "--expected-source-tree" in replay
    assert "--verify-existing-receipt" in replay
    assert "--receipt-output" not in replay


def test_malformed_runtime_environment_blocks_runtime_receipt_verifier(
    tmp_path: Path,
) -> None:
    replay = _function(
        _source(), "verify_runtime_review_transport_unlogged", "launch_fresh_container"
    )
    trace = tmp_path / "runtime-replay.trace"
    harness = f"""
set -euo pipefail
TRACE={shlex.quote(str(trace))}

{replay}

require_runtime_environment() {{
  printf 'runtime-environment-rejected\n' >>"$TRACE"
  return 37
}}
validate_transport_execution_sources() {{
  printf 'transport-validation-ran\n' >>"$TRACE"
}}
run_bootstrap_verifier() {{
  printf 'verifier-dispatched\n' >>"$TRACE"
}}
die() {{
  printf 'die:%s\n' "$*" >>"$TRACE"
  return 99
}}

verify_runtime_review_transport_unlogged
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 37, result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == ["runtime-environment-rejected"]


def test_deferred_runtime_logger_finalizes_when_reviewed_digest_is_unset(
    tmp_path: Path,
) -> None:
    logger = _function(_source(), "run_logged_deferred_evidence", "capture_new")
    publication_start = logger.index(
        "  /usr/bin/python3 -I -S - \\\n",
        logger.index("local publication_status=0"),
    )
    publication_end = logger.index("\nPY\n  publication_status=$?", publication_start)
    logger = (
        logger[:publication_start]
        + '  publication_stub >"$publication_stdout" 2>"$publication_stderr"'
        + logger[publication_end + len("\nPY") :]
    )

    deferred = tmp_path / "deferred"
    evidence = tmp_path / "evidence"
    deferred.mkdir()
    evidence.mkdir()
    observed_status = tmp_path / "logger-status"
    harness = f"""
set -uo pipefail
P30_DEFERRED_ROOT={shlex.quote(str(deferred))}
P30_HOST_EVIDENCE={shlex.quote(str(evidence))}
P30_RUNTIME_REVIEW_RECEIPT={shlex.quote(str(tmp_path / "receipt.json"))}
OBSERVED_STATUS={shlex.quote(str(observed_status))}
unset P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256

{logger}

die() {{ return 99; }}
write_new_status() {{
  ( set -o noclobber; printf '%s\n' "$2" >"$1" )
}}
seal_retained_file() {{
  if [[ "$1" == "$P30_DEFERRED_ROOT/"* ]]; then
    chmod 400 "$1"
  else
    chmod 444 "$1"
  fi
}}
failing_runtime_verifier() {{ return 37; }}
publication_stub() {{
  printf 'publisher rejected missing digest\n' >&2
  return 41
}}

set +e
run_logged_deferred_evidence \
  p30-pre-marker-runtime-receipt-replay failing_runtime_verifier
logger_status=$?
set -e
printf '%s\n' "$logger_status" >"$OBSERVED_STATUS"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert observed_status.read_bytes() == b"41\n"
    prefix = deferred / ".deferred-p30-pre-marker-runtime-receipt-replay"
    assert Path(f"{prefix}.exit-status.txt").read_bytes() == b"37\n"
    assert Path(f"{prefix}.publication.exit-status.txt").read_bytes() == b"41\n"
    assert Path(f"{prefix}.publication-failure.exit-status.txt").read_bytes() == b"41\n"
    for path in deferred.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o400


def test_prepare_uses_exact_p23_ten_mount_contract() -> None:
    source = _source()
    prepare = _function(source, "prepare_runtime", "stage_source")
    mounts = re.findall(r"--mount type=bind,[^\n]+", prepare)
    assert len(mounts) == 10
    destinations = {
        re.search(r",dst=([^,\\ ]+)", mount).group(1)  # type: ignore[union-attr]
        for mount in mounts
    }
    assert destinations == {
        "/workspace/OptimizationML",
        "/workspace/inputs/nanoGPT",
        "/workspace/inputs/muon",
        "/private/tmp/optimizationml-p22-data",
        "/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py",
        "/workspace/evidence/p23",
        "/mounted-host-evidence/image-inspect.json",
        "/mounted-host-evidence/running-container-inspect.json",
        "/mounted-host-evidence/running-mountinfo.txt",
        "/mounted-host-evidence/nvidia-smi.csv",
    }
    assert prepare.count("launch_fresh_container --detach") == 1
    assert "p30_docker run" in _function(
        source, "launch_fresh_container", "read_fresh_container_id"
    )
    assert "--network none" in prepare
    assert "--read-only" in prepare
    assert "--user 0:0" in prepare
    assert "--userns host" in prepare
    assert 'get("UsernsMode") == "host"' in source
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824" in prepare
    assert "freeze-runtime" in prepare
    assert "run_p30_umask_bound_control_seal.py" not in prepare


def test_persistent_authority_checkout_rejects_git_object_indirections() -> None:
    source = _source()
    authority = _function(source, "validate_authority_and_inputs", "validate_control_sources")
    layout = _function(source, "transport_checkout_layout_binding", "authority_layout_binding")
    assert "O_DIRECTORY" in layout
    assert "O_NOFOLLOW" in layout
    assert '"authority-185e444"' in layout
    assert "checkout != transport / basename" in layout
    assert "owner or mode differs" in layout
    assert "has an ACL" in layout
    assert "is an unexpected mount point" in layout
    assert "crosses a mount-ID boundary" in layout
    assert 'authority_binding_before="$(authority_layout_binding)"' in authority
    assert 'authority_binding_after="$(authority_layout_binding)"' in authority
    assert '"$authority_binding_after" == "$authority_binding_before"' in authority
    assert "rev-parse --is-shallow-repository" in authority
    assert '"$P30_AUTHORITY_REPO/.git/objects/info/alternates"' in authority
    assert '"$P30_AUTHORITY_REPO/.git/info/grafts"' in authority
    assert '! -e "$authority_indirection" && ! -L "$authority_indirection"' in authority
    assert "for-each-ref" in authority
    assert "refs/replace/" in authority
    assert "P30 authority checkout has forbidden replace refs" in authority
    assert "extensions\\.partialclone" in authority
    assert "promisor|partialclonefilter" in authority
    assert "forbidden promisor or partial-clone configuration" in authority
    checkout = _function(source, "assert_clean_checkout", "validate_namespace_layout")
    assert '"$repository/.git/commondir"' in checkout
    assert "includeif.*.path" in checkout
    assert "forbidden local Git configuration" in checkout


def test_prepare_stops_at_freeze_and_has_no_cleanup_or_retry_commands() -> None:
    source = _source()
    forbidden = re.compile(r"\bdocker\s+(?:rm|stop|start|restart|kill)\b")
    assert forbidden.search(source) is None
    prepare = _function(source, "prepare_runtime", "stage_source")
    assert "run_logged p30-freeze-runtime freeze_runtime_from_bound_container" in prepare
    assert "run_logged p30-freeze-runtime freeze_runtime_from_bound_container ||" not in prepare
    assert prepare.index("set +e\n  run_logged p30-freeze-runtime") < prepare.index(
        "freeze_status=$?\n  set -e"
    )
    freeze = _function(
        source, "freeze_runtime_from_bound_container", "capture_running_container_inspection"
    )
    assert "p30_cuda_runtime_lock.json" in freeze
    assert "p30_host_attestation.json" in freeze
    assert "git commit" not in prepare
    assert "git add" not in prepare
    assert "install -m 0644" not in prepare.split("freeze-runtime", 1)[1]


def test_freeze_capture_does_not_suppress_producer_errexit(tmp_path: Path) -> None:
    run_logged = _function(_source(), "run_logged", "create_admission_log_placeholders")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    side_effect = tmp_path / "must-not-exist"
    observed_status = tmp_path / "observed-status"
    harness = f"""
set -uo pipefail
P30_HOST_EVIDENCE={shlex.quote(str(evidence))}
SIDE_EFFECT={shlex.quote(str(side_effect))}
OBSERVED_STATUS={shlex.quote(str(observed_status))}

{run_logged}

die() {{ return 99; }}
write_new_status() {{ printf '%s\n' "$2" >"$1"; }}
seal_retained_file() {{ chmod 444 "$1"; }}
seal_evidence_inventory() {{ return 0; }}
freeze_runtime_from_bound_container() {{
  false
  printf 'errexit-was-suppressed\n' >"$SIDE_EFFECT"
}}

set -e
freeze_status=0
set +e
run_logged p30-freeze-runtime freeze_runtime_from_bound_container
freeze_status=$?
set -e
printf '%s\n' "$freeze_status" >"$OBSERVED_STATUS"
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert observed_status.read_bytes() == b"1\n"
    assert not side_effect.exists()
    assert (evidence / "p30-freeze-runtime.exit-status.txt").read_bytes() == b"1\n"


def test_inventory_sealing_continues_after_an_earlier_child_failure(
    tmp_path: Path,
) -> None:
    function_source = _function(_source(), "seal_evidence_inventory", "run_logged")
    python_source = _heredoc_python(function_source)
    uid = os.getuid()
    gid = os.getgid()
    python_source = python_source.replace("(0, 0, 0o555)", f"({uid}, {gid}, 0o555)").replace(
        "(0, 0)", f"({uid}, {gid})"
    )
    python_source = python_source.replace("os.listxattr(file_descriptor)", "[]")
    python_source = python_source.replace(
        "            os.fchmod(file_descriptor, 0o444)",
        "            if name == 'a-first-fails.txt':\n"
        "                raise OSError('injected first-child seal failure')\n"
        "            os.fchmod(file_descriptor, 0o444)",
        1,
    )
    evidence = tmp_path / "evidence"
    evidence.mkdir(mode=0o755)
    first = evidence / "a-first-fails.txt"
    later = evidence / "z-later-output.json"
    first.write_bytes(b"first\n")
    later.write_bytes(b"later\n")
    first.chmod(0o644)
    later.chmod(0o644)
    evidence.chmod(0o555)

    result = subprocess.run(
        [sys.executable, "-I", "-S", "-", str(evidence)],
        input=python_source,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "a-first-fails.txt" in result.stderr
    assert stat.S_IMODE(first.stat().st_mode) == 0o644
    assert stat.S_IMODE(later.stat().st_mode) == 0o444, result.stderr


def test_nonempty_deferred_publication_records_terminal_failure_sentinel() -> None:
    logger = _function(_source(), "run_logged_deferred_evidence", "capture_new")
    publication = logger.split("local publication_status=0", 1)[1].split(
        "local publication_finalization_failed=0", 1
    )[0]
    assert "set -o noclobber" in publication
    assert '>"$publication_stdout" 2>"$publication_stderr"' in publication
    assert "set +o noclobber" in publication
    nonempty = logger.split(
        'if [[ -s "$publication_stdout" || -s "$publication_stderr" ]]; then', 1
    )[1].split("\n  fi", 1)[0]
    assert 'write_new_status "$publication_failure" 125' in nonempty
    assert 'seal_retained_file "$publication_failure"' in nonempty
    assert "return 125" in nonempty


def test_runtime_review_requires_direct_child_and_exact_two_file_delta() -> None:
    source = _source()
    source_history = _function(
        source, "validate_source_freeze_history", "validate_runtime_review_history"
    )
    assert '"$P30_SOURCE_FREEZE_COMMIT^{tree}"' in source_history
    assert '"$P30_SOURCE_FREEZE_TREE"' in source_history
    assert "$P30_SOURCE_FREEZE_COMMIT $P30_TERMINAL_P29_COMMIT" in source_history
    assert "terminal P29 commit/tree binding differs" in source_history
    review = _function(source, "validate_runtime_review_history", "run_contract_reconstruction")
    assert "$P30_RUNTIME_REVIEW_COMMIT $P30_SOURCE_FREEZE_COMMIT" in review
    assert review.count("A\\texperiments/training/p30_cuda_runtime_lock.json") == 1
    assert review.count("A\\texperiments/training/p30_host_attestation.json") == 1
    assert "validate_source_freeze_history" in review


def test_permission_safe_namespace_and_attempt_layouts_are_revalidated() -> None:
    source = _source()
    namespace = _function(source, "validate_namespace_layout", "validate_fresh_attempt_layout")
    assert "O_DIRECTORY" in namespace
    assert "O_NOFOLLOW" in namespace
    assert "system.posix_acl_access" in namespace
    assert "system.posix_acl_default" in namespace
    assert 'pathlib.PurePosixPath("/secure/p30")' in namespace
    assert 'namespace / "transport-20260906-03"' in namespace
    assert 'transport / "control-source"' in namespace
    assert 'transport / "authority-185e444"' in namespace
    assert "unexpected mount point" in namespace
    assert "cross a mount-ID boundary" in namespace
    attempt = _function(source, "validate_fresh_attempt_layout", "validate_authority_and_inputs")
    assert 'namespace / "attempt-20260906-03"' in attempt
    assert "P30 attempt component authority differs" in attempt
    prepare = _function(source, "prepare_runtime", "stage_source")
    creation = prepare.index("create_fresh_attempt_layout")
    validation = prepare.index("validate_prepare_snapshot_boundary", creation)
    container = prepare.index("launch_fresh_container --detach")
    assert creation < validation < container


def test_executing_orchestrator_is_bound_to_reviewed_control_bytes() -> None:
    source = _source()
    validation = _function(source, "validate_control_sources", "validate_source_freeze_history")
    assert "${BASH_SOURCE[0]}" in validation
    assert '"$executing_orchestrator" == "$P30_SEALED_ORCHESTRATOR"' in validation
    assert "executing P30 host orchestrator bytes differ" in validation
    transport_validation = _function(
        source, "validate_transport_execution_sources", "verify_source_transport_unlogged"
    )
    assert "${BASH_SOURCE[0]}" in transport_validation
    assert "executing P30 host orchestrator bytes differ before transport replay" in (
        transport_validation
    )
    assert '"$(sha256_file "$P30_BOOTSTRAP_VERIFIER")"' in transport_validation
    assert '"$(/usr/bin/stat -c \'%a\' "$P30_BOOTSTRAP_VERIFIER")" == "600"' in (
        transport_validation
    )


def test_run_stages_exactly_four_hash_bound_sources_in_existing_evidence_mount() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    assert "stage_source" not in source
    assert (
        "/private/tmp/optimizationml-p22-data/.p30-tools/"
        "run_p27_cuda_deleted_mapping_localization.py"
    ) in run
    assert (
        "/private/tmp/optimizationml-p22-data/.p30-tools/ingest_p26_trace_off_a_failure.py"
    ) in run
    assert (
        "/private/tmp/optimizationml-p22-data/.p30-tools/p23_deterministic_cuda_shadow_trace.py"
    ) in run
    assert (
        "/private/tmp/optimizationml-p22-data/.p30-tools/"
        "sanitize_p27_cuda_deleted_mapping_localization.py"
    ) in _function(source, "sanitize_localization_artifact", "validate_localization_entry_state")
    assert "docker cp" not in run
    assert "--mount " not in run


def test_p26_ingestion_is_host_root_one_read_bridge_before_cuda_localizer() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    ingestion = run.index("run_logged p30-p26-failure-ingestion run_p26_ingestion_once")
    localization = run.index("run_logged p30-localization run_localizer_once")
    sanitizer = run.index("run_logged p30-localization-sanitizer sanitize_localization_artifact")
    assert ingestion < localization < sanitizer
    assert (
        "--source /private/tmp/optimizationml-p22-data/.p30-tools/"
        "p26-trace-off-a-failure.authenticated.json"
    ) in run
    assert "--output-root /workspace/evidence/p23" in run
    assert 'sha256_file "$P30_P26_NATIVE"' not in run
    assert 'root_sha256_file "$P30_P26_NATIVE"' not in run
    assert "before localization" in run


def test_localizer_is_one_shot_and_preserves_nonzero_status_before_sanitizing() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    assert run.count("run_logged p30-localization run_localizer_once") == 1
    assert "run-localization.invoked" in source
    assert "no rerun is allowed" in run
    command = run[run.index("run_logged p30-localization run_localizer_once") :]
    assert command.count("--expected-p25-source-sha256") == 1
    assert command.count("--expected-p25-contract-sha256") == 1
    sanitizer = run.index("run_logged p30-localization-sanitizer sanitize_localization_artifact")
    assert run.index("run_logged p30-localization run_localizer_once") < sanitizer
    assert "--output-root /workspace/evidence/p23" in run
    assert (
        "--runner-source /private/tmp/optimizationml-p22-data/.p30-tools/"
        "run_p27_cuda_deleted_mapping_localization.py"
    ) in run
    assert "--repository-root /workspace/OptimizationML" in run
    dispatch = _dispatch_branch(source, "run-localization")
    assert "write_localization_exit_status" in dispatch
    assert "sanitized failure evidence is retained" in run


def test_live_revalidation_covers_identity_mounts_gpu_and_runtime_bytes() -> None:
    source = _source()
    live = _function(source, "validate_live_runtime", "validate_reviewed_runtime_artifacts")
    for token in (
        '"container_id"',
        '"running"',
        '"pid"',
        '"restart"',
        '"image_reference"',
        '"image_id"',
        '"network"',
        '"rootfs"',
        '"tmpfs"',
        '"mountinfo"',
        '"gpu_query"',
        '"gpu_uuid"',
        '"exact_ten_mounts"',
        '"environment"',
        '"device_request"',
    ):
        assert token in live
    run = _function(source, "run_localization", "usage")
    assert "validate_live_runtime p30-pre-ingestion" in run
    assert "validate_live_runtime p30-pre-localization" in run
    artifacts = _function(source, "validate_reviewed_runtime_artifacts", "assert_staged_source")
    assert artifacts.count("validate_root_evidence_file") == 2
    assert '"$native_lock" "$P30_EXPECTED_RUNTIME_LOCK_SHA256" - 0444 0 0' in artifacts
    assert '"$native_attestation" "$P30_EXPECTED_HOST_ATTESTATION_SHA256" - 0444 0 0' in artifacts


def test_no_training_or_scientific_acquisition_entrypoint_is_exposed() -> None:
    source = _source()
    case = source[source.rindex('case "$1" in') :]
    for forbidden in (
        "trace-off-a",
        "trace-on",
        "run-acquisition",
        "forward",
        "backward",
        "optimizer-step",
        "candidate",
    ):
        assert f"{forbidden})" not in case


def test_terminal_p27_attempt_is_never_reused_or_mutated() -> None:
    source = _source()
    assert "/secure/p27/attempt-20260906-01" not in source
    assert "p27-localization-20260906-01" not in source
    assert "ec63550331925ded158e3f389e294e4d1f12db3a" in source
    assert "d8efa72fda9ee41fde0b5d15b126aa87397cd48d" in source


def test_preexisting_valid_prepare_token_blocks_burn_and_prepare_dispatch(
    tmp_path: Path,
) -> None:
    creator_result, token = _run_token_creator(tmp_path / "creator", "preexisting_token")
    assert creator_result.returncode != 0
    assert token.read_bytes() == b"prepare-runtime\n"
    assert stat.S_IMODE(token.stat().st_mode) == 0o400

    source = _source()
    burn = _function(source, "burn_prepare_invocation", "authorize_localization")
    dispatch = _dispatch_branch(source, "prepare-runtime")
    trace = tmp_path / "prepare-dispatch.trace"
    harness = f"""
set -uo pipefail
TRACE={shlex.quote(str(trace))}
P30_PREPARE_INVOCATION={shlex.quote(str(token))}

{burn}

create_localization_token() {{
  [[ "$1" == prepare ]] || return 90
  if [[ -f "$P30_PREPARE_INVOCATION" &&
        "$(<"$P30_PREPARE_INVOCATION")" == prepare-runtime ]]; then
    printf 'valid-prepare-token-observed\n' >>"$TRACE"
    return 73
  fi
  return 0
}}

prepare_runtime() {{
  printf 'prepare-runtime-entered\n' >>"$TRACE"
  return 0
}}

die() {{
  printf 'die:%s\n' "$*" >>"$TRACE"
  exit 99
}}

{dispatch}
"""
    dispatch_result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert dispatch_result.returncode == 99, dispatch_result.stderr
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "valid-prepare-token-observed",
        "die:prepare-runtime was already invoked or its admission token could not be retained",
    ]


def test_token_creation_authenticates_journal_and_sealed_orchestrator_before_o_excl() -> None:
    function_source = _function(
        _source(), "create_localization_token", "burn_localization_invocation"
    )
    python_source = _heredoc_python(function_source)
    token_open = python_source.index("token_fd = os.open(")
    deferred_open = python_source.index('deferred_fd = os.open(\n        "deferred"')
    orchestrator_open = python_source.index("orchestrator_fd = os.open(")
    assert deferred_open < orchestrator_open < token_open
    before_token = python_source[:token_open]
    assert "os.O_DIRECTORY | os.O_NOFOLLOW" in before_token
    assert "(0, 0, 0o700)" in before_token
    assert "os.listxattr(deferred_fd)" in before_token
    assert "system.posix_acl_default" in before_token
    assert '"run_p30_umask_bound_control_seal.sh"' in before_token
    assert "os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW" in before_token
    assert "fields(before) != fields(after)" in before_token
    assert "fields(after) != fields(by_name)" in before_token
    assert "(0, 0, 0o555)" in before_token
    assert "digest.hexdigest() != expected_orchestrator_sha256" in before_token
    assert "os.listxattr(orchestrator_fd)" in before_token
    assert "$P30_EXPECTED_ORCHESTRATOR_SHA256" in function_source.split("<<'PY'", 1)[0]
    assert "os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW" in python_source[token_open:]


@pytest.mark.parametrize(
    "mutation",
    [
        "deferred_symlink",
        "deferred_mode",
        "deferred_acl",
        "orchestrator_symlink",
        "orchestrator_mode",
        "orchestrator_acl",
        "orchestrator_unstable",
        "orchestrator_digest",
    ],
)
def test_token_creation_rejects_untrusted_journal_or_orchestrator_before_o_excl(
    tmp_path: Path, mutation: str
) -> None:
    result, token = _run_token_creator(tmp_path, mutation)
    assert result.returncode != 0
    assert not token.exists() and not token.is_symlink()


def test_token_creation_succeeds_only_after_trust_inputs_authenticate(tmp_path: Path) -> None:
    result, token = _run_token_creator(tmp_path, "valid")
    assert result.returncode == 0, result.stderr
    assert token.read_bytes() == b"prepare-runtime\n"
    assert stat.S_IMODE(token.stat().st_mode) == 0o400


def test_complete_prepare_admission_is_root_journal_logged_before_attempt_creation() -> None:
    source = _source()
    prepare = _function(source, "prepare_runtime", "validate_live_runtime")
    admission_call = prepare.index("run_root_journal_logged p30-prepare-pre-attempt-admission")
    attempt_call = prepare.index("run_logged_deferred_evidence p30-prepare-attempt-layout")
    assert admission_call < attempt_call
    assert prepare.count("prepare_runtime_admission") == 1
    admission = _function(source, "prepare_runtime_admission", "prepare_runtime")
    for boundary in (
        "require_source_environment",
        "validate_localization_ledger present absent absent absent",
        "verify_source_transport_unlogged",
        "validate_control_sources",
        "validate_source_freeze_history",
        "validate_authority_and_inputs",
        "reviewed P30 image digest is unavailable",
    ):
        assert boundary in admission
    assert "create_fresh_attempt_layout" not in admission

    logger = _function(source, "run_root_journal_logged", "run_logged_deferred_evidence")
    assert '"$label" == "p30-prepare-pre-attempt-admission"' in logger
    assert logger.index("create_admission_log_placeholders") < logger.index('"$@"')
    assert logger.count(":600:0:0:1") == 2
    assert 'write_new_status "$status_path" "$status"' in logger
    for retained in ("$stdout_path", "$stderr_path", "$status_path"):
        assert f'seal_retained_file "{retained}"' in logger
    assert "return 125" in logger
    sealing = _function(source, "seal_retained_file", "seal_evidence_inventory")
    assert "deferred: ((0, 0, 0o700), 0o400)" in sealing


def test_deferred_journal_inventory_includes_three_prepare_admission_files() -> None:
    journal = _function(_source(), "validate_deferred_journal", "validate_localization_ledger")
    assert '"after-admission": ()' in journal
    assert 'if stage != "empty"' in journal
    for suffix in ("stdout.log", "stderr.log", "exit-status.txt"):
        assert f'".deferred-p30-prepare-pre-attempt-admission.{suffix}"' in journal
    assert "(0, 0, 0o400)" in journal
    assert 'expected_status = b"0\\n"' in journal
    assert 'stage == "after-admission"' in journal
    assert 'expected_status = f"{admission_status}\\n".encode("ascii")' in journal
    assert "bytes(raw) != expected_status" in journal


@pytest.mark.parametrize(
    "logger_name",
    [
        "run_logged",
        "run_root_journal_logged",
        "run_logged_deferred_evidence",
        "capture_new",
        "refresh_bound_file",
    ],
)
@pytest.mark.parametrize("status_fault", ["o_excl", "partial_write"])
def test_logger_status_failure_after_producer_is_terminal_and_seals_every_output(
    tmp_path: Path, logger_name: str, status_fault: str
) -> None:
    result, trace, status_path = _run_logger_status_fault(tmp_path, logger_name, status_fault)
    assert result.returncode == 0, result.stderr
    assert "status-after-producer:yes" in trace
    assert "status-value:37" in trace
    assert "return:125" in trace
    assert "return:37" not in trace

    if logger_name in {
        "run_root_journal_logged",
        "run_logged_deferred_evidence",
    }:
        label = (
            "p30-prepare-pre-attempt-admission"
            if logger_name == "run_root_journal_logged"
            else "p30-prepare-attempt-layout"
        )
        prefix = tmp_path / f"deferred/.deferred-{label}"
        expected_seals = {
            f"seal:{prefix}.stdout.log",
            f"seal:{prefix}.stderr.log",
            f"seal:{prefix}.exit-status.txt",
        }
        assert expected_seals.issubset(trace)
        assert not any(line.startswith("inventory-status:") for line in trace)
        for suffix in ("stdout.log", "stderr.log", "exit-status.txt"):
            assert stat.S_IMODE(Path(f"{prefix}.{suffix}").stat().st_mode) == 0o400
    else:
        evidence = tmp_path / "evidence"
        output = (
            evidence / "fault-injection.stdout.log"
            if logger_name == "run_logged"
            else evidence / "bound-output.json"
        )
        assert f"seal:{output}" in trace
        assert f"seal:{evidence / 'fault-injection.stderr.log'}" in trace
        assert "inventory-status:present" in trace
        assert stat.S_IMODE(status_path.stat().st_mode) == 0o444

    assert status_path.read_text(encoding="utf-8") in {"collision\n", "partial\n"}
    assert status_path.read_text(encoding="utf-8") != "37\n"


def test_root_admission_logger_retains_producer_status_and_validates_transcripts(
    tmp_path: Path,
) -> None:
    result, trace, status_path = _run_logger_status_fault(
        tmp_path, "run_root_journal_logged", "none"
    )
    assert result.returncode == 0, result.stderr
    assert "status-after-producer:yes" in trace
    assert "status-value:37" in trace
    assert "journal-validation:after-admission:37" in trace
    assert "return:37" in trace
    assert "return:125" not in trace
    assert status_path.read_bytes() == b"37\n"
    prefix = tmp_path / "deferred/.deferred-p30-prepare-pre-attempt-admission"
    for suffix in ("stdout.log", "stderr.log", "exit-status.txt"):
        assert stat.S_IMODE(Path(f"{prefix}.{suffix}").stat().st_mode) == 0o400


def test_successful_invocation_token_owner_runs_and_closes_terminal_status(
    tmp_path: Path,
) -> None:
    source = _source()
    burn = _function(source, "burn_localization_invocation", "burn_prepare_invocation")
    dispatch = _dispatch_branch(source, "run-localization")
    trace = tmp_path / "dispatch.trace"
    invocation = tmp_path / "run-localization.invoked"
    terminal_status = tmp_path / "run-localization.exit-status.txt"
    harness = f"""
set -uo pipefail
TRACE={shlex.quote(str(trace))}
P30_LOCALIZATION_INVOCATION={shlex.quote(str(invocation))}
P30_LOCALIZATION_EXIT_STATUS={shlex.quote(str(terminal_status))}

{burn}

create_localization_token() {{
  [[ "$1" == invocation ]] || return 90
  printf 'run-localization\n' >"$P30_LOCALIZATION_INVOCATION"
  printf 'token-created\n' >>"$TRACE"
  return 0
}}

validate_localization_ledger() {{
  printf 'post-create-validation:%s\n' "$*" >>"$TRACE"
  return 41
}}

run_localization() {{
  printf 'run-localization-called\n' >>"$TRACE"
  return 41
}}

write_localization_exit_status() {{
  printf 'terminal-status-attempt:%s\n' "$1" >>"$TRACE"
  printf '%s\n' "$1" >"$P30_LOCALIZATION_EXIT_STATUS"
  return 0
}}

die() {{
  printf 'die:%s\n' "$*" >>"$TRACE"
  exit 99
}}

{dispatch}
"""
    result = subprocess.run(
        ["bash", "-c", harness],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 41, result.stderr
    assert invocation.read_text(encoding="utf-8") == "run-localization\n"
    assert terminal_status.read_text(encoding="utf-8") == "41\n"
    trace_lines = trace.read_text(encoding="utf-8").splitlines()
    assert trace_lines == [
        "token-created",
        "run-localization-called",
        "terminal-status-attempt:41",
    ]


def test_internal_post_create_token_failure_is_incomplete_and_non_retryable(
    tmp_path: Path,
) -> None:
    transaction_root = tmp_path / "transaction"
    failed, token = _run_token_creator(transaction_root, "post_create_failure")
    assert failed.returncode != 0
    assert "injected post-create validation failure" in failed.stderr
    assert token.read_bytes() == b"prepare-runtime\n"
    assert stat.S_IMODE(token.stat().st_mode) == 0o400
    replay, replay_token = _run_token_creator(transaction_root, "valid", initialize=False)
    assert replay.returncode != 0
    assert replay_token == token
    assert token.read_bytes() == b"prepare-runtime\n"

    source = _source()
    burn = _function(source, "burn_localization_invocation", "burn_prepare_invocation")
    dispatch = _dispatch_branch(source, "run-localization")
    trace = tmp_path / "incomplete.trace"
    invocation = tmp_path / "incomplete-run-localization.invoked"
    terminal_status = tmp_path / "incomplete-run-localization.exit-status.txt"
    harness = f"""
set -uo pipefail
TRACE={shlex.quote(str(trace))}
P30_LOCALIZATION_INVOCATION={shlex.quote(str(invocation))}

{burn}

create_localization_token() {{
  [[ "$1" == invocation ]] || return 90
  if [[ ! -e "$P30_LOCALIZATION_INVOCATION" ]]; then
    printf 'run-localization\n' >"$P30_LOCALIZATION_INVOCATION"
    printf 'token-created-but-incomplete\n' >>"$TRACE"
    return 74
  fi
  printf 'incomplete-token-blocked-replay\n' >>"$TRACE"
  return 75
}}

run_localization() {{
  printf 'run-localization-called\n' >>"$TRACE"
}}

write_localization_exit_status() {{
  printf 'terminal-status-attempt:%s\n' "$1" >>"$TRACE"
  printf '%s\n' "$1" >{shlex.quote(str(terminal_status))}
}}

die() {{
  printf 'die:%s\n' "$*" >>"$TRACE"
  exit 99
}}

{dispatch}
"""
    first = subprocess.run(
        ["bash", "-c", harness], cwd=tmp_path, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        ["bash", "-c", harness], cwd=tmp_path, text=True, capture_output=True, check=False
    )
    assert first.returncode == second.returncode == 99
    assert invocation.read_bytes() == b"run-localization\n"
    assert not terminal_status.exists()
    assert trace.read_text(encoding="utf-8").splitlines() == [
        "token-created-but-incomplete",
        "die:run-localization was already invoked or its invocation token could not be retained",
        "incomplete-token-blocked-replay",
        "die:run-localization was already invoked or its invocation token could not be retained",
    ]


def test_concurrent_invocation_replay_cannot_steal_terminal_status(
    tmp_path: Path,
) -> None:
    source = _source()
    burn = _function(source, "burn_localization_invocation", "burn_prepare_invocation")
    dispatch = _dispatch_branch(source, "run-localization")
    trace = tmp_path / "concurrent.trace"
    invocation = tmp_path / "run-localization.invoked"
    terminal_status = tmp_path / "run-localization.exit-status.txt"
    owner_ready = tmp_path / "owner.ready"
    release_owner = tmp_path / "release.owner"
    harness = f"""
set -uo pipefail
ACTOR="$1"
TRACE={shlex.quote(str(trace))}
P30_LOCALIZATION_INVOCATION={shlex.quote(str(invocation))}
P30_LOCALIZATION_EXIT_STATUS={shlex.quote(str(terminal_status))}
OWNER_READY={shlex.quote(str(owner_ready))}
RELEASE_OWNER={shlex.quote(str(release_owner))}

{burn}

create_localization_token() {{
  [[ "$1" == invocation ]] || return 90
  if (set -o noclobber; printf 'run-localization\n' >"$P30_LOCALIZATION_INVOCATION") \
      2>/dev/null; then
    printf 'token-owner:%s\n' "$ACTOR" >>"$TRACE"
    : >"$OWNER_READY"
    while [[ ! -e "$RELEASE_OWNER" ]]; do sleep 0.01; done
    return 0
  fi
  printf 'token-replay-rejected:%s\n' "$ACTOR" >>"$TRACE"
  return 73
}}

run_localization() {{
  printf 'run-localization:%s\n' "$ACTOR" >>"$TRACE"
  return 37
}}

write_localization_exit_status() {{
  if (set -o noclobber; printf '%s\n' "$1" >"$P30_LOCALIZATION_EXIT_STATUS") \
      2>/dev/null; then
    printf 'terminal-status-owner:%s:%s\n' "$ACTOR" "$1" >>"$TRACE"
    return 0
  fi
  printf 'terminal-status-collision:%s:%s\n' "$ACTOR" "$1" >>"$TRACE"
  return 88
}}

die() {{
  printf 'die:%s:%s\n' "$ACTOR" "$*" >>"$TRACE"
  exit 99
}}

{dispatch}
"""
    owner = subprocess.Popen(
        ["bash", "-c", harness, "p30-test", "owner"],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not owner_ready.exists() and owner.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert owner_ready.exists(), owner.communicate(timeout=1)

    replay = subprocess.run(
        ["bash", "-c", harness, "p30-test", "replay"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
        timeout=5,
    )
    release_owner.touch()
    owner_stdout, owner_stderr = owner.communicate(timeout=5)
    assert replay.returncode == 99, replay.stderr
    assert owner.returncode == 37, owner_stderr or owner_stdout
    assert invocation.read_bytes() == b"run-localization\n"
    assert terminal_status.read_bytes() == b"37\n"
    trace_lines = trace.read_text(encoding="utf-8").splitlines()
    assert "token-owner:owner" in trace_lines
    assert "token-replay-rejected:replay" in trace_lines
    assert "run-localization:owner" in trace_lines
    assert "terminal-status-owner:owner:37" in trace_lines
    assert not any("terminal-status-owner:replay" in line for line in trace_lines)
    assert not any("terminal-status-collision" in line for line in trace_lines)
