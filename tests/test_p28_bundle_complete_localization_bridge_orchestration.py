from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p28_bundle_complete_localization_bridge.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def _function(source: str, name: str, following: str) -> str:
    start = source.index(f"{name}() {{")
    end = source.index(f"\n{following}() {{", start)
    return source[start:end]


def test_shell_syntax_and_only_two_public_phases() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], cwd=ROOT, check=True)
    source = _source()
    case = source[source.rindex('case "$1" in') :]
    assert "prepare-runtime) prepare_runtime ;;" in case
    assert "run-localization) run_localization ;;" in case
    assert "cleanup)" not in case
    assert "retry)" not in case
    assert "run-acquisition)" not in case


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
    assert result.returncode == 2
    assert "no cleanup" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_attempt_image_gpu_and_authority_are_fixed_and_env_checked() -> None:
    source = _source()
    assert "readonly P28_ATTEMPT_ROOT=/secure/p28/attempt-20260906-01" in source
    assert "readonly P28_CONTAINER=p28-localization-20260906-01" in source
    assert "readonly P28_TRANSPORT_ROOT=/secure/p28/transport-20260906-01" in source
    assert (
        'readonly P28_BOOTSTRAP_VERIFIER="$P28_TRANSPORT_ROOT/verify_p28_control_bundle.py"'
        in source
    )
    assert "readonly P28_CONTROL_REPO=/secure/p28/control-source" in source
    assert "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0" in source
    assert "185e444afc0b44ca0a09b1bde49a6b6fa3973355" in source
    assert "24f4bdac331a57bd7c1b807747d7c7fba253ee5a" in source
    required = _function(source, "require_source_environment", "require_runtime_environment")
    assert "require_var P28_ATTEMPT_ID" in required
    assert '"$P28_ATTEMPT_ID" == "$P28_REQUIRED_ATTEMPT_ID"' in required
    assert "require_var P28_GPU_UUID" in required
    assert '"$P28_GPU_UUID" == "$P28_REQUIRED_GPU_UUID"' in required


def test_source_bundle_receipt_and_p27_reconstruction_precede_attempt_write() -> None:
    source = _source()
    prepare = _function(source, "prepare_runtime", "stage_source")
    source_replay = prepare.index("verify_source_transport_unlogged")
    p27_reconstruction = prepare.index("run_p27_reconstruction_unlogged")
    attempt_write = prepare.index("sudo /usr/bin/install -d")
    container_write = prepare.index("sudo docker run --detach")
    assert source_replay < p27_reconstruction < attempt_write < container_write
    assert prepare.index("run_p28_reconstruction_unlogged") < attempt_write
    assert prepare.count("verify_source_transport_unlogged") == 1
    assert prepare.count("run_p27_reconstruction_unlogged") == 1
    assert "run_logged " not in prepare[:attempt_write]
    assert "capture_new " not in prepare[:attempt_write]

    replay = _function(
        source, "verify_source_transport_unlogged", "verify_runtime_review_transport_unlogged"
    )
    assert "--phase source" in replay
    assert '/usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER"' in replay
    assert '--bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER"' in replay
    assert '--expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256"' in replay
    assert '"$P28_SOURCE_BUNDLE"' in replay
    assert '"$P28_SOURCE_CLOSURE"' in replay
    assert '"$P28_SOURCE_RECEIPT"' in replay
    assert "--verify-existing-receipt" in replay
    assert "--receipt-output" not in replay
    assert '"$P28_EXPECTED_SOURCE_RECEIPT_SHA256"' in replay


def test_runtime_review_receipt_replay_precedes_localization_marker() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    runtime_replay = run.index("verify_runtime_review_transport_unlogged")
    marker = run.index('write_new_status "$invocation_marker" run-localization')
    ingestion = run.index("run_logged p28-p26-failure-ingestion")
    localization = run.index("run_logged p28-localization sudo docker exec")
    assert runtime_replay < marker < ingestion < localization
    assert run.index("run_p27_reconstruction_unlogged") < marker
    assert run.index("validate_reviewed_runtime_artifacts") < marker

    replay = _function(source, "verify_runtime_review_transport_unlogged", "prepare_runtime")
    assert "--phase runtime-review" in replay
    assert '/usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER"' in replay
    assert '--bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER"' in replay
    assert '--expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256"' in replay
    assert '"$P28_RUNTIME_REVIEW_BUNDLE"' in replay
    assert '"$P28_RUNTIME_REVIEW_CLOSURE"' in replay
    assert '"$P28_RUNTIME_REVIEW_RECEIPT"' in replay
    assert "--expected-source-commit" in replay
    assert "--expected-source-tree" in replay
    assert "--verify-existing-receipt" in replay
    assert "--receipt-output" not in replay


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
    assert prepare.count("sudo docker run --detach") == 1
    assert "--network none" in prepare
    assert "--read-only" in prepare
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824" in prepare
    assert "freeze-runtime" in prepare
    assert "run_p28_bundle_complete_localization_bridge.py" not in prepare


def test_persistent_authority_checkout_rejects_git_object_indirections() -> None:
    source = _source()
    authority = _function(source, "validate_authority_and_inputs", "validate_control_sources")
    assert "rev-parse --is-shallow-repository" in authority
    assert '"$P28_AUTHORITY_REPO/.git/objects/info/alternates"' in authority
    assert '"$P28_AUTHORITY_REPO/.git/info/grafts"' in authority
    assert '! -e "$authority_indirection" && ! -L "$authority_indirection"' in authority
    assert "for-each-ref" in authority
    assert "refs/replace/" in authority
    assert "P28 authority checkout has forbidden replace refs" in authority
    assert "extensions\\.partialclone" in authority
    assert "promisor|partialclonefilter" in authority
    assert "forbidden promisor or partial-clone configuration" in authority


def test_prepare_stops_at_freeze_and_has_no_cleanup_or_retry_commands() -> None:
    source = _source()
    forbidden = re.compile(r"\bdocker\s+(?:rm|stop|start|restart|kill)\b")
    assert forbidden.search(source) is None
    prepare = _function(source, "prepare_runtime", "stage_source")
    assert "p28_cuda_runtime_lock.json" in prepare
    assert "p28_host_attestation.json" in prepare
    assert "git commit" not in prepare
    assert "git add" not in prepare
    assert "install -m 0644" not in prepare.split("freeze-runtime", 1)[1]


def test_runtime_review_requires_direct_child_and_exact_two_file_delta() -> None:
    source = _source()
    source_history = _function(
        source, "validate_source_freeze_history", "validate_runtime_review_history"
    )
    assert '"$P28_SOURCE_FREEZE_COMMIT^{tree}"' in source_history
    assert '"$P28_SOURCE_FREEZE_TREE"' in source_history
    assert "$P28_SOURCE_FREEZE_COMMIT $P28_TERMINAL_P27_COMMIT" in source_history
    assert "terminal P27 commit/tree binding differs" in source_history
    review = _function(source, "validate_runtime_review_history", "run_contract_reconstruction")
    assert "$P28_RUNTIME_REVIEW_COMMIT $P28_SOURCE_FREEZE_COMMIT" in review
    assert review.count("A\\texperiments/training/p28_cuda_runtime_lock.json") == 1
    assert review.count("A\\texperiments/training/p28_host_attestation.json") == 1
    assert "validate_source_freeze_history" in review


def test_executing_orchestrator_is_bound_to_reviewed_control_bytes() -> None:
    source = _source()
    validation = _function(source, "validate_control_sources", "validate_source_freeze_history")
    assert "${BASH_SOURCE[0]}" in validation
    assert '"$executing_orchestrator" == "$(/usr/bin/realpath -- "$orchestrator")"' in validation
    assert "executing P28 host orchestrator bytes differ" in validation
    transport_validation = _function(
        source, "validate_transport_execution_sources", "verify_source_transport_unlogged"
    )
    assert "${BASH_SOURCE[0]}" in transport_validation
    assert "executing P28 host orchestrator bytes differ before transport replay" in (
        transport_validation
    )
    assert '"$(sha256_file "$P28_BOOTSTRAP_VERIFIER")"' in transport_validation
    assert '"$(/usr/bin/stat -c \'%a\' "$P28_BOOTSTRAP_VERIFIER")" == "600"' in (
        transport_validation
    )


def test_run_stages_exactly_four_hash_bound_sources_in_existing_evidence_mount() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    assert run.count("stage_source ") == 4
    assert "run_p27_cuda_deleted_mapping_localization.py" in run
    assert "ingest_p26_trace_off_a_failure.py" in run
    assert "p23_deterministic_cuda_shadow_trace.py" in run
    assert "sanitize_p27_cuda_deleted_mapping_localization.py" in run
    assert "install -m 0400 -o root -g root" in _function(
        source, "stage_source", "validate_live_runtime"
    )
    assert "docker cp" not in run
    assert "--mount " not in run


def test_p26_ingestion_is_host_root_one_read_bridge_before_cuda_localizer() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    ingestion = run.index(
        'run_logged p28-p26-failure-ingestion sudo /usr/bin/python3 "$staged_ingester"'
    )
    localization = run.index("run_logged p28-localization sudo docker exec")
    sanitizer = run.index(
        'run_logged p28-localization-sanitizer sudo /usr/bin/python3 "$staged_sanitizer"'
    )
    assert ingestion < localization < sanitizer
    assert '--source "$P28_P26_NATIVE"' in run
    assert '--output-root "$P28_HOST_EVIDENCE"' in run
    assert 'sha256_file "$P28_P26_NATIVE"' not in run
    assert 'root_sha256_file "$P28_P26_NATIVE"' not in run
    assert "before localization" in run


def test_localizer_is_one_shot_and_preserves_nonzero_status_before_sanitizing() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    assert run.count("run_logged p28-localization sudo docker exec") == 1
    assert "p28-run-localization.invoked" in run
    assert "retry is forbidden" in run
    command = run[run.index("run_logged p28-localization sudo docker exec") :]
    assert command.count("--expected-p25-source-sha256") == 1
    assert command.count("--expected-p25-contract-sha256") == 1
    status_write = run.index("p28-localization.exit-status.txt")
    sanitizer = run.index(
        'run_logged p28-localization-sanitizer sudo /usr/bin/python3 "$staged_sanitizer"'
    )
    assert status_write < sanitizer
    assert '--output-root "$P28_HOST_EVIDENCE"' in run
    assert '--runner-source "$staged_localizer"' in run
    assert '--repository-root "$P28_CONTROL_REPO"' in run
    assert 'if [[ ! -e "$native"' in run
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
    assert "validate_live_runtime p28-pre-ingestion" in run
    assert "validate_live_runtime p28-pre-localization" in run
    assert "sudo /usr/bin/cmp -s" in _function(
        source, "validate_reviewed_runtime_artifacts", "assert_staged_source"
    )


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
