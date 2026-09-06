from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p27_cuda_deleted_mapping_localization.sh"


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
    assert "readonly P27_ATTEMPT_ROOT=/secure/p27/attempt-20260906-01" in source
    assert "readonly P27_CONTAINER=p27-localization-20260906-01" in source
    assert "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0" in source
    assert "185e444afc0b44ca0a09b1bde49a6b6fa3973355" in source
    assert "24f4bdac331a57bd7c1b807747d7c7fba253ee5a" in source
    required = _function(source, "require_source_environment", "require_runtime_environment")
    assert "require_var P27_ATTEMPT_ID" in required
    assert '"$P27_ATTEMPT_ID" == "$P27_REQUIRED_ATTEMPT_ID"' in required
    assert "require_var P27_GPU_UUID" in required
    assert '"$P27_GPU_UUID" == "$P27_REQUIRED_GPU_UUID"' in required


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
    assert "run_p27_cuda_deleted_mapping_localization.py" not in prepare


def test_prepare_stops_at_freeze_and_has_no_cleanup_or_retry_commands() -> None:
    source = _source()
    forbidden = re.compile(r"\bdocker\s+(?:rm|stop|start|restart|kill)\b")
    assert forbidden.search(source) is None
    prepare = _function(source, "prepare_runtime", "stage_source")
    assert "p27_cuda_runtime_lock.json" in prepare
    assert "p27_host_attestation.json" in prepare
    assert "git commit" not in prepare
    assert "git add" not in prepare
    assert "install -m 0644" not in prepare.split("freeze-runtime", 1)[1]


def test_runtime_review_requires_direct_child_and_exact_two_file_delta() -> None:
    source = _source()
    source_history = _function(
        source, "validate_source_freeze_history", "validate_runtime_review_history"
    )
    assert '"$P27_SOURCE_FREEZE_COMMIT^{tree}"' in source_history
    assert '"$P27_SOURCE_FREEZE_TREE"' in source_history
    review = _function(source, "validate_runtime_review_history", "run_contract_reconstruction")
    assert "$P27_RUNTIME_REVIEW_COMMIT $P27_SOURCE_FREEZE_COMMIT" in review
    assert review.count("A\\texperiments/training/p27_cuda_runtime_lock.json") == 1
    assert review.count("A\\texperiments/training/p27_host_attestation.json") == 1
    assert "validate_source_freeze_history" in review


def test_executing_orchestrator_is_bound_to_reviewed_control_bytes() -> None:
    source = _source()
    validation = _function(source, "validate_control_sources", "validate_source_freeze_history")
    assert "${BASH_SOURCE[0]}" in validation
    assert '"$executing_orchestrator" == "$(/usr/bin/realpath -- "$orchestrator")"' in validation
    assert "executing P27 host orchestrator bytes differ" in validation


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
        'run_logged p27-p26-failure-ingestion sudo /usr/bin/python3 "$staged_ingester"'
    )
    localization = run.index("run_logged p27-localization sudo docker exec")
    sanitizer = run.index(
        'run_logged p27-localization-sanitizer sudo /usr/bin/python3 "$staged_sanitizer"'
    )
    assert ingestion < localization < sanitizer
    assert '--source "$P27_P26_NATIVE"' in run
    assert '--output-root "$P27_HOST_EVIDENCE"' in run
    assert 'sha256_file "$P27_P26_NATIVE"' not in run
    assert 'root_sha256_file "$P27_P26_NATIVE"' not in run
    assert "before localization" in run


def test_localizer_is_one_shot_and_preserves_nonzero_status_before_sanitizing() -> None:
    source = _source()
    run = _function(source, "run_localization", "usage")
    assert run.count("run_logged p27-localization sudo docker exec") == 1
    assert "p27-run-localization.invoked" in run
    assert "retry is forbidden" in run
    command = run[run.index("run_logged p27-localization sudo docker exec") :]
    assert command.count("--expected-p25-source-sha256") == 1
    assert command.count("--expected-p25-contract-sha256") == 1
    status_write = run.index("p27-localization.exit-status.txt")
    sanitizer = run.index(
        'run_logged p27-localization-sanitizer sudo /usr/bin/python3 "$staged_sanitizer"'
    )
    assert status_write < sanitizer
    assert '--output-root "$P27_HOST_EVIDENCE"' in run
    assert '--runner-source "$staged_localizer"' in run
    assert '--repository-root "$P27_CONTROL_REPO"' in run
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
    assert "validate_live_runtime p27-pre-ingestion" in run
    assert "validate_live_runtime p27-pre-localization" in run
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
