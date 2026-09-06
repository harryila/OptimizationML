from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rehearse_p30_uid1000_root_handoff.sh"
WORKFLOW = ROOT / ".github/workflows/p30-uid1000-root-engineering-rehearsal.yml"
DOCUMENT = ROOT / "experiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_rehearsal_script_has_valid_shell_syntax_and_fails_closed_off_actions() -> None:
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True, cwd=ROOT)
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin"},
        text=True,
    )
    assert result.returncode != 0
    assert "must be absolute" in result.stderr


def test_rehearsal_uses_real_uid_boundary_and_real_reconstructors() -> None:
    source = _source()
    assert "--reuid=1000 --regid=1000 --clear-groups --no-new-privs" in source
    assert '"$controller_uid" != 0 && "$controller_uid" != 1000' in source
    assert "client controller must be distinct from root and UID 1000" in source
    assert '"$P30_BOOTSTRAP_VERIFIER"' in source
    assert "/usr/bin/python3 -I -S" in source
    assert "1000 1000 0 0" in source
    assert '"p30_contract": 15' in source
    assert '"p29_contract": 15' in source
    assert '"p29_terminal_outcome": 11' in source
    assert '"p28_contract": 12' in source
    assert '"p28_terminal_outcome": 10' in source
    assert '"p27_contract": 23' in source
    assert "monkeypatch" not in source
    assert "run_fixture_reconstruction" not in source
    assert "top-level P30 reconstruction was stubbed" in source


def test_root_executes_commit_bound_seal_program_not_uid1000_checkout_code() -> None:
    source = _source()
    function = source[
        source.index("seal_reviewed_orchestrator_as_root() {") : source.index(
            "verify_root_seal() {"
        )
    ]
    assert 'clean_git -C "$SCRIPT_ROOT" show' in function
    assert '"$SOURCE_COMMIT:experiments/training/' in function
    assert "$P30_CONTROL_REPO/experiments/training" not in function
    assert "/usr/bin/sudo -n /usr/bin/env -i" in function


def test_preverifier_git_reads_ignore_replacements_and_ambient_config() -> None:
    source = _source()
    assert "GIT_CONFIG_GLOBAL=/dev/null" in source
    assert "GIT_CONFIG_NOSYSTEM=1" in source
    assert "GIT_NO_REPLACE_OBJECTS=1" in source
    assert "GIT_TERMINAL_PROMPT=0" in source
    assert "the hosted checkout contains local Git include directives" in source


def test_reviewed_orchestrator_tracked_mode_matches_verifier_authority() -> None:
    record = subprocess.run(
        [
            "git",
            "ls-files",
            "--stage",
            "--",
            "scripts/run_p30_umask_bound_control_seal.sh",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    assert record.startswith("100644 ")


def test_success_requires_rehearsal_manifest_finalization() -> None:
    source = _source()
    function = source[source.index("write_manifest() {") : source.index("run_logged() {")]
    assert 'if [[ "$overall_status" -eq 0 && "$manifest_status" -ne 0 ]]' in function
    assert "manifest finalization failed" in function
    assert "exit 125" in function


def test_logged_shell_function_cannot_mask_an_intermediate_failure(tmp_path: Path) -> None:
    source = _source()
    function = source[source.index("finalize_logged_failure() {") : source.index("as_uid1000() {")]
    log_root = tmp_path / "logs"
    marker = tmp_path / "masked-success"
    log_root.mkdir()
    program = f"""
set -Eeuo pipefail
LOG_ROOT={str(log_root)!r}
{function}
probe() {{
  false
  printf 'should-not-run\\n' >{str(marker)!r}
}}
run_logged 00-probe probe
"""
    result = subprocess.run(
        ["bash", "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert not marker.exists()
    assert (log_root / "00-probe.exit-status.txt").read_text(encoding="ascii") != "0\n"
    assert "nvidia-smi" not in source
    assert "docker " not in source
    assert "prepare-runtime" not in source
    assert "run-localization" not in source


def test_successful_logged_function_preserves_parent_state(tmp_path: Path) -> None:
    source = _source()
    function = source[source.index("finalize_logged_failure() {") : source.index("as_uid1000() {")]
    log_root = tmp_path / "logs"
    log_root.mkdir()
    program = f"""
set -Eeuo pipefail
LOG_ROOT={str(log_root)!r}
{function}
VALUE=before
probe() {{
  VALUE=after
}}
run_logged 00-probe probe
[[ "$VALUE" == after ]]
"""
    result = subprocess.run(
        ["bash", "-c", program],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (log_root / "00-probe.exit-status.txt").read_bytes() == b"0\n"


def test_receipt_is_created_reviewed_replayed_then_root_sealed() -> None:
    source = _source()
    main = source[source.index("main() {") :]
    ordered = [
        "05-source-verifier-create",
        "06-source-receipt-review",
        "07-source-verifier-replay",
        "08-root-orchestrator-seal",
        "09-root-seal-verification",
        "10-final-evidence-boundary",
    ]
    positions = [main.index(value) for value in ordered]
    assert positions == sorted(positions)
    assert "--receipt-output" in source
    assert "--verify-existing-receipt" in source
    assert "--expected-receipt-sha256" in source


def test_all_client_boundaries_have_stream_status_triplets_and_manifest() -> None:
    source = _source()
    assert "$label.stdout.log" in source
    assert "$label.stderr.log" in source
    assert "$label.exit-status.txt" in source
    assert "rehearsal-manifest.json" in source
    assert '"scientific_acquisition": False' in source
    assert '"gpu_used": False' in source
    assert '"attempt_root_created": attempt_root_created' in source
    assert '"execution_root_created": execution_root_created' in source
    assert '"real_host_attempt_identity_consumed": False' in source
    assert "successful rehearsal stream inventory differs" in source
    assert "successful rehearsal command failed" in source
    assert "set -o noclobber" in source
    assert 'exec 8>"$stdout_path"' in source
    assert 'exec 9>"$stderr_path"' in source
    for label in (
        "02-namespace-provision",
        "05-source-verifier-create",
        "06-source-receipt-review",
        "08-root-orchestrator-seal",
    ):
        assert f"run_logged {label}" in source


def test_workflow_rehearses_on_temporary_branch_or_manual_dispatch() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "push:" in workflow
    assert "p30-engineering-rehearsal" in workflow
    assert "pull_request:" not in workflow
    assert "${{ inputs.source_commit || github.sha }}" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "persist-credentials: false" in workflow
    assert "P30_REHEARSAL_LOG_ROOT: ${{ runner.temp }}/" in workflow
    assert "if: ${{ always() }}" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "retention-days: 90" in workflow
    assert "nvidia-smi" not in workflow
    assert "docker" not in workflow


def test_document_separates_rehearsal_from_scientific_acquisition() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    assert "logged engineering rehearsal" in document
    assert "not a frozen scientific acquisition" in document
    assert "not another research phase" in document
    assert "no GPU" in document
    assert "Historical failed rehearsals are retained rather than overwritten" in document
    assert "mirrors P30's frozen `/secure/p30`" in document
    assert "never creates the frozen attempt root" in document
    assert "cannot consume the distinct real-host attempt identity" in document
