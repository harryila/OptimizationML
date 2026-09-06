from __future__ import annotations

import json
import re
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


def test_runtime_review_materialization_preserves_git_and_physical_modes() -> None:
    source = _source()
    start = source.index("author_runtime_review_as_uid1000() {")
    end = source.index("verify_final_evidence_boundary() {")
    rehearsal = source[start:end]

    assert "as_uid1000_umask077" in rehearsal
    assert "umask 077" in source[source.index("as_uid1000_umask077() {") : start]
    assert "p30_cuda_runtime_lock.json" in rehearsal
    assert "p30_host_attestation.json" in rehearsal
    assert "engineering_rehearsal_not_scientific_acquisition" in rehearsal
    assert "runtime-review staged delta differs" in rehearsal
    assert "awk '$1 != \"100644\" {print}'" in rehearsal
    assert "materialized runtime-lock authority differs" in rehearsal
    assert "materialized host-attestation authority differs" in rehearsal
    assert "1000:1000:600:1" in rehearsal
    assert "runtime-control inode changed across root handoff" in rehearsal
    assert "runtime-control file has a forbidden ACL" in rehearsal
    assert "root-observed runtime-review parent differs" in rehearsal
    assert "root-observed runtime-review delta differs" in rehearsal
    assert "runtime_review_tracked_modes=100644,100644" in rehearsal
    assert "runtime_review_materialized_authority=1000:1000:0600" in rehearsal
    assert "/usr/bin/sudo -n /usr/bin/python3 -I -S" in rehearsal


def test_runtime_review_source_closure_is_checked_by_its_uid1000_owner() -> None:
    source = _source()
    function = source[
        source.index("author_runtime_review_as_uid1000() {") : source.index(
            "materialize_runtime_review_as_uid1000() {"
        )
    ]

    # The controller cannot traverse the UID-1000-owned mode-0700 transport
    # directory.  Availability checks must cross the same identity boundary as
    # the clone that immediately consumes the closure.
    assert 'as_uid1000 /usr/bin/test -d "$P30_SOURCE_CLOSURE"' in function
    assert 'as_uid1000 /usr/bin/test ! -L "$P30_SOURCE_CLOSURE"' in function
    assert '[[ -d "$P30_SOURCE_CLOSURE"' not in function


def test_runtime_review_rehearsal_is_synthetic_and_outside_frozen_checkout() -> None:
    source = _source()
    final = source[source.index("verify_final_evidence_boundary() {") : source.index("main() {")]
    assert 'readonly P30_REHEARSED_RUNTIME_ROOT="$STAGE_ROOT/runtime-review"' in source
    assert "synthetic runtime-review artifacts entered the authoritative source checkout" in final
    assert '"cuda_initialized": False' in source
    assert '"real_host_attempt_identity_consumed": False' in source


def test_root_handoff_binding_preserves_stat_percent_a_octal_spelling() -> None:
    source = _source()
    handoff = source[
        source.index("verify_runtime_review_handoff_as_root() {") : source.index(
            "verify_final_evidence_boundary() {"
        )
    ]
    assert 'format(binding_fields[4], "o")' in handoff
    assert '":".join(str(value) for value in fields(after))' not in handoff


def test_runtime_review_rehearses_the_exact_37_name_root_environment() -> None:
    source = _source()
    block = source[
        source.index("local -a localization_environment=(") : source.index(
            "printf 'localization_root_environment_handoff_count=37"
        )
    ]
    names = re.findall(r'^\s+"(P30_[A-Z0-9_]+)=', block, flags=re.MULTILINE)
    expected = {
        "P30_ATTEMPT_ID",
        "P30_GPU_UUID",
        "P30_AUTHORITY_REPO",
        "P30_NANOGPT_HOST",
        "P30_MUON_HOST",
        "P30_DATA_HOST",
        "P30_SOURCE_FREEZE_COMMIT",
        "P30_SOURCE_FREEZE_TREE",
        "P30_EXPECTED_SOURCE_BUNDLE_SHA256",
        "P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT",
        "P30_EXPECTED_SOURCE_RECEIPT_SHA256",
        "P30_EXPECTED_CONTRACT_SHA256",
        "P30_EXPECTED_ORCHESTRATOR_SHA256",
        "P30_EXPECTED_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_BUNDLE_VERIFIER_SHA256",
        "P30_EXPECTED_P29_CONTRACT_SHA256",
        "P30_EXPECTED_P29_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_P29_OUTCOME_SHA256",
        "P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_P28_CONTRACT_SHA256",
        "P30_EXPECTED_P28_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_P28_OUTCOME_SHA256",
        "P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_P27_CONTRACT_SHA256",
        "P30_EXPECTED_P27_RECONSTRUCTOR_SHA256",
        "P30_EXPECTED_P27_LOCALIZER_SHA256",
        "P30_EXPECTED_INGESTER_SHA256",
        "P30_EXPECTED_P23_CORE_SHA256",
        "P30_EXPECTED_P27_SANITIZER_SHA256",
        "P30_RUNTIME_REVIEW_COMMIT",
        "P30_RUNTIME_REVIEW_TREE",
        "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256",
        "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT",
        "P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256",
        "P30_EXPECTED_RUNTIME_LOCK_SHA256",
        "P30_EXPECTED_HOST_ATTESTATION_SHA256",
        "P30_EXPECTED_CONTAINER_ID",
    }
    assert len(names) == len(set(names)) == 37
    assert set(names) == expected
    assert "/usr/bin/sudo -n /usr/bin/env -i" in block
    assert 'actual={name for name in os.environ if name.startswith("P30_")}' in block
    assert "37 if runtime_root_handoff_succeeded else 0" in source


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
        "10-runtime-review-authoring",
        "11-runtime-review-materialization",
        "12-runtime-review-root-handoff",
        "13-final-evidence-boundary",
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
    assert '"runtime_review_checkout_rehearsed": runtime_materialization_succeeded' in source
    assert '"runtime_review_expected_tracked_mode": "100644"' in source
    assert '"runtime_review_expected_materialized_owner": "1000:1000"' in source
    assert '"runtime_review_expected_materialized_mode_octal": "0600"' in source
    assert (
        '"runtime_review_tracked_mode": "100644" if runtime_authoring_succeeded else None' in source
    )
    assert '"1000:1000" if runtime_materialization_succeeded else None' in source
    assert '"0600" if runtime_materialization_succeeded else None' in source
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
        "10-runtime-review-authoring",
        "11-runtime-review-materialization",
        "12-runtime-review-root-handoff",
    ):
        assert f"run_logged {label}" in source


def test_early_failure_manifest_does_not_claim_unreached_runtime_observations(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    stage_root = tmp_path / "stage"
    environment = {
        "PATH": "/usr/bin:/bin",
        "P30_REHEARSAL_LOG_ROOT": str(log_root),
        "P30_REHEARSAL_STAGE_ROOT": str(stage_root),
        "P30_REHEARSAL_SOURCE_COMMIT": "0" * 40,
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        cwd=ROOT,
        env=environment,
        text=True,
    )

    assert result.returncode != 0
    manifest = json.loads((log_root / "rehearsal-manifest.json").read_text(encoding="utf-8"))
    assert manifest["overall_exit_status"] != 0
    assert manifest["runtime_review_checkout_rehearsed"] is False
    assert manifest["runtime_review_root_handoff_verified"] is False
    assert manifest["localization_root_environment_handoff_count"] == 0
    assert manifest["runtime_review_tracked_mode"] is None
    assert manifest["runtime_review_materialized_owner"] is None
    assert manifest["runtime_review_materialized_mode_octal"] is None
    assert manifest["runtime_review_expected_tracked_mode"] == "100644"
    assert manifest["runtime_review_expected_materialized_owner"] == "1000:1000"
    assert manifest["runtime_review_expected_materialized_mode_octal"] == "0600"


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
    assert (
        "P30_REHEARSAL_STAGE_ROOT: "
        "/tmp/p30-engineering-rehearsal-${{ github.run_id }}-${{ github.run_attempt }}" in workflow
    )
    assert "P30_REHEARSAL_STAGE_ROOT: ${{ runner.temp }}/" not in workflow
    assert "if: ${{ always() }}" in workflow
    assert "actions/upload-artifact@v7" in workflow
    assert "retention-days: 90" in workflow
    assert "nvidia-smi" not in workflow
    assert "docker" not in workflow


def test_rehearsal_executes_the_production_mode_bridge_as_root() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "verify_native_mode_bridge_as_root()" in source
    bridge = source.split("verify_native_mode_bridge_as_root()", 1)[1].split(
        "\nverify_root_seal()", 1
    )[0]
    assert "set_native_private_mode_for_p27_sanitizer()" in bridge
    assert "/usr/bin/sudo -n /usr/bin/python3 -I -S" in bridge
    assert '"$digest" "$byte_count" "$device" "$inode"' in bridge
    assert "root_native_mode_bridge=passed" in bridge
    root_seal = source.split("verify_root_seal()", 1)[1].split(
        "\nauthor_runtime_review_as_uid1000()", 1
    )[0]
    assert "verify_native_mode_bridge_as_root" in root_seal


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
    assert "tracked as mode `100644`" in document
    assert "physically UID/GID `1000:1000` mode `0600`" in document
    assert "synthetic, credential-free placeholders" in document
    assert "all 37 localization variable names" in document
    assert "actual verifier runtime-review materialization path" in document
