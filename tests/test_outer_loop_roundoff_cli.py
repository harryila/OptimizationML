from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_generator(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_outer_loop_roundoff.py"), *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p10_cli_records_exact_composition_guards_and_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_generator(root)

    assert payload["schema_version"] == "passive-muon-outer-loop-roundoff-certificate-v1"
    scope = payload["claim_scope"]
    assert "q10" in scope["guarantee"]
    assert "4096x11008" in scope["matrix_domain"]
    assert "logical W=high+middle+low" in scope["gradient_boundary"]
    assert any("unconditional invariant" in item for item in scope["not_claimed"])
    assert any("full-parameter ISS" in item for item in scope["not_claimed"])

    operator = payload["operator"]
    assert operator["floor_c"]["exact"] == "1"
    assert operator["epsilon"].startswith("none")
    assert operator["newton_schulz_iteration_count"] == 5
    assert operator["polynomial_coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    arithmetic = payload["arithmetic_contract"]
    assert arithmetic["rounding"].endswith("named FP32 operation")
    assert arithmetic["gradual_underflow_required"]
    assert not arithmetic["ftz_daz_allowed"]
    assert not arithmetic["fma_allowed"]
    assert arithmetic["seed"].startswith("none")
    assert "r_tilde_m" in arithmetic["raw_port_formulas"]

    reduction = payload["port_reduction"]
    assert reduction["shape"] == [4_096, 11_008]
    assert reduction["sqrt_entries_upper"] == 6_715
    assert reduction["form"].endswith("slope*sqrt(V)+intercept")
    assert Fraction(reduction["p9_binary32_operator_error"]["slope"]["exact"]) == Fraction(
        60_114_853, 549_755_813_888
    )
    assert set(reduction["affine_storage_envelopes"]) == {
        "momentum_port_r_m",
        "signal_port_r_s",
        "actual_fp32_signal",
        "p9_operator_output",
        "master_update_port_r_W",
        "effective_p7_gradient_error",
        "effective_p7_operator_error",
    }

    locked = payload["locked_certificate"]
    assert Fraction(locked["rate_q10"]["exact"]) == Fraction(549_700_907_325, 549_755_813_888) < 1
    assert Fraction(locked["constant_forcing_D10"]["exact"]) == Fraction(
        2_162_331, 1_099_511_627_776
    )
    assert Fraction(locked["function_gap_ultimate"]["exact"]) < 1

    guard = payload["guard_closure"]
    assert guard["storage_radius"]["exact"] == "1"
    assert guard["certificate_operator_output_max_abs"]["exact"] == str(2**15)
    assert guard["runtime_operator_output_max_abs"]["exact"] == str(2**16)
    assert guard["rounded_step_max_abs"]["exact"] == "2"
    assert guard["rounded_step_quantity"] == "entrywise maxabs"
    assert guard["ema_intermediates_are_finite"]
    assert all(guard["middle_low_invariant_checks"].values())
    assert "conditional" in guard["high_word_guard"]

    assert payload["actual_p9_fp32_stalling_witness"]["certified"]
    assert payload["audit"]["all_exact_checks_passed"]
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/finite_precision_outer_loop.py" in snapshot
    assert "src/passive_muon/outer_loop_roundoff_certificate.py" in snapshot
    assert "src/passive_muon/specs.py" in snapshot
    assert "src/passive_muon/upstream_momentum.py" in snapshot
    assert "src/passive_muon/mixed_precision_certificate.py" in snapshot
    assert "src/passive_muon/floored_certificate.py" in snapshot
    assert "scripts/reconstruct_outer_loop_roundoff.py" in snapshot
    assert "theory/finite_precision_outer_loop_certificate.md" in snapshot
    assert "theory/audits/P10_HUMAN_PROOF_AUDIT.md" in snapshot
    assert "results/summaries/P10_RESULTS.md" in snapshot
    assert ".github/workflows/p10-finite-precision.yml" in snapshot
    assert "tests/test_result_manifests.py" in snapshot
    assert all(len(digest) == 64 for digest in snapshot.values())


def test_p10_cli_output_file_matches_stdout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p10.json"
    payload = _run_generator(root, "--output", str(output))

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_checked_p10_source_snapshot_is_current_and_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical_path = root / "results/summaries/outer_loop_roundoff_certificate.json"
    assert canonical_path.is_file(), "the committed P10 exact certificate is mandatory"

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    git = canonical["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p10-finite-precision-outer-loop"
    assert git["dirty"] is False
    snapshot = canonical["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        source = root / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
