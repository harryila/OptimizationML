from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_mixed_precision_cli_replays_scope_invariant_and_p7_closure() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_mixed_precision.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-mixed-precision-certificate-v1"
    scope = payload["claim_scope"]
    assert "11/100000" in scope["guarantee"]
    assert "2x2" in scope["guarantee"]
    assert "finite FP32" in scope["matrix_domain"]
    assert "overflow" in scope["overflow_scope"]
    assert any("end-to-end FP32" in item for item in scope["not_claimed"])
    assert any("arbitrary matrix shape" in item for item in scope["not_claimed"])

    contract = payload["implementation_contract"]
    assert contract["shape"] == [2, 2]
    assert Fraction(contract["maximum_input_absolute_value"]["exact"]) == 2**116
    assert contract["stage_count"] == 5
    assert contract["stage_storage"] == "BF16 normalized input plus five BF16 stage outputs"
    assert "two separately rounded FP32 multiplies" in contract["matrix_product"]
    assert "one separately rounded FP32 addition" in contract["matrix_product"]
    assert "not assumed" in contract["fma_policy"]
    assert contract["normalization_rule_target"].endswith("no additive epsilon")

    polynomial = payload["polynomial_range_certificate"]
    assert polynomial["claim"] == "0<=q(s)<121/100 for every 0<=s<=5/4"
    assert polynomial["endpoint_variations"] == [3, 3]
    assert polynomial["same_variation_proves_no_roots"]
    assert len(polynomial["sturm_chain_coefficients_ascending"]) == 6

    normalization = payload["normalization_certificate"]
    assert "max_abs>1/2" in normalization["branch_qualification"]
    assert Fraction(normalization["values"]["normalized_output_error"]["exact"]) < Fraction(
        1, 500_000
    )

    invariant = payload["stage_invariant_certificate"]
    recurrence = invariant["one_stage_recurrence"]
    assert Fraction(recurrence["epsilon_y"]["exact"]) < Fraction(1, 20_000)
    assert Fraction(recurrence["next_frobenius"]["exact"]) < Fraction(7, 4)
    assert Fraction(recurrence["next_spectral"]["exact"]) < Fraction(5, 4)
    assert "pre-BF16" in invariant["important_qualification"]

    error = payload["operator_error_certificate"]
    binary = error["binary32_input_bound"]
    adapter = error["real_boundary_adapter_bound"]
    assert Fraction(binary["A32"]["exact"]) == Fraction(11, 100_000)
    assert Fraction(binary["B32"]["exact"]) == Fraction(347, 100)
    assert Fraction(error["repair_rounding"]["complete_slope"]["exact"]) < Fraction(
        binary["A32"]["exact"]
    )
    assert Fraction(adapter["A_real"]["exact"]) == Fraction(1, 5_000)
    assert Fraction(adapter["B_real"]["exact"]) == Fraction(347, 100)
    assert "RN32" in adapter["formula"]

    closure = payload["p7_closure"]
    assert Fraction(closure["modified_rate_q8"]["exact"]) == Fraction(
        41_597_186_684_695_561,
        41_601_344_000_000_000,
    )
    assert Fraction(closure["constant_forcing"]["exact"]) == Fraction(4_936_769, 819_840_000_000)
    assert Fraction(closure["zero_gradient_noise_function_gap_limsup"]["exact"]) == Fraction(
        462_392_438_350_000_000,
        207_695_315_294_468_001,
    )
    assert "post-operator error port" in closure["outer_loop_scope"]
    assert "real-boundary adapter" in closure["outer_loop_scope"]
    safe_range = closure["safe_range_invariant"]
    assert Fraction(safe_range["initial_storage_upper"]["exact"]) == Fraction(
        closure["values"]["safe_storage_radius"]["exact"]
    )
    assert Fraction(safe_range["forcing_capacity"]["exact"]) == Fraction(
        closure["values"]["safe_forcing_capacity"]["exact"]
    )
    assert Fraction(closure["constant_forcing"]["exact"]) <= Fraction(
        safe_range["forcing_capacity"]["exact"]
    )
    assert "V_0<=H" in safe_range["induction"]

    assert payload["audit"]["all_exact_checks_passed"]
    assert all(payload["audit"]["checks"].values())
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/mixed_precision_certificate.py" in snapshot
    assert "scripts/reconstruct_mixed_precision.py" in snapshot
    assert all(len(value) == 64 for value in snapshot.values())


def test_checked_certificate_source_snapshot_is_current() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical_path = root / "results" / "summaries" / "mixed_precision_certificate.json"
    if not canonical_path.exists():
        return

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    snapshot = canonical["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        source = root / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
