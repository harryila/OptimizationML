from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_generator(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_scalable_mixed_precision.py"), *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p9_cli_replays_shape_bounds_obstructions_and_p7_rates() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_generator(root)

    assert payload["schema_version"] == "passive-muon-scalable-mixed-precision-certificate-v1"
    scope = payload["claim_scope"]
    assert "A_rc" in scope["guarantee"]
    assert "seven fixed Transformer shapes" in scope["matrix_domain"]
    assert "no additive epsilon" in scope["normalization"]
    assert "not a global" in scope["qualified_frontier"]
    assert any("production-throughput" in item for item in scope["not_claimed"])
    assert any("state compression" in item for item in scope["not_claimed"])
    assert any("FP32 EMA/Nesterov" in item for item in scope["not_claimed"])

    contract = payload["implementation_contract"]
    assert contract["orientation"].startswith("transpose once")
    assert "balanced FP32 norm" in contract["normalizer"]
    assert contract["stage_dag"] == [
        "G=balanced_fp32(X X^T)",
        "T=fl32(c32*G+b32*I)",
        "D=balanced_fp32(T G)",
        "E=fl32(D+a32*I)",
        "Y=balanced_fp32(E X)",
    ]
    assert "low=RN_bf16" in contract["boundary"]
    assert contract["upper_rationalization"].endswith("denominator 2^40")

    certificate = payload["certificate"]
    assert certificate["upper_grid_denominator"] == str(2**40)
    assert certificate["locked_stages"] == 5
    assert certificate["locked_max_entries"] == 2**52
    assert certificate["rank_gates"] == {
        "ordinary_one_term": 71,
        "ideal_two_term": 4_693_632,
        "sterbenz_free_two_term": 4_656_751,
    }
    serial = certificate["serial_normalizer_obstruction"]
    assert serial["shape"] == [4_096, 11_008]
    assert serial["returned_singular_value_squared"] == "43/16"
    assert serial["leaves_spectral_tube"]

    shapes = certificate["shape_audits"]
    assert len(shapes) == 7
    assert all(item["certified"] for item in shapes)
    assert all(item["polynomial"]["certified"] for item in shapes)
    assert all(item["normalizer"]["certified"] for item in shapes)
    assert all(len(item["stages"]) == 5 for item in shapes)
    assert all(
        Fraction(item["p7"]["values"]["rate"]) == Fraction(137_425_214_491, 137_438_953_472) < 1
        for item in shapes
    )
    largest = next(item for item in shapes if item["shape"] == [4_096, 11_008])
    assert Fraction(largest["operator_bound"]["real_slope"]) == Fraction(
        102_465_557, 549_755_813_888
    )
    assert Fraction(largest["operator_bound"]["real_intercept"]) == Fraction(
        2_179_083_213_031, 1_099_511_627_776
    )
    assert Fraction(largest["p7"]["values"]["function_gap_ultimate"]) == Fraction(
        798_350_562_999, 1_099_511_627_776
    )

    frontier = payload["qualified_negative_controls"]["audited_4_to_1_case"]
    assert frontier == {
        "shape": [4_608, 18_432],
        "failed_checks": ["stage_5_input_in_spectral_tube"],
        "interpretation": "specific recurrence boundary, not a global rank frontier",
    }
    assert payload["audit"]["all_representative_exact_checks_passed"]

    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert "src/passive_muon/scalable_mixed_precision_certificate.py" in snapshot
    assert "src/passive_muon/mixed_precision_certificate.py" in snapshot
    assert "src/passive_muon/floored_certificate.py" in snapshot
    assert "src/passive_muon/pl_convergence.py" in snapshot
    assert "scripts/reconstruct_scalable_mixed_precision.py" in snapshot
    assert "theory/scalable_mixed_precision_certificate.md" in snapshot
    assert "theory/audits/P9_HUMAN_PROOF_AUDIT.md" in snapshot
    assert all(len(digest) == 64 for digest in snapshot.values())


def test_p9_cli_output_file_matches_stdout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p9.json"
    payload = _run_generator(root, "--output", str(output))

    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_checked_p9_source_snapshot_is_current_when_canonical_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical_path = root / "results/summaries/scalable_mixed_precision_certificate.json"
    if not canonical_path.exists():
        return

    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    git = canonical["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p9-scalable-mixed-precision"
    assert git["dirty"] is False
    snapshot = canonical["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        source = root / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
