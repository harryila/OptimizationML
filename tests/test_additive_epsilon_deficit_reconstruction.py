from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest


def _run(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_additive_epsilon_deficit.py"),
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def _generate(root: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_additive_epsilon_deficit.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture(scope="module")
def independent_payload(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    missing = tmp_path_factory.mktemp("p12-reconstruction") / "missing.json"
    return _run(root, "--canonical", str(missing))


@pytest.fixture(scope="module")
def fresh_primary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path_factory.mktemp("p12-primary") / "fresh.json"
    _generate(root, output)
    return output


def test_reconstruction_is_standard_library_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_additive_epsilon_deficit.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])
    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint(
        {"flint", "numpy", "scipy", "sympy", "torch", "mpmath", "cvxpy"}
    )


def test_independent_decimal_and_exact_reconstruction_closes(
    independent_payload: dict[str, object],
) -> None:
    assert independent_payload["all_internal_checks_passed"] is True
    assert independent_payload["all_checks_passed"] is False
    assert independent_payload["canonical_comparison"]["status"] == "not_found"
    reconstruction = independent_payload["reconstruction"]
    assert all(reconstruction["checks"].values())
    interval = reconstruction["decimal_interval_replay"]
    assert interval["precision_decimal_digits"] == 90
    assert interval["global"]["leaf_count"] == 24_338
    assert interval["global"]["maximum_power"] == 32
    assert interval["global"]["trace_sha256"] == (
        "5224e1af1fed5fb677f47de03a8f0f11701aef2a90618f438266e664915dfce1"
    )
    assert [item["leaf_count"] for item in interval["prefixes"]] == [276, 367, 909]


def test_independent_reconstruction_recovers_bracket_and_deployed_scale(
    independent_payload: dict[str, object],
) -> None:
    reconstruction = independent_payload["reconstruction"]
    assert Fraction(reconstruction["unit_epsilon_upper"]) == Fraction(
        6_602_082_433_275_499_863, 41_641_817_600_000_000
    )
    assert Fraction(reconstruction["pair"]["strict_lower"]) == Fraction(98_823_281, 625_000)
    assert reconstruction["pair"]["sha256"] == (
        "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"
    )
    assert Fraction(reconstruction["deployed_strict_lower"]) == 1_581_172_496
    assert Fraction(reconstruction["deployed_upper"]) > 1_585_000_000


def test_reconstruction_matches_a_fresh_primary_certificate(fresh_primary: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run(root, "--canonical", str(fresh_primary), "--require-canonical")
    assert payload["all_checks_passed"] is True
    comparison = payload["canonical_comparison"]
    assert comparison["status"] == "matched"
    assert comparison["all_exact_fields_match"] is True
    assert all(comparison["comparisons"].values())
    assert {
        "claim_scope",
        "operator_scope",
        "deficit_scope",
        "operator_steps_and_coefficients",
        "bracket",
        "pair_parameters",
        "origin_gain",
        "repair_comparison",
        "epsilon_evaluations",
        "all_primary_arb_passes",
        "source_snapshot",
    } <= comparison["comparisons"].keys()


def test_require_canonical_rejects_corrupted_headline_fields(
    fresh_primary: Path, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(fresh_primary.read_text(encoding="utf-8"))
    payload["operator"]["coefficients_exact"]["a"] = "1"
    payload["arb_interval_certificate"]["passes"][1]["global_derivative_cover"][
        "ordered_leaf_trace_sha256"
    ] = "0" * 64
    corrupted = tmp_path / "corrupted.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_additive_epsilon_deficit.py"),
            "--canonical",
            str(corrupted),
            "--require-canonical",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr


def test_reconstruction_matches_committed_certificate_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/additive_epsilon_deficit_certificate.json"
    if not canonical.exists():
        pytest.skip("committed additive-epsilon certificate has not landed yet")
    payload = _run(root, "--require-canonical")
    assert payload["all_checks_passed"] is True
    assert payload["canonical_comparison"]["status"] == "matched"


def test_require_canonical_rejects_a_missing_artifact(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_additive_epsilon_deficit.py"),
            "--canonical",
            str(tmp_path / "missing.json"),
            "--require-canonical",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr
