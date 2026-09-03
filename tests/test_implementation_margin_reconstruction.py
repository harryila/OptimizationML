from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

from passive_muon.implementation_margin_certificate import audit_implementation_margin


def _run_reconstruction(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_implementation_margin.py"),
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p11_reconstruction_is_stdlib_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_implementation_margin.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint(
        {"numpy", "scipy", "sympy", "torch", "flint", "mpmath", "cvxpy"}
    )


def test_p11_reconstruction_rebuilds_locked_exact_values() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_reconstruction(root)

    assert payload["schema_version"] == "passive-muon-p11-independent-reconstruction-v1"
    assert payload["implementation_scope"]["project_package_imported"] is False
    assert payload["implementation_scope"]["numerical_library_imported"] is False
    assert payload["all_internal_exact_checks_passed"]
    rebuilt = payload["reconstruction"]
    assert all(rebuilt["checks"].values())
    assert Fraction(rebuilt["zero_error"]["rate"]) == Fraction(549_700_907_325, 549_755_813_888)
    assert Fraction(rebuilt["zero_error"]["forcing"]) == Fraction(2_162_331, 1_099_511_627_776)
    assert Fraction(rebuilt["zero_error"]["function_gap"]) == Fraction(
        399_957_341_889, 549_755_813_888
    )
    assert {
        axis: int(Fraction(boundary["accepted_value"]) * 2**40)
        for axis, boundary in rebuilt["axis_boundaries"].items()
    } == {
        "gradient_slope": 10_815_225_547,
        "operator_slope": 513_245_498_810,
        "gradient_intercept": 10_879_487_718,
        "operator_intercept": 13_351_103_462_525,
    }
    assert len(rebuilt["slope_frontier"]) == len(rebuilt["intercept_frontier"]) == 9
    assert rebuilt["jointly_nonzero"]["certified"]


def test_p11_reconstruction_matches_primary_exact_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    rebuilt = _run_reconstruction(root)["reconstruction"]
    primary = audit_implementation_margin()

    for name, value in primary.sensitivities.__dict__.items():
        assert Fraction(rebuilt["external_port_sensitivities"][name]) == value
    assert Fraction(rebuilt["zero_error"]["rate"]) == primary.zero_error.rate
    assert Fraction(rebuilt["zero_error"]["forcing"]) == primary.zero_error.constant_forcing
    for axis, boundary in primary.axis_boundaries.items():
        candidate = rebuilt["axis_boundaries"][axis]
        assert Fraction(candidate["accepted_value"]) == boundary.accepted_value
        assert Fraction(candidate["rejected_value"]) == boundary.rejected_value
        assert candidate["rejected_checks"] == list(boundary.rejected_checks)
    for rebuilt_rows, primary_rows in (
        (rebuilt["slope_frontier"], primary.slope_frontier),
        (rebuilt["intercept_frontier"], primary.intercept_frontier),
    ):
        for candidate, row in zip(rebuilt_rows, primary_rows, strict=True):
            assert Fraction(candidate["gradient_value"]) == row.gradient_value
            assert Fraction(candidate["operator_value"]) == row.operator_value
            assert Fraction(candidate["rejected_gradient_value"]) == row.rejected_gradient_value
            assert Fraction(candidate["rejected_operator_value"]) == row.rejected_operator_value
            assert Fraction(candidate["rate"]) == row.rate
            assert Fraction(candidate["forcing"]) == row.forcing
            assert Fraction(candidate["rejected_gradient_invariance_slack"]) == (
                row.rejected_gradient_invariance_slack
            )
            assert Fraction(candidate["rejected_operator_invariance_slack"]) == (
                row.rejected_operator_invariance_slack
            )


def test_p11_reconstruction_matches_fresh_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "fresh-p11.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_implementation_margin.py"),
            "--output",
            str(generated),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _run_reconstruction(
        root,
        "--canonical",
        str(generated),
        "--require-canonical",
    )

    comparison = payload["canonical_comparison"]
    assert comparison["status"] == "matched"
    assert comparison["all_exact_fields_match"]
    assert all(comparison["comparisons"].values())
    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"]


def test_p11_reconstruction_requires_and_matches_committed_canonical() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/implementation_margin_certificate.json"
    assert canonical.is_file(), "the committed P11 exact certificate is mandatory"
    payload = _run_reconstruction(root, "--require-canonical")
    comparison = payload["canonical_comparison"]

    assert comparison["status"] == "matched"
    assert comparison["all_exact_fields_match"]
    assert all(comparison["comparisons"].values())
    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"]


def test_missing_p11_canonical_never_counts_as_full_replay(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing-p11.json"
    payload = _run_reconstruction(root, "--canonical", str(missing))

    assert payload["all_internal_exact_checks_passed"]
    assert payload["canonical_comparison"]["status"] == "not_found"
    assert payload["all_exact_checks_passed"] is False

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_implementation_margin.py"),
            "--canonical",
            str(missing),
            "--require-canonical",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "canonical P11 comparison failed" in completed.stderr
