from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_reconstruction(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_outer_loop_roundoff.py"),
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p10_reconstruction_is_stdlib_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_outer_loop_roundoff.py"
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


def test_p10_reconstruction_rebuilds_locked_exact_values() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_reconstruction(root)

    assert payload["schema_version"] == "passive-muon-p10-independent-reconstruction-v1"
    assert payload["implementation_scope"]["project_package_imported"] is False
    assert payload["implementation_scope"]["numerical_library_imported"] is False
    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"] is (
        payload["canonical_comparison"]["status"] == "matched"
    )
    rebuilt = payload["reconstruction"]
    assert all(rebuilt["checks"].values())
    assert Fraction(rebuilt["certificate"]["rate_q10"]) == Fraction(
        549_700_907_325, 549_755_813_888
    )
    assert Fraction(rebuilt["certificate"]["constant_forcing_D10"]) == Fraction(
        2_162_331, 1_099_511_627_776
    )
    assert Fraction(rebuilt["certificate"]["function_gap_ultimate"]) < 1
    assert all(rebuilt["guards"]["master_checks"].values())


def test_p10_reconstruction_matches_fresh_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "fresh-p10.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_outer_loop_roundoff.py"),
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


def test_p10_reconstruction_requires_and_matches_committed_canonical() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/outer_loop_roundoff_certificate.json"
    assert canonical.is_file(), "the committed P10 exact certificate is mandatory"
    payload = _run_reconstruction(root, "--require-canonical")
    comparison = payload["canonical_comparison"]

    assert comparison["status"] == "matched"
    assert comparison["all_exact_fields_match"]
    assert all(comparison["comparisons"].values())
    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"]


def test_missing_canonical_never_counts_as_all_exact_checks_passed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    missing = tmp_path / "missing-p10.json"
    payload = _run_reconstruction(root, "--canonical", str(missing))

    assert payload["all_internal_exact_checks_passed"]
    assert payload["canonical_comparison"]["status"] == "not_found"
    assert payload["all_exact_checks_passed"] is False

    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_outer_loop_roundoff.py"),
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
    assert "canonical P10 comparison failed" in completed.stderr
