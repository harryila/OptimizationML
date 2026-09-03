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
            str(root / "scripts/reconstruct_scalable_mixed_precision.py"),
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p9_reconstruction_is_an_independent_stdlib_only_code_path() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_scalable_mixed_precision.py"
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


def test_p9_internal_reconstruction_rebuilds_every_exact_shape_row() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_reconstruction(root)

    assert payload["schema_version"] == "passive-muon-p9-independent-reconstruction-v1"
    scope = payload["implementation_scope"]
    assert scope["project_package_imported"] is False
    assert scope["numerical_library_imported"] is False
    assert scope["canonical_read_order"].startswith("only after complete independent")
    assert "not a human proof audit" in scope["qualification"]
    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"]

    reconstruction = payload["reconstruction"]
    assert reconstruction["all_internal_exact_checks_passed"]
    assert all(reconstruction["checks"].values())
    assert reconstruction["qualified_frontier_failed_checks"] == ["stage_5_input_in_spectral_tube"]
    certificate = reconstruction["certificate"]
    assert certificate["rank_gates"]["ordinary_one_term"] == 71
    assert certificate["rank_gates"]["sterbenz_free_two_term"] == 4_656_751
    assert (
        certificate["serial_normalizer_obstruction"]["returned_singular_value_squared"] == "43/16"
    )

    audits = certificate["shape_audits"]
    assert len(audits) == 7
    assert all(audit["certified"] for audit in audits)
    assert all(audit["polynomial"]["certified"] for audit in audits)
    assert all(len(audit["stages"]) == 5 for audit in audits)
    assert all(
        Fraction(audit["operator_bound"]["real_slope"]) == Fraction(102_465_557, 549_755_813_888)
        for audit in audits
    )
    assert all(
        Fraction(audit["p7"]["values"]["rate"]) == Fraction(137_425_214_491, 137_438_953_472)
        for audit in audits
    )


def test_p9_reconstruction_matches_fresh_generator_output(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    generated = tmp_path / "fresh-p9.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_scalable_mixed_precision.py"),
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


def test_p9_reconstruction_matches_committed_canonical_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/scalable_mixed_precision_certificate.json"
    payload = _run_reconstruction(root)
    comparison = payload["canonical_comparison"]

    if canonical.exists():
        assert comparison["status"] == "matched"
        assert comparison["all_exact_fields_match"]
        assert all(comparison["comparisons"].values())
    else:
        assert comparison == {
            "all_exact_fields_match": None,
            "comparisons": {},
            "path": "results/summaries/scalable_mixed_precision_certificate.json",
            "status": "not_found",
        }
