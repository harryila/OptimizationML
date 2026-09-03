from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_reconstruction(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/reconstruct_mixed_precision.py"), *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p8_reconstruction_is_an_independent_stdlib_only_code_path() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_mixed_precision.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint", "mpmath"})


def test_p8_internal_reconstruction_rebuilds_all_exact_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_reconstruction(root)

    assert payload["schema_version"] == "passive-muon-p8-independent-reconstruction-v1"
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
    assert reconstruction["sturm_endpoint_signs"] == [
        [1, -1, -1, 1, -1, -1],
        [1, -1, -1, -1, 1, -1],
    ]
    assert reconstruction["sturm_variations"] == [3, 3]
    assert len(reconstruction["sturm_chain"]) == 6

    recurrence = reconstruction["recurrence"]
    assert Fraction(recurrence["gamma4"]) == Fraction(1, 4_194_303)
    assert Fraction(recurrence["epsilon_y"]) < Fraction(1, 20_000)
    assert Fraction(recurrence["next_frobenius"]) < Fraction(7, 4)
    assert Fraction(recurrence["next_spectral"]) < Fraction(5, 4)

    repair = reconstruction["repair"]
    assert Fraction(repair["complete_slope"]) < Fraction(11, 100_000)
    assert Fraction(repair["complete_intercept"]) < Fraction(347, 100)
    assert Fraction(repair["real_adapter_slope"]) < Fraction(1, 5_000)
    assert Fraction(repair["real_adapter_intercept"]) < Fraction(347, 100)
    closure = reconstruction["p7_closure"]
    assert Fraction(closure["rate"]) == Fraction(
        41_597_186_684_695_561,
        41_601_344_000_000_000,
    )
    assert Fraction(closure["forcing"]) == Fraction(4_936_769, 819_840_000_000)
    assert Fraction(closure["function_gap_ultimate"]) == Fraction(
        462_392_438_350_000_000,
        207_695_315_294_468_001,
    )
    assert Fraction(closure["safe_storage_radius"]) > 0
    assert Fraction(closure["forcing"]) <= Fraction(closure["safe_forcing_capacity"])


def test_p8_reconstruction_matches_canonical_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/mixed_precision_certificate.json"
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
            "path": "results/summaries/mixed_precision_certificate.json",
            "status": "not_found",
        }
