from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p6_reconstruction_uses_an_independent_stdlib_only_code_path() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_pl_convergence.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint"})


def test_p6_independent_reconstruction_matches_complete_canonical_lmi() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/reconstruct_pl_convergence.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-p6-independent-reconstruction-v1"
    scope = payload["implementation_scope"]
    assert scope["project_package_imported"] is False
    assert scope["numerical_algebra_library_imported"] is False
    assert scope["canonical_read_order"].startswith("only after complete independent")
    assert "not a human proof audit" in scope["qualification"]

    revision_roles = payload["revision_roles"]
    assert revision_roles == {
        "certificate_generation_source_commit": "a8f650f6c60dcbc5d2f83647fd367348f4c67548",
        "distinction": (
            "a8f650f6 is the certificate generation/source commit; "
            "ef88d8f5, tagged p6-checkpoint, is the final P6 checkpoint"
        ),
        "final_checkpoint_commit": "ef88d8f5b26148af0ec1ca70b506048938bf9bef",
        "final_checkpoint_tag": "p6-checkpoint",
    }

    assert payload["all_exact_checks_passed"]
    comparison = payload["canonical_comparison"]
    assert (
        comparison["canonical_generation_source_commit"]
        == revision_roles["certificate_generation_source_commit"]
    )
    assert comparison["all_exact_fields_match"]
    assert all(comparison["comparisons"].values())

    reconstruction = payload["reconstruction"]
    assert reconstruction["all_internal_exact_checks_passed"]
    assert all(reconstruction["checks"].values())
    assert len(reconstruction["lmi_matrix"]) == 4
    assert all(len(row) == 4 for row in reconstruction["lmi_matrix"])
    assert (
        reconstruction["function_coefficients"] == reconstruction["expected_function_coefficients"]
    )
    assert len(reconstruction["storage_leading_principal_minors"]) == 2
    assert len(reconstruction["negative_lmi_leading_principal_minors"]) == 4
    assert all(
        Fraction(value) > 0
        for value in reconstruction["storage_leading_principal_minors"]
        + reconstruction["negative_lmi_leading_principal_minors"]
    )
