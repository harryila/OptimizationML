from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def test_p5_reconstruction_uses_an_independent_stdlib_only_code_path() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_nonquadratic_convergence.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint"})


def test_p5_independent_reconstruction_matches_every_exact_canonical_field() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/reconstruct_nonquadratic_convergence.py")],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["schema_version"] == "passive-muon-p5-independent-reconstruction-v1"
    scope = payload["implementation_scope"]
    assert scope["project_package_imported"] is False
    assert scope["numerical_algebra_library_imported"] is False
    assert "not a human proof audit" in scope["qualification"]
    assert payload["all_exact_checks_passed"]
    assert payload["canonical_comparison"]["all_exact_fields_match"]
    assert all(payload["canonical_comparison"]["comparisons"].values())

    reconstruction = payload["reconstruction"]
    assert reconstruction["storage_positive_definite"]
    assert reconstruction["lmi_negative_definite"]
    assert reconstruction["function_values_cancel"]
    assert len(reconstruction["lmi_matrix"]) == 5
    assert len(reconstruction["negative_lmi_leading_minors"]) == 5
    assert all(Fraction(value) > 0 for value in reconstruction["negative_lmi_leading_minors"])
