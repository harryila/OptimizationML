from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_inexact_yosida_robustness.py"),
            *arguments,
        ],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def _generate(root: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_inexact_yosida_robustness.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_p15_reconstruction_is_standard_library_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_inexact_yosida_robustness.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])

    assert "passive_muon" not in imported
    assert imported.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint", "mpmath", "cvxpy"})


def test_p15_reconstruction_closes_every_internal_exact_check() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(_run(root).stdout)

    assert payload["schema_version"] == "passive-muon-p15-independent-reconstruction-v1"
    assert payload["implementation_scope"]["project_package_imported"] is False
    assert payload["implementation_scope"]["numerical_library_imported"] is False
    assert payload["all_internal_exact_checks_passed"]
    assert all(payload["reconstruction"]["checks"].values())
    assert payload["canonical_comparison"]["status"] in {"not_found", "matched"}

    reconstruction = payload["reconstruction"]
    assert reconstruction["residual_envelope"] == {
        "absolute_output_from_rbar": "500",
        "base_centered_radius": "250",
        "effective_centered_radius": "252",
        "equation_to_output_gain": "500",
        "forward_strong_monotonicity": "2",
        "relative_output_radius": "2",
        "relative_residual": "1/250",
    }
    assert len(reconstruction["certificate"]["lmi"]) == 5
    assert all(len(row) == 5 for row in reconstruction["certificate"]["lmi"])
    assert reconstruction["controls"]["critical_penalty_determinant"] == "0"
    assert reconstruction["controls"]["abstract_ascent_p_at_one"] == "-1/1280"


def test_p15_reconstruction_matches_fresh_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p15.json"
    _generate(root, fresh)
    payload = json.loads(_run(root, "--canonical", str(fresh), "--require-canonical").stdout)

    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]
    assert all(payload["canonical_comparison"]["comparisons"].values())


def test_p15_reconstruction_matches_committed_canonical_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/inexact_yosida_robustness_certificate.json"
    if not canonical.is_file():
        pytest.skip("committed P15 canonical certificate has not landed yet")
    payload = json.loads(_run(root, "--require-canonical").stdout)
    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]


def test_p15_reconstruction_rejects_tampered_forcing_gain(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p15.json"
    _generate(root, fresh)
    payload = json.loads(fresh.read_text(encoding="utf-8"))
    payload["reconstruction_fields"]["certificate"]["absolute_forcing_gain"] = "0"
    tampered = tmp_path / "tampered-p15.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(root, "--canonical", str(tampered), "--require-canonical", check=False)
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr
