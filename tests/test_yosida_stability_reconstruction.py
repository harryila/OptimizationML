from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest


def _run(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "scripts/reconstruct_yosida_stability.py"), *arguments],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def _generate(root: Path, output: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_yosida_stability.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def test_p14_reconstruction_is_standard_library_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_yosida_stability.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])

    assert "passive_muon" not in imported
    assert imported.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint", "mpmath", "cvxpy"})


def test_p14_reconstruction_closes_every_internal_exact_check() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(_run(root).stdout)

    assert payload["schema_version"] == "passive-muon-p14-independent-reconstruction-v1"
    assert payload["implementation_scope"]["project_package_imported"] is False
    assert payload["implementation_scope"]["numerical_library_imported"] is False
    assert payload["all_internal_exact_checks_passed"]
    assert all(payload["reconstruction"]["checks"].values())
    assert payload["canonical_comparison"]["status"] in {"not_found", "matched"}

    reconstructed = payload["reconstruction"]
    assert reconstructed["sector"] == {
        "center": "750",
        "dimensionless_step": "15/64",
        "lower": "500",
        "residual_radius": "250",
        "residual_ratio": "1/3",
        "surplus_factor": "500",
        "upper": "1000",
    }
    assert all(
        Fraction(value) > 0 for value in reconstructed["certificate"]["storage_leading_minors"]
    )
    assert all(
        Fraction(value) > 0 for value in reconstructed["certificate"]["negative_lmi_leading_minors"]
    )
    assert len(reconstructed["certificate"]["lmi_matrix"]) == 4
    assert all(len(row) == 4 for row in reconstructed["certificate"]["lmi_matrix"])
    assert len(reconstructed["certificate"]["transition"]) == 2
    assert len(reconstructed["certificate"]["signal_selector"]) == 4
    assert reconstructed["sharpness"]["skew_centered_residual"] == "0"
    assert reconstructed["sharpness"]["skew_sector_residual"] == "0"


def test_p14_reconstruction_matches_fresh_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p14.json"
    _generate(root, fresh)
    payload = json.loads(_run(root, "--canonical", str(fresh), "--require-canonical").stdout)

    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]
    assert all(payload["canonical_comparison"]["comparisons"].values())


def test_p14_reconstruction_matches_committed_canonical_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/yosida_stability_certificate.json"
    if not canonical.is_file():
        pytest.skip("committed P14 canonical certificate has not landed yet")
    payload = json.loads(_run(root, "--require-canonical").stdout)
    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]


def test_p14_reconstruction_rejects_tampered_rate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p14.json"
    _generate(root, fresh)
    payload = json.loads(fresh.read_text(encoding="utf-8"))
    payload["exact_pl_certificate"]["q"]["exact"] = "0"
    tampered = tmp_path / "tampered-p14.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(root, "--canonical", str(tampered), "--require-canonical", check=False)
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr


def test_p14_reconstruction_rejects_tampered_lmi_entry(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p14.json"
    _generate(root, fresh)
    payload = json.loads(fresh.read_text(encoding="utf-8"))
    payload["exact_pl_certificate"]["reconstruction_fields"]["lmi_matrix"][0][0] = "0"
    tampered = tmp_path / "tampered-p14-lmi.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(root, "--canonical", str(tampered), "--require-canonical", check=False)
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr
