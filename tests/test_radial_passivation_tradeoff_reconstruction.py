from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


def _run(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_radial_passivation_tradeoff.py"),
            *arguments,
        ],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def test_p13_reconstruction_is_standard_library_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_radial_passivation_tradeoff.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert "passive_muon" not in imported
    assert imported.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint", "mpmath", "cvxpy"})


def test_p13_reconstruction_closes_every_internal_exact_check() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(_run(root).stdout)
    assert payload["schema_version"] == "passive-muon-p13-independent-reconstruction-v1"
    assert payload["all_internal_exact_checks_passed"]
    assert all(payload["reconstruction"]["checks"].values())
    assert payload["canonical_comparison"]["status"] in {"not_found", "matched"}


def test_p13_reconstruction_matches_fresh_generator(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    fresh = tmp_path / "fresh-p13.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_radial_passivation_tradeoff.py"),
            "--output",
            str(fresh),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(_run(root, "--canonical", str(fresh), "--require-canonical").stdout)
    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]
    assert all(payload["canonical_comparison"]["comparisons"].values())


def test_p13_reconstruction_requires_committed_canonical() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/radial_passivation_tradeoff_certificate.json"
    assert canonical.is_file(), "the committed P13 canonical certificate is mandatory"
    payload = json.loads(_run(root, "--require-canonical").stdout)
    assert payload["canonical_comparison"]["status"] == "matched"
    assert payload["all_exact_checks_passed"]


def test_p13_reconstruction_rejects_tampered_canonical(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/radial_passivation_tradeoff_certificate.json"
    if not canonical.is_file():
        return
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    payload["stiffness_tradeoff"]["relative_gap"]["exact"] = "0"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    completed = _run(root, "--canonical", str(tampered), "--require-canonical", check=False)
    assert completed.returncode != 0
    assert "missing or mismatched" in completed.stderr
