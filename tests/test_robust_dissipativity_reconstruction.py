from __future__ import annotations

import ast
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

EXPECTED_NEGATIVE_LMI_MINORS = (
    "588366669441078114305592272723841308118612219507/"
    "35513159127688596684800000000000000000000000000000",
    "111653153906865859147084187785008643627781514215908602092179/"
    "1420526365107543867392000000000000000000000000000000000000000000",
    "30245282351805947095058111344751251587087360962215007639796547873/"
    "71026318255377193369600000000000000000000000000000000000000000000000000",
    "106378237649828474982433094173183543070031301965649332843580933465570058399/"
    "2841052730215087734784000000000000000000000000000000000000000000000000000000000000000",
    "358878445547025338702743793976131456516389191266001498332974313243385014147631510993/"
    "568210546043017546956800000000000000000000000000000000000000000000000000000000000000000000000",
    "25157750088920031541989779883715177798784065261265275305093677552588813246840551502751954845110066654533994497393/"
    "3079063650460282933136543676107220582400000000000000000000000000000000000000000000000000000000000000000000000000000000000",
)


def _run_reconstruction(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/reconstruct_robust_dissipativity.py"),
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p7_reconstruction_uses_an_independent_stdlib_only_code_path() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts/reconstruct_robust_dissipativity.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert "passive_muon" not in imported_roots
    assert imported_roots.isdisjoint({"numpy", "scipy", "sympy", "torch", "flint"})


def test_p7_internal_reconstruction_replays_all_six_locked_minors() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_reconstruction(root)

    assert payload["schema_version"] == "passive-muon-p7-independent-reconstruction-v1"
    scope = payload["implementation_scope"]
    assert scope["project_package_imported"] is False
    assert scope["numerical_algebra_library_imported"] is False
    assert scope["determinant_method"] == "local exact Bareiss elimination"
    assert scope["canonical_read_order"].startswith("only after complete independent")
    assert "not a human proof audit" in scope["qualification"]

    assert payload["all_internal_exact_checks_passed"]
    assert payload["all_exact_checks_passed"]
    reconstruction = payload["reconstruction"]
    assert reconstruction["all_internal_exact_checks_passed"]
    assert all(reconstruction["checks"].values())
    assert len(reconstruction["lmi_matrix"]) == 6
    assert all(len(row) == 6 for row in reconstruction["lmi_matrix"])
    assert (
        reconstruction["function_coefficients"] == reconstruction["expected_function_coefficients"]
    )
    assert reconstruction["storage_leading_principal_minors"] == [
        "14487/20000",
        "650021/250000000",
    ]
    assert tuple(reconstruction["negative_lmi_leading_principal_minors"]) == (
        EXPECTED_NEGATIVE_LMI_MINORS
    )
    assert all(Fraction(value) > 0 for value in EXPECTED_NEGATIVE_LMI_MINORS)

    parameters = reconstruction["parameters"]
    assert parameters["learning_rate"] == "1/32000"
    assert parameters["rate"] == "399960001/400000000"
    assert parameters["physical_gains"] == {
        "gradient_noise": "1/2",
        "implementation_error": "1/2000000",
    }
    assert parameters["normalized_penalties"]["gradient_noise"] == "50"
    dynamics = reconstruction["dynamics"]
    assert dynamics["noisy_gradient_selector"] == ["0", "1", "0", "0", "1", "0"]


def test_p7_reconstruction_matches_canonical_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    canonical = root / "results/summaries/robust_dissipativity_certificate.json"
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
            "path": "results/summaries/robust_dissipativity_certificate.json",
            "status": "not_found",
        }
