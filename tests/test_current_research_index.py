from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_current_overviews_record_both_p5_guarantees() -> None:
    for relative_path in (
        "README.md",
        "theory/claims.md",
        "results/summaries/RESULTS.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "1/640000" in text
        assert "1/32000" in text
        assert "incremental" in text
        assert "trajectory-to-minimizer" in text


def test_current_ledgers_record_p5_and_the_completed_p6_step() -> None:
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")

    assert "P5 full-step" in tasks
    assert "Polyak--Lojasiewicz" in tasks
    assert "P6 rational value--momentum LMI" in tasks
    assert "nonquadratic_convergence_certificate.json" in results_index
    assert "nonquadratic_convergence_falsification.json" in results_index
    assert "pl_convergence_certificate.json" in results_index
    assert "pl_falsification.json" in results_index
    assert "P4" in experiments
    assert "P5" in experiments
    assert "P6" in experiments


def test_double_blind_material_is_guarded_from_the_public_repository() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        "paper/",
        "submission/",
        "supplement/",
        "anonymous-supplement/",
        "anonymized-supplement/",
    ):
        assert pattern in ignore.splitlines()
