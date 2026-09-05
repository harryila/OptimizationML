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


def test_current_indexes_record_scoped_p7_p8_and_p9_results() -> None:
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    results_flat = " ".join(results.split())

    for artifact in (
        "robust_dissipativity_certificate.json",
        "mixed_precision_certificate.json",
        "scalable_mixed_precision_certificate.json",
        "scalable_mixed_precision_diagnostic.json",
        "P9_RESULTS.md",
    ):
        assert artifact in results_index

    for command in (
        "certify_scalable_mixed_precision.py",
        "reconstruct_scalable_mixed_precision.py",
        "run_scalable_mixed_precision_diagnostic.py",
    ):
        assert command in experiments
        assert command in results

    assert "standard-library-only" in results_index
    assert "native-matmul" in experiments
    assert "falsification" in experiments
    assert "## 14. P9 scalable mixed-precision certificate" in results
    assert "seven locked Transformer shapes" in results
    assert "137425214491" in results
    assert "137438953472" in results
    assert "4608` is not a universal rank frontier" in results
    assert "## 17. Unrun gate" in results
    assert "model forward/backward" in results_flat
    assert "three-word master" in results_flat


def test_current_indexes_record_scoped_p10_outer_loop_result() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "outer_loop_roundoff_certificate.json",
        "finite_precision_outer_loop_diagnostic.json",
        "P10_RESULTS.md",
    ):
        assert artifact in results_index

    for command in (
        "certify_outer_loop_roundoff.py",
        "reconstruct_outer_loop_roundoff.py",
        "run_finite_precision_outer_loop_diagnostic.py",
    ):
        assert command in experiments
        assert command in results

    for text in (readme, results, claims):
        assert "549700907325" in text
        assert "549755813888" in text
        assert "three-word" in text

    assert "three-word" in tasks

    assert "## 15. P10 finite-precision outer-loop certificate" in results
    assert "## C15. Finite-precision outer-loop storage certificate" in claims
    assert "## C17. Circuit and broader optimization consequences" in claims
    assert "standard-library-only" in results_index
    assert "human proof audit of C15" in tasks


def test_current_indexes_record_scoped_p11_implementation_margin_result() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "implementation_margin_certificate.json",
        "P11_RESULTS.md",
    ):
        assert artifact in results_index

    for command in (
        "certify_implementation_margin.py",
        "reconstruct_implementation_margin.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results

    for text in (readme, results, claims):
        assert "10815225547" in text
        assert "13351103462525" in text
        assert "2^-40" in text

    assert "## 16. P11 certified implementation margins" in results
    assert "## C16. Certified implementation margins above P10" in claims
    assert "## C17. Circuit and broader optimization consequences" in claims
    assert "human proof audit of C16" in tasks
    assert "P10 human proof audit remains pending" in tasks


def test_current_indexes_record_scoped_p12_additive_epsilon_result() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "additive_epsilon_deficit_certificate.json",
        "P12_ADDITIVE_EPSILON_RESULTS.md",
    ):
        assert artifact in results_index

    for command in (
        "certify_additive_epsilon_deficit.py",
        "reconstruct_additive_epsilon_deficit.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results

    for text in (readme, results, claims):
        assert "6602082433275499863" in text
        assert "41641817600000000" in text
        assert "158.1172496" in text
        assert "1/epsilon" in text
        assert "exact-real" in text
        assert "BF16" in text

    assert "## 18. P12 additive-epsilon deficit" in results
    assert "## C18. Additive-epsilon full-matrix deficit" in claims
    assert "human proof audit" in tasks
    assert "catastrophically large" in " ".join(results.split())


def test_current_indexes_record_scoped_p13_radial_passivation_tradeoff() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "radial_passivation_tradeoff_certificate.json",
        "P13_RADIAL_PASSIVATION_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_radial_passivation_tradeoff.py",
        "reconstruct_radial_passivation_tradeoff.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims):
        assert "2571.857826" in text
        assert "3086.959580" in text
        assert "158.1172496" in text
        assert "1/epsilon" in text
        assert "BF16" in text
    assert "## 19. P13 radial passivation tradeoff" in results
    assert "## C19. Nonlinear radial passivation and unavoidable stiffness" in claims
    assert "human proof audit of C19" in tasks
    assert "scalar" in results and "Jury" in results


def test_current_indexes_record_scoped_p14_yosida_stability() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "yosida_stability_certificate.json",
        "P14_YOSIDA_STABILITY_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_yosida_stability.py",
        "reconstruct_yosida_stability.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims):
        assert "249001" in text
        assert "250000" in text
        assert "1/32000" in text
        assert "Yosida" in text
        assert "exact-real" in text
        assert "BF16" in text
    assert "## 20. P14 Yosida stability" in results
    assert "## C20. Yosida regularization and full-step PL stability" in claims
    assert "human proof audit of C20" in tasks
    assert "D14=0" in " ".join(results.split())


def test_current_indexes_record_scoped_p15_inexact_yosida_robustness() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")

    for artifact in (
        "inexact_yosida_robustness_certificate.json",
        "P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_inexact_yosida_robustness.py",
        "reconstruct_inexact_yosida_robustness.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims):
        assert "249001" in text
        assert "250000" in text
        assert "6250000000000" in text
        assert "312929757" in text
        assert "1/250" in text
        assert "exact-real" in text
        assert "BF16" in text
    assert "## 21. P15 inexact-Yosida robustness" in results
    assert "## C21. Inexact-Yosida residual robustness" in claims
    assert "human proof audit of C21" in tasks
    assert "q15=249001/250000" in " ".join(results.split())
    assert "C15=5/2" in " ".join(results.split())
    assert "not an algorithm" in " ".join(claims.split())


def test_current_indexes_record_scoped_p16_equivariant_resolvent_solver() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    p16_summary = (
        ROOT / "results/summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md"
    ).read_text(encoding="utf-8")

    for artifact in (
        "equivariant_resolvent_solver_certificate.json",
        "p16_solver_study.json",
        "P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_equivariant_resolvent_solver.py",
        "reconstruct_equivariant_resolvent_solver.py",
        "run_p16_solver_study.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, p16_summary):
        assert "1e-7" in text or "1/10000000" in text
        assert "6889/2000" in text
        assert "-191/40" in text
        assert "4063/2000" in text
        assert "five" in text.lower()
        assert "1/1000" in text
        assert "mu=1000" in text
        assert "1/250" in text
        assert "5.879e-6" in text
        assert "8.403e-5" in text
        assert "effectively scalar" in text
        assert "FP64" in text
        assert "rounding" in text
    assert "## 22. P16 equivariant resolvent solver" in results
    assert "## C22. Equivariant structured resolvent solve" in claims
    assert "human proof audit of C22" in tasks
    assert "nineteen technical goals" in readme
    assert "not a useful uniform" in " ".join(claims.split())
    assert "fails its overall meaningful-Muon acceptance gate" in results


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
