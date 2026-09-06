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
    assert "technical goals" in readme
    assert "not a useful uniform" in " ".join(claims.split())
    assert "fails its overall meaningful-Muon acceptance gate" in results


def test_current_indexes_record_scoped_p17_shape_preserving_resolvent() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    p17_summary = (ROOT / "results/summaries/P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md").read_text(
        encoding="utf-8"
    )

    for artifact in (
        "shape_preserving_resolvent_certificate.json",
        "p17_shape_preserving_study.json",
        "P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_shape_preserving_resolvent.py",
        "reconstruct_shape_preserving_resolvent.py",
        "run_p17_shape_preserving_study.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, p17_summary):
        assert "1e-7" in text or "1/10000000" in text
        assert "6889/2000" in text
        assert "-191/40" in text
        assert "4063/2000" in text
        assert "1/1000" in text
        assert "mu=1000" in text
        assert "3/4" in text
        assert "4096" in text
        assert "1/128000" in text
        assert "1/8" in text
        assert "8192" in text
        assert "1/32000" in text
        assert "pointwise" in text.lower()
        assert "exact-real" in text
    combined = " ".join((readme, results, claims, p17_summary))
    assert "281474943156225/281474976710656" in combined
    assert "281474741829681/281474976710656" in combined
    assert "not incremental" in combined or "not an incremental" in combined
    assert "inexact-solver" in combined or "approximate-solver" in combined
    assert "## 23. P17 shape-preserving gated resolvent" in results
    assert "## C23. Shape-preserving gated resolvent" in claims
    assert "human proof audit of C23" in tasks
    assert "technical goals" in readme


def test_current_indexes_record_scoped_p18_sector_projected_useful_rate() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    p18_summary = (
        ROOT / "results/summaries/P18_SECTOR_PROJECTED_USEFUL_RATE_RESULTS.md"
    ).read_text(encoding="utf-8")

    for artifact in (
        "sector_projected_useful_rate_certificate.json",
        "p18_sector_projected_study.json",
        "P18_SECTOR_PROJECTED_USEFUL_RATE_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_sector_projected_useful_rate.py",
        "reconstruct_sector_projected_useful_rate.py",
        "run_p18_sector_projected_study.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, p18_summary):
        assert "1e-7" in text or "1/10000000" in text
        assert "6889/2000" in text
        assert "-191/40" in text
        assert "4063/2000" in text
        assert "1/1000" in text
        assert "mu=1000" in text
        assert "3/4" in text
        assert "1024" in text
        assert "125/1024" in text or r"\frac{125}{1024}" in text
        assert "509/512" in text or r"\frac{509}{512}" in text
        assert "1/83" in text
        assert "999598040401" in text
        assert "pointwise" in text.lower()
        assert "exact-real" in text
    combined = " ".join((readme, results, claims, p18_summary))
    assert "3768360579178620269" in combined
    assert "not incremental" in combined or "not an incremental" in combined
    assert "inexact-solver" in combined or "approximate-solver" in combined
    assert "192/327" in combined
    assert "2176" in combined
    assert "## 24. P18 sector-projected useful-rate resolvent" in results
    assert "## C24. Sector-projected shape interface" in claims
    assert "human proof audit of C24" in tasks
    assert "technical goals" in readme


def test_current_indexes_record_scoped_p19_sector_shield() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    p19_summary = (
        ROOT / "results/summaries/P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_RESULTS.md"
    ).read_text(encoding="utf-8")
    p19_theorem = (ROOT / "theory/sector_shielded_inexact_resolvent.md").read_text(encoding="utf-8")

    for artifact in (
        "sector_shielded_inexact_resolvent_certificate.json",
        "p19_sector_shielded_study.json",
        "P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_sector_shielded_inexact_resolvent.py",
        "reconstruct_sector_shielded_inexact_resolvent.py",
        "run_p19_sector_shielded_study.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, p19_summary, p19_theorem):
        assert "1e-7" in text or "1/10000000" in text
        assert "6889/2000" in text
        assert "-191/40" in text
        assert "4063/2000" in text
        assert "1/1000" in text
        assert "mu=1000" in text or "\\mu=1000" in text
        assert "125/1024" in text or r"\frac{125}{1024}" in text
        assert "509/512" in text or r"\frac{509}{512}" in text
        assert "1143/2048" in text or r"\frac{1143}{2048}" in text
        assert "893/2048" in text or r"\frac{893}{2048}" in text
        assert "1/83" in text or r"\frac1{83}" in text
        assert "1/120" in text or r"\frac1{120}" in text
        assert "999598040401" in text
        assert "624350169" in text
        assert "pointwise" in text.lower()
    combined = " ".join((readme, results, claims, p19_summary, p19_theorem))
    assert "S=0" in combined or "S = 0" in combined
    assert "nonexpansive" in combined
    assert "successful return" in combined
    assert "subnormal" in combined
    assert "2688" in combined
    assert "2176" in combined
    assert "## 25. P19 sector-shielded inexact resolvent" in results
    assert "## C25. Sector shield makes arbitrary finite candidates" in claims
    assert "human proof audit of C25" in tasks
    assert "technical goals" in readme


def test_current_indexes_record_scoped_p20_scalable_sector_shield() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    theorem = (ROOT / "theory/scalable_mixed_precision_sector_shield.md").read_text(
        encoding="utf-8"
    )

    for artifact in (
        "scalable_sector_shield_certificate.json",
        "p20_scalable_sector_shield_study.json",
        "P20_SCALABLE_MIXED_PRECISION_SECTOR_SHIELD_RESULTS.md",
    ):
        assert artifact in results_index
    for command in (
        "certify_scalable_sector_shield.py",
        "reconstruct_scalable_sector_shield.py",
        "run_p20_scalable_sector_shield_study.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, theorem):
        assert "125/1024" in text or r"\frac{125}{1024}" in text
        assert "509/512" in text or r"\frac{509}{512}" in text
        assert "1143/2048" in text or r"\frac{1143}{2048}" in text
        assert "893/2048" in text or r"\frac{893}{2048}" in text
        assert "4096 x 11008" in text
        assert "4096 x 14336" in text
        assert "999598040401" in text
        assert "624350169" in text
        assert "pointwise" in text.lower()
        assert "BF16" in text
        assert "FP32" in text
    combined = " ".join((readme, results, claims, theorem))
    assert "71710053325847" in combined
    assert "1152921504606846976" in combined
    assert "2671/2688" in combined
    assert "2176/2176" in combined
    assert "761/2688" in combined
    assert "1927" in combined
    assert "radial clip" in combined or "radial-clip" in combined
    assert "runtime big-integer" in combined or "runtime rational" in combined
    assert "not P19's metric projection" in combined
    assert "not an incremental" in combined
    assert "P21" in combined
    assert "## 26. P20 scalable mixed-precision sector shield" in results
    assert "## C26. Shape-locked mixed-precision sector containment" in claims
    assert "human proof audit of C26" in tasks
    assert "technical goals" in readme


def test_current_indexes_record_scoped_p21_outer_loop_composition() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    theorem = (ROOT / "theory/certified_outer_loop_composition.md").read_text(encoding="utf-8")
    summary = (
        ROOT / "results/summaries/P21_CERTIFIED_OUTER_LOOP_COMPOSITION_RESULTS.md"
    ).read_text(encoding="utf-8")
    protocol = (ROOT / "theory/p21_shadow_trace_protocol.md").read_text(encoding="utf-8")

    for artifact in (
        "certified_outer_loop_composition_certificate.json",
        "p21_synthetic_shadow_trace.json",
        "P21_CERTIFIED_OUTER_LOOP_COMPOSITION_RESULTS.md",
        "p21_shadow_trace_protocol.json",
    ):
        assert artifact in results_index
    for command in (
        "certify_outer_loop_composition.py",
        "reconstruct_outer_loop_composition.py",
        "run_p21_synthetic_shadow_trace.py",
    ):
        assert command in readme
        assert command in experiments
        assert command in results
    for text in (readme, results, claims, theorem, summary):
        assert "624350169" in text
        assert "999598040401" in text
        assert "16384" in text
        assert "32768" in text
        assert "125/1024" in text or r"\frac{125}{1024}" in text
        assert "509/512" in text or r"\frac{509}{512}" in text
        assert "1e-7" in text or "1/10000000" in text or "10^{-7}" in text
        assert "6889/2000" in text
        assert "-191/40" in text
        assert "4063/2000" in text
        assert "five" in text.lower()
        assert "stored" in text.lower()
        assert "pointwise" in text.lower()
        assert "incremental" in text.lower()
    combined = " ".join((readme, results, claims, theorem, summary, protocol))
    assert "1098368546995" in combined
    assert "549534922135" in combined
    assert "14267/274877906944" in combined
    assert "57049/549755813888" in combined
    assert "1/4096" in combined
    assert "2^-127" in combined
    assert "weight_decay=0" in combined or "weight decay to zero" in combined
    assert "768 x 2304" in combined
    assert "no real-gradient" in combined.lower() or "no real gradient" in combined.lower()
    assert "## 27. P21 certified stored-signal outer-loop composition" in results
    assert "## C27. Stored-signal finite-precision outer-loop composition" in claims
    assert "human proof audit of C27" in tasks
    assert "twenty-five technical goals" in readme


def test_current_indexes_record_p22_blocked_mps_repeatability_diagnostic() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    tasks = (ROOT / "TASKS.md").read_text(encoding="utf-8")
    results_index = (ROOT / "results/README.md").read_text(encoding="utf-8")
    experiments = (ROOT / "experiments/README.md").read_text(encoding="utf-8")
    results = (ROOT / "results/summaries/RESULTS.md").read_text(encoding="utf-8")
    claims = (ROOT / "theory/claims.md").read_text(encoding="utf-8")
    summary = (ROOT / "results/summaries/P22_REAL_GRADIENT_SHADOW_TRACE_RESULTS.md").read_text(
        encoding="utf-8"
    )
    protocol = (ROOT / "theory/p22_real_gradient_shadow_trace_protocol.md").read_text(
        encoding="utf-8"
    )
    erratum = (ROOT / "theory/p22_real_gradient_shadow_trace_protocol_erratum.md").read_text(
        encoding="utf-8"
    )

    combined = " ".join(
        (readme, tasks, results_index, experiments, results, claims, summary, protocol, erratum)
    )
    for text in (readme, results, claims, summary):
        assert "303" in text
        assert "step 0" in text
        assert "step 2" in text
        assert "trace-on" in text.lower()
        assert "CUDA" in text
    for operator_detail in (
        "1e-7",
        "6889/2000",
        "-191/40",
        "4063/2000",
        "f98f1cacc0263b04290753e32be8d498c1efc806",
    ):
        assert operator_detail in combined
    assert "## 28. P22 blocked Apple-MPS baseline-repeatability diagnostic" in results
    assert "## C28. P22 real-gradient shadow trace" in claims
    for artifact in (
        "P22_REAL_GRADIENT_SHADOW_TRACE_RESULTS.md",
        "p22_repeatability_failure_evidence.json",
        "p22_real_gradient_preflight_evidence.json",
        "p22_fineweb_materialization_evidence.json",
        "p22_scalable_shield_shape_extension_certificate.json",
        "P22_REAL_GRADIENT_SHADOW_TRACE_AUDIT.md",
        "p22_real_gradient_shadow_trace_protocol_erratum.md",
    ):
        assert artifact in results_index
    for text in (readme, results, results_index, summary):
        assert "p22_repeatability_failure_evidence.json" in text
    for text in (readme, experiments, results, results_index):
        assert "build_p22_repeatability_failure_evidence.py" in text
    assert "no trace-on run" in combined.lower()
    assert "not real-gradient fidelity evidence" in combined.lower()
    assert "twenty-five technical goals" in readme
    assert "79f33ec" in " ".join((claims, summary))
    assert "provenance caveat" in " ".join((claims, summary)).lower()
    assert "casts" in erratum
    assert "before" in erratum
    assert "normalization" in erratum
    assert "acquisition hash" in erratum


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
