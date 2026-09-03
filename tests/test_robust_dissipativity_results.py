from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_PATH = ROOT / "results/summaries/robust_dissipativity_certificate.json"
PROBE_PATH = ROOT / "results/summaries/robust_dissipativity_falsification.json"
SOURCE_COMMIT = "c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a"
P6_CHECKPOINT = "ef88d8f5b26148af0ec1ca70b506048938bf9bef"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_source_snapshot(snapshot: dict[str, str]) -> None:
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        assert _sha256(ROOT / relative_path) == expected_digest


def test_p7_result_locks_scope_operator_update_and_gains() -> None:
    payload = _load(CERTIFICATE_PATH)
    assert payload["schema_version"] == ("passive-muon-robust-dissipativity-certificate-v1")
    assert payload["git"] == {
        "branch": "p7-robust-dissipativity",
        "dirty": False,
        "sha": SOURCE_COMMIT,
    }

    scope = payload["claim_scope"]
    assert "for every disturbance realization" in scope["guarantee"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert "global PL inequality with constant 1" in scope["objective_domain"]
    assert scope["initialization"] == "arbitrary finite same-shaped W_0 and m_0"
    assert "storage/output ISS" in scope["iss_scope"]
    assert "not full-state ISS in W" in scope["iss_scope"]
    assert any("BF16" in item for item in scope["not_claimed"])

    operator = payload["operator"]
    assert operator["normalization_rule"] == ("M/max(c,||M||_F), exact fixed Frobenius max floor")
    assert operator["epsilon"].startswith("none")
    assert Fraction(operator["floor_c"]["exact"]) == 1
    assert operator["polynomial_coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["newton_schulz_iteration_count"] == 5
    assert Fraction(operator["repair_deficit_upper"]["exact"]) == Fraction(
        41_528_474_059_081, 260_261_360_000
    )
    assert Fraction(operator["repair_margin_mu"]["exact"]) == 648
    assert Fraction(operator["constant_repair_rho"]["exact"]) == Fraction(
        210_177_835_339_081, 260_261_360_000
    )

    update = payload["disturbed_update"]
    assert "grad_f(W_t)+xi_t" in update["gradient_measurement"]
    assert "g_hat_t" in update["momentum"]
    assert "g_hat_t" in update["nesterov_signal"]
    assert "R(s_(t+1))+e_t" in update["parameter_update"]
    assert "arbitrary finite same-shape" in update["disturbance_assumption_for_pathwise_result"]

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["beta"]["exact"]) == Fraction(19, 20)
    assert Fraction(locked["smoothness_L"]["exact"]) == 10
    assert Fraction(locked["pl_constant_ell_PL"]["exact"]) == 1
    assert Fraction(locked["rate_q"]["exact"]) == Fraction(399_960_001, 400_000_000)
    assert Fraction(locked["physical_gains"]["gamma_g"]["exact"]) == Fraction(1, 2)
    assert Fraction(locked["physical_gains"]["gamma_R"]["exact"]) == Fraction(1, 2_000_000)


def test_p7_result_locks_exact_lmi_and_deterministic_stochastic_consequences() -> None:
    payload = _load(CERTIFICATE_PATH)
    locked = payload["locked_exact_certificate"]
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["function_coefficients"]["cancel_exactly"]
    assert locked["all_exact_checks_passed"]
    assert len(locked["lmi_matrix"]) == 6
    assert all(len(row) == 6 for row in locked["lmi_matrix"])
    minors = [Fraction(item["exact"]) for item in locked["negative_lmi_leading_principal_minors"]]
    assert len(minors) == 6
    assert all(value > 0 for value in minors)

    consequences = payload["consequences"]
    ultimate = consequences["ultimate_storage_coefficients_for_norm_bounds"]
    assert Fraction(ultimate["gradient_noise_X_squared"]["exact"]) == Fraction(200_000_000, 39_999)
    assert Fraction(ultimate["implementation_error_E_squared"]["exact"]) == Fraction(200, 39_999)
    assert "V_t->0" in consequences["square_summable_inputs"]
    assert "do not imply convergence of W_t" in consequences["iterate_qualification"]
    assert "bounded second moments" in consequences["bounded_second_moment_stochastic"]
    assert "E[V_0]<infinity" in consequences["stochastic_assumptions"]
    assert "variance bound" in consequences["unbiasedness_note"]


def test_p7_result_preserves_p6_and_source_snapshot() -> None:
    payload = _load(CERTIFICATE_PATH)
    p6 = payload["p6_provenance"]
    assert p6["source_generation_commit"] == ("a8f650f6c60dcbc5d2f83647fd367348f4c67548")
    assert p6["final_checkpoint_commit"] == P6_CHECKPOINT
    assert p6["checkpoint_tag"] == "p6-checkpoint"
    assert p6["annotated_tag_git_oid"] == ("678e4ead01732c2a0d8db786147a063f0ae119ec")
    p6_manifest = ROOT / p6["result_manifest_path"]
    assert _sha256(p6_manifest) == p6["result_manifest_sha256"]
    _assert_source_snapshot(payload["proof_replay_provenance"]["source_snapshot"])


def test_p7_probe_locks_default_grid_and_no_candidate_violation() -> None:
    payload = _load(PROBE_PATH)
    assert payload["schema_version"] == ("passive-muon-robust-dissipativity-falsification-v1")
    assert payload["experiment_provenance"]["git"] == {
        "branch": "p7-robust-dissipativity",
        "dirty": False,
        "sha": SOURCE_COMMIT,
    }
    assert payload["experiment_provenance"]["seed"] == 2_026_090_3
    assert payload["experiment_provenance"]["dtype"] == "float64"
    assert payload["experiment_provenance"]["hardware"]["torch_device"] == "cpu"

    config = payload["config"]
    assert config["iterations"] == 120
    assert config["harmonic_horizon"] == 10_000
    assert config["shapes"] == [[2, 2], [2, 3]]
    assert config["rank_modes"] == ["full_rank", "codimension_one"]
    assert config["transition_scales"] == [0.001, 1.0, 100.0]
    assert config["radius_multipliers"] == [math.sqrt(2.0), 4.0]
    assert config["momentum_modes"] == ["zero", "random"]
    assert config["disturbance_profiles"] == [
        "deterministic_bounded",
        "seeded_stochastic",
        "implementation_only",
    ]
    assert math.isclose(config["learning_rate"], 1 / 32_000, rel_tol=0.0, abs_tol=1e-18)

    trials = payload["trials"]
    actual_cases = {
        (
            tuple(trial["shape"]),
            trial["rank_mode"],
            trial["transition_scale"],
            trial["radius_multiplier"],
            trial["momentum_mode"],
            trial["disturbance_profile"],
        )
        for trial in trials
    }
    expected_cases = set(
        itertools.product(
            map(tuple, config["shapes"]),
            config["rank_modes"],
            config["transition_scales"],
            config["radius_multipliers"],
            config["momentum_modes"],
            config["disturbance_profiles"],
        )
    )
    assert len(trials) == 144
    assert actual_cases == expected_cases
    assert sum(trial["executed_iterations"] for trial in trials) == 17_280
    assert all(trial["executed_iterations"] == 120 for trial in trials)

    summary = payload["summary"]
    assert summary["trial_count"] == 144
    assert summary["rank_deficient_case_count"] == 72
    assert summary["deterministic_bounded_case_count"] == 48
    assert summary["seeded_stochastic_case_count"] == 48
    assert summary["implementation_only_case_count"] == 48
    assert summary["candidate_violation_count"] == 0
    assert summary["inequality_violation_count"] == 0
    assert summary["nonfinite_count"] == 0
    assert summary["divergence_count"] == 0
    assert summary["maximum_dissipation_excess"] < 0


def test_p7_probe_locks_harmonic_boundary_and_provenance() -> None:
    payload = _load(PROBE_PATH)
    harmonic = payload["harmonic_drift_counterexample"]
    assert harmonic["maximum_objective_gap"] == 0
    assert harmonic["maximum_lyapunov_value"] == 0
    assert harmonic["zero_signal_operator_residual"] == 0
    assert harmonic["finite_prefix_error_energy"] < math.pi**2 / 6
    assert harmonic["finite_prefix_absolute_error_mass"] > math.log(harmonic["horizon"])
    assert harmonic["final_iterate_displacement"] > 0
    assert "do not imply convergence of W_t" in harmonic["interpretation"]

    p6 = payload["p6_reference"]
    assert p6["checkpoint"] == P6_CHECKPOINT
    assert p6["checkpoint_tag"] == "p6-checkpoint"
    assert _sha256(ROOT / p6["result_manifest_path"]) == p6["result_manifest_sha256"]
    _assert_source_snapshot(payload["experiment_provenance"]["source_snapshot"])
