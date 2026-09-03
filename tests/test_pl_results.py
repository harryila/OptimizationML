from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE_PATH = ROOT / "results/summaries/pl_convergence_certificate.json"
PROBE_PATH = ROOT / "results/summaries/pl_falsification.json"
CODE_COMMIT = "a8f650f6c60dcbc5d2f83647fd367348f4c67548"
P5_CHECKPOINT = "a549fb4c206335ef9ec264524e0f581216b250d4"
P5_TAG_OBJECT = "ad54ad81034a4e87bdf043d12afc0fa8f054cf8e"
P5_FULL_STEP_SHA256 = "f794b5075c16ef0393a6c2b6da06b77a1801cf8c2220cb768e2aa7a4fb3d7223"
P5_INCREMENTAL_SHA256 = "d9dac9bbd762586e4b84b921d5a015811d915b207d7898fcd1eedf76f83c1d9f"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_source_snapshot(snapshot: dict[str, str]) -> None:
    assert snapshot
    for relative_path, expected_digest in snapshot.items():
        assert _sha256(ROOT / relative_path) == expected_digest


def test_pl_result_locks_scope_operator_step_and_rate() -> None:
    payload = _load(CERTIFICATE_PATH)
    assert payload["schema_version"] == "passive-muon-pl-convergence-certificate-v1"
    assert payload["git"] == {
        "branch": "p6-pl-convergence",
        "dirty": False,
        "frozen_p5_checkpoint_commit": P5_CHECKPOINT,
        "frozen_p5_checkpoint_tag": "p5-checkpoint",
        "sha": CODE_COMMIT,
    }

    scope = payload["claim_scope"]
    assert "globally 10-smooth" in scope["objective_domain"]
    assert "global PL inequality with constant 1" in scope["objective_domain"]
    assert "nonconvex and nonunique-minimizer objectives" in scope["objective_domain"]
    assert scope["matrix_domain"] == "R^(m x n) for every fixed finite positive m,n"
    assert "function-value convergence and momentum decay" in scope["guarantee"]
    assert "not arbitrary-pair incremental" in scope["incremental_distinction"]
    assert "trajectory-dependent global minimizer" in scope["iterate_corollary"]
    assert any("unique minimizer" in item for item in scope["not_claimed"])
    assert any("BF16" in item for item in scope["not_claimed"])

    objective = payload["objective_class"]
    assert objective["finite_infimum_assumption"] == "f_star=inf_W f(W)>-infinity"
    assert Fraction(objective["smoothness_L"]["exact"]) == 10
    assert Fraction(objective["pl_constant_ell_PL"]["exact"]) == 1
    assert Fraction(objective["normalized_pl_constant_k"]["exact"]) == Fraction(1, 10)
    assert objective["weighted_function_coefficients"]["cancel_exactly"]

    operator = payload["operator"]
    assert operator["formula"] == "R(M)=H_h(M/max(c,||M||_F))+rho*M"
    assert operator["normalization"] == "exact fixed Frobenius max floor"
    assert operator["additive_epsilon"] == "none; denominator is max(c,||M||_F)"
    assert operator["orthogonalizer"] == "jordan_quintic"
    assert operator["newton_schulz_steps"] == 5
    assert operator["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert Fraction(operator["floor_c"]["exact"]) == 1
    assert Fraction(operator["constant_repair_rho"]["exact"]) == Fraction(
        210_177_835_339_081, 260_261_360_000
    )

    locked = payload["locked_exact_certificate"]
    assert Fraction(locked["learning_rate_eta"]["exact"]) == Fraction(1, 32_000)
    assert Fraction(locked["beta"]["exact"]) == Fraction(19, 20)
    assert Fraction(locked["exponential_norm_rate_tau"]["exact"]) == Fraction(19_999, 20_000)
    assert Fraction(locked["function_value_rate_q"]["exact"]) == Fraction(399_960_001, 400_000_000)
    assert (
        Fraction(locked["function_value_rate_q"]["exact"])
        == Fraction(locked["exponential_norm_rate_tau"]["exact"]) ** 2
    )


def test_pl_result_locks_exact_positive_storage_and_negative_lmi() -> None:
    locked = _load(CERTIFICATE_PATH)["locked_exact_certificate"]
    assert locked["storage_positive_definite"]
    assert locked["lmi_negative_definite"]
    assert locked["function_values_cancel"]
    assert locked["all_exact_checks_passed"]
    assert Fraction(locked["storage_normalization_trace_P_plus_c_F"]["exact"]) == 1

    storage_minors = [
        Fraction(item["exact"]) for item in locked["storage_leading_principal_minors"]
    ]
    negative_lmi_minors = [
        Fraction(item["exact"]) for item in locked["negative_lmi_leading_principal_minors"]
    ]
    assert len(storage_minors) == 2
    assert len(negative_lmi_minors) == 4
    assert all(minor > 0 for minor in storage_minors)
    assert all(minor > 0 for minor in negative_lmi_minors)

    p = [[Fraction(entry) for entry in row] for row in locked["storage_P"]]
    lmi = [[Fraction(entry) for entry in row] for row in locked["lmi_matrix"]]
    assert p == [
        [Fraction(14_487, 20_000), Fraction(-637, 20_000)],
        [Fraction(-637, 20_000), Fraction(499, 100_000)],
    ]
    assert all(p[i][j] == p[j][i] for i in range(2) for j in range(2))
    assert all(lmi[i][j] == lmi[j][i] for i in range(4) for j in range(4))


def test_pl_result_preserves_p5_and_source_snapshots() -> None:
    payload = _load(CERTIFICATE_PATH)
    p5 = payload["p5_checkpoint"]
    assert p5 == {
        "annotated_tag_git_oid": P5_TAG_OBJECT,
        "full_step_manifest_path": "results/summaries/nonquadratic_convergence_certificate.json",
        "full_step_manifest_sha256": P5_FULL_STEP_SHA256,
        "incremental_manifest_path": "results/summaries/nonquadratic_stability_certificate.json",
        "incremental_manifest_sha256": P5_INCREMENTAL_SHA256,
        "peeled_commit": P5_CHECKPOINT,
        "status": "frozen and unchanged; p6 is additive on a separate branch",
        "tag": "p5-checkpoint",
    }
    assert _sha256(ROOT / p5["full_step_manifest_path"]) == P5_FULL_STEP_SHA256
    assert _sha256(ROOT / p5["incremental_manifest_path"]) == P5_INCREMENTAL_SHA256
    _assert_source_snapshot(payload["proof_replay_provenance"]["source_snapshot"])


def test_pl_probe_locks_default_configuration_and_complete_factorial_coverage() -> None:
    payload = _load(PROBE_PATH)
    assert payload["schema_version"] == "passive-muon-pl-falsification-v1"
    assert "sampled passes do not prove" in payload["claim_scope"]["qualification"]
    assert payload["experiment_provenance"]["git"] == {
        "branch": "p6-pl-convergence",
        "dirty": False,
        "sha": CODE_COMMIT,
    }
    assert payload["experiment_provenance"]["seed"] == 2_026_090_2
    assert payload["experiment_provenance"]["dtype"] == "float64"
    assert payload["experiment_provenance"]["hardware"]["torch_device"] == "cpu"

    config = payload["config"]
    assert config["iterations"] == 500
    assert config["seed"] == 2_026_090_2
    assert config["shapes"] == [[2, 2], [3, 3]]
    assert config["rank_modes"] == ["full_rank", "codimension_one"]
    assert config["transition_scales"] == [0.0001, 1.0, 100.0]
    assert config["radius_multipliers"] == [0.5, math.sqrt(2.0), 4.0]
    assert config["momentum_modes"] == ["zero", "random"]
    assert math.isclose(config["beta"], 0.95, rel_tol=0.0, abs_tol=1e-15)
    assert math.isclose(config["learning_rate"], 1 / 32_000, rel_tol=0.0, abs_tol=1e-18)

    trials = payload["trials"]
    actual_cases = {
        (
            tuple(trial["shape"]),
            trial["rank_mode"],
            trial["transition_scale"],
            trial["radius_multiplier"],
            trial["momentum_mode"],
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
        )
    )
    assert len(trials) == 72
    assert actual_cases == expected_cases
    assert all(trial["requested_iterations"] == 500 for trial in trials)
    assert all(trial["executed_iterations"] == 500 for trial in trials)


def test_pl_probe_records_nonconvex_nonunique_stress_and_no_candidate_violation() -> None:
    payload = _load(PROBE_PATH)
    summary = payload["summary"]
    assert summary["trial_count"] == 72
    assert summary["rank_deficient_case_count"] == 36
    assert summary["orientation_changing_case_count"] == 72
    assert summary["negative_curvature_case_count"] > 0
    assert summary["objective_increasing_case_count"] > 0
    assert summary["maximum_orientation_commutator"] > 0.25
    assert summary["sampled_hessian_eigenvalue_minimum"] <= -0.37
    assert summary["sampled_hessian_eigenvalue_minimum"] >= -0.375 - 3e-14
    assert summary["sampled_hessian_eigenvalue_maximum"] <= 9.0 + 3e-14
    assert summary["minimum_sampled_pl_ratio"] >= 1.0 - 3e-12

    assert summary["candidate_violation_count"] == 0
    assert summary["objective_bound_violating_case_count"] == 0
    assert summary["lyapunov_rate_violating_case_count"] == 0
    assert summary["nonpositive_lyapunov_case_count"] == 0
    assert summary["unresolved_lyapunov_resurgence_case_count"] == 0
    assert summary["divergence_count"] == 0
    assert summary["nonfinite_count"] == 0
    q = float(Fraction(payload["locked_pl_storage"]["rate_squared"]))
    assert summary["maximum_lyapunov_ratio"] < q
    assert summary["maximum_normalized_lyapunov_rate_excess"] < 0
    assert summary["maximum_resolved_one_step_gap_ratio"] > 1
    assert summary["worst_final_objective_gap_ratio"] < 1e-12
    assert summary["worst_final_momentum_to_peak_ratio"] < 1e-6


def test_pl_probe_locks_operator_references_and_source_snapshot() -> None:
    payload = _load(PROBE_PATH)
    assert payload["locked_parameters"] == {
        "beta": "19/20",
        "floor": "1",
        "learning_rate_eta": "1/32000",
        "objective_PL_lower": "1",
        "objective_sharper_smoothness_upper": "9",
        "objective_smoothness_upper_used_by_target": "10",
        "repair_rho": "210177835339081/260261360000",
    }
    assert payload["operator"]["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert payload["operator"]["iterations"] == 5
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["additive_epsilon"] == (
        "not_applicable; denominator uses a max floor"
    )

    certificate_reference = payload["certificate_reference"]
    assert _sha256(ROOT / certificate_reference["source"]) == certificate_reference["source_sha256"]
    assert (
        _sha256(ROOT / certificate_reference["theory_note"])
        == certificate_reference["theory_note_sha256"]
    )
    p5 = payload["p5_reference"]
    assert p5["checkpoint"] == P5_CHECKPOINT
    assert p5["manifest_sha256"] == P5_FULL_STEP_SHA256
    assert _sha256(ROOT / p5["manifest_path"]) == P5_FULL_STEP_SHA256
    _assert_source_snapshot(payload["experiment_provenance"]["source_snapshot"])
