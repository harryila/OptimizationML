from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "mixed_precision" / "run_p20_scalable_sector_shield_study.py"
SPEC = importlib.util.spec_from_file_location("p20_scalable_sector_shield_study", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P20 scalable-sector-shield study")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

CANDIDATE_P18 = EXPERIMENT.CANDIDATE_P18
CANDIDATE_UPSTREAM = EXPERIMENT.CANDIDATE_UPSTREAM
LOCKED_DEFAULT_DECISION_DIGEST = EXPERIMENT.LOCKED_DEFAULT_DECISION_DIGEST
P20StudyConfig = EXPERIMENT.P20StudyConfig
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
canonical_json = EXPERIMENT.canonical_json
run_study = EXPERIMENT.run_study


def _small_config(**changes: object) -> object:
    return replace(P20StudyConfig(operating_annulus_grid_points=5), **changes)


@pytest.fixture(scope="module")
def small_payload() -> dict[str, object]:
    return run_study(_small_config())


@pytest.fixture(scope="module")
def default_payload() -> dict[str, object]:
    return run_study()


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_scope_separates_theorem_sampling_and_literal_backend(
    small_payload: dict[str, object],
) -> None:
    assert small_payload["schema_version"] == SCHEMA_VERSION
    scope = small_payload["claim_scope"]
    assert "independent P20 exact" in scope["proof_boundary"]
    assert "any finite" in scope["candidate_independence"]
    assert "canonical and annulus 2x2" in scope["literal_upstream_boundary"]
    assert "without backend parity" in scope["literal_upstream_boundary"]
    assert "not global" in scope["sampled_boundary"]

    provenance = small_payload["upstream_formula_provenance"]
    assert provenance["revision"] == "f98f1cacc0263b04290753e32be8d498c1efc806"
    assert provenance["audited_file_sha256"] == (
        "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
    )
    assert provenance["normalization"].endswith("+1e-7)")
    assert "before aspect scaling" in provenance["candidate_scope"]


def test_canonical_literal_upstream_and_p18_candidates_are_inactive(
    small_payload: dict[str, object],
) -> None:
    canonical = small_payload["canonical_diag_3_4"]
    assert "literal clean-room" in canonical["execution_scope"]
    assert "full 2x2 P20 shield" in canonical["execution_scope"]
    assert canonical["reduced_helper_matches_literal_diagonal_bits"]
    assert canonical["p18_solver_residual_passes"]
    assert canonical["shielded_p18_fidelity"]["meaningful_fidelity_gate_passes"]
    for family in (CANDIDATE_P18, CANDIDATE_UPSTREAM):
        record = canonical["candidate_records"][family]
        assert record["bitwise_identity"]
        assert record["offline_exact_p19_disk_check_passes"]
        assert not record["shield_diagnostics"]["active"]
        assert not record["shield_diagnostics"]["reduced_spectrum_diagnostic"]


def test_full_dense_fp32_and_bf16_reference_paths_execute(
    small_payload: dict[str, object],
) -> None:
    smoke = small_payload["dense_reference_smoke"]
    assert "full configured 768x768" in smoke["scope"]
    for name, expected_dtype in (
        ("fp32_inputs", "torch.float32"),
        ("bf16_inputs", "torch.bfloat16"),
    ):
        record = smoke[name]
        assert record["bitwise_identity"]
        assert record["offline_exact_p19_disk_check_passes"]
        diagnostics = record["shield_diagnostics"]
        assert diagnostics["signal_input_dtype"] == expected_dtype
        assert diagnostics["candidate_input_dtype"] == expected_dtype
        assert diagnostics["output_dtype"] == "torch.float32"
        assert not diagnostics["reduced_spectrum_diagnostic"]
        assert diagnostics["signal_norm"]["reduction_depth"] == 20


def test_p18_annulus_is_normally_inactive_and_upstream_candidate_is_safely_screened(
    small_payload: dict[str, object],
) -> None:
    annulus = small_payload["operating_annulus"]
    assert "literal 2x2 CPU paths" in annulus["scope"]
    assert annulus["case_count"] > 100
    assert annulus["candidate_evaluation_count"] == 2 * annulus["case_count"]

    p18 = annulus["candidate_families"][CANDIDATE_P18]
    assert p18["inactive"] > p18["active"]
    assert p18["bitwise_identity"] == p18["inactive"]
    assert p18["clipped"] == p18["active"]
    assert p18["fallback"] == 0
    assert p18["active_accounting_closes"]
    assert p18["all_outputs_pass_offline_exact_disk_check"]
    assert p18["all_informative_fidelity_pass"]

    upstream = annulus["candidate_families"][CANDIDATE_UPSTREAM]
    assert 0 < upstream["active"] < upstream["evaluated"]
    assert upstream["clipped"] == upstream["active"]
    assert upstream["fallback"] == 0
    assert upstream["active_accounting_closes"]
    assert upstream["bitwise_identity"] == upstream["inactive"]
    assert upstream["all_outputs_pass_offline_exact_disk_check"]
    assert upstream["minimum_cosine"] > 0.7
    interpretation = annulus["guarded_p18_inactivity_interpretation"]
    assert interpretation["normally_inactive"]
    assert interpretation["half_fallback_count"] == 0
    assert "no entire-annulus" in interpretation["qualification"]


def test_all_seven_transformer_shapes_and_both_candidates_are_exercised(
    small_payload: dict[str, object],
) -> None:
    transformer = small_payload["transformer_spectrum_diagnostics"]
    expected_shapes = [
        [768, 768],
        [768, 3_072],
        [768, 50_257],
        [3_072, 12_288],
        [4_096, 4_096],
        [4_096, 11_008],
        [4_096, 14_336],
    ]
    assert transformer["declared_shapes"] == expected_shapes
    assert "no dense allocation" in transformer["scope"]
    assert "or claim of literal" in transformer["scope"]
    assert "pinned-formula" in transformer["upstream_scope"]
    assert len(transformer["exact_shape_margin_references"]) == 7
    assert all(item["certified"] for item in transformer["exact_shape_margin_references"])
    assert all(
        float(item["inward_margin"]["decimal"]) > 0.0
        for item in transformer["exact_shape_margin_references"]
    )

    summary = transformer["summary"]
    assert summary["shape_count"] == 7
    assert summary["case_count"] == 21
    assert summary["candidate_evaluation_count"] == 42
    assert summary["all_p18_solver_residuals_pass"]
    for family in (CANDIDATE_P18, CANDIDATE_UPSTREAM):
        assert summary[family]["evaluated"] == 21
        assert summary[family]["all_outputs_pass_offline_exact_disk_check"]
        assert summary[family]["fallback"] == 0
        assert summary[family]["clipped"] == summary[family]["active"]
    interpretation = summary["guarded_p18_inactivity_interpretation"]
    assert interpretation["operating_spectrum_case_count"] == 14
    assert interpretation["operating_spectrum_inactive_count"] == 14
    assert interpretation["all_operating_spectrum_cases_inactive"]
    assert interpretation["flat_boundary_stress_case_count"] == 7
    assert interpretation["flat_boundary_stress_active_count"] == 7
    assert "intentionally activates" in interpretation["qualification"]
    assert summary[CANDIDATE_UPSTREAM]["active"] > 0

    assert all(
        record["candidate_records"][family]["shield_diagnostics"]["reduced_spectrum_diagnostic"]
        for record in transformer["cases"]
        for family in (CANDIDATE_P18, CANDIDATE_UPSTREAM)
    )


def test_adversarial_and_one_ulp_candidates_are_fail_closed_or_shielded(
    small_payload: dict[str, object],
) -> None:
    controls = small_payload["adversarial_controls"]
    records = controls["finite_and_nonfinite_candidate_controls"]
    assert set(records) == {
        "zero",
        "outward",
        "anti_aligned",
        "orthogonal_corruption",
        "one_ulp_outward",
        "nan_candidate",
        "infinite_candidate",
    }
    for record in records.values():
        assert record["returned"]
        assert record["offline_exact_p19_disk_check_passes"]
        assert record["shield_diagnostics"]["active"]
    one_ulp = records["one_ulp_outward"]
    assert one_ulp["candidate_was_inside_original_disk"] is False
    assert one_ulp["candidate_offline_exact_p19_disk_margin"].startswith("-")
    finite_names = {
        "zero",
        "outward",
        "anti_aligned",
        "orthogonal_corruption",
        "one_ulp_outward",
    }
    for name in finite_names:
        assert records[name]["shield_diagnostics"]["candidate_clipped"]
        assert not records[name]["shield_diagnostics"]["used_fallback"]
    for name in ("nan_candidate", "infinite_candidate"):
        assert not records[name]["shield_diagnostics"]["candidate_clipped"]
        assert records[name]["shield_diagnostics"]["used_fallback"]
        assert records[name]["shield_diagnostics"]["fail_closed"]

    zero = controls["zero_signal_controls"]
    assert zero["zero_candidate"]["bitwise_identity"]
    assert not zero["zero_candidate"]["shield_diagnostics"]["active"]
    assert zero["finite_nonzero_candidate"]["shield_diagnostics"]["active"]
    assert zero["nonfinite_candidate"]["shield_diagnostics"]["fail_closed"]
    assert set(controls["nonfinite_signal_rejections"]) == {
        "nan_signal",
        "infinite_signal",
    }
    assert all(
        not record["returned"] for record in controls["nonfinite_signal_rejections"].values()
    )


def test_normal_subnormal_and_ftz_policies_are_explicit(
    small_payload: dict[str, object],
) -> None:
    near_zero = small_payload["near_zero_and_ftz_controls"]
    assert "gradual underflow" in near_zero["ftz_policy"]
    assert "near-zero finite-precision neighborhood" in near_zero["finite_precision_neighborhood"]
    cases = near_zero["cases"]
    assert cases["minimum_normal"]["returned"]
    assert cases["minimum_normal"]["offline_exact_p19_disk_check_passes"]
    assert cases["even_subnormal_exact_halving"]["returned"]
    assert cases["even_subnormal_exact_halving"]["shield_diagnostics"]["exact_halving_guard"]
    assert cases["even_subnormal_exact_halving"]["offline_exact_p19_disk_check_passes"]
    for name in ("maximum_subnormal", "minimum_subnormal"):
        assert not cases[name]["returned"]
        assert cases[name]["exception_type"] == "NearZeroUnrepresentable"


def test_runtime_contract_locks_arithmetic_and_has_no_big_integer_postcheck(
    small_payload: dict[str, object],
) -> None:
    contract = small_payload["locked_runtime_contract"]
    manifest = contract["representative_manifest"]
    assert manifest["matrix_domain"]["shape"] == [4_096, 11_008]
    assert manifest["matrix_domain"]["input_dtypes"] == [
        "torch.float32",
        "torch.bfloat16",
    ]
    assert manifest["matrix_domain"]["output_dtype"] == "torch.float32"
    assert manifest["operation_graph"]["fma_allowed"] is False
    assert manifest["operation_graph"]["reassociation_allowed"] is False
    assert manifest["backend"]["per_output_fraction_or_big_integer_postcheck"] is False
    assert manifest["backend"]["runtime_uses_generated_shape_table"] is True
    assert contract["backend_self_check"]["ftz_daz"] is False
    assert contract["backend_self_check"]["gradual_underflow"] is True


def test_cross_platform_strategy_hashes_decisions_not_metric_extrema(
    small_payload: dict[str, object],
) -> None:
    replay = small_payload["cross_platform_replay"]
    assert replay["runtime_big_integer_check"] is False
    assert replay["offline_small_control_exact_check"] is True
    assert "macOS arm64" in replay["strategy"]
    assert "Ubuntu x86_64" in replay["strategy"]
    assert "archive raw bit hashes separately" in replay["strategy"]
    assert "does not by itself prove" in replay["cross_platform_status"]
    assert replay["locked_default_decision_digest"] == LOCKED_DEFAULT_DECISION_DIGEST
    assert replay["this_run_uses_locked_default_grid"] is False
    assert replay["locked_default_digest_matches"] is None
    assert len(small_payload["decision_digest"]) == 64
    hardware = small_payload["experiment_provenance"]["hardware"]
    assert hardware["system"]
    assert hardware["machine"]


def test_full_default_discrete_decisions_are_locked_across_platforms(
    default_payload: dict[str, object],
) -> None:
    assert default_payload["decision_digest"] == LOCKED_DEFAULT_DECISION_DIGEST
    replay = default_payload["cross_platform_replay"]
    assert replay["this_run_uses_locked_default_grid"] is True
    assert replay["locked_default_digest_matches"] is True

    annulus = default_payload["operating_annulus"]
    p18 = annulus["candidate_families"][CANDIDATE_P18]
    assert annulus["case_count"] == 2_688
    assert p18["inactive"] == 2_671
    assert p18["clipped"] == 17
    assert p18["fallback"] == 0
    assert p18["informative"] == p18["fidelity_passes"] == 2_176
    upstream = annulus["candidate_families"][CANDIDATE_UPSTREAM]
    assert upstream["inactive"] == 761
    assert upstream["clipped"] == 1_927
    assert upstream["fallback"] == 0

    transformer = default_payload["transformer_spectrum_diagnostics"]["summary"]
    assert transformer[CANDIDATE_P18]["bitwise_identity"] == 14
    assert transformer[CANDIDATE_P18]["clipped"] == 7
    assert transformer[CANDIDATE_P18]["fallback"] == 0
    assert transformer[CANDIDATE_UPSTREAM]["clipped"] == 18
    assert transformer[CANDIDATE_UPSTREAM]["fallback"] == 0


def test_small_payload_is_deterministic_ignoring_provenance() -> None:
    config = _small_config()
    first = run_study(config)
    second = run_study(config)
    assert first["decision_digest"] == second["decision_digest"]
    assert _without_provenance(first) == _without_provenance(second)


@pytest.mark.parametrize(
    ("change", "exception", "match"),
    [
        ({"seed": -1}, ValueError, "nonnegative"),
        ({"seed": True}, ValueError, "nonnegative"),
        ({"operating_annulus_grid_points": 4}, ValueError, "at least five"),
        ({"operating_annulus_grid_points": True}, ValueError, "at least five"),
        ({"include_transformer_spectra": 1}, TypeError, "Boolean"),
        ({"torch_threads": 0}, ValueError, "positive"),
        ({"torch_threads": True}, ValueError, "positive"),
        ({"solver": object()}, TypeError, "EquivariantResolventSolverConfig"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        replace(P20StudyConfig(), **change)


def test_cli_writes_canonical_json_for_small_grid(tmp_path: Path) -> None:
    output = tmp_path / "p20-study.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "20",
            "--operating-annulus-grid-points",
            "5",
            "--skip-transformer-spectra",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert output.read_text(encoding="utf-8") == canonical_json(payload)
    assert payload["config"]["seed"] == 20
    assert payload["transformer_spectrum_diagnostics"]["cases"] == []
    assert payload["canonical_target"] == (
        "results/summaries/p20_scalable_sector_shield_study.json"
    )
