from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "mixed_precision" / "run_finite_precision_outer_loop_diagnostic.py"
CANONICAL = ROOT / "results" / "summaries" / "finite_precision_outer_loop_diagnostic.json"
SPEC = importlib.util.spec_from_file_location("finite_precision_outer_loop_diagnostic", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P10 finite-precision outer-loop diagnostic")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

DiagnosticConfig = EXPERIMENT.DiagnosticConfig
DEFAULT_SHAPES = EXPERIMENT.DEFAULT_SHAPES
EMA_CASE_FAMILIES = EXPERIMENT.EMA_CASE_FAMILIES
MASTER_CASE_FAMILIES = EXPERIMENT.MASTER_CASE_FAMILIES
MAX_DIAGNOSTIC_ENTRIES = EXPERIMENT.MAX_DIAGNOSTIC_ENTRIES
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
build_ema_cases = EXPERIMENT.build_ema_cases
build_master_cases = EXPERIMENT.build_master_cases
canonical_json = EXPERIMENT.canonical_json
run_diagnostic = EXPERIMENT.run_diagnostic


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_default_diagnostic_is_complete_exact_on_selected_entries_and_scoped() -> None:
    payload = run_diagnostic(DiagnosticConfig())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["config"]["seed"] == 20_260_903
    assert payload["config"]["shapes"] == [list(shape) for shape in DEFAULT_SHAPES]
    assert payload["config"]["maximum_diagnostic_entries"] == MAX_DIAGNOSTIC_ENTRIES
    assert payload["config"]["ema_case_families"] == list(EMA_CASE_FAMILIES)
    assert payload["config"]["master_case_families"] == list(MASTER_CASE_FAMILIES)
    assert payload["config"]["entries_checked_exhaustively_within_each_selected_case"] is True

    summary = payload["summary"]
    assert summary["shape_count"] == 3
    assert summary["ema_case_count"] == 9
    assert summary["master_case_count"] == 6
    assert summary["exact_ema_residual_entry_check_count"] == 810
    assert summary["exact_master_residual_entry_check_count"] == 270
    assert summary["exact_two_sum_identity_check_count"] == 544
    assert summary["exact_logical_update_identity_check_count"] == 270
    assert summary["exact_ema_residual_violation_count"] == 0
    assert summary["exact_master_residual_violation_count"] == 0
    assert summary["exact_identity_violation_count"] == 0
    assert summary["actual_p9_stalling_witness_certified"] is True
    assert summary["all_selected_falsification_checks_pass"] is True

    scope = payload["claim_scope"]
    assert "falsification" in scope["status"]
    assert "not a global certificate" in scope["status"]
    assert "cannot prove" in scope["exact_check_qualification"]
    assert "not a convergence theorem" in scope["stalling_qualification"]
    assert "only the small 2x2" in scope["operator_qualification"]


def test_every_reported_worst_entry_satisfies_its_exact_fraction_envelope() -> None:
    payload = run_diagnostic(DiagnosticConfig(shapes=((2, 3), (4, 5))))

    for case in payload["ema_nesterov_cases"]:
        for port in ("r_m", "r_s"):
            audit = case[port]
            worst = audit["worst_entry"]
            assert audit["all_entries_within_exact_envelope"] is True
            assert audit["violation_count"] == 0
            assert Fraction(worst["absolute_residual_exact"]) <= Fraction(worst["bound_exact"])
            assert Fraction(worst["slack_exact"]) >= 0
            assert Fraction(worst["utilization_exact"]) <= 1

    for case in payload["compensated_master_cases"]:
        audit = case["r_W"]
        worst = audit["worst_entry"]
        identities = case["exact_identities"]
        assert audit["all_entries_within_exact_envelope"] is True
        assert Fraction(worst["absolute_residual_exact"]) <= Fraction(worst["bound_exact"])
        assert identities["middle_two_sum_violation_count"] == 0
        assert identities["high_two_sum_violation_count"] == 0
        assert identities["logical_update_violation_count"] == 0
        assert case["all_exact_checks_pass"] is True


def test_standalone_two_sum_and_actual_p9_stall_records_are_exact() -> None:
    payload = run_diagnostic(DiagnosticConfig(shapes=((1, 1),)))

    two_sum = payload["standalone_two_sum"]
    assert two_sum["case_count"] == 4
    assert two_sum["violation_count"] == 0
    for case in two_sum["cases"]:
        assert case["identity_holds"] is True
        assert Fraction(case["rounded_exact"]) + Fraction(case["residual_exact"]) == (
            Fraction(case["left_exact"]) + Fraction(case["right_exact"])
        )

    witness = payload["actual_p9_raw_stall_vs_compensation_witness"]
    assert witness["certified"] is True
    assert all(witness["checks"].values())
    assert witness["scope"] == "actual P9 repaired-operator parameter-stalling witness"
    assert witness["raw_final_bits_hex"] == witness["initial_fp32_bits_hex"]
    assert Fraction(witness["rounded_step_exact"]) > 0
    assert Fraction(witness["compensated_first_middle_exact"]) < 0
    assert witness["compensated_first_high_move_iteration"] is not None
    assert Fraction(witness["compensated_final"]["logical_exact"]) < Fraction(
        witness["initial_exact"]
    )


def test_case_builders_are_seeded_and_keep_structural_edges_fixed() -> None:
    left_ema = build_ema_cases((3, 4), seed=11, shape_index=0)
    right_ema = build_ema_cases((3, 4), seed=12, shape_index=0)
    assert [case[0] for case in left_ema] == list(EMA_CASE_FAMILIES)
    assert torch.equal(left_ema[0][1], right_ema[0][1])
    assert torch.equal(left_ema[0][2], right_ema[0][2])
    assert not torch.equal(left_ema[1][1], right_ema[1][1])
    assert not torch.equal(left_ema[2][1], right_ema[2][1])

    left_master = build_master_cases((3, 4), seed=11, shape_index=0)
    right_master = build_master_cases((3, 4), seed=12, shape_index=0)
    assert [case[0] for case in left_master] == list(MASTER_CASE_FAMILIES)
    assert torch.equal(left_master[0][1].high, right_master[0][1].high)
    assert torch.equal(left_master[0][2], right_master[0][2])
    assert not torch.equal(left_master[1][1].high, right_master[1][1].high)
    assert not torch.equal(left_master[1][2], right_master[1][2])


@pytest.mark.parametrize(
    ("change", "exception", "match"),
    [
        ({"seed": -1}, ValueError, "nonnegative"),
        ({"seed": True}, ValueError, "nonnegative"),
        ({"shapes": ()}, ValueError, "at least one"),
        ({"shapes": ((2, 0),)}, ValueError, "positive"),
        ({"shapes": ((65, 65),)}, ValueError, "at most"),
        ({"shapes": ((2, 2), (2, 2))}, ValueError, "unique"),
        ({"torch_threads": 0}, ValueError, "positive"),
        ({"torch_threads": True}, ValueError, "positive"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object], exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        replace(DiagnosticConfig(), **change)


def test_default_numerical_payload_is_deterministic_ignoring_provenance() -> None:
    config = DiagnosticConfig(shapes=((2, 3),))
    first = run_diagnostic(config)
    second = run_diagnostic(config)
    assert _without_provenance(first) == _without_provenance(second)


def test_provenance_and_operator_contract_are_complete() -> None:
    payload = run_diagnostic(DiagnosticConfig(shapes=((2, 3),)))

    operator = payload["config"]["bound_operator"]
    outer_loop = payload["config"]["outer_loop"]
    assert outer_loop["ema_nesterov_operation_order"] == (
        "bg=fl32(a32*g); bm=fl32(beta32*m); m_next=fl32(bm+bg); "
        "bs=fl32(beta32*m_next); s_next=fl32(bs+bg)"
    )
    assert operator["domain"] == "finite matrices of each locked shape"
    assert operator["normalization"] == "s/max(1,||s||_F)"
    assert operator["epsilon"] == "none"
    assert operator["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["iteration_count"] == 5
    assert operator["repair_rho_exact"]

    provenance = payload["experiment_provenance"]
    assert len(provenance["git"]["sha"]) == 40
    assert isinstance(provenance["git"]["dirty"], bool)
    assert provenance["seed"] == 20_260_903
    assert provenance["hardware"]["device"] == "cpu"
    assert provenance["software"]["python"]
    assert provenance["software"]["numpy"]
    assert provenance["software"]["torch"]
    assert provenance["runtime"]["torch_num_threads"] == 1
    assert provenance["runtime"]["deterministic_algorithms_enabled"] is True
    assert set(provenance["source_snapshot"]) == {
        "experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py",
        "src/passive_muon/finite_precision_outer_loop.py",
        "src/passive_muon/scalable_mixed_precision.py",
        "src/passive_muon/scalable_mixed_precision_certificate.py",
        "src/passive_muon/structure_aware_stability.py",
        "tests/test_finite_precision_outer_loop_diagnostic.py",
        "pyproject.toml",
        "uv.lock",
    }


def test_cli_writes_only_when_explicitly_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    arguments = EXPERIMENT.parse_args()
    assert arguments.output is None
    assert arguments.write_canonical is False

    output = tmp_path / "p10-diagnostic.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "17",
            "--torch-threads",
            "1",
            "--shape",
            "2x3",
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
    assert payload["config"]["seed"] == 17
    assert payload["config"]["shapes"] == [[2, 3]]
    assert payload["summary"]["ema_case_count"] == 3
    assert payload["summary"]["master_case_count"] == 2
    assert payload["canonical_target"] == (
        "results/summaries/finite_precision_outer_loop_diagnostic.json"
    )


def test_committed_diagnostic_is_mandatory_clean_and_hash_current() -> None:
    assert CANONICAL.is_file(), "the committed P10 diagnostic artifact is mandatory"
    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    git = payload["experiment_provenance"]["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p10-finite-precision-outer-loop"
    assert git["dirty"] is False
    assert payload["config"]["seed"] == 20_260_903
    assert payload["summary"]["all_selected_falsification_checks_pass"] is True
    for relative_path, expected_digest in payload["experiment_provenance"][
        "source_snapshot"
    ].items():
        source = ROOT / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
