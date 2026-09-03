from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "mixed_precision" / "run_scalable_mixed_precision_diagnostic.py"
CANONICAL = ROOT / "results" / "summaries" / "scalable_mixed_precision_diagnostic.json"
SPEC = importlib.util.spec_from_file_location("scalable_mixed_precision_diagnostic", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P9 scalable mixed-precision diagnostic")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

DiagnosticConfig = EXPERIMENT.DiagnosticConfig
DEFAULT_SHAPES = EXPERIMENT.DEFAULT_SHAPES
POLICIES = EXPERIMENT.POLICIES
REALISTIC_SHAPES = EXPERIMENT.REALISTIC_SHAPES
SCHEMA_VERSION = EXPERIMENT.SCHEMA_VERSION
build_cases = EXPERIMENT.build_cases
canonical_json = EXPERIMENT.canonical_json
evaluate_policy = EXPERIMENT.evaluate_policy
run_diagnostic = EXPERIMENT.run_diagnostic


def _without_provenance(payload: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in payload.items() if key != "experiment_provenance"}


def test_default_diagnostic_is_complete_scoped_and_finite() -> None:
    payload = run_diagnostic(DiagnosticConfig())

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["config"]["seed"] == 20_260_902
    assert payload["config"]["resolved_shapes"] == [list(shape) for shape in DEFAULT_SHAPES]
    assert payload["config"]["policies"] == list(POLICIES)
    assert payload["config"]["stages"] == 5
    assert payload["summary"]["shape_count"] == len(DEFAULT_SHAPES)
    assert payload["summary"]["case_count"] == 4 * len(DEFAULT_SHAPES)
    assert payload["summary"]["policy_evaluation_count"] == 12 * len(DEFAULT_SHAPES)
    assert set(payload["summary"]["policy_summary"]) == set(POLICIES)
    assert all(
        summary["nonfinite_count"] == 0 for summary in payload["summary"]["policy_summary"].values()
    )
    assert "not a certificate" in payload["claim_scope"]["status"]
    assert "not the fixed balanced reduction" in payload["claim_scope"]["native_matmul_caveat"]
    assert "not exact arithmetic" in payload["claim_scope"]["target_caveat"]

    obstruction = payload["serial_normalizer_obstruction"]
    assert obstruction["matrix_shape"] == [4_096, 11_008]
    assert obstruction["allocated"] is False
    assert obstruction["unit_square_serial_sum"] == 2**24
    assert obstruction["unit_square_exact_sum"] == 4_096 * 11_008
    assert obstruction["returned_singular_value_squared"] == "43/16"
    assert obstruction["leaves_five_quarters_spectral_tube"] is True

    provenance = payload["experiment_provenance"]
    assert len(provenance["git"]["sha"]) == 40
    assert isinstance(provenance["git"]["dirty"], bool)
    assert provenance["seed"] == 20_260_902
    assert provenance["hardware"]["device"] == "cpu"
    assert provenance["software"]["python"]
    assert provenance["software"]["numpy"]
    assert provenance["software"]["torch"]
    assert provenance["runtime"]["torch_num_threads"] == 1
    assert provenance["runtime"]["deterministic_algorithms_enabled"] is True
    assert set(provenance["source_snapshot"]) == {
        "experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py",
        "src/passive_muon/scalable_mixed_precision.py",
        "src/passive_muon/scalable_mixed_precision_certificate.py",
        "tests/test_scalable_mixed_precision_diagnostic.py",
        "pyproject.toml",
        "uv.lock",
    }


def test_default_numerical_payload_is_deterministic_ignoring_provenance() -> None:
    first = run_diagnostic(DiagnosticConfig())
    second = run_diagnostic(DiagnosticConfig())
    assert _without_provenance(first) == _without_provenance(second)


def test_seed_changes_seeded_cases() -> None:
    left = build_cases((4, 6), seed=11, shape_index=0)
    right = build_cases((4, 6), seed=12, shape_index=0)

    assert [label for label, _ in left] == [label for label, _ in right]
    assert not torch.equal(left[0][1], right[0][1])
    assert not torch.equal(left[1][1], right[1][1])
    assert torch.equal(left[2][1], right[2][1])
    assert torch.equal(left[3][1], right[3][1])


def test_realistic_shapes_require_explicit_opt_in() -> None:
    default = DiagnosticConfig(shapes=((2, 3),))
    opted_in = replace(default, include_realistic_shapes=True)

    assert default.resolved_shapes == ((2, 3),)
    assert opted_in.resolved_shapes == ((2, 3), *REALISTIC_SHAPES)


@pytest.mark.parametrize(
    ("change", "exception", "match"),
    [
        ({"seed": -1}, ValueError, "nonnegative"),
        ({"seed": True}, ValueError, "nonnegative"),
        ({"shapes": ()}, ValueError, "at least one"),
        ({"shapes": ((2, 0),)}, ValueError, "positive"),
        ({"shapes": ((2, 2), (2, 2))}, ValueError, "unique"),
        ({"include_realistic_shapes": 1}, TypeError, "Boolean"),
        ({"torch_threads": 0}, ValueError, "positive"),
        ({"torch_threads": True}, ValueError, "positive"),
    ],
)
def test_invalid_config_is_rejected(
    change: dict[str, object], exception: type[Exception], match: str
) -> None:
    with pytest.raises(exception, match=match):
        replace(DiagnosticConfig(), **change)


def test_policy_evaluator_rejects_unknown_policy_and_bad_inputs() -> None:
    signal = torch.eye(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="unknown"):
        evaluate_policy(signal, "not-a-policy")
    with pytest.raises(ValueError, match="CPU FP32"):
        evaluate_policy(signal.to(torch.float64), POLICIES[0])
    signal[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_policy(signal, POLICIES[0])


def test_boundary_policies_are_distinct_on_a_dense_control() -> None:
    signal = torch.tensor(
        [[0.1137, -0.7221, 0.3019], [0.5513, 0.0417, -0.3911]],
        dtype=torch.float32,
    )
    polynomials = {policy: evaluate_policy(signal, policy)[0] for policy in POLICIES}

    assert any(
        not torch.equal(polynomials[left], polynomials[right])
        for index, left in enumerate(POLICIES)
        for right in POLICIES[index + 1 :]
    )


def test_cli_writes_canonical_json_with_requested_small_shape(tmp_path: Path) -> None:
    output = tmp_path / "p9-diagnostic.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--seed",
            "17",
            "--torch-threads",
            "1",
            "--shape",
            "3x5",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert output.read_text() == canonical_json(payload)
    assert payload["config"]["seed"] == 17
    assert payload["config"]["resolved_shapes"] == [[3, 5]]
    assert payload["summary"]["case_count"] == 4
    assert payload["canonical_target"] == (
        "results/summaries/scalable_mixed_precision_diagnostic.json"
    )


def test_committed_diagnostic_provenance_is_clean_and_hash_current_when_present() -> None:
    if not CANONICAL.exists():
        return

    payload = json.loads(CANONICAL.read_text(encoding="utf-8"))
    git = payload["experiment_provenance"]["git"]
    assert len(git["sha"]) == 40
    assert git["branch"] == "p9-scalable-mixed-precision"
    assert git["dirty"] is False
    assert payload["config"]["seed"] == 20_260_902
    assert payload["summary"]["policy_evaluation_count"] == 48
    assert payload["summary"]["compensated_no_larger_than_repeated_count"] == 16
    for relative_path, expected_digest in payload["experiment_provenance"][
        "source_snapshot"
    ].items():
        source = ROOT / relative_path
        assert source.is_file(), relative_path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
