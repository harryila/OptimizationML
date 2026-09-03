from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from passive_muon.pl_convergence import (
    LOCKED_PL_CONVERGENCE_LEARNING_RATE,
    LOCKED_PL_CONVERGENCE_TAU,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "experiments" / "nonconvex" / "run_robust_dissipativity_falsification.py"
SPEC = importlib.util.spec_from_file_location("robust_dissipativity_falsification", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the robust dissipativity experiment")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXPERIMENT
SPEC.loader.exec_module(EXPERIMENT)

GRADIENT_NOISE_GAIN = EXPERIMENT.GRADIENT_NOISE_GAIN
IMPLEMENTATION_ERROR_GAIN = EXPERIMENT.IMPLEMENTATION_ERROR_GAIN
ROBUST_RATE = EXPERIMENT.ROBUST_RATE
RobustProbeConfig = EXPERIMENT.RobustProbeConfig
run_harmonic_drift = EXPERIMENT.run_harmonic_drift
run_probe = EXPERIMENT.run_probe
run_robust_trial = EXPERIMENT.run_robust_trial


def _small_config(
    *,
    profiles: tuple[str, ...] = (
        "deterministic_bounded",
        "seeded_stochastic",
        "implementation_only",
    ),
    iterations: int = 20,
    harmonic_horizon: int = 256,
) -> RobustProbeConfig:
    return replace(
        RobustProbeConfig(),
        shapes=((2, 2),),
        rank_modes=("codimension_one",),
        transition_scales=(1.0,),
        radius_multipliers=(math.sqrt(2.0),),
        momentum_modes=("random",),
        disturbance_profiles=profiles,
        iterations=iterations,
        harmonic_horizon=harmonic_horizon,
    )


def test_locked_physical_gains_and_update_scale_are_exact() -> None:
    config = RobustProbeConfig()

    assert ROBUST_RATE == LOCKED_PL_CONVERGENCE_TAU**2
    assert Fraction(399_960_001, 400_000_000) == ROBUST_RATE
    assert Fraction(1, 2) == GRADIENT_NOISE_GAIN
    assert Fraction(1, 2_000_000) == IMPLEMENTATION_ERROR_GAIN
    assert config.learning_rate == float(LOCKED_PL_CONVERGENCE_LEARNING_RATE)
    assert config.case_count == 144

    with pytest.raises(ValueError, match="gradient_noise_gain"):
        replace(config, gradient_noise_gain=0.49)
    with pytest.raises(ValueError, match="disturbance_profiles"):
        replace(config, disturbance_profiles=("unknown",))


def test_sampled_disturbed_pl_grid_replays_candidate_pathwise_bound() -> None:
    config = _small_config()

    payload = run_probe(config)

    assert payload["schema_version"] == ("passive-muon-robust-dissipativity-falsification-v1")
    assert payload["summary"]["trial_count"] == 3
    assert payload["summary"]["rank_deficient_case_count"] == 3
    assert payload["summary"]["deterministic_bounded_case_count"] == 1
    assert payload["summary"]["seeded_stochastic_case_count"] == 1
    assert payload["summary"]["implementation_only_case_count"] == 1
    assert payload["summary"]["candidate_violation_count"] == 0
    assert payload["summary"]["inequality_violation_count"] == 0
    assert payload["summary"]["nonfinite_count"] == 0
    assert payload["summary"]["divergence_count"] == 0
    assert payload["summary"]["maximum_dissipation_excess"] < 0.0
    assert payload["candidate_inequality"]["q_bar_exact"] == str(ROBUST_RATE)
    assert payload["candidate_inequality"]["gamma_g_exact"] == "1/2"
    assert payload["candidate_inequality"]["gamma_R_exact"] == "1/2000000"
    assert "sampled passes do not prove" in payload["claim_scope"]["qualification"]

    implementation_only = next(
        trial
        for trial in payload["trials"]
        if trial["disturbance_profile"] == "implementation_only"
    )
    assert implementation_only["total_gradient_noise_energy"] == 0.0
    assert implementation_only["total_implementation_error_energy"] > 0.0
    assert implementation_only["inequality_violation_count"] == 0


def test_seeded_stochastic_replay_is_bitwise_deterministic_on_cpu() -> None:
    config = _small_config(profiles=("seeded_stochastic",), iterations=8)
    arguments = {
        "config": config,
        "shape": (2, 2),
        "rank_mode": "codimension_one",
        "transition_scale": 1.0,
        "radius_multiplier": math.sqrt(2.0),
        "momentum_mode": "random",
        "disturbance_profile": "seeded_stochastic",
        "case_index": 7,
    }

    first = run_robust_trial(**arguments)
    second = run_robust_trial(**arguments)

    assert first == second
    assert first.total_gradient_noise_energy > 0.0
    assert first.total_implementation_error_energy > 0.0
    assert first.inequality_violation_count == 0
    assert not first.candidate_implementation_violation


def test_harmonic_flat_direction_is_l2_but_iterates_do_not_converge() -> None:
    short = run_harmonic_drift(_small_config(harmonic_horizon=128))
    long = run_harmonic_drift(_small_config(harmonic_horizon=512))

    assert 0.0 < short.finite_prefix_error_energy < math.pi**2 / 6.0
    assert short.infinite_error_energy_upper_bound == pytest.approx(math.pi**2 / 6.0)
    assert short.finite_prefix_absolute_error_mass > math.log(short.horizon)
    assert long.finite_prefix_error_energy > short.finite_prefix_error_energy
    assert long.finite_prefix_error_energy < math.pi**2 / 6.0
    assert long.final_iterate_displacement > short.final_iterate_displacement
    assert short.final_iterate_displacement == pytest.approx(
        short.expected_harmonic_displacement,
        rel=2e-14,
    )
    assert long.final_iterate_displacement == pytest.approx(
        long.expected_harmonic_displacement,
        rel=2e-14,
    )
    assert long.maximum_objective_gap == 0.0
    assert long.maximum_lyapunov_value == 0.0
    assert long.zero_signal_operator_residual == 0.0
    assert "do not imply convergence of W_t" in long.interpretation


def test_robust_falsification_cli_records_full_provenance() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "experiments" / "nonconvex" / "run_robust_dissipativity_falsification.py"),
            "--quick",
            "--iterations",
            "3",
            "--harmonic-horizon",
            "16",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["config"]["iterations"] == 3
    assert payload["config"]["harmonic_horizon"] == 16
    assert payload["summary"]["trial_count"] == 3
    assert payload["summary"]["candidate_violation_count"] == 0
    assert payload["operator"]["normalization"] == "fixed_frobenius_floor"
    assert payload["operator"]["additive_epsilon"].startswith("not_applicable")
    assert payload["operator"]["iterations"] == 5
    assert payload["p6_reference"]["checkpoint_tag"] == "p6-checkpoint"
    assert payload["p6_reference"]["result_manifest_path"] == (
        "results/summaries/pl_convergence_certificate.json"
    )
    assert len(payload["p6_reference"]["result_manifest_sha256"]) == 64
    assert payload["upstream_provenance"]["revision"]
    provenance = payload["experiment_provenance"]
    assert provenance["seed"] == 2_026_090_3
    assert provenance["dtype"] == "float64"
    assert provenance["hardware"]["torch_device"] == "cpu"
    assert provenance["software"]["numpy"]
    assert provenance["software"]["torch"]
    assert len(provenance["git"]["sha"]) == 40
    assert provenance["source_snapshot"][
        "experiments/nonconvex/run_robust_dissipativity_falsification.py"
    ]
    assert provenance["source_snapshot"]["src/passive_muon/nonquadratic_experiment.py"]
    assert provenance["source_snapshot"]["src/passive_muon/nonquadratic_stability.py"]
    assert provenance["source_snapshot"]["src/passive_muon/floored_certificate.py"]
    assert provenance["source_snapshot"]["src/passive_muon/momentum_iqc.py"]
    assert provenance["source_snapshot"]["tests/test_robust_dissipativity_experiment.py"]
