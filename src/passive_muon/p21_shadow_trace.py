"""CPU-only shadow diagnostics for the proposed P21 optimizer composition.

This module observes a stored optimizer signal and an already-computed
five-stage BF16 Muon candidate.  It applies the pinned upstream aspect factor
*before* the P20 shield, records scale and fidelity diagnostics, and aggregates
the predeclared P21 trace gates.

The synthetic runner exercises the observer on deterministic CPU tensors only.
A real-gradient acquisition must use an externally pinned trainer and copy the
candidate actually computed by that trainer; recomputing a CPU candidate is not
backend parity.  In either mode the observer is shadow-only and does not make
the shielded output drive training.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from passive_muon.scalable_sector_shield import (
    LOCKED_CERTIFIED_SHAPES,
    NearZeroUnrepresentable,
    ScalableSectorShieldConfig,
    shield_sector_candidate_mixed_precision,
)

P21_SHADOW_TRACE_SCHEMA_VERSION: Final = "passive-muon-p21-shadow-trace-v1"
P21_PROTOCOL_RELATIVE_PATH: Final = Path("experiments/training/p21_shadow_trace_protocol.json")

TOTAL_OPTIMIZER_STEPS: Final = 256
EARLY_CAPTURE_STEPS: Final = tuple(range(0, 8))
MIDDLE_CAPTURE_STEPS: Final = tuple(range(124, 132))
LATE_CAPTURE_STEPS: Final = tuple(range(248, 256))
CAPTURE_STEPS: Final = EARLY_CAPTURE_STEPS + MIDDLE_CAPTURE_STEPS + LATE_CAPTURE_STEPS

PRIMARY_LEARNING_RATE: Final = Fraction(1, 120)
SECONDARY_LEARNING_RATE: Final = Fraction(1, 83)

# Frozen P16/P18 diagnostic gates.
INFORMATIVE_CANDIDATE_DEPARTURE: Final = 1.0e-2
MINIMUM_OUTPUT_DEPARTURE: Final = 1.0e-3
MINIMUM_SHAPING_RETENTION: Final = 1.0e-1
ANNULUS_SIGNAL_NORM_LOWER: Final = Fraction(3, 4)
ANNULUS_SIGNAL_NORM_UPPER: Final = Fraction(25)
ANNULUS_OUTPUT_TO_CANDIDATE_LOWER: Final = Fraction(1, 4)
ANNULUS_OUTPUT_TO_CANDIDATE_UPPER: Final = Fraction(8)
OUTPUT_TO_SIGNAL_LOWER: Final = Fraction(1, 10)
OUTPUT_TO_SIGNAL_UPPER: Final = Fraction(1)
PRIMARY_EFFECTIVE_GAIN_LOWER: Final = Fraction(1, 1_000)
PRIMARY_EFFECTIVE_GAIN_UPPER: Final = Fraction(1, 80)

# Frozen mild-intervention gates.  These are empirical decision rules, not
# consequences of the sector theorem.
MAXIMUM_OVERALL_ACTIVATION_RATE: Final = Fraction(1, 4)
MAXIMUM_CELL_ACTIVATION_RATE: Final = Fraction(1, 2)
MINIMUM_CELL_COUNT: Final = 20
MAXIMUM_MEDIAN_RELATIVE_CORRECTION: Final = Fraction(1, 20)
MAXIMUM_P95_RELATIVE_CORRECTION: Final = Fraction(1, 4)
MAXIMUM_RELATIVE_CORRECTION: Final = Fraction(1)
MINIMUM_P05_COSINE: Final = Fraction(19, 20)
MINIMUM_COSINE: Final = Fraction(4, 5)
MINIMUM_P05_OUTPUT_TO_CANDIDATE: Final = Fraction(3, 4)
MAXIMUM_P95_OUTPUT_TO_CANDIDATE: Final = Fraction(5, 4)

METRIC_BLOCK_SIZE: Final = 2**20


class ShadowTraceProtocolError(ValueError):
    """Raised when a record does not obey the frozen shadow-trace protocol."""


def repository_root() -> Path:
    """Return the source-tree root containing the frozen protocol."""

    return Path(__file__).resolve().parents[2]


def frozen_protocol_path() -> Path:
    """Return the absolute path to the predeclared P21 protocol JSON."""

    return repository_root() / P21_PROTOCOL_RELATIVE_PATH


def load_frozen_protocol() -> dict[str, object]:
    """Load the frozen protocol and check its schedule and version."""

    path = frozen_protocol_path()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != P21_SHADOW_TRACE_SCHEMA_VERSION:
        raise ShadowTraceProtocolError("P21 shadow-trace protocol version mismatch")
    schedule = payload.get("capture_schedule")
    if not isinstance(schedule, dict) or schedule.get("steps") != list(CAPTURE_STEPS):
        raise ShadowTraceProtocolError("P21 shadow-trace capture schedule mismatch")
    return payload


def frozen_protocol_sha256() -> str:
    """Hash the exact protocol bytes included with every result."""

    return hashlib.sha256(frozen_protocol_path().read_bytes()).hexdigest()


def phase_for_step(step: int) -> str:
    """Map one predeclared optimizer step to its frozen phase."""

    if step in EARLY_CAPTURE_STEPS:
        return "early"
    if step in MIDDLE_CAPTURE_STEPS:
        return "middle"
    if step in LATE_CAPTURE_STEPS:
        return "late"
    raise ShadowTraceProtocolError(f"step {step} is not in the frozen capture schedule")


def nearest_rank(values: Sequence[float], numerator: int, denominator: int) -> float:
    """Return the frozen nearest-rank empirical quantile.

    For sorted values ``x[0] <= ... <= x[n-1]`` and ``p=a/b`` in ``(0,1]``,
    the returned value is ``x[ceil(p*n)-1]``.  Missing values must be removed
    by the caller; nonfinite values are rejected rather than silently omitted.
    """

    if not values:
        raise ShadowTraceProtocolError("nearest-rank quantiles require at least one value")
    if (
        not isinstance(numerator, int)
        or isinstance(numerator, bool)
        or not isinstance(denominator, int)
        or isinstance(denominator, bool)
        or denominator <= 0
        or numerator <= 0
        or numerator > denominator
    ):
        raise ShadowTraceProtocolError("quantile must be an integer fraction in (0,1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ShadowTraceProtocolError("quantile inputs must be finite")
    index = (numerator * len(ordered) + denominator - 1) // denominator - 1
    return ordered[index]


def _check_matrix(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ShadowTraceProtocolError(f"{name} must be a matrix")
    if value.device.type != "cpu":
        raise ShadowTraceProtocolError(f"{name} must be copied to CPU before observation")
    if value.dtype not in (torch.float32, torch.bfloat16):
        raise ShadowTraceProtocolError(f"{name} must be stored FP32 or BF16")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ShadowTraceProtocolError(f"{name} must be contiguous and strided")


def _canonical_orientation(shape: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    if shape in LOCKED_CERTIFIED_SHAPES:
        return shape, False
    reversed_shape = (shape[1], shape[0])
    if reversed_shape in LOCKED_CERTIFIED_SHAPES:
        return reversed_shape, True
    raise ShadowTraceProtocolError(
        f"matrix shape {shape} has neither orientation in the P20 certified table"
    )


def upstream_aspect_factor(shape: tuple[int, int]) -> float:
    """Return the pinned upstream post-Jordan aspect multiplier."""

    rows, columns = shape
    if rows <= 0 or columns <= 0:
        raise ShadowTraceProtocolError("matrix dimensions must be positive")
    return math.sqrt(max(1.0, rows / columns))


def _tensor_sha256(value: Tensor) -> str:
    host = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for offset in range(0, host.numel(), METRIC_BLOCK_SIZE):
        digest.update(host[offset : offset + METRIC_BLOCK_SIZE].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _dot(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        raise ShadowTraceProtocolError("metric tensors must have equal shape")
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    partials: list[float] = []
    for offset in range(0, left_flat.numel(), METRIC_BLOCK_SIZE):
        left_block = left_flat[offset : offset + METRIC_BLOCK_SIZE].to(torch.float64)
        right_block = right_flat[offset : offset + METRIC_BLOCK_SIZE].to(torch.float64)
        partials.append(float(torch.dot(left_block, right_block)))
    result = math.fsum(partials)
    if not math.isfinite(result):
        raise ShadowTraceProtocolError("diagnostic dot product is nonfinite")
    return result


def _norm(value: Tensor) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _difference_norm(left: Tensor, right: Tensor) -> float:
    if left.shape != right.shape:
        raise ShadowTraceProtocolError("metric tensors must have equal shape")
    left_flat = left.detach().reshape(-1)
    right_flat = right.detach().reshape(-1)
    partials: list[float] = []
    for offset in range(0, left_flat.numel(), METRIC_BLOCK_SIZE):
        difference = left_flat[offset : offset + METRIC_BLOCK_SIZE].to(torch.float64) - right_flat[
            offset : offset + METRIC_BLOCK_SIZE
        ].to(torch.float64)
        partials.append(float(torch.dot(difference, difference)))
    square = math.fsum(partials)
    if not math.isfinite(square):
        raise ShadowTraceProtocolError("diagnostic difference norm is nonfinite")
    return math.sqrt(max(0.0, square))


def _best_scalar_departure(signal: Tensor, target: Tensor) -> float:
    signal_square = _dot(signal, signal)
    target_square = _dot(target, target)
    if target_square == 0.0:
        return 0.0
    if signal_square == 0.0:
        return 1.0
    cross = _dot(signal, target)
    orthogonal_square = max(0.0, target_square - cross * cross / signal_square)
    return math.sqrt(orthogonal_square / target_square)


def _optional_ratio(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator != 0.0:
        return numerator / denominator, "finite_denominator"
    if numerator == 0.0:
        return None, "both_zero"
    return None, "nonzero_over_zero"


def _optional_cosine(left: Tensor, right: Tensor) -> tuple[float | None, str]:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None, "both_zero" if left_norm == right_norm == 0.0 else "one_zero"
    cosine = _dot(left, right) / (left_norm * right_norm)
    return max(-1.0, min(1.0, cosine)), "finite_denominator"


def observe_aspect_scaled_candidate(
    signal: Tensor,
    raw_candidate: Tensor,
    *,
    optimizer_step: int,
    tokens_seen: int,
    seed: int,
    parameter_name: str,
    layer: int,
    role: str,
    learning_rate: float,
    beta: float,
    weight_decay: float,
) -> dict[str, object]:
    """Observe one stored signal and pre-aspect BF16/FP32 candidate.

    ``raw_candidate`` must be the candidate captured from the training backend,
    before its pinned upstream aspect multiplier.  The observer performs that
    multiplier in the candidate's stored dtype, then applies P20 on CPU.  It
    never mutates either caller tensor and never updates model parameters.
    """

    _check_matrix("signal", signal)
    _check_matrix("raw_candidate", raw_candidate)
    if signal.shape != raw_candidate.shape:
        raise ShadowTraceProtocolError("signal and candidate shapes must match")
    if not bool(torch.isfinite(signal).all()):
        raise ShadowTraceProtocolError("a nonfinite signal is a hard trace-gate failure")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool) or tokens_seen < 0:
        raise ShadowTraceProtocolError("tokens_seen must be a nonnegative integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ShadowTraceProtocolError("seed must be a nonnegative integer")
    if not parameter_name or not role:
        raise ShadowTraceProtocolError("parameter_name and role must be nonempty")
    if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
        raise ShadowTraceProtocolError("layer must be a nonnegative integer")
    for name, value in (
        ("learning_rate", learning_rate),
        ("beta", beta),
        ("weight_decay", weight_decay),
    ):
        if not math.isfinite(value):
            raise ShadowTraceProtocolError(f"{name} must be finite")

    phase = phase_for_step(optimizer_step)
    original_shape = tuple(signal.shape)
    oriented_shape, transposed = _canonical_orientation(original_shape)  # type: ignore[arg-type]
    aspect = upstream_aspect_factor(original_shape)  # type: ignore[arg-type]

    aspect_candidate = raw_candidate.clone()
    aspect_candidate.mul_(aspect)
    signal_for_shield = signal.mT.contiguous() if transposed else signal.clone()
    candidate_for_shield = (
        aspect_candidate.mT.contiguous() if transposed else aspect_candidate.clone()
    )
    config = ScalableSectorShieldConfig(oriented_shape)

    dead_zone = False
    dead_zone_bound = None
    try:
        shield = shield_sector_candidate_mixed_precision(
            signal_for_shield,
            candidate_for_shield,
            config,
        )
        oriented_output = shield.output
        output = oriented_output.mT.contiguous() if transposed else oriented_output
        action = shield.diagnostics.action.value
        active = shield.diagnostics.active
        sector_certified = True
        signal_guard = shield.diagnostics.signal_guard_class.value
        shield_reason = shield.diagnostics.reason
    except NearZeroUnrepresentable as error:
        # This is the proposed P21 wrapper policy, not a successful P20 call.
        # The analytic P21 theorem must absorb the distance to the exact S/2
        # sector point as an absolute update disturbance.
        output = torch.zeros_like(signal, dtype=torch.float32)
        action = "dead_zone_zero"
        active = True
        sector_certified = False
        signal_guard = "subnormal_unrepresentable"
        shield_reason = str(error)
        dead_zone = True
        dead_zone_bound = 0.5 * _norm(signal)

    signal_norm = _norm(signal)
    raw_candidate_norm = _norm(raw_candidate) if bool(torch.isfinite(raw_candidate).all()) else None
    aspect_candidate_finite = bool(torch.isfinite(aspect_candidate).all())
    candidate_norm = _norm(aspect_candidate) if aspect_candidate_finite else None
    output_norm = _norm(output)
    correction_norm = (
        _difference_norm(output, aspect_candidate) if aspect_candidate_finite else None
    )
    output_to_signal, output_to_signal_status = _optional_ratio(output_norm, signal_norm)
    output_to_candidate, output_to_candidate_status = _optional_ratio(
        output_norm,
        candidate_norm if candidate_norm is not None else 0.0,
    )
    relative_correction, relative_correction_status = _optional_ratio(
        correction_norm if correction_norm is not None else 0.0,
        candidate_norm if candidate_norm is not None else 0.0,
    )
    cosine, cosine_status = (
        _optional_cosine(output, aspect_candidate)
        if aspect_candidate_finite
        else (None, "nonfinite_candidate")
    )
    candidate_departure = (
        _best_scalar_departure(signal, aspect_candidate) if aspect_candidate_finite else None
    )
    output_departure = _best_scalar_departure(signal, output)
    informative = bool(
        candidate_departure is not None and candidate_departure >= INFORMATIVE_CANDIDATE_DEPARTURE
    )
    shaping_retention = (
        output_departure / candidate_departure
        if informative and candidate_departure is not None
        else None
    )
    in_annulus = float(ANNULUS_SIGNAL_NORM_LOWER) <= signal_norm <= float(ANNULUS_SIGNAL_NORM_UPPER)
    primary_effective_gain = (
        float(PRIMARY_LEARNING_RATE) * output_to_signal if output_to_signal is not None else None
    )
    secondary_effective_gain = (
        float(SECONDARY_LEARNING_RATE) * output_to_signal if output_to_signal is not None else None
    )

    return {
        "schema_version": P21_SHADOW_TRACE_SCHEMA_VERSION,
        "evidence_kind": "shadow_observation",
        "optimizer_step": optimizer_step,
        "tokens_seen": tokens_seen,
        "phase": phase,
        "seed": seed,
        "parameter": {
            "name": parameter_name,
            "layer": layer,
            "role": role,
            "original_shape": list(original_shape),
            "shield_shape": list(oriented_shape),
            "transposed_for_shield": transposed,
            "entry_count": signal.numel(),
        },
        "optimizer": {
            "beta": beta,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "upstream_aspect_factor": aspect,
            "aspect_applied_before_shield": True,
        },
        "storage": {
            "signal_dtype": str(signal.dtype),
            "raw_candidate_dtype": str(raw_candidate.dtype),
            "aspect_candidate_dtype": str(aspect_candidate.dtype),
            "output_dtype": str(output.dtype),
            "device": "cpu",
            "raw_candidate_finite": bool(torch.isfinite(raw_candidate).all()),
            "aspect_candidate_finite": aspect_candidate_finite,
            "signal_sha256": _tensor_sha256(signal),
            "raw_candidate_sha256": _tensor_sha256(raw_candidate),
            "aspect_candidate_sha256": _tensor_sha256(aspect_candidate),
            "output_sha256": _tensor_sha256(output),
        },
        "shield": {
            "action": action,
            "active": active,
            "sector_certified_by_successful_p20_call": sector_certified,
            "signal_guard": signal_guard,
            "reason": shield_reason,
            "dead_zone_zero": dead_zone,
            "dead_zone_update_disturbance_upper": dead_zone_bound,
        },
        "metrics": {
            "signal_norm": signal_norm,
            "raw_candidate_norm": raw_candidate_norm,
            "aspect_candidate_norm": candidate_norm,
            "output_norm": output_norm,
            "correction_norm": correction_norm,
            "relative_correction": relative_correction,
            "relative_correction_status": relative_correction_status,
            "candidate_output_cosine": cosine,
            "candidate_output_cosine_status": cosine_status,
            "output_to_signal_amplitude": output_to_signal,
            "output_to_signal_status": output_to_signal_status,
            "output_to_candidate_amplitude": output_to_candidate,
            "output_to_candidate_status": output_to_candidate_status,
            "candidate_best_scalar_departure": candidate_departure,
            "output_best_scalar_departure": output_departure,
            "shaping_retention": shaping_retention,
            "informative_candidate": informative,
            "in_frozen_operating_annulus": in_annulus,
            "primary_eta_1_over_120_effective_gain": primary_effective_gain,
            "secondary_eta_1_over_83_effective_gain": secondary_effective_gain,
            "observed_training_step_norm": learning_rate * output_norm,
            "primary_theorem_step_norm": float(PRIMARY_LEARNING_RATE) * output_norm,
        },
        "per_observation_gates": {
            "successful_p20_sector_call": sector_certified,
            "frozen_fidelity": (
                output_departure >= MINIMUM_OUTPUT_DEPARTURE
                and shaping_retention is not None
                and shaping_retention >= MINIMUM_SHAPING_RETENTION
                if informative
                else None
            ),
            "frozen_annulus_output_to_candidate": (
                output_to_candidate is not None
                and float(ANNULUS_OUTPUT_TO_CANDIDATE_LOWER)
                <= output_to_candidate
                <= float(ANNULUS_OUTPUT_TO_CANDIDATE_UPPER)
                if in_annulus
                else None
            ),
            "frozen_output_to_signal": (
                output_to_signal is not None
                and float(OUTPUT_TO_SIGNAL_LOWER)
                <= output_to_signal
                <= float(OUTPUT_TO_SIGNAL_UPPER)
            ),
            "frozen_primary_effective_gain": (
                primary_effective_gain is not None
                and float(PRIMARY_EFFECTIVE_GAIN_LOWER)
                <= primary_effective_gain
                <= float(PRIMARY_EFFECTIVE_GAIN_UPPER)
            ),
        },
    }


def _records_by_phase_role(
    observations: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], list[Mapping[str, object]]]:
    groups: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for observation in observations:
        parameter = observation["parameter"]
        assert isinstance(parameter, Mapping)
        groups[(str(observation["phase"]), str(parameter["role"]))].append(observation)
    return dict(groups)


def _metric_values(observations: Iterable[Mapping[str, object]], metric: str) -> list[float]:
    values: list[float] = []
    for observation in observations:
        metrics = observation["metrics"]
        assert isinstance(metrics, Mapping)
        value = metrics.get(metric)
        if value is not None:
            converted = float(value)
            if not math.isfinite(converted):
                raise ShadowTraceProtocolError(f"metric {metric} contains a nonfinite value")
            values.append(converted)
    return values


def _quantile_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "minimum": min(values),
        "p05_nearest_rank": nearest_rank(values, 5, 100),
        "p50_nearest_rank": nearest_rank(values, 50, 100),
        "p95_nearest_rank": nearest_rank(values, 95, 100),
        "maximum": max(values),
    }


def summarize_shadow_trace(
    observations: Sequence[Mapping[str, object]],
    *,
    evidence_kind: str,
    provenance: Mapping[str, object],
    intended_parameter_numel: Mapping[str, int],
) -> dict[str, object]:
    """Aggregate observations under the frozen protocol and decision gates."""

    if evidence_kind not in {"synthetic_cpu_diagnostic", "real_gradient_shadow_trace"}:
        raise ShadowTraceProtocolError("unknown evidence_kind")
    if not observations:
        raise ShadowTraceProtocolError("a trace must contain observations")
    if not intended_parameter_numel or any(
        not name or not isinstance(numel, int) or isinstance(numel, bool) or numel <= 0
        for name, numel in intended_parameter_numel.items()
    ):
        raise ShadowTraceProtocolError(
            "intended_parameter_numel must map every intended parameter to a positive integer"
        )
    observed_steps = sorted({int(observation["optimizer_step"]) for observation in observations})
    schedule_complete = observed_steps == list(CAPTURE_STEPS)
    phases_present = sorted({str(observation["phase"]) for observation in observations})
    expected_phases = ["early", "late", "middle"]
    phase_complete = phases_present == expected_phases

    expected_names = set(intended_parameter_numel)
    observations_by_step: dict[int, list[str]] = defaultdict(list)
    observed_names: set[str] = set()
    inventory_numel_consistent = True
    for observation in observations:
        parameter = observation["parameter"]
        assert isinstance(parameter, Mapping)
        step = int(observation["optimizer_step"])
        name = str(parameter["name"])
        numel = int(parameter["entry_count"])
        observations_by_step[step].append(name)
        observed_names.add(name)
        inventory_numel_consistent &= (
            name in intended_parameter_numel and intended_parameter_numel.get(name) == numel
        )
    inventory_complete_each_step = inventory_numel_consistent and all(
        len(names) == len(expected_names) and set(names) == expected_names
        for names in observations_by_step.values()
    )
    inventory_complete_each_step &= set(observations_by_step) == set(CAPTURE_STEPS)
    covered_names = observed_names & expected_names
    covered_numel = sum(intended_parameter_numel[name] for name in covered_names)
    intended_numel = sum(intended_parameter_numel.values())

    active_count = sum(bool(observation["shield"]["active"]) for observation in observations)  # type: ignore[index]
    activation_rate = active_count / len(observations)
    dead_zone_count = sum(
        bool(observation["shield"]["dead_zone_zero"])
        for observation in observations  # type: ignore[index]
    )
    successful_count = sum(
        bool(observation["shield"]["sector_certified_by_successful_p20_call"])  # type: ignore[index]
        for observation in observations
    )
    nonfinite_candidate_count = sum(
        not bool(observation["storage"]["aspect_candidate_finite"])  # type: ignore[index]
        for observation in observations
    )
    informative = [
        observation
        for observation in observations
        if bool(observation["metrics"]["informative_candidate"])  # type: ignore[index]
    ]
    annulus = [
        observation
        for observation in observations
        if bool(observation["metrics"]["in_frozen_operating_annulus"])  # type: ignore[index]
    ]

    relative_corrections = _metric_values(observations, "relative_correction")
    cosines = _metric_values(observations, "candidate_output_cosine")
    output_to_candidate = _metric_values(observations, "output_to_candidate_amplitude")
    relative_summary = _quantile_summary(relative_corrections)
    cosine_summary = _quantile_summary(cosines)
    amplitude_summary = _quantile_summary(output_to_candidate)

    cells: dict[str, object] = {}
    evaluated_cell_rates: list[float] = []
    for (phase, role), records in sorted(_records_by_phase_role(observations).items()):
        cell_active = sum(bool(record["shield"]["active"]) for record in records)  # type: ignore[index]
        rate = cell_active / len(records)
        evaluated = len(records) >= MINIMUM_CELL_COUNT
        if evaluated:
            evaluated_cell_rates.append(rate)
        cells[f"{phase}:{role}"] = {
            "count": len(records),
            "active": cell_active,
            "activation_rate": rate,
            "minimum_count_reached": evaluated,
            "passes_if_evaluated": (
                rate <= float(MAXIMUM_CELL_ACTIVATION_RATE) if evaluated else None
            ),
        }

    annulus_phases = {str(observation["phase"]) for observation in annulus}
    all_informative_fidelity_pass = bool(informative) and all(
        observation["per_observation_gates"]["frozen_fidelity"] is True  # type: ignore[index]
        for observation in informative
    )
    all_annulus_amplitude_pass = bool(annulus) and all(
        observation["per_observation_gates"]["frozen_annulus_output_to_candidate"] is True  # type: ignore[index]
        for observation in annulus
    )
    annulus_all_phases = annulus_phases == {"early", "middle", "late"}
    all_sector_amplitude_pass = all(
        observation["per_observation_gates"]["frozen_output_to_signal"] is True  # type: ignore[index]
        for observation in observations
    )
    all_effective_gain_pass = all(
        observation["per_observation_gates"]["frozen_primary_effective_gain"] is True  # type: ignore[index]
        for observation in observations
    )

    mild_gate_checks = {
        "overall_activation_rate_at_most_1_over_4": (
            activation_rate <= float(MAXIMUM_OVERALL_ACTIVATION_RATE)
        ),
        "all_evaluated_phase_role_cells_at_most_1_over_2": (
            bool(evaluated_cell_rates)
            and all(rate <= float(MAXIMUM_CELL_ACTIVATION_RATE) for rate in evaluated_cell_rates)
        ),
        "relative_correction_p50_at_most_1_over_20": (
            relative_summary is not None
            and relative_summary["p50_nearest_rank"] <= float(MAXIMUM_MEDIAN_RELATIVE_CORRECTION)
        ),
        "relative_correction_p95_at_most_1_over_4": (
            relative_summary is not None
            and relative_summary["p95_nearest_rank"] <= float(MAXIMUM_P95_RELATIVE_CORRECTION)
        ),
        "relative_correction_maximum_at_most_1": (
            relative_summary is not None
            and relative_summary["maximum"] <= float(MAXIMUM_RELATIVE_CORRECTION)
        ),
        "cosine_p05_at_least_19_over_20": (
            cosine_summary is not None
            and cosine_summary["p05_nearest_rank"] >= float(MINIMUM_P05_COSINE)
        ),
        "cosine_minimum_at_least_4_over_5": (
            cosine_summary is not None and cosine_summary["minimum"] >= float(MINIMUM_COSINE)
        ),
        "output_to_candidate_p05_at_least_3_over_4": (
            amplitude_summary is not None
            and amplitude_summary["p05_nearest_rank"] >= float(MINIMUM_P05_OUTPUT_TO_CANDIDATE)
        ),
        "output_to_candidate_p95_at_most_5_over_4": (
            amplitude_summary is not None
            and amplitude_summary["p95_nearest_rank"] <= float(MAXIMUM_P95_OUTPUT_TO_CANDIDATE)
        ),
    }
    hard_checks = {
        "capture_schedule_complete": schedule_complete,
        "all_three_phases_present": phase_complete,
        "zero_nonfinite_candidate_events": nonfinite_candidate_count == 0,
        "all_calls_successful_p20_sector_calls": successful_count == len(observations),
        "zero_dead_zone_events": dead_zone_count == 0,
        "all_output_to_signal_gates_pass": all_sector_amplitude_pass,
        "all_primary_effective_gain_gates_pass": all_effective_gain_pass,
        "intended_muon_inventory_complete_every_capture": inventory_complete_each_step,
    }
    frozen_fidelity_checks = {
        "informative_observations_exist": bool(informative),
        "all_informative_observations_pass": all_informative_fidelity_pass,
        "annulus_observations_exist_in_every_phase": annulus_all_phases,
        "all_annulus_output_to_candidate_gates_pass": all_annulus_amplitude_pass,
    }
    all_checks = {**hard_checks, **frozen_fidelity_checks, **mild_gate_checks}

    return {
        "schema_version": P21_SHADOW_TRACE_SCHEMA_VERSION,
        "evidence_kind": evidence_kind,
        "real_gradient_evidence": evidence_kind == "real_gradient_shadow_trace",
        "claim_scope": (
            "deterministic synthetic CPU diagnostic only; no model, dataset, backward pass, "
            "accelerator, or real gradient is exercised"
            if evidence_kind == "synthetic_cpu_diagnostic"
            else "shadow-mode observations; the shield output did not update training parameters"
        ),
        "protocol": {
            "path": P21_PROTOCOL_RELATIVE_PATH.as_posix(),
            "sha256": frozen_protocol_sha256(),
            "capture_steps": list(CAPTURE_STEPS),
            "quantile_definition": "nearest-rank x[ceil(p*n)-1] after removing only None",
        },
        "provenance": dict(provenance),
        "coverage": {
            "observation_count": len(observations),
            "observed_steps": observed_steps,
            "schedule_complete": schedule_complete,
            "phases": phases_present,
            "roles": sorted(
                {
                    str(observation["parameter"]["role"])  # type: ignore[index]
                    for observation in observations
                }
            ),
            "supported_shape_observations": len(observations),
            "unsupported_shape_observations": 0,
            "intended_parameter_count": len(expected_names),
            "covered_parameter_count": len(covered_names),
            "parameter_count_coverage": len(covered_names) / len(expected_names),
            "intended_parameter_numel": intended_numel,
            "covered_parameter_numel": covered_numel,
            "parameter_numel_coverage": covered_numel / intended_numel,
            "inventory_numel_consistent": inventory_numel_consistent,
            "inventory_complete_every_capture": inventory_complete_each_step,
        },
        "shield_actions": {
            "active": active_count,
            "inactive": len(observations) - active_count,
            "activation_rate": activation_rate,
            "dead_zone_zero": dead_zone_count,
            "successful_p20_sector_calls": successful_count,
            "nonfinite_candidate_events": nonfinite_candidate_count,
        },
        "diagnostic_quantiles": {
            "relative_correction": relative_summary,
            "candidate_output_cosine": cosine_summary,
            "output_to_candidate_amplitude": amplitude_summary,
        },
        "diagnostic_sample_counts": {
            "relative_correction": len(relative_corrections),
            "candidate_output_cosine": len(cosines),
            "output_to_candidate_amplitude": len(output_to_candidate),
        },
        "phase_role_cells": cells,
        "frozen_gate_counts": {
            "informative_fidelity": len(informative),
            "annulus": len(annulus),
        },
        "hard_gate_checks": hard_checks,
        "frozen_fidelity_gate_checks": frozen_fidelity_checks,
        "mild_intervention_gate_checks": mild_gate_checks,
        "all_predeclared_checks_pass": all(all_checks.values()),
        "interpretation": (
            "Passing synthetic gates tests the observer and decision logic only. It cannot "
            "satisfy or predict the blocked real-gradient trace gate."
            if evidence_kind == "synthetic_cpu_diagnostic"
            else "These are declared real-gradient shadow observations. Passing evaluates "
            "the frozen fidelity gates for this acquisition only; this summary alone does "
            "not deeply validate their acquisition provenance, and the shield output did "
            "not update training parameters."
        ),
        "observations": [dict(observation) for observation in observations],
    }


__all__ = [
    "CAPTURE_STEPS",
    "EARLY_CAPTURE_STEPS",
    "LATE_CAPTURE_STEPS",
    "MIDDLE_CAPTURE_STEPS",
    "P21_SHADOW_TRACE_SCHEMA_VERSION",
    "PRIMARY_LEARNING_RATE",
    "SECONDARY_LEARNING_RATE",
    "ShadowTraceProtocolError",
    "frozen_protocol_sha256",
    "load_frozen_protocol",
    "nearest_rank",
    "observe_aspect_scaled_candidate",
    "phase_for_step",
    "summarize_shadow_trace",
    "upstream_aspect_factor",
]
