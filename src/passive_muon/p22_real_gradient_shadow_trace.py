"""P22 observer for actual post-aspect accelerator Muon candidates.

P21's frozen observer accepts a pre-aspect candidate and reproduces the
aspect multiplication on CPU.  P22 instead records the value that the pinned
accelerator backend actually returns *after* that multiplication.  This
module keeps the P21 decision gates unchanged while extending the certified
shape inventory by the separately generated ``768 x 2304`` P22 certificate.

The CPU shield result is diagnostic only.  It is never returned to the
training optimizer and therefore cannot change the baseline update.
"""

from __future__ import annotations

import math
from typing import Final

import torch
from torch import Tensor

from passive_muon.p21_shadow_trace import (
    ANNULUS_OUTPUT_TO_CANDIDATE_LOWER,
    ANNULUS_OUTPUT_TO_CANDIDATE_UPPER,
    ANNULUS_SIGNAL_NORM_LOWER,
    ANNULUS_SIGNAL_NORM_UPPER,
    INFORMATIVE_CANDIDATE_DEPARTURE,
    MINIMUM_OUTPUT_DEPARTURE,
    MINIMUM_SHAPING_RETENTION,
    OUTPUT_TO_SIGNAL_LOWER,
    OUTPUT_TO_SIGNAL_UPPER,
    PRIMARY_EFFECTIVE_GAIN_LOWER,
    PRIMARY_EFFECTIVE_GAIN_UPPER,
    PRIMARY_LEARNING_RATE,
    SECONDARY_LEARNING_RATE,
    ShadowTraceProtocolError,
    _best_scalar_departure,
    _difference_norm,
    _norm,
    _optional_cosine,
    _optional_ratio,
    _tensor_sha256,
    phase_for_step,
    upstream_aspect_factor,
)
from passive_muon.p22_scalable_sector_shield_extension import (
    P22ScalableSectorShieldConfig,
    p22_canonical_shield_orientation,
    p22_shield_sector_candidate_mixed_precision,
)
from passive_muon.scalable_sector_shield import NearZeroUnrepresentable

P22_SHADOW_OBSERVATION_SCHEMA_VERSION: Final = "passive-muon-p22-post-aspect-shadow-observation-v1"


def _check_stored_matrix(name: str, value: Tensor) -> None:
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


def observe_post_aspect_candidate(
    signal: Tensor,
    post_aspect_candidate: Tensor,
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
    accelerator_backend: str,
    pre_aspect_candidate_sha256: str | None = None,
) -> dict[str, object]:
    """Observe one actual stored signal/candidate pair without recomputation.

    ``post_aspect_candidate`` is copied from the return value of the pinned
    accelerator ``muon_update``.  The CPU proof-reference consumes that exact
    stored value; no CPU aspect multiplication or Jordan recomputation occurs.
    """

    _check_stored_matrix("signal", signal)
    _check_stored_matrix("post_aspect_candidate", post_aspect_candidate)
    if signal.shape != post_aspect_candidate.shape:
        raise ShadowTraceProtocolError("signal and candidate shapes must match")
    if not bool(torch.isfinite(signal).all()):
        raise ShadowTraceProtocolError("a nonfinite signal is a hard trace-gate failure")
    if accelerator_backend not in {"cuda", "mps"}:
        raise ShadowTraceProtocolError("accelerator backend must be cuda or mps")
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
    oriented_shape, transposed = p22_canonical_shield_orientation(original_shape)  # type: ignore[arg-type]
    signal_for_shield = signal.mT.contiguous() if transposed else signal.clone()
    candidate_for_shield = (
        post_aspect_candidate.mT.contiguous() if transposed else post_aspect_candidate.clone()
    )
    config = P22ScalableSectorShieldConfig(oriented_shape)

    dead_zone = False
    dead_zone_bound = None
    try:
        shield = p22_shield_sector_candidate_mixed_precision(
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
        output = torch.zeros_like(signal, dtype=torch.float32)
        action = "dead_zone_zero"
        active = True
        sector_certified = False
        signal_guard = "subnormal_unrepresentable"
        shield_reason = str(error)
        dead_zone = True
        dead_zone_bound = 0.5 * _norm(signal)

    signal_norm = _norm(signal)
    candidate_finite = bool(torch.isfinite(post_aspect_candidate).all())
    candidate_norm = _norm(post_aspect_candidate) if candidate_finite else None
    output_norm = _norm(output)
    correction_norm = _difference_norm(output, post_aspect_candidate) if candidate_finite else None
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
        _optional_cosine(output, post_aspect_candidate)
        if candidate_finite
        else (None, "nonfinite_candidate")
    )
    candidate_departure = (
        _best_scalar_departure(signal, post_aspect_candidate) if candidate_finite else None
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
        "schema_version": P22_SHADOW_OBSERVATION_SCHEMA_VERSION,
        "evidence_kind": "real_gradient_shadow_observation",
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
            "upstream_aspect_factor": upstream_aspect_factor(original_shape),  # type: ignore[arg-type]
            "candidate_captured_after_accelerator_aspect": True,
            "aspect_applied_by_observer": False,
        },
        "storage": {
            "signal_dtype": str(signal.dtype),
            "raw_candidate_dtype": None,
            "aspect_candidate_dtype": str(post_aspect_candidate.dtype),
            "output_dtype": str(output.dtype),
            "source_device": accelerator_backend,
            "device": "cpu",
            "raw_candidate_finite": None,
            "aspect_candidate_finite": candidate_finite,
            "signal_sha256": _tensor_sha256(signal),
            "raw_candidate_sha256": pre_aspect_candidate_sha256,
            "aspect_candidate_sha256": _tensor_sha256(post_aspect_candidate),
            "output_sha256": _tensor_sha256(output),
        },
        "shield": {
            "action": action,
            "active": active,
            "sector_certified_by_successful_p20_call": sector_certified,
            "sector_certificate_scope": "P20 plus exact P22 768x2304 extension",
            "signal_guard": signal_guard,
            "reason": shield_reason,
            "dead_zone_zero": dead_zone,
            "dead_zone_update_disturbance_upper": dead_zone_bound,
        },
        "metrics": {
            "signal_norm": signal_norm,
            "raw_candidate_norm": None,
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


__all__ = [
    "P22_SHADOW_OBSERVATION_SCHEMA_VERSION",
    "observe_post_aspect_candidate",
]
