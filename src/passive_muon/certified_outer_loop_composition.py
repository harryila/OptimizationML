"""Stored FP32/BF16 outer-loop composition around the P20 sector shield.

This module composes already-frozen public P10 and P20 building blocks without
modifying either checkpoint.  The executable order is

``gradient cast -> FP32 EMA/Nesterov -> BF16 Jordan -> aspect scale ->
P20 shield -> compensated FP32 master update``.

The P20 sector is asserted relative to the *stored* FP32 Nesterov signal.  No
incremental comparison between rounded and exact operator calls is made.  A
nonrepresentable all-subnormal signal is the one exception: the wrapper emits
zero and records its exact error relative to the conceptual sector point
``signal/2``.  Nonfinite signals are never converted into an update.

This remains a CPU proof-reference graph.  In particular, candidate quality is
not needed for safety, and the BF16 Jordan path is not a GPU/tensor-core parity
claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Final

import torch
from torch import Tensor

from passive_muon.deployed import keller_jordan_map
from passive_muon.finite_precision_outer_loop import (
    FinitePrecisionOuterLoopConfig,
    FP32EmaNesterovResult,
    ThreeWordFP32Master,
    TwoSumResult,
    fp32_ema_nesterov,
    two_sum_fp32,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_CERTIFIED_SHAPES,
    LOCKED_REPRESENTATIVE_SHAPES,
    NearZeroUnrepresentable,
    ScalableSectorShieldConfig,
    ScalableSectorShieldDiagnostics,
    shield_sector_candidate_mixed_precision,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION: Final = (
    "passive-muon-certified-outer-loop-composition-v1"
)
LOCKED_CANDIDATE_EPSILON: Final = 1e-7
LOCKED_CANDIDATE_STEPS: Final = 5
LOCKED_OPERATOR_OUTPUT_MAX_ABS: Final = 64.0
LOCKED_TOTAL_STEP_MAX_ABS: Final = 1.0

PRIMARY_LEARNING_RATE_FP32_BITS: Final = 0x3C08_8889
MAXIMUM_LEARNING_RATE_FP32_BITS: Final = 0x3C45_65C8
PRIMARY_LEARNING_RATE_FP32_EXACT: Final = Fraction(8_947_849, 1_073_741_824)
MAXIMUM_LEARNING_RATE_FP32_EXACT: Final = Fraction(1_617_081, 134_217_728)


class P21OperatingPoint(StrEnum):
    """The two P19/P20 operating points replayed by P21."""

    PRIMARY = "primary_eta_1_over_120"
    MAXIMUM_STEP = "maximum_eta_1_over_83"

    @property
    def learning_rate(self) -> Fraction:
        if self is P21OperatingPoint.PRIMARY:
            return Fraction(1, 120)
        return Fraction(1, 83)

    @property
    def fp32_bits(self) -> int:
        if self is P21OperatingPoint.PRIMARY:
            return PRIMARY_LEARNING_RATE_FP32_BITS
        return MAXIMUM_LEARNING_RATE_FP32_BITS

    @property
    def fp32_exact(self) -> Fraction:
        if self is P21OperatingPoint.PRIMARY:
            return PRIMARY_LEARNING_RATE_FP32_EXACT
        return MAXIMUM_LEARNING_RATE_FP32_EXACT


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fp32_bits(value: Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("bit inspection requires one CPU float32 scalar")
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _fp32_fraction(value: Tensor) -> Fraction:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("exact conversion requires one CPU float32 scalar")
    scalar = float(value.reshape(1).item())
    if not math.isfinite(scalar):
        raise ValueError("exact conversion requires a finite value")
    return Fraction.from_float(scalar)


def _check_matrix(
    name: str,
    value: Tensor,
    shape: tuple[int, int],
    *,
    dtypes: tuple[torch.dtype, ...] = (torch.float32,),
    finite: bool = True,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or tuple(value.shape) != shape:
        raise ValueError(f"{name} must have configured shape {shape}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on the locked CPU backend")
    if value.dtype not in dtypes:
        names = " or ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must use {names}")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if finite and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _canonical_shield_shape(shape: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    if shape in LOCKED_CERTIFIED_SHAPES:
        return shape, False
    transposed = (shape[1], shape[0])
    if transposed in LOCKED_CERTIFIED_SHAPES:
        return transposed, True
    raise ValueError(
        "matrix_shape or its transpose must be in the generated P20 certified shape table"
    )


@dataclass(frozen=True)
class CertifiedOuterLoopConfig:
    """Locked shape, rate, and decoupled-weight-decay scalar for one shell."""

    matrix_shape: tuple[int, int]
    operating_point: P21OperatingPoint = P21OperatingPoint.PRIMARY
    weight_decay_fp32_bits: int = 0

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.matrix_shape
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        _canonical_shield_shape(self.matrix_shape)
        if not isinstance(self.operating_point, P21OperatingPoint):
            raise TypeError("operating_point must be a P21OperatingPoint")
        if (
            not isinstance(self.weight_decay_fp32_bits, int)
            or isinstance(self.weight_decay_fp32_bits, bool)
            or not 0 <= self.weight_decay_fp32_bits <= 0xFFFF_FFFF
        ):
            raise ValueError("weight_decay_fp32_bits must be one unsigned 32-bit integer")
        weight_decay = _fp32_from_bits(self.weight_decay_fp32_bits)
        if not bool(torch.isfinite(weight_decay)) or bool(weight_decay < 0.0):
            raise ValueError("weight decay must be a finite nonnegative binary32 scalar")
        # Construction runs the frozen backend probes in both dependencies.
        FinitePrecisionOuterLoopConfig(self.matrix_shape)
        ScalableSectorShieldConfig(self.shield_shape)

    @property
    def shield_shape(self) -> tuple[int, int]:
        return _canonical_shield_shape(self.matrix_shape)[0]

    @property
    def transpose_for_shield(self) -> bool:
        return _canonical_shield_shape(self.matrix_shape)[1]

    @property
    def learning_rate(self) -> Fraction:
        return self.operating_point.learning_rate

    @property
    def learning_rate_fp32(self) -> Tensor:
        return _fp32_from_bits(self.operating_point.fp32_bits)

    @property
    def weight_decay_fp32(self) -> Tensor:
        return _fp32_from_bits(self.weight_decay_fp32_bits)


@dataclass(frozen=True)
class StoredGradientResult:
    """The immutable FP32 gradient sample consumed by the EMA graph."""

    value: Tensor
    source_dtype: torch.dtype
    widened_from_bf16: bool


def store_gradient_fp32(gradient: Tensor, config: CertifiedOuterLoopConfig) -> StoredGradientResult:
    """Copy a finite CPU BF16/FP32 sample into the stored FP32 boundary."""

    _check_matrix(
        "gradient",
        gradient,
        config.matrix_shape,
        dtypes=(torch.float32, torch.bfloat16),
    )
    stored = gradient.to(torch.float32).contiguous().clone()
    return StoredGradientResult(
        value=stored,
        source_dtype=gradient.dtype,
        widened_from_bf16=gradient.dtype == torch.bfloat16,
    )


@dataclass(frozen=True)
class AspectScaledCandidateResult:
    """Pinned BF16 Jordan output and the candidate presented to P20."""

    raw_candidate: Tensor
    candidate: Tensor
    aspect_factor_binary64: float
    aspect_factor_binary64_hex: str
    raw_candidate_finite: bool
    candidate_finite: bool
    upstream_revision: str = PINNED_KELLER_JORDAN_MUON_REVISION
    upstream_sha256: str = PINNED_MUON_PY_SHA256


def pinned_bf16_aspect_candidate(
    signal: Tensor,
    config: CertifiedOuterLoopConfig,
) -> AspectScaledCandidateResult:
    """Run the pinned five-stage BF16 candidate and scale it before shielding."""

    _check_matrix("stored signal", signal, config.matrix_shape)
    raw = keller_jordan_map(
        signal,
        steps=LOCKED_CANDIDATE_STEPS,
        eps=LOCKED_CANDIDATE_EPSILON,
    ).contiguous()
    if raw.dtype != torch.bfloat16:
        raise TypeError("the pinned Jordan candidate must return BF16 storage")
    rows, columns = config.matrix_shape
    aspect = max(1, rows / columns) ** 0.5
    candidate = raw.clone()
    candidate.mul_(aspect)
    return AspectScaledCandidateResult(
        raw_candidate=raw,
        candidate=candidate.contiguous(),
        aspect_factor_binary64=aspect,
        aspect_factor_binary64_hex=aspect.hex(),
        raw_candidate_finite=bool(torch.isfinite(raw).all()),
        candidate_finite=bool(torch.isfinite(candidate).all()),
    )


@dataclass(frozen=True)
class DeadZoneDiagnostics:
    """Absolute port created when a positive-sector output is unrepresentable."""

    active: bool
    reason: str | None
    conceptual_reference_gain: Fraction
    output_error_frobenius_strict_upper: Fraction
    parameter_error_frobenius_strict_upper: Fraction
    signal_frobenius_strict_upper: Fraction


@dataclass(frozen=True)
class ExactDeadZonePort:
    """Exact output error ``zero - signal/2`` for the dead-zone branch."""

    signal: Tensor
    learning_rate: Fraction

    def exact_output_error_entry(self, row: int, column: int) -> Fraction:
        return -_fp32_fraction(self.signal[row, column]) / 2

    def exact_parameter_error_entry(self, row: int, column: int) -> Fraction:
        return self.learning_rate * _fp32_fraction(self.signal[row, column]) / 2


@dataclass(frozen=True)
class StoredShieldResult:
    """FP32 operator output plus either P20 or dead-zone diagnostics."""

    output: Tensor
    p20_diagnostics: ScalableSectorShieldDiagnostics | None
    dead_zone_diagnostics: DeadZoneDiagnostics
    dead_zone_port: ExactDeadZonePort | None
    transposed_for_shield: bool


def shield_or_zero_dead_zone(
    signal: Tensor,
    candidate: Tensor,
    config: CertifiedOuterLoopConfig,
) -> StoredShieldResult:
    """Apply P20, catching only its unrepresentable all-subnormal case."""

    _check_matrix("stored signal", signal, config.matrix_shape)
    _check_matrix(
        "aspect-scaled candidate",
        candidate,
        config.matrix_shape,
        dtypes=(torch.float32, torch.bfloat16),
        finite=False,
    )
    shield_signal = signal.mT.contiguous() if config.transpose_for_shield else signal
    shield_candidate = candidate.mT.contiguous() if config.transpose_for_shield else candidate
    p20_config = ScalableSectorShieldConfig(config.shield_shape)
    try:
        shielded = shield_sector_candidate_mixed_precision(
            shield_signal,
            shield_candidate,
            p20_config,
        )
    except NearZeroUnrepresentable:
        output = torch.zeros_like(signal)
        # Use ceil(sqrt(m*n))*2^-126 rather than the exception's binary64
        # display.  This is an exact rational upper bound on the Frobenius
        # radius of every all-subnormal signal, including nonsquare shapes.
        entries = math.prod(config.matrix_shape)
        root = math.isqrt(entries)
        sqrt_entries_upper = root if root * root == entries else root + 1
        signal_bound = Fraction(sqrt_entries_upper, 2**126)
        output_error_bound = signal_bound / 2
        return StoredShieldResult(
            output=output,
            p20_diagnostics=None,
            dead_zone_diagnostics=DeadZoneDiagnostics(
                active=True,
                reason="all_subnormal_positive_sector_unrepresentable_zero_update",
                conceptual_reference_gain=Fraction(1, 2),
                output_error_frobenius_strict_upper=output_error_bound,
                parameter_error_frobenius_strict_upper=(config.learning_rate * output_error_bound),
                signal_frobenius_strict_upper=signal_bound,
            ),
            dead_zone_port=ExactDeadZonePort(signal=signal, learning_rate=config.learning_rate),
            transposed_for_shield=config.transpose_for_shield,
        )

    output = shielded.output
    if config.transpose_for_shield:
        output = output.mT.contiguous()
    return StoredShieldResult(
        output=output,
        p20_diagnostics=shielded.diagnostics,
        dead_zone_diagnostics=DeadZoneDiagnostics(
            active=False,
            reason=None,
            conceptual_reference_gain=Fraction(1, 2),
            output_error_frobenius_strict_upper=Fraction(0),
            parameter_error_frobenius_strict_upper=Fraction(0),
            signal_frobenius_strict_upper=Fraction(0),
        ),
        dead_zone_port=None,
        transposed_for_shield=config.transpose_for_shield,
    )


@dataclass(frozen=True)
class ExactWeightDecayResidualPort:
    """Rounding error in ``decay_step ~= eta*weight_decay*represented``."""

    represented_weight: Tensor
    rounded_decay_step: Tensor
    learning_rate: Fraction
    weight_decay: Fraction

    def exact_entry(self, row: int, column: int) -> Fraction:
        return _fp32_fraction(self.rounded_decay_step[row, column]) - (
            self.learning_rate
            * self.weight_decay
            * _fp32_fraction(self.represented_weight[row, column])
        )


@dataclass(frozen=True)
class ExactComposedMasterResidualPort:
    """Master residual relative to stored operator and decay steps."""

    previous: ThreeWordFP32Master
    updated: ThreeWordFP32Master
    operator_output: Tensor
    rounded_decay_step: Tensor
    learning_rate: Fraction

    def exact_entry(self, row: int, column: int) -> Fraction:
        return (
            self.updated.exact_entry(row, column)
            - self.previous.exact_entry(row, column)
            + self.learning_rate * _fp32_fraction(self.operator_output[row, column])
            + _fp32_fraction(self.rounded_decay_step[row, column])
        )


@dataclass(frozen=True)
class ExactBaselineMasterResidualPort:
    """Zero-decay master residual isolated before the decay subtraction.

    The error-free ``TwoSum`` stages preserve the logical sum, so the
    hypothetical no-decay logical change is exactly
    ``pending_after_operator - previous.low``.  This is the quantity bounded
    by the reused P10 one-subtraction envelope.
    """

    previous_low: Tensor
    pending_after_operator: Tensor
    operator_output: Tensor
    learning_rate: Fraction

    def exact_entry(self, row: int, column: int) -> Fraction:
        return (
            _fp32_fraction(self.pending_after_operator[row, column])
            - _fp32_fraction(self.previous_low[row, column])
            + self.learning_rate * _fp32_fraction(self.operator_output[row, column])
        )


@dataclass(frozen=True)
class ExactDecayDisplacementPort:
    """Complete logical displacement introduced by the decay subgraph.

    This includes the rounded decay step *and* rounding in
    ``pending_after_decay = fl32(pending_after_operator-decay_step)``.  The
    P21 bounded-decay theorem assumes a norm budget on this exact port; it
    does not infer that budget merely from a configured decay coefficient.
    """

    pending_after_operator: Tensor
    pending_after_decay: Tensor

    def exact_entry(self, row: int, column: int) -> Fraction:
        return _fp32_fraction(self.pending_after_decay[row, column]) - _fp32_fraction(
            self.pending_after_operator[row, column]
        )


@dataclass(frozen=True)
class ComposedMasterUpdateTrace:
    """Every named stored operation in the P21 master update."""

    rounded_operator_step: Tensor
    rounded_eta_weight_decay: Tensor
    rounded_decay_step: Tensor
    pending_after_operator: Tensor
    pending_after_decay: Tensor
    middle_update: TwoSumResult
    high_update: TwoSumResult
    total_step_max_abs: float
    weight_decay_enabled: bool


@dataclass(frozen=True)
class ComposedMasterUpdateResult:
    """Updated three-word master with separate arithmetic and decay ports."""

    master: ThreeWordFP32Master
    master_residual_port: ExactComposedMasterResidualPort
    baseline_master_residual_port: ExactBaselineMasterResidualPort
    decay_displacement_port: ExactDecayDisplacementPort
    weight_decay_residual_port: ExactWeightDecayResidualPort
    trace: ComposedMasterUpdateTrace


def compensated_master_update_at_operating_point(
    master: ThreeWordFP32Master,
    operator_output: Tensor,
    config: CertifiedOuterLoopConfig,
) -> ComposedMasterUpdateResult:
    """Apply one rate-selected compensated update with an explicit decay port.

    The locked stored-operation order is

    ``operator_step = fl32(eta32 * operator_output)``,
    ``eta_decay = fl32(eta32 * weight_decay32)``,
    ``decay_step = fl32(eta_decay * master.high)``,
    ``pending_1 = fl32(master.low - operator_step)``, then
    ``pending_2 = fl32(pending_1 - decay_step)``.

    ``pending_2`` is accumulated through the two public P10 ``TwoSum`` calls.
    Thus decay is evaluated at the represented *pre-update* FP32 high word and
    is a separately diagnosed port; it is neither folded into the candidate nor
    recomputed at the updated master.
    """

    if master.matrix_shape != config.matrix_shape:
        raise ValueError(f"master must have configured shape {config.matrix_shape}")
    _check_matrix("operator output", operator_output, config.matrix_shape)
    output_maximum = float(torch.amax(torch.abs(operator_output)))
    if output_maximum > LOCKED_OPERATOR_OUTPUT_MAX_ABS:
        raise ValueError("operator output exceeds the locked P21 max-absolute range 2^6")

    eta = config.learning_rate_fp32
    rounded_operator_step = torch.mul(eta, operator_output)
    if not bool(torch.isfinite(rounded_operator_step).all()):
        raise FloatingPointError("nonfinite FP32 operator step")

    weight_decay = config.weight_decay_fp32
    represented = master.high
    weight_decay_enabled = bool(weight_decay != 0.0)
    if weight_decay_enabled:
        rounded_eta_weight_decay = torch.mul(eta, weight_decay)
        rounded_decay_step = torch.mul(rounded_eta_weight_decay, represented)
        if not bool(torch.isfinite(rounded_decay_step).all()):
            raise FloatingPointError("nonfinite FP32 weight-decay step")
    else:
        rounded_eta_weight_decay = torch.zeros((), dtype=torch.float32, device="cpu")
        rounded_decay_step = torch.zeros_like(operator_output)

    total_step_max_abs = float(
        torch.amax(torch.add(torch.abs(rounded_operator_step), torch.abs(rounded_decay_step)))
    )
    if not math.isfinite(total_step_max_abs) or total_step_max_abs > LOCKED_TOTAL_STEP_MAX_ABS:
        raise ValueError("combined operator and decay steps exceed the locked max-absolute range 1")

    pending_after_operator = torch.sub(master.low, rounded_operator_step)
    pending_after_decay = (
        torch.sub(pending_after_operator, rounded_decay_step)
        if weight_decay_enabled
        else pending_after_operator
    )
    if not bool(torch.isfinite(pending_after_decay).all()):
        raise FloatingPointError("nonfinite FP32 pending master update")
    middle_update = two_sum_fp32(
        master.middle,
        pending_after_decay,
        name="P21 middle+pending TwoSum",
    )
    high_update = two_sum_fp32(
        master.high,
        middle_update.rounded,
        name="P21 high+middle TwoSum",
    )
    updated = ThreeWordFP32Master(
        high=high_update.rounded,
        middle=high_update.residual,
        low=middle_update.residual,
    )
    return ComposedMasterUpdateResult(
        master=updated,
        master_residual_port=ExactComposedMasterResidualPort(
            previous=master,
            updated=updated,
            operator_output=operator_output,
            rounded_decay_step=rounded_decay_step,
            learning_rate=config.learning_rate,
        ),
        baseline_master_residual_port=ExactBaselineMasterResidualPort(
            previous_low=master.low,
            pending_after_operator=pending_after_operator,
            operator_output=operator_output,
            learning_rate=config.learning_rate,
        ),
        decay_displacement_port=ExactDecayDisplacementPort(
            pending_after_operator=pending_after_operator,
            pending_after_decay=pending_after_decay,
        ),
        weight_decay_residual_port=ExactWeightDecayResidualPort(
            represented_weight=represented,
            rounded_decay_step=rounded_decay_step,
            learning_rate=config.learning_rate,
            weight_decay=_fp32_fraction(weight_decay),
        ),
        trace=ComposedMasterUpdateTrace(
            rounded_operator_step=rounded_operator_step,
            rounded_eta_weight_decay=rounded_eta_weight_decay,
            rounded_decay_step=rounded_decay_step,
            pending_after_operator=pending_after_operator,
            pending_after_decay=pending_after_decay,
            middle_update=middle_update,
            high_update=high_update,
            total_step_max_abs=total_step_max_abs,
            weight_decay_enabled=weight_decay_enabled,
        ),
    )


@dataclass(frozen=True)
class ExactModelReconstructionPort:
    """Pre-update gradient-model difference from the logical three-word master.

    Gradients are evaluated at ``master.high``.  This port therefore belongs to
    the state entering the step, not the updated state returned by the step.
    """

    master: ThreeWordFP32Master

    def exact_entry(self, row: int, column: int) -> Fraction:
        return -(
            _fp32_fraction(self.master.middle[row, column])
            + _fp32_fraction(self.master.low[row, column])
        )

    def float64_diagnostic(self) -> Tensor:
        return -(self.master.middle.double() + self.master.low.double())


@dataclass(frozen=True)
class CertifiedOuterLoopState:
    """Stored optimizer state; the FP32 high word is the represented model."""

    master: ThreeWordFP32Master
    momentum: Tensor

    @property
    def represented_model(self) -> Tensor:
        return self.master.high


@dataclass(frozen=True)
class CertifiedOuterLoopStepResult:
    """One complete P21 stored-computation result and its named ports.

    ``model_reconstruction_port`` diagnoses the pre-update represented model at
    which the supplied gradient was evaluated.
    """

    state: CertifiedOuterLoopState
    stored_gradient: StoredGradientResult
    ema_nesterov: FP32EmaNesterovResult
    candidate: AspectScaledCandidateResult
    shield: StoredShieldResult
    master_update: ComposedMasterUpdateResult
    model_reconstruction_port: ExactModelReconstructionPort


def certified_outer_loop_step(
    state: CertifiedOuterLoopState,
    gradient: Tensor,
    config: CertifiedOuterLoopConfig,
) -> CertifiedOuterLoopStepResult:
    """Run one atomic, non-mutating P21 proof-reference optimizer step."""

    if state.master.matrix_shape != config.matrix_shape:
        raise ValueError(f"master must have configured shape {config.matrix_shape}")
    _check_matrix("momentum", state.momentum, config.matrix_shape)
    stored_gradient = store_gradient_fp32(gradient, config)
    p10_config = FinitePrecisionOuterLoopConfig(config.matrix_shape)
    ema = fp32_ema_nesterov(state.momentum, stored_gradient.value, p10_config)
    # P20's theorem is relative to this exact stored tensor.  Do not rebuild it
    # from the residual ports or compare an operator call at an ideal signal.
    candidate = pinned_bf16_aspect_candidate(ema.signal, config)
    shield = shield_or_zero_dead_zone(ema.signal, candidate.candidate, config)
    update = compensated_master_update_at_operating_point(state.master, shield.output, config)
    next_state = CertifiedOuterLoopState(master=update.master, momentum=ema.momentum)
    return CertifiedOuterLoopStepResult(
        state=next_state,
        stored_gradient=stored_gradient,
        ema_nesterov=ema,
        candidate=candidate,
        shield=shield,
        master_update=update,
        model_reconstruction_port=ExactModelReconstructionPort(state.master),
    )


def supported_matrix_orientations() -> tuple[tuple[int, int], ...]:
    """Return representative P20 shapes and their nonduplicate transposes."""

    shapes = list(LOCKED_REPRESENTATIVE_SHAPES)
    shapes.extend(
        (columns, rows) for rows, columns in LOCKED_REPRESENTATIVE_SHAPES if rows != columns
    )
    return tuple(shapes)


__all__ = [
    "CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION",
    "LOCKED_CANDIDATE_EPSILON",
    "LOCKED_CANDIDATE_STEPS",
    "LOCKED_OPERATOR_OUTPUT_MAX_ABS",
    "LOCKED_TOTAL_STEP_MAX_ABS",
    "MAXIMUM_LEARNING_RATE_FP32_BITS",
    "MAXIMUM_LEARNING_RATE_FP32_EXACT",
    "PRIMARY_LEARNING_RATE_FP32_BITS",
    "PRIMARY_LEARNING_RATE_FP32_EXACT",
    "AspectScaledCandidateResult",
    "CertifiedOuterLoopConfig",
    "CertifiedOuterLoopState",
    "CertifiedOuterLoopStepResult",
    "ComposedMasterUpdateResult",
    "ComposedMasterUpdateTrace",
    "DeadZoneDiagnostics",
    "ExactBaselineMasterResidualPort",
    "ExactComposedMasterResidualPort",
    "ExactDeadZonePort",
    "ExactDecayDisplacementPort",
    "ExactModelReconstructionPort",
    "ExactWeightDecayResidualPort",
    "P21OperatingPoint",
    "StoredGradientResult",
    "StoredShieldResult",
    "certified_outer_loop_step",
    "compensated_master_update_at_operating_point",
    "pinned_bf16_aspect_candidate",
    "shield_or_zero_dead_zone",
    "store_gradient_fp32",
    "supported_matrix_orientations",
]
