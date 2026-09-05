from __future__ import annotations

import math
from fractions import Fraction

import pytest
import torch

from passive_muon.certified_outer_loop_composition import (
    LOCKED_CANDIDATE_EPSILON,
    LOCKED_CANDIDATE_STEPS,
    MAXIMUM_LEARNING_RATE_FP32_BITS,
    MAXIMUM_LEARNING_RATE_FP32_EXACT,
    PRIMARY_LEARNING_RATE_FP32_BITS,
    PRIMARY_LEARNING_RATE_FP32_EXACT,
    CertifiedOuterLoopConfig,
    CertifiedOuterLoopState,
    ExactModelReconstructionPort,
    P21OperatingPoint,
    certified_outer_loop_step,
    compensated_master_update_at_operating_point,
    pinned_bf16_aspect_candidate,
    shield_or_zero_dead_zone,
    store_gradient_fp32,
    supported_matrix_orientations,
)
from passive_muon.deployed import keller_jordan_map
from passive_muon.finite_precision_outer_loop import (
    FinitePrecisionOuterLoopConfig,
    ThreeWordFP32Master,
    fp32_ema_nesterov,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_REPRESENTATIVE_SHAPES,
    ShieldAction,
)


def _bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _from_bits(bits: int) -> torch.Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _fraction(value: torch.Tensor) -> Fraction:
    return Fraction.from_float(float(value.reshape(1).item()))


def _float32_bits(value: float) -> int:
    return _bits(torch.tensor(value, dtype=torch.float32))


def _exact_disk_margin(output: torch.Tensor, signal: torch.Tensor) -> Fraction:
    center = Fraction(1_143, 2_048)
    radius = Fraction(893, 2_048)
    exact_output = tuple(Fraction.from_float(float(value)) for value in output.ravel())
    exact_signal = tuple(Fraction.from_float(float(value)) for value in signal.ravel())
    return radius**2 * sum(value**2 for value in exact_signal) - sum(
        (value - center * source) ** 2
        for value, source in zip(exact_output, exact_signal, strict=True)
    )


def test_operating_point_bits_exact_values_and_supported_orientations() -> None:
    primary = CertifiedOuterLoopConfig((2, 2))
    maximum = CertifiedOuterLoopConfig(
        (2, 2),
        operating_point=P21OperatingPoint.MAXIMUM_STEP,
    )

    assert primary.learning_rate == Fraction(1, 120)
    assert maximum.learning_rate == Fraction(1, 83)
    assert _bits(primary.learning_rate_fp32) == PRIMARY_LEARNING_RATE_FP32_BITS
    assert _bits(maximum.learning_rate_fp32) == MAXIMUM_LEARNING_RATE_FP32_BITS
    assert _fraction(primary.learning_rate_fp32) == PRIMARY_LEARNING_RATE_FP32_EXACT
    assert _fraction(maximum.learning_rate_fp32) == MAXIMUM_LEARNING_RATE_FP32_EXACT
    assert PRIMARY_LEARNING_RATE_FP32_BITS == 0x3C088889
    assert MAXIMUM_LEARNING_RATE_FP32_BITS == 0x3C4565C8

    orientations = supported_matrix_orientations()
    for shape in LOCKED_REPRESENTATIVE_SHAPES:
        assert shape in orientations
        assert CertifiedOuterLoopConfig(shape).shield_shape == shape
        reverse = (shape[1], shape[0])
        assert reverse in orientations
        reverse_config = CertifiedOuterLoopConfig(reverse)
        if reverse != shape:
            assert reverse_config.transpose_for_shield
            assert reverse_config.shield_shape == shape


def test_stored_gradient_is_an_immutable_fp32_boundary() -> None:
    config = CertifiedOuterLoopConfig((2, 2))
    source = torch.tensor([[1.0, -2.0], [0.25, 3.0]], dtype=torch.bfloat16)
    source_before = source.clone()

    stored = store_gradient_fp32(source, config)
    source[0, 0] = 7.0

    assert stored.source_dtype == torch.bfloat16
    assert stored.widened_from_bf16
    assert stored.value.dtype == torch.float32
    assert torch.equal(stored.value, source_before.float())
    with pytest.raises(ValueError, match="finite"):
        store_gradient_fp32(
            torch.tensor([[math.inf, 0.0], [0.0, 0.0]], dtype=torch.float32),
            config,
        )


def test_pinned_candidate_uses_literal_bf16_graph_and_aspect_before_shield() -> None:
    config = CertifiedOuterLoopConfig((2, 1))
    signal = torch.tensor([[3.0], [4.0]], dtype=torch.float32)
    expected_raw = keller_jordan_map(
        signal,
        steps=LOCKED_CANDIDATE_STEPS,
        eps=LOCKED_CANDIDATE_EPSILON,
    )
    expected_candidate = expected_raw.clone()
    expected_candidate.mul_(math.sqrt(2.0))

    result = pinned_bf16_aspect_candidate(signal, config)

    assert result.raw_candidate.dtype == torch.bfloat16
    assert result.candidate.dtype == torch.bfloat16
    assert torch.equal(result.raw_candidate, expected_raw)
    assert torch.equal(result.candidate, expected_candidate)
    assert result.aspect_factor_binary64 == math.sqrt(2.0)
    assert result.aspect_factor_binary64_hex == math.sqrt(2.0).hex()
    assert result.raw_candidate_finite
    assert result.candidate_finite

    shielded = shield_or_zero_dead_zone(signal, result.candidate, config)
    assert shielded.transposed_for_shield
    assert shielded.output.shape == signal.shape
    assert shielded.output.dtype == torch.float32
    assert _exact_disk_margin(shielded.output, signal) >= 0


def test_p20_all_subnormal_boundary_is_closed_by_an_explicit_zero_port() -> None:
    config = CertifiedOuterLoopConfig((1, 2))
    minimum_subnormal = _from_bits(0x00000001)
    signal = torch.stack((minimum_subnormal, torch.tensor(0.0))).reshape(1, 2)
    candidate = torch.zeros((1, 2), dtype=torch.bfloat16)

    result = shield_or_zero_dead_zone(signal, candidate, config)

    assert torch.equal(result.output, torch.zeros_like(signal))
    assert result.p20_diagnostics is None
    assert result.dead_zone_diagnostics.active
    assert result.dead_zone_diagnostics.conceptual_reference_gain == Fraction(1, 2)
    assert result.dead_zone_port is not None
    assert result.dead_zone_port.exact_output_error_entry(0, 0) == -Fraction(1, 2**150)
    assert result.dead_zone_port.exact_parameter_error_entry(0, 0) == Fraction(1, 120 * 2**150)
    assert result.dead_zone_diagnostics.output_error_frobenius_strict_upper > 0.0
    assert result.dead_zone_diagnostics.parameter_error_frobenius_strict_upper > 0.0

    exactly_halvable = torch.stack((_from_bits(0x00000002), torch.tensor(0.0))).reshape(1, 2)
    ordinary = shield_or_zero_dead_zone(exactly_halvable, candidate, config)
    assert ordinary.p20_diagnostics is not None
    assert ordinary.p20_diagnostics.action is ShieldAction.HALF_FALLBACK
    assert not ordinary.dead_zone_diagnostics.active
    assert _bits(ordinary.output[0, 0]) == 0x00000001


def test_nonfinite_signal_is_never_downgraded_to_a_zero_update() -> None:
    config = CertifiedOuterLoopConfig((1, 2))
    signal = torch.tensor([[math.inf, 0.0]], dtype=torch.float32)
    candidate = torch.zeros((1, 2), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="finite"):
        shield_or_zero_dead_zone(signal, candidate, config)


@pytest.mark.parametrize(
    "operating_point",
    (P21OperatingPoint.PRIMARY, P21OperatingPoint.MAXIMUM_STEP),
)
def test_rate_parametric_master_update_replays_exact_logical_port(
    operating_point: P21OperatingPoint,
) -> None:
    config = CertifiedOuterLoopConfig((2, 2), operating_point=operating_point)
    master = ThreeWordFP32Master(
        high=torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32),
        middle=torch.tensor([[2.0**-24, 0.0], [-(2.0**-25), 2.0**-23]], dtype=torch.float32),
        low=torch.tensor([[0.0, 2.0**-40], [2.0**-42, 0.0]], dtype=torch.float32),
    )
    output = torch.tensor([[0.5, -0.25], [1.0, -2.0]], dtype=torch.float32)

    result = compensated_master_update_at_operating_point(master, output, config)

    assert _bits(result.trace.rounded_operator_step[0, 0] / output[0, 0]) == (
        operating_point.fp32_bits
    )
    assert not result.trace.weight_decay_enabled
    assert torch.count_nonzero(result.trace.rounded_decay_step) == 0
    for row in range(2):
        for column in range(2):
            old = master.exact_entry(row, column)
            new = result.master.exact_entry(row, column)
            pending = _fraction(result.trace.pending_after_decay[row, column])
            old_low = _fraction(master.low[row, column])
            assert new - old == pending - old_low
            assert result.master_residual_port.exact_entry(row, column) == (
                new
                - old
                + config.learning_rate * _fraction(output[row, column])
                + _fraction(result.trace.rounded_decay_step[row, column])
            )
            assert result.baseline_master_residual_port.exact_entry(row, column) == (
                pending - old_low + config.learning_rate * _fraction(output[row, column])
            )
            assert result.decay_displacement_port.exact_entry(row, column) == 0


def test_weight_decay_is_a_separate_stored_update_and_rounding_port() -> None:
    weight_decay_bits = _float32_bits(0.01)
    config = CertifiedOuterLoopConfig((2, 2), weight_decay_fp32_bits=weight_decay_bits)
    master = ThreeWordFP32Master.from_primary(
        torch.tensor([[1.0, -2.0], [0.5, -0.25]], dtype=torch.float32)
    )
    output = torch.tensor([[0.5, -0.25], [1.0, -2.0]], dtype=torch.float32)

    result = compensated_master_update_at_operating_point(master, output, config)
    eta = config.learning_rate_fp32
    weight_decay = config.weight_decay_fp32
    expected_eta_decay = torch.mul(eta, weight_decay)
    expected_decay_step = torch.mul(expected_eta_decay, master.high)
    expected_operator_step = torch.mul(eta, output)
    expected_pending_after_operator = torch.sub(master.low, expected_operator_step)
    expected_pending_after_decay = torch.sub(
        expected_pending_after_operator,
        expected_decay_step,
    )

    assert result.trace.weight_decay_enabled
    assert torch.equal(result.trace.rounded_operator_step, expected_operator_step)
    assert torch.equal(result.trace.rounded_eta_weight_decay, expected_eta_decay)
    assert torch.equal(result.trace.rounded_decay_step, expected_decay_step)
    assert torch.equal(result.trace.pending_after_operator, expected_pending_after_operator)
    assert torch.equal(result.trace.pending_after_decay, expected_pending_after_decay)
    for row in range(2):
        for column in range(2):
            expected_residual = _fraction(expected_decay_step[row, column]) - (
                config.learning_rate * _fraction(weight_decay) * _fraction(master.high[row, column])
            )
            assert result.weight_decay_residual_port.exact_entry(row, column) == expected_residual
            decay_displacement = _fraction(expected_pending_after_decay[row, column]) - _fraction(
                expected_pending_after_operator[row, column]
            )
            assert result.decay_displacement_port.exact_entry(row, column) == decay_displacement
            logical_change = result.master.exact_entry(row, column) - master.exact_entry(
                row, column
            )
            assert (
                logical_change + config.learning_rate * _fraction(output[row, column])
                == result.baseline_master_residual_port.exact_entry(row, column)
                + decay_displacement
            )


def test_represented_model_is_high_word_with_an_exact_reconstruction_port() -> None:
    master = ThreeWordFP32Master(
        high=torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32),
        middle=torch.tensor([[2.0**-24, 0.0], [-(2.0**-25), 2.0**-23]], dtype=torch.float32),
        low=torch.tensor([[0.0, 2.0**-40], [2.0**-42, 0.0]], dtype=torch.float32),
    )
    state = CertifiedOuterLoopState(master=master, momentum=torch.zeros((2, 2)))
    port = ExactModelReconstructionPort(master)

    assert state.represented_model.data_ptr() == master.high.data_ptr()
    for row in range(2):
        for column in range(2):
            assert port.exact_entry(row, column) == (
                _fraction(master.high[row, column]) - master.exact_entry(row, column)
            )


def test_complete_step_uses_stored_signal_and_does_not_mutate_inputs() -> None:
    config = CertifiedOuterLoopConfig((2, 2))
    primary = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    middle = torch.full((2, 2), 2.0**-24, dtype=torch.float32)
    low = torch.full((2, 2), 2.0**-40, dtype=torch.float32)
    momentum = torch.tensor([[0.25, -0.5], [0.75, -1.0]], dtype=torch.float32)
    gradient = torch.tensor([[-1.0, 2.0], [0.5, 1.25]], dtype=torch.bfloat16)
    state = CertifiedOuterLoopState(
        master=ThreeWordFP32Master(high=primary, middle=middle, low=low),
        momentum=momentum,
    )
    primary_before = primary.clone()
    momentum_before = momentum.clone()
    gradient_before = gradient.clone()

    result = certified_outer_loop_step(state, gradient, config)
    independent_ema = fp32_ema_nesterov(
        momentum,
        gradient.float(),
        FinitePrecisionOuterLoopConfig((2, 2)),
    )

    assert torch.equal(result.stored_gradient.value, gradient_before.float())
    assert torch.equal(result.ema_nesterov.signal, independent_ema.signal)
    assert result.candidate.raw_candidate.dtype == torch.bfloat16
    assert result.candidate.candidate.dtype == torch.bfloat16
    assert result.shield.output.dtype == torch.float32
    assert result.state.momentum.dtype == torch.float32
    assert result.state.represented_model.dtype == torch.float32
    assert result.model_reconstruction_port.master is state.master
    assert result.model_reconstruction_port.master is not result.state.master
    assert result.model_reconstruction_port.exact_entry(0, 0) == -(
        _fraction(middle[0, 0]) + _fraction(low[0, 0])
    )
    assert _exact_disk_margin(result.shield.output, result.ema_nesterov.signal) >= 0
    assert torch.equal(primary, primary_before)
    assert torch.equal(momentum, momentum_before)
    assert torch.equal(gradient, gradient_before)


def test_master_update_range_guard_fails_before_state_is_returned() -> None:
    config = CertifiedOuterLoopConfig((2, 2))
    master = ThreeWordFP32Master.from_primary(torch.zeros((2, 2), dtype=torch.float32))
    with pytest.raises(ValueError, match=r"2\^6"):
        compensated_master_update_at_operating_point(
            master,
            torch.full((2, 2), 65.0, dtype=torch.float32),
            config,
        )
