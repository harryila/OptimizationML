from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_BITS,
    BETA_FP32_EXACT,
    CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER,
    EMA_GRADIENT_ENVELOPE_COEFFICIENT,
    EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
    EMA_RESIDUAL_CRUMB_PER_ENTRY,
    FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LEARNING_RATE_FP32_BITS,
    LEARNING_RATE_FP32_EXACT,
    LOCKED_MASTER_HIGH_MAX_ABS_POWER,
    LOCKED_MASTER_LOW_MAX_ABS,
    LOCKED_MASTER_LOW_MAX_ABS_POWER,
    LOCKED_MASTER_MIDDLE_MAX_ABS_POWER,
    LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER,
    LOCKED_OUTER_BACKEND,
    LOCKED_ROUNDED_STEP_MAX_ABS_POWER,
    MASTER_LOW_WORD_ENVELOPE_COEFFICIENT,
    MASTER_OUTPUT_ENVELOPE_COEFFICIENT,
    MASTER_RESIDUAL_CRUMB_PER_ENTRY,
    ONE_MINUS_BETA_FP32_BITS,
    ONE_MINUS_BETA_FP32_EXACT,
    FinitePrecisionOuterLoopConfig,
    P9RepairedOperatorAdapter,
    ThreeWordFP32Master,
    compensated_master_weight_update,
    finite_precision_outer_loop_manifest,
    finite_precision_outer_step,
    fp32_ema_nesterov,
    fp32_master_stalling_witness,
    locked_outer_backend_self_check,
    outer_residual_envelope,
    raw_fp32_parameter_update,
    two_sum_fp32,
)
from passive_muon.scalable_mixed_precision import ScalableMixedPrecisionConfig
from passive_muon.structure_aware_stability import LOCKED_BETA, LOCKED_LEARNING_RATE


def _from_bits(bits: int) -> torch.Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _fraction(value: torch.Tensor) -> Fraction:
    return Fraction.from_float(float(value.reshape(1).item()))


class _RecordingAffineOperator:
    def __init__(self, matrix_shape: tuple[int, int]) -> None:
        self.matrix_shape = matrix_shape
        self.interface_id = "test-affine-fp32-operator-v1"
        self.calls: list[torch.Tensor] = []

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        self.calls.append(signal.clone())
        return torch.add(torch.mul(signal, signal.new_tensor(2.0)), signal.new_tensor(0.25))

    def manifest_reference(self) -> dict[str, object]:
        return {
            "interface_id": self.interface_id,
            "matrix_shape": list(self.matrix_shape),
            "source_file": "tests/test_finite_precision_outer_loop.py",
            "source_sha256": "test-only",
        }


def test_runtime_scalar_bits_and_backend_contract_are_locked() -> None:
    report = locked_outer_backend_self_check()

    assert BETA_FP32_BITS == 0x3F733333
    assert ONE_MINUS_BETA_FP32_BITS == 0x3D4CCCCD
    assert LEARNING_RATE_FP32_BITS == 0x3803126F
    assert _fraction(_from_bits(BETA_FP32_BITS)) == Fraction(15_938_355, 16_777_216)
    assert _fraction(_from_bits(ONE_MINUS_BETA_FP32_BITS)) == Fraction(13_421_773, 268_435_456)
    assert _fraction(_from_bits(LEARNING_RATE_FP32_BITS)) == Fraction(8_589_935, 274_877_906_944)
    assert _fraction(_from_bits(BETA_FP32_BITS)) == BETA_FP32_EXACT
    assert _fraction(_from_bits(ONE_MINUS_BETA_FP32_BITS)) == ONE_MINUS_BETA_FP32_EXACT
    assert _fraction(_from_bits(LEARNING_RATE_FP32_BITS)) == LEARNING_RATE_FP32_EXACT
    assert report["rounding"] == "IEEE-754 roundTiesToEven"
    assert report["gradual_underflow"] is True
    assert report["ftz_daz"] is False


def test_residual_envelopes_are_exact_fractions_with_shape_scaled_crumbs() -> None:
    config = FinitePrecisionOuterLoopConfig((4_096, 11_008))
    envelope = outer_residual_envelope(config)
    two_operation_factor = 2 * FP32_UNIT_ROUNDOFF + FP32_UNIT_ROUNDOFF**2
    sqrt_entries_upper = 6_715

    assert envelope.entry_count == 45_088_768
    assert envelope.sqrt_entry_count_upper == sqrt_entries_upper
    assert (
        abs(BETA_FP32_EXACT - LOCKED_BETA) + two_operation_factor * BETA_FP32_EXACT
    ) == EMA_MOMENTUM_ENVELOPE_COEFFICIENT
    assert (
        abs(ONE_MINUS_BETA_FP32_EXACT - (1 - LOCKED_BETA))
        + two_operation_factor * ONE_MINUS_BETA_FP32_EXACT
    ) == EMA_GRADIENT_ENVELOPE_COEFFICIENT
    assert EMA_RESIDUAL_CRUMB_PER_ENTRY == (3 + 2 * FP32_UNIT_ROUNDOFF) * FP32_HALF_MIN_SUBNORMAL
    assert (
        abs(LEARNING_RATE_FP32_EXACT - LOCKED_LEARNING_RATE)
        + two_operation_factor * LEARNING_RATE_FP32_EXACT
    ) == MASTER_OUTPUT_ENVELOPE_COEFFICIENT
    assert MASTER_LOW_WORD_ENVELOPE_COEFFICIENT == FP32_UNIT_ROUNDOFF
    assert MASTER_RESIDUAL_CRUMB_PER_ENTRY == (2 + FP32_UNIT_ROUNDOFF) * FP32_HALF_MIN_SUBNORMAL
    assert envelope.ema_absolute_crumb == sqrt_entries_upper * EMA_RESIDUAL_CRUMB_PER_ENTRY
    assert envelope.master_absolute_crumb == (sqrt_entries_upper * MASTER_RESIDUAL_CRUMB_PER_ENTRY)
    assert envelope.master_absolute_part_under_low_guard == (
        sqrt_entries_upper * FP32_UNIT_ROUNDOFF * LOCKED_MASTER_LOW_MAX_ABS
        + envelope.master_absolute_crumb
    )
    assert all(
        isinstance(value, Fraction)
        for value in (
            envelope.ema_momentum_coefficient,
            envelope.ema_gradient_coefficient,
            envelope.ema_absolute_crumb,
            envelope.master_output_coefficient,
            envelope.master_low_word_coefficient,
            envelope.master_absolute_crumb,
            envelope.master_absolute_bound_under_output_and_low_guards,
            envelope.master_absolute_bound_under_certificate_guards,
        )
    )


def test_ema_nesterov_matches_independent_literal_graph_and_exact_ports() -> None:
    config = FinitePrecisionOuterLoopConfig((2, 3))
    momentum = torch.tensor(
        [[0.25, -2.0, 17.0], [-0.0, 2.0**-120, -(2.0**20)]],
        dtype=torch.float32,
    )
    gradient = torch.tensor(
        [[-1.0, 3.0, -11.0], [2.0**-130, -(2.0**-119), 2.0**19]],
        dtype=torch.float32,
    )
    beta = _from_bits(BETA_FP32_BITS)
    weight = _from_bits(ONE_MINUS_BETA_FP32_BITS)
    expected_bg = torch.mul(weight, gradient)
    expected_bm = torch.mul(beta, momentum)
    expected_momentum = torch.add(expected_bm, expected_bg)
    expected_bs = torch.mul(beta, expected_momentum)
    expected_signal = torch.add(expected_bs, expected_bg)

    result = fp32_ema_nesterov(momentum, gradient, config)

    assert torch.equal(result.trace.weighted_gradient, expected_bg)
    assert torch.equal(result.trace.beta_momentum, expected_bm)
    assert torch.equal(result.momentum, expected_momentum)
    assert torch.equal(result.trace.beta_momentum_next, expected_bs)
    assert torch.equal(result.signal, expected_signal)
    assert result.residual_ports.r_m.symbol == "r^m"
    assert result.residual_ports.r_s.symbol == "r^s"

    for row in range(2):
        for column in range(3):
            m = _fraction(momentum[row, column])
            g = _fraction(gradient[row, column])
            m_next = _fraction(result.momentum[row, column])
            signal = _fraction(result.signal[row, column])
            r_m = result.residual_ports.r_m.exact_entry(row, column)
            r_s = result.residual_ports.r_s.exact_entry(row, column)
            assert m_next == LOCKED_BETA * m + (1 - LOCKED_BETA) * g + r_m
            assert signal == LOCKED_BETA * m_next + (1 - LOCKED_BETA) * g + r_s


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (2.0**30, -(2.0**4 + 0.375)),
        (1.0, 2.0**-24),
        (2.0**-126, 2.0**-149),
        (-12345.5, 0.03125),
    ],
)
def test_two_sum_is_entrywise_error_free_on_locked_edge_cases(left: float, right: float) -> None:
    left_tensor = torch.tensor([[left]], dtype=torch.float32)
    right_tensor = torch.tensor([[right]], dtype=torch.float32)

    result = two_sum_fp32(left_tensor, right_tensor)

    assert _fraction(result.rounded) + _fraction(result.residual) == (
        _fraction(left_tensor) + _fraction(right_tensor)
    )


def test_three_word_master_update_matches_literal_graph_and_exact_logical_port() -> None:
    config = FinitePrecisionOuterLoopConfig((2, 2))
    master = ThreeWordFP32Master(
        high=torch.tensor([[2.0**30, -8.0], [0.5, -(2.0**20)]], dtype=torch.float32),
        middle=torch.tensor([[0.0, 2.0**-22], [-(2.0**-26), 0.25]], dtype=torch.float32),
        low=torch.tensor([[0.0, -(2.0**-47)], [2.0**-51, -(2.0**-24)]], dtype=torch.float32),
    )
    operator_output = torch.tensor([[2.0**15, -17.0], [0.25, -(2.0**12)]], dtype=torch.float32)
    eta = _from_bits(LEARNING_RATE_FP32_BITS)
    expected_step = torch.mul(eta, operator_output)
    expected_pending = torch.sub(master.low, expected_step)
    expected_middle = two_sum_fp32(master.middle, expected_pending)
    expected_high = two_sum_fp32(master.high, expected_middle.rounded)

    result = compensated_master_weight_update(master, operator_output, config)

    assert torch.equal(result.trace.rounded_step, expected_step)
    assert torch.equal(result.trace.pending, expected_pending)
    assert torch.equal(result.master.high, expected_high.rounded)
    assert torch.equal(result.master.middle, expected_high.residual)
    assert torch.equal(result.master.low, expected_middle.residual)
    for row in range(2):
        for column in range(2):
            old = master.exact_entry(row, column)
            new = result.master.exact_entry(row, column)
            pending = _fraction(expected_pending[row, column])
            old_low = _fraction(master.low[row, column])
            assert new - old == pending - old_low
            assert result.residual_port.exact_entry(row, column) == (
                new - old + LOCKED_LEARNING_RATE * _fraction(operator_output[row, column])
            )


def test_exact_stalling_witness_separates_raw_and_compensated_master() -> None:
    witness = fp32_master_stalling_witness()

    assert witness["certified"] is True
    assert all(witness["checks"].values())
    assert witness["scope"] == "actual P9 repaired-operator parameter-stalling witness"
    assert witness["operator_interface_id"].startswith("passive-muon-scalable-mixed-precision")
    assert witness["initial_fp32_bits_hex"] == "0x4e800000"
    assert witness["raw_final_bits_hex"] == witness["initial_fp32_bits_hex"]
    assert Fraction(witness["rounded_step_exact"]) > 0
    assert Fraction(witness["compensated_first_middle_exact"]) < 0
    assert witness["compensated_first_high_move_iteration"] is not None
    assert Fraction(witness["compensated_final"]["high_exact"]) < Fraction(witness["initial_exact"])
    assert witness["synthetic_arithmetic_control"]["certified"] is True
    assert witness["synthetic_arithmetic_control"]["scope"].startswith("synthetic")


def test_complete_outer_step_uses_one_clean_operator_call_and_no_input_mutation() -> None:
    config = FinitePrecisionOuterLoopConfig((2, 3))
    operator = _RecordingAffineOperator(config.matrix_shape)
    primary = torch.tensor([[1.0, -2.0, 0.5], [3.0, -0.25, 4.0]], dtype=torch.float32)
    master = ThreeWordFP32Master.from_primary(primary)
    momentum = torch.tensor([[0.25, -0.5, 1.0], [0.0, 0.75, -1.5]], dtype=torch.float32)
    gradient = torch.tensor([[-1.0, 2.0, 0.5], [3.0, -4.0, 1.25]], dtype=torch.float32)
    primary_before = primary.clone()
    momentum_before = momentum.clone()
    gradient_before = gradient.clone()

    result = finite_precision_outer_step(master, momentum, gradient, operator, config)
    ema = fp32_ema_nesterov(momentum, gradient, config)
    expected_output = torch.add(torch.mul(ema.signal, 2.0), 0.25)
    expected_update = compensated_master_weight_update(master, expected_output, config)

    assert len(operator.calls) == 1
    assert torch.equal(operator.calls[0], ema.signal)
    assert torch.equal(result.signal, ema.signal)
    assert torch.equal(result.operator_output, expected_output)
    assert torch.equal(result.master.high, expected_update.master.high)
    assert torch.equal(result.master.middle, expected_update.master.middle)
    assert torch.equal(result.master.low, expected_update.master.low)
    assert torch.equal(primary, primary_before)
    assert torch.equal(momentum, momentum_before)
    assert torch.equal(gradient, gradient_before)


def test_p9_adapter_matches_public_p9_operator_on_small_shape() -> None:
    p9_config = ScalableMixedPrecisionConfig((2, 3))
    operator = P9RepairedOperatorAdapter(p9_config)
    signal = torch.tensor([[1.0, -2.0, 0.25], [0.5, 3.0, -1.0]], dtype=torch.float32)

    first = operator(signal)
    second = operator(signal)

    assert operator.matrix_shape == (2, 3)
    assert torch.equal(first, second)
    reference = operator.manifest_reference()
    assert reference["interface_id"].startswith("passive-muon-scalable-mixed-precision")
    assert reference["matrix_shape"] == [2, 3]
    assert len(reference["source_sha256"]) == 64


def test_domain_operator_return_and_overflow_guards() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        FinitePrecisionOuterLoopConfig((0, 2))
    with pytest.raises(ValueError, match="positive integers"):
        FinitePrecisionOuterLoopConfig((True, 2))
    with pytest.raises(ValueError, match=r"2\^52"):
        FinitePrecisionOuterLoopConfig((1, 2**52 + 1))
    with pytest.raises(ValueError, match="requires backend"):
        FinitePrecisionOuterLoopConfig((2, 2), backend="native")

    config = FinitePrecisionOuterLoopConfig((2, 2))
    master = ThreeWordFP32Master.from_primary(torch.zeros((2, 2), dtype=torch.float32))
    good = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(TypeError, match="float32"):
        fp32_ema_nesterov(good.double(), good, config)
    with pytest.raises(ValueError, match="contiguous"):
        fp32_ema_nesterov(good.mT, good, config)
    nonfinite = good.clone()
    nonfinite[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        fp32_ema_nesterov(good, nonfinite, config)

    wrong_shape_operator = _RecordingAffineOperator((1, 4))
    with pytest.raises(ValueError, match="operator must have configured shape"):
        finite_precision_outer_step(master, good, good, wrong_shape_operator, config)

    class BadDtypeOperator(_RecordingAffineOperator):
        def __call__(self, signal: torch.Tensor) -> torch.Tensor:
            return signal.double()

    with pytest.raises(TypeError, match="float32"):
        finite_precision_outer_step(master, good, good, BadDtypeOperator((2, 2)), config)

    with pytest.raises(ValueError, match=r"master high exceeds.*2\^30"):
        ThreeWordFP32Master.from_primary(
            torch.full((2, 2), torch.finfo(torch.float32).max, dtype=torch.float32)
        )
    with pytest.raises(ValueError, match=r"master middle exceeds.*2\^7"):
        ThreeWordFP32Master(
            high=torch.zeros((2, 2), dtype=torch.float32),
            middle=torch.full((2, 2), 2.0**8, dtype=torch.float32),
            low=torch.zeros((2, 2), dtype=torch.float32),
        )
    with pytest.raises(ValueError, match=r"master low exceeds.*2\^-16"):
        ThreeWordFP32Master(
            high=torch.zeros((2, 2), dtype=torch.float32),
            middle=torch.zeros((2, 2), dtype=torch.float32),
            low=torch.full((2, 2), 2.0**-15, dtype=torch.float32),
        )
    with pytest.raises(ValueError, match=r"operator_output exceeds.*2\^16"):
        compensated_master_weight_update(
            master,
            torch.full((2, 2), 2.0**17, dtype=torch.float32),
            config,
        )


def test_manifest_locks_graph_ports_master_contract_scope_and_provenance() -> None:
    config = FinitePrecisionOuterLoopConfig((2, 3))
    operator = P9RepairedOperatorAdapter(ScalableMixedPrecisionConfig((2, 3)))
    manifest = finite_precision_outer_loop_manifest(config, operator)

    assert manifest["schema_version"] == FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION
    assert manifest["matrix_domain"]["shape"] == [2, 3]
    assert manifest["runtime_scalars"]["beta"]["source_exact"] == "19/20"
    assert manifest["runtime_scalars"]["beta"]["runtime_fp32_bits_hex"] == "0x3f733333"
    assert manifest["runtime_scalars"]["one_minus_beta"]["source_exact"] == "1/20"
    assert manifest["runtime_scalars"]["learning_rate"]["source_exact"] == "1/32000"
    assert {"r^m", "r^s", "placement", "frobenius_envelope"} == set(
        manifest["exact_residual_ports"]
    )
    assert "represented FP32 gradient sample" in manifest["ema_nesterov_graph"][0]
    assert manifest["matrix_domain"]["master_word_max_abs_inclusive"] == {
        "high": f"2^{LOCKED_MASTER_HIGH_MAX_ABS_POWER}",
        "middle": f"2^{LOCKED_MASTER_MIDDLE_MAX_ABS_POWER}",
        "low": f"2^{LOCKED_MASTER_LOW_MAX_ABS_POWER}",
    }
    assert manifest["matrix_domain"]["operator_output_max_abs_inclusive"] == (
        f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}"
    )
    assert (
        manifest["matrix_domain"]["certified_storage_profile_operator_output_max_abs_inclusive"]
        == f"2^{CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER}"
    )
    assert manifest["matrix_domain"]["rounded_step_max_abs_inclusive"] == (
        f"2^{LOCKED_ROUNDED_STEP_MAX_ABS_POWER}"
    )
    assert manifest["master_update"]["logical_value"] == "high+middle+low over the reals"
    assert "TwoSum" in " ".join(manifest["master_update"]["graph"])
    assert manifest["master_update"]["word_guard_audit"]["certified"] is True
    assert manifest["master_update"]["stalling_witness"]["certified"] is True
    assert manifest["operator_interface"]["matrix_shape"] == [2, 3]
    assert manifest["operator_interface"]["operator"]["epsilon"] == "none"
    assert manifest["operator_interface"]["operator"]["steps"] == 5
    assert manifest["ordering_context"]["pinned_upstream_revision"]
    assert len(manifest["ordering_context"]["pinned_muon_py_sha256"]) == 64
    assert manifest["precision_contract"]["fma_allowed"] is False
    assert manifest["precision_contract"]["ftz_daz_allowed"] is False
    exclusions = " ".join(manifest["excluded_claims"])
    assert "no stability theorem" in manifest["claim_scope"]
    assert "upstream" in exclusions
    assert "gradient-evaluation" in exclusions
    assert "model-forward" in exclusions
    assert "nonunique PL" in exclusions
    assert manifest["provenance"]["source_file"].endswith("finite_precision_outer_loop.py")
    assert len(manifest["provenance"]["source_sha256"]) == 64
    assert set(manifest["provenance"]["git"]) == {"sha", "branch", "dirty"}


def test_raw_update_helper_is_exactly_the_uncompensated_control() -> None:
    config = FinitePrecisionOuterLoopConfig((1, 1))
    position = torch.tensor([[2.0**25]], dtype=torch.float32)
    operator_output = torch.tensor([[2.0**14]], dtype=torch.float32)
    expected = torch.sub(position, torch.mul(_from_bits(LEARNING_RATE_FP32_BITS), operator_output))

    assert torch.equal(raw_fp32_parameter_update(position, operator_output, config), expected)
    assert Fraction(1, 32_000) == LOCKED_LEARNING_RATE
    assert LOCKED_OUTER_BACKEND.startswith("torch-cpu-eager-ieee-rne-fp32")
