from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.mixed_precision import (
    COEFFICIENT_A_FP32_BITS,
    COEFFICIENT_B_FP32_BITS,
    COEFFICIENT_C_FP32_BITS,
    LOCKED_BACKEND,
    LOCKED_JORDAN_STEPS,
    LOCKED_MATRIX_SHAPE,
    LOCKED_SAFE_MAX_ABS_POWER,
    MIXED_PRECISION_SCHEMA_VERSION,
    REPAIR_RHO_FP32_BITS,
    MixedPrecisionConfig,
    bf16_jordan_five_stage,
    fp32_max_floor_normalize,
    locked_backend_self_check,
    mixed_precision_ema_nesterov_step,
    mixed_precision_manifest,
    mixed_precision_repaired_operator,
)
from passive_muon.structure_aware_stability import LOCKED_LEARNING_RATE, LOCKED_REPAIR_RHO


def _from_bits(bits: int) -> torch.Tensor:
    """Test-side bit constructor, independent of the implementation helper."""

    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _matrix(values: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.stack((torch.stack(values[:2]), torch.stack(values[2:])))


def _literal_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    entries = []
    for row in range(2):
        for column in range(2):
            first = torch.mul(left[row, 0], right[0, column])
            second = torch.mul(left[row, 1], right[1, column])
            entries.append(torch.add(first, second))
    return _matrix(tuple(entries))  # type: ignore[arg-type]


def _literal_normalize(signal: torch.Tensor) -> torch.Tensor:
    entries = (signal[0, 0], signal[0, 1], signal[1, 0], signal[1, 1])
    maximum = torch.abs(entries[0])
    for entry in entries[1:]:
        maximum = torch.maximum(maximum, torch.abs(entry))
    if bool(maximum <= torch.tensor(0.5, dtype=torch.float32)):
        return signal.clone()

    scaled = tuple(torch.div(entry, maximum) for entry in entries)
    squares = tuple(torch.mul(entry, entry) for entry in scaled)
    total = squares[0]
    for square in squares[1:]:
        total = torch.add(total, square)
    norm = torch.mul(maximum, torch.sqrt(total))
    denominator = torch.maximum(torch.tensor(1.0, dtype=torch.float32), norm)
    return _matrix(tuple(torch.div(entry, denominator) for entry in entries))


def _literal_stages(normalized: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Independent literal replay of all locked stage-boundary bits."""

    a = _from_bits(0x405C72B0)
    b = _from_bits(0xC098CCCD)
    c = _from_bits(0x40020419)
    zero = torch.tensor(0.0, dtype=torch.float32)
    state = normalized.to(torch.bfloat16)
    outputs = []
    for _ in range(5):
        x = state.to(torch.float32)
        gram = _literal_mm(x, x.mT)
        t_entries = []
        for row in range(2):
            for column in range(2):
                t_entries.append(
                    torch.add(b if row == column else zero, torch.mul(c, gram[row, column]))
                )
        d_matrix = _literal_mm(_matrix(tuple(t_entries)), gram)  # type: ignore[arg-type]
        e_entries = []
        for row in range(2):
            for column in range(2):
                e_entries.append(torch.add(a if row == column else zero, d_matrix[row, column]))
        state = _literal_mm(_matrix(tuple(e_entries)), x).to(torch.bfloat16)  # type: ignore[arg-type]
        outputs.append(state.clone())
    return tuple(outputs)


@pytest.mark.parametrize(
    "signal",
    [
        torch.tensor([[0.125, -0.25], [0.5, 0.0625]], dtype=torch.float32),
        torch.tensor([[3.0, -4.0], [0.5, 2.0]], dtype=torch.float32),
        torch.tensor([[0.734375, -0.15625], [0.28125, 0.609375]], dtype=torch.float32),
    ],
)
def test_locked_kernel_matches_independent_literal_serial_reference(signal: torch.Tensor) -> None:
    config = MixedPrecisionConfig()
    expected_normalized = _literal_normalize(signal)
    normalized = fp32_max_floor_normalize(signal, config)
    output, stages = bf16_jordan_five_stage(normalized, config, return_stage_outputs=True)
    expected_stages = _literal_stages(expected_normalized)

    assert torch.equal(normalized, expected_normalized)
    assert len(stages) == LOCKED_JORDAN_STEPS
    assert all(stage.dtype == torch.bfloat16 for stage in stages)
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(stages, expected_stages, strict=True)
    )
    assert output.dtype == torch.float32
    assert torch.equal(output, expected_stages[-1].float())


def test_subnormal_and_half_threshold_take_exact_identity_branch() -> None:
    config = MixedPrecisionConfig()
    smallest_subnormal = _from_bits(0x00000001)
    signal = torch.tensor(
        [[float(smallest_subnormal), -float(smallest_subnormal)], [0.5, -0.25]],
        dtype=torch.float32,
    )

    normalized = fp32_max_floor_normalize(signal, config)

    assert torch.equal(normalized, signal)
    assert _bits(normalized[0, 0]) == 0x00000001
    assert _bits(normalized[0, 1]) == 0x80000001


def test_scaled_branch_can_still_return_identity_below_frobenius_floor() -> None:
    config = MixedPrecisionConfig()
    signal = torch.tensor([[0.75, 0.0], [0.0, 0.25]], dtype=torch.float32)

    assert torch.equal(fp32_max_floor_normalize(signal, config), signal)


def test_three_four_witness_has_locked_serial_normalization() -> None:
    config = MixedPrecisionConfig()
    signal = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float32))
    expected = torch.diag(torch.tensor([0.6, 0.8], dtype=torch.float32))

    actual = fp32_max_floor_normalize(signal, config)

    assert torch.equal(actual, expected)
    assert torch.equal(actual, _literal_normalize(signal))


def test_all_runtime_coefficient_bits_are_locked_fp32_dyadics() -> None:
    assert COEFFICIENT_A_FP32_BITS == 0x405C72B0
    assert COEFFICIENT_B_FP32_BITS == 0xC098CCCD
    assert COEFFICIENT_C_FP32_BITS == 0x40020419
    assert REPAIR_RHO_FP32_BITS == 0x4449E421
    assert Fraction.from_float(float(_from_bits(COEFFICIENT_A_FP32_BITS))) == Fraction(
        902_955, 262_144
    )
    assert Fraction.from_float(float(_from_bits(COEFFICIENT_B_FP32_BITS))) == Fraction(
        -10_013_901, 2_097_152
    )
    assert Fraction.from_float(float(_from_bits(COEFFICIENT_C_FP32_BITS))) == Fraction(
        8_520_729, 4_194_304
    )


def test_repair_uses_unnormalized_signal_and_two_final_fp32_operations() -> None:
    config = MixedPrecisionConfig()
    signal = torch.tensor([[2.0, -1.0], [0.5, 3.0]], dtype=torch.float32)
    normalized = fp32_max_floor_normalize(signal, config)
    polynomial = bf16_jordan_five_stage(normalized, config)
    assert isinstance(polynomial, torch.Tensor)
    repair = torch.mul(signal, _from_bits(0x4449E421))
    expected = torch.add(polynomial, repair)

    actual = mixed_precision_repaired_operator(signal, config)

    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


def test_kernel_is_bitwise_deterministic() -> None:
    config = MixedPrecisionConfig()
    signal = torch.tensor([[1.25, -2.5], [3.75, 0.125]], dtype=torch.float32)

    outputs = [mixed_precision_repaired_operator(signal, config) for _ in range(4)]

    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_certified_shape_backend_dtype_layout_and_finiteness_are_enforced() -> None:
    with pytest.raises(ValueError, match="exactly"):
        MixedPrecisionConfig(matrix_shape=(2, 3))
    with pytest.raises(ValueError, match="exactly"):
        MixedPrecisionConfig(matrix_shape=(1, 1))
    with pytest.raises(ValueError, match="requires backend"):
        MixedPrecisionConfig(backend="native-bf16")

    config = MixedPrecisionConfig()
    with pytest.raises(TypeError, match="float32"):
        mixed_precision_repaired_operator(torch.eye(2, dtype=torch.float64), config)
    with pytest.raises(ValueError, match="certified shape"):
        mixed_precision_repaired_operator(torch.ones((2, 3), dtype=torch.float32), config)
    with pytest.raises(ValueError, match="contiguous"):
        mixed_precision_repaired_operator(torch.ones((2, 2), dtype=torch.float32).mT, config)
    nonfinite = torch.eye(2, dtype=torch.float32)
    nonfinite[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        mixed_precision_repaired_operator(nonfinite, config)


def test_inclusive_safe_range_runs_and_larger_finite_input_is_rejected() -> None:
    config = MixedPrecisionConfig()
    at_bound = torch.tensor([[2.0**116, 0.0], [0.0, -(2.0**116)]], dtype=torch.float32)
    outside = torch.tensor([[2.0**117, 0.0], [0.0, 0.0]], dtype=torch.float32)

    assert bool(torch.isfinite(mixed_precision_repaired_operator(at_bound, config)).all())
    with pytest.raises(ValueError, match=r"2\^116"):
        mixed_precision_repaired_operator(outside, config)


def test_backend_self_check_records_rne_and_gradual_underflow() -> None:
    report = locked_backend_self_check()

    assert report["rounding"] == "IEEE-754 roundTiesToEven"
    assert report["fp32_add_halfway_bits_hex"] == "0x3f800000"
    assert report["fp32_min_subnormal_bits_hex"] == "0x00000001"
    assert report["fp32_subnormal_input_add_bits_hex"] == "0x00800001"
    assert report["bf16_cast_halfway_bits_hex"] == "0x3f80"
    assert report["bf16_min_subnormal_bits_hex"] == "0x0001"
    assert report["bf16_min_subnormal_to_fp32_bits_hex"] == "0x00010000"
    assert report["gradual_underflow"] is True
    assert report["ftz_daz"] is False


def test_manifest_locks_contract_and_excludes_upstream_and_full_fp32_loop() -> None:
    manifest = mixed_precision_manifest(MixedPrecisionConfig())

    assert manifest["schema_version"] == MIXED_PRECISION_SCHEMA_VERSION
    assert manifest["matrix_domain"]["shape"] == list(LOCKED_MATRIX_SHAPE)
    assert manifest["matrix_domain"]["max_abs_inclusive"] == (f"2^{LOCKED_SAFE_MAX_ABS_POWER}")
    assert manifest["operator"]["floor_exact"] == "1"
    assert manifest["operator"]["epsilon"] == "none"
    assert manifest["operator"]["steps"] == 5
    assert manifest["operator"]["repair_rho_exact"] == str(LOCKED_REPAIR_RHO)
    assert manifest["backend"]["name"] == LOCKED_BACKEND
    assert manifest["backend"]["torch_matmul_used"] is False
    assert "two-term dot" in manifest["precision_contract"]["stage_dot_product"]
    assert manifest["precision_contract"]["gradual_underflow_required"] is True
    assert manifest["precision_contract"]["ftz_daz_allowed"] is False
    assert manifest["outer_loop_prototype"]["status"] == (
        "executable only; outside the P7/P8 theorem"
    )
    exclusions = " ".join(manifest["excluded_claims"])
    assert "upstream Muon" in exclusions
    assert "end-to-end P7 stability" in exclusions
    assert "arbitrary matrix shapes" in exclusions
    assert len(manifest["provenance"]["source_sha256"]) == 64
    assert {
        name: record["fp32_bits_hex"]
        for name, record in manifest["operator"]["coefficients"].items()
    } == {"a": "0x405c72b0", "b": "0xc098cccd", "c": "0x40020419"}


def test_fp32_outer_loop_helper_is_deterministic_but_only_a_prototype() -> None:
    config = MixedPrecisionConfig()
    position = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    momentum = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
    gradient = torch.tensor([[-1.0, 0.5], [2.0, -0.25]], dtype=torch.float32)

    first = mixed_precision_ema_nesterov_step(position, momentum, gradient, config)
    second = mixed_precision_ema_nesterov_step(position, momentum, gradient, config)

    assert all(
        value.dtype == torch.float32
        for value in (first.position, first.momentum, first.signal, first.operator_output)
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            (first.position, first.momentum, first.signal, first.operator_output),
            (second.position, second.momentum, second.signal, second.operator_output),
            strict=True,
        )
    )


def test_fp32_parameter_update_can_stall_in_flat_direction_counterexample() -> None:
    """Document why the exact-real P7 shell cannot silently become FP32."""

    config = MixedPrecisionConfig()
    position = torch.tensor([[2.0**30, 0.0], [0.0, 0.0]], dtype=torch.float32)
    signal = torch.tensor([[64.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    output = mixed_precision_repaired_operator(signal, config)
    decrement = torch.mul(torch.tensor(float(LOCKED_LEARNING_RATE), dtype=torch.float32), output)
    updated = torch.sub(position, decrement)

    assert 0.0 < float(decrement[0, 0]) < 32.0
    assert _bits(updated[0, 0]) == _bits(position[0, 0])
    assert float(updated[0, 0]) == 2.0**30
