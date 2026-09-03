from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.scalable_mixed_precision import (
    COEFFICIENT_A_FP32_BITS,
    COEFFICIENT_B_FP32_BITS,
    COEFFICIENT_C_FP32_BITS,
    LOCKED_BACKEND,
    LOCKED_JORDAN_STEPS,
    LOCKED_MAX_ENTRIES,
    LOCKED_SAFE_MAX_ABS_POWER,
    REPAIR_RHO_FP32_BITS,
    SCALABLE_MIXED_PRECISION_SCHEMA_VERSION,
    ScalableMixedPrecisionConfig,
    fp32_balanced_max_floor_normalize,
    scalable_mixed_precision_manifest,
    scalable_mixed_precision_repaired_operator,
    serial_normalizer_obstruction,
    two_term_bf16_jordan_five_stage,
)


def _from_bits(bits: int) -> torch.Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _literal_balanced_sum(values: torch.Tensor) -> torch.Tensor:
    level = values.reshape(-1).clone()
    while level.numel() > 1:
        next_level = []
        for index in range(0, level.numel() - 1, 2):
            next_level.append(torch.add(level[index], level[index + 1]))
        if level.numel() % 2:
            next_level.append(level[-1].clone())
        level = torch.stack(next_level)
    return level[0]


def _literal_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    entries = []
    for row in range(left.shape[0]):
        row_entries = []
        for column in range(right.shape[1]):
            products = torch.mul(left[row, :], right[:, column])
            row_entries.append(_literal_balanced_sum(products))
        entries.append(torch.stack(row_entries))
    return torch.stack(entries)


def _literal_normalize(signal: torch.Tensor) -> torch.Tensor:
    one = torch.tensor(1.0, dtype=torch.float32)
    sigma = torch.maximum(one, torch.amax(torch.abs(signal)))
    scaled = torch.div(signal, sigma)
    squares = torch.mul(scaled.reshape(-1), scaled.reshape(-1))
    norm = torch.sqrt(_literal_balanced_sum(squares))
    denominator = torch.maximum(torch.div(one, sigma), norm)
    return torch.div(scaled, denominator)


def _literal_encode(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    high = value.to(torch.bfloat16)
    residual = torch.sub(value, high.to(torch.float32))
    low = residual.to(torch.bfloat16)
    reconstructed = torch.add(high.to(torch.float32), low.to(torch.float32))
    return high, low, reconstructed


def _literal_stages(normalized: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    a = _from_bits(COEFFICIENT_A_FP32_BITS)
    b = _from_bits(COEFFICIENT_B_FP32_BITS)
    c = _from_bits(COEFFICIENT_C_FP32_BITS)
    high, low, state = _literal_encode(normalized)
    del high, low
    outputs = []
    for _ in range(LOCKED_JORDAN_STEPS):
        gram = _literal_mm(state, state.mT)
        shifted = torch.add(torch.mul(c, gram), torch.eye(state.shape[0]) * b)
        product = _literal_mm(shifted, gram)
        affine = torch.add(product, torch.eye(state.shape[0]) * a)
        body = _literal_mm(affine, state)
        high, low, state = _literal_encode(body)
        outputs.append((high, low))
    return tuple(outputs)


@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (3, 5)])
def test_scalable_kernel_matches_independent_literal_graph(shape: tuple[int, int]) -> None:
    config = ScalableMixedPrecisionConfig(shape)
    signal = torch.linspace(-2.0, 3.0, steps=shape[0] * shape[1], dtype=torch.float32).reshape(
        shape
    )
    expected_normalized = _literal_normalize(signal)
    actual_normalized = fp32_balanced_max_floor_normalize(signal, config)
    oriented = expected_normalized.mT.contiguous() if shape[0] > shape[1] else expected_normalized
    expected_states = _literal_stages(oriented)

    polynomial, actual_states = two_term_bf16_jordan_five_stage(oriented, return_stage_states=True)

    assert torch.equal(actual_normalized, expected_normalized)
    assert len(actual_states) == LOCKED_JORDAN_STEPS
    assert all(state.high.dtype == torch.bfloat16 for state in actual_states)
    assert all(state.low.dtype == torch.bfloat16 for state in actual_states)
    for actual, (expected_high, expected_low) in zip(actual_states, expected_states, strict=True):
        assert torch.equal(actual.high, expected_high)
        assert torch.equal(actual.low, expected_low)
    final_high, final_low = expected_states[-1]
    assert torch.equal(polynomial, torch.add(final_high.float(), final_low.float()))

    rho = _from_bits(REPAIR_RHO_FP32_BITS)
    restored = polynomial.mT.contiguous() if shape[0] > shape[1] else polynomial
    expected = torch.add(restored, torch.mul(signal, rho))
    assert torch.equal(scalable_mixed_precision_repaired_operator(signal, config), expected)


def test_scale_free_normalizer_handles_zero_subnormal_floor_and_large_input() -> None:
    config = ScalableMixedPrecisionConfig((2, 3))
    smallest = _from_bits(0x00000001)
    cases = (
        torch.zeros((2, 3), dtype=torch.float32),
        torch.tensor(
            [[float(smallest), -float(smallest), 0.0], [0.25, -0.5, 0.125]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.75, 0.0, 0.0], [0.0, 0.25, 0.0]], dtype=torch.float32),
        torch.tensor([[2.0**116, 0.0, 0.0], [0.0, -(2.0**116), 0.0]], dtype=torch.float32),
    )

    for signal in cases:
        actual = fp32_balanced_max_floor_normalize(signal, config)
        assert torch.equal(actual, _literal_normalize(signal))
        assert bool(torch.isfinite(actual).all())
        assert float(torch.linalg.vector_norm(actual)) <= 1.000001

    assert _bits(fp32_balanced_max_floor_normalize(cases[1], config)[0, 0]) == 0x00000001


def test_serial_normalizer_obstruction_is_exact_for_transformer_shape() -> None:
    report = serial_normalizer_obstruction((4_096, 11_008))
    saturation = torch.tensor(float(2**24), dtype=torch.float32)
    absorbed = torch.add(saturation, torch.tensor(1.0, dtype=torch.float32))

    assert _bits(saturation) == 0x4B800000
    assert _bits(absorbed) == 0x4B800000
    assert report["entry_count"] == 45_088_768
    assert report["unit_square_serial_sum"] == 2**24
    assert report["unit_square_exact_sum"] == 43 * 2**20
    assert report["returned_singular_value_squared"] == "43/16"
    assert report["leaves_five_quarters_spectral_tube"] is True
    assert report["standard_gamma_n_defined"] is False
    assert report["balanced_addition_depth"] == 26


def test_shape_dtype_layout_finiteness_magnitude_and_backend_guards() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        ScalableMixedPrecisionConfig((0, 2))
    with pytest.raises(ValueError, match="positive integers"):
        ScalableMixedPrecisionConfig((True, 2))
    with pytest.raises(ValueError, match=r"2\^52"):
        ScalableMixedPrecisionConfig((1, LOCKED_MAX_ENTRIES + 1))
    with pytest.raises(ValueError, match="requires backend"):
        ScalableMixedPrecisionConfig((2, 3), backend="native-bf16")

    config = ScalableMixedPrecisionConfig((2, 3))
    with pytest.raises(TypeError, match="float32"):
        scalable_mixed_precision_repaired_operator(torch.ones((2, 3), dtype=torch.float64), config)
    with pytest.raises(ValueError, match="configured shape"):
        scalable_mixed_precision_repaired_operator(torch.ones((3, 2)), config)
    with pytest.raises(ValueError, match="contiguous"):
        scalable_mixed_precision_repaired_operator(torch.ones((3, 2)).mT, config)
    nonfinite = torch.ones((2, 3), dtype=torch.float32)
    nonfinite[0, 0] = torch.inf
    with pytest.raises(ValueError, match="finite"):
        scalable_mixed_precision_repaired_operator(nonfinite, config)
    outside = torch.zeros((2, 3), dtype=torch.float32)
    outside[0, 0] = 2.0**117
    with pytest.raises(ValueError, match=r"2\^116"):
        scalable_mixed_precision_repaired_operator(outside, config)


def test_manifest_locks_shape_arithmetic_and_scope() -> None:
    config = ScalableMixedPrecisionConfig((11_008, 4_096))
    manifest = scalable_mixed_precision_manifest(config)

    assert manifest["schema_version"] == SCALABLE_MIXED_PRECISION_SCHEMA_VERSION
    assert manifest["matrix_domain"]["shape"] == [11_008, 4_096]
    assert manifest["matrix_domain"]["oriented_shape"] == [4_096, 11_008]
    assert manifest["matrix_domain"]["max_entries_inclusive"] == (f"2^52 ({LOCKED_MAX_ENTRIES})")
    assert manifest["matrix_domain"]["max_abs_inclusive"] == (f"2^{LOCKED_SAFE_MAX_ABS_POWER}")
    assert manifest["operator"]["epsilon"] == "none"
    assert manifest["operator"]["steps"] == 5
    assert manifest["backend"]["name"] == LOCKED_BACKEND
    assert manifest["precision_contract"]["torch_matmul_used"] is False
    assert manifest["precision_contract"]["stored_stage_boundary"].startswith("two BF16")
    exclusions = " ".join(manifest["excluded_claims"])
    assert "upstream Muon" in exclusions
    assert "production performance" in exclusions
    assert "state compression" in exclusions
    assert "FP32 EMA/Nesterov" in exclusions
    assert len(manifest["provenance"]["source_sha256"]) == 64


def test_transpose_orientations_are_numerically_consistent() -> None:
    signal = torch.tensor([[1.25, -0.5, 2.0], [-1.0, 0.75, 0.125]], dtype=torch.float32)
    wide = scalable_mixed_precision_repaired_operator(
        signal, ScalableMixedPrecisionConfig(tuple(signal.shape))
    )
    tall = scalable_mixed_precision_repaired_operator(
        signal.mT.contiguous(), ScalableMixedPrecisionConfig(tuple(signal.mT.shape))
    )

    assert torch.allclose(wide, tall.mT, rtol=2e-6, atol=2e-6)
    assert Fraction.from_float(float(_from_bits(REPAIR_RHO_FP32_BITS))) == Fraction(
        13_231_137, 16_384
    )
