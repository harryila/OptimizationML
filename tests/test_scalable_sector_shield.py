from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
import torch

from passive_muon.scalable_sector_shield import (
    ACCEPTANCE_RADIUS_FP32_BITS,
    CLIP_COEFFICIENT_FP32_BITS,
    FALLBACK_GAIN_FP32_BITS,
    FP32_MIN_NORMAL,
    LOCKED_ACCEPTANCE_RADIUS,
    LOCKED_CLIP_COEFFICIENT,
    LOCKED_FALLBACK_GAIN,
    LOCKED_REDUCTION_BLOCK_SIZE,
    LOCKED_REPRESENTATIVE_SHAPES,
    LOCKED_RESERVED_INWARD_MARGIN,
    LOCKED_SCALABLE_SHIELD_BACKEND,
    LOCKED_SHIELD_CENTER,
    LOCKED_SHIELD_RADIUS,
    SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION,
    SHIELD_CENTER_FP32_BITS,
    SHIELD_RADIUS_FP32_BITS,
    NearZeroUnrepresentable,
    ScalableSectorShieldConfig,
    ScalableSectorShieldFailure,
    ShieldAction,
    SignalGuardClass,
    _all_entries_halved_exactly_fp32,
    _balanced_sum_fp32,
    _balanced_sum_one_block,
    locked_scalable_shield_backend_self_check,
    scalable_sector_shield_manifest,
    shield_reduced_spectrum_mixed_precision,
    shield_sector_candidate_mixed_precision,
)
from passive_muon.sector_projected_resolvent import (
    evaluate_sector_projected_singular_values_fp64,
)


def _from_bits(bits: int) -> torch.Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32).view(torch.float32)[0]


def _bits(value: torch.Tensor) -> int:
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _exact_disk_margin(output: torch.Tensor, signal: torch.Tensor) -> Fraction:
    exact_output = tuple(Fraction.from_float(float(value)) for value in output.float().ravel())
    exact_signal = tuple(Fraction.from_float(float(value)) for value in signal.float().ravel())
    center = Fraction(1_143, 2_048)
    radius = Fraction(893, 2_048)
    return radius**2 * sum(value**2 for value in exact_signal) - sum(
        (value - center * source) ** 2
        for value, source in zip(exact_output, exact_signal, strict=True)
    )


def test_constants_bits_and_backend_are_locked() -> None:
    report = locked_scalable_shield_backend_self_check()

    assert LOCKED_SHIELD_CENTER == 1_143 / 2_048
    assert LOCKED_SHIELD_RADIUS == 893 / 2_048
    assert LOCKED_ACCEPTANCE_RADIUS == 891 / 2_048
    assert LOCKED_CLIP_COEFFICIENT == 890 / 2_048
    assert LOCKED_RESERVED_INWARD_MARGIN == 1 / 1_024
    assert LOCKED_FALLBACK_GAIN == 0.5
    assert SHIELD_CENTER_FP32_BITS == 0x3F0EE000
    assert SHIELD_RADIUS_FP32_BITS == 0x3EDF4000
    assert ACCEPTANCE_RADIUS_FP32_BITS == 0x3EDEC000
    assert CLIP_COEFFICIENT_FP32_BITS == 0x3EDE8000
    assert FALLBACK_GAIN_FP32_BITS == 0x3F000000
    assert _bits(torch.tensor(LOCKED_SHIELD_CENTER, dtype=torch.float32)) == SHIELD_CENTER_FP32_BITS
    assert _bits(torch.tensor(LOCKED_SHIELD_RADIUS, dtype=torch.float32)) == SHIELD_RADIUS_FP32_BITS
    assert report["rounding"] == "IEEE-754 roundTiesToEven"
    assert report["gradual_underflow"] is True
    assert report["ftz_daz"] is False
    assert report["fp64_to_fp32_halfway_bits_hex"] == "0x3f800000"
    assert report["fp32_one_nextafter_zero_bits_hex"] == "0x3f7fffff"


def test_all_seven_transformer_shapes_construct_with_positive_static_margin() -> None:
    assert LOCKED_REPRESENTATIVE_SHAPES == (
        (768, 768),
        (768, 3_072),
        (768, 50_257),
        (3_072, 12_288),
        (4_096, 4_096),
        (4_096, 11_008),
        (4_096, 14_336),
    )
    for shape in LOCKED_REPRESENTATIVE_SHAPES:
        config = ScalableSectorShieldConfig(shape)
        assert config.certified_inward_margin > 0.0
        assert config.certified_inward_radius < LOCKED_SHIELD_RADIUS
        assert config.entry_count == math.prod(shape)


def test_fp32_inside_candidate_passes_through_bit_for_bit() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([3.0, -4.0, 0.5, -0.25], dtype=torch.float32)
    candidate = torch.mul(signal, torch.tensor(0.5, dtype=torch.float32))
    bits = candidate.view(torch.int32).clone()

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.candidate_accepted
    assert not result.diagnostics.active
    assert not result.diagnostics.used_fallback
    assert not result.diagnostics.candidate_clipped
    assert result.diagnostics.action is ShieldAction.PASS_THROUGH
    assert not result.diagnostics.fail_closed
    assert result.diagnostics.signal_guard_class is SignalGuardClass.NORMAL_ANCHORED
    assert result.output.dtype == torch.float32
    assert torch.equal(result.output.view(torch.int32), bits)
    assert _exact_disk_margin(result.output, signal) > 0


@pytest.mark.parametrize("signal_dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("candidate_dtype", (torch.float32, torch.bfloat16))
def test_fp32_and_bf16_inputs_widen_once_and_return_fp32(
    signal_dtype: torch.dtype,
    candidate_dtype: torch.dtype,
) -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([3.0, 4.0, -2.0, 1.0], dtype=signal_dtype)
    candidate = torch.tensor([1.5, 2.0, -1.0, 0.5], dtype=candidate_dtype)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.candidate_accepted
    assert result.diagnostics.signal_input_dtype == str(signal_dtype)
    assert result.diagnostics.candidate_input_dtype == str(candidate_dtype)
    assert result.diagnostics.widened_dtype == "torch.float32"
    assert result.diagnostics.output_dtype == "torch.float32"
    assert result.output.dtype == torch.float32
    assert torch.equal(result.output, candidate.float())
    assert _exact_disk_margin(result.output, signal.float()) >= 0


def test_guarded_p18_candidate_is_normally_inactive() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([3.0, 4.0], dtype=torch.float32)
    evaluated = evaluate_sector_projected_singular_values_fp64(
        np.asarray([3.0, 4.0], dtype=np.float64)
    )
    candidate = torch.from_numpy(evaluated.output.astype(np.float32))

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.reduced_spectrum_diagnostic
    assert result.diagnostics.candidate_accepted
    assert not result.diagnostics.active
    assert torch.equal(result.output, candidate)
    assert _exact_disk_margin(result.output, signal) > 0


@pytest.mark.parametrize(
    "candidate",
    (
        torch.tensor([100.0, -200.0], dtype=torch.float32),
        torch.tensor([-3.0e30, 2.0e30], dtype=torch.float32),
    ),
)
def test_finite_corrupted_candidates_are_radially_clipped_and_exactly_safe(
    candidate: torch.Tensor,
) -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.active
    assert result.diagnostics.candidate_clipped
    assert not result.diagnostics.used_fallback
    assert not result.diagnostics.candidate_accepted
    assert not result.diagnostics.fail_closed
    assert result.diagnostics.action is ShieldAction.RADIAL_CLIP
    assert result.diagnostics.reason == "candidate_outside_inward_screen_radial_clip"
    assert result.diagnostics.clip_ratio_fp64_downward is not None
    assert result.diagnostics.clip_scale_fp64_downward is not None
    assert result.diagnostics.clip_alpha_fp32 is not None
    assert result.diagnostics.clip_alpha_fp32_bits is not None
    assert result.diagnostics.signal_norm is not None
    assert result.diagnostics.displacement_norm is not None
    assert result.diagnostics.signal_norm.returned_norm is not None
    assert result.diagnostics.displacement_norm.returned_norm is not None
    expected_ratio = math.nextafter(
        result.diagnostics.signal_norm.returned_norm
        / result.diagnostics.displacement_norm.returned_norm,
        0.0,
    )
    expected_scale = math.nextafter(LOCKED_CLIP_COEFFICIENT * expected_ratio, 0.0)
    assert result.diagnostics.clip_ratio_fp64_downward == expected_ratio
    assert result.diagnostics.clip_scale_fp64_downward == expected_scale
    rne_alpha = torch.tensor(
        result.diagnostics.clip_scale_fp64_downward,
        dtype=torch.float64,
    ).to(torch.float32)
    expected_alpha = (
        torch.nextafter(rne_alpha, torch.zeros((), dtype=torch.float32))
        if bool(rne_alpha != 0.0)
        else rne_alpha
    )
    assert result.diagnostics.clip_alpha_fp32_bits == _bits(expected_alpha)
    assert result.diagnostics.clip_alpha_fp32 <= result.diagnostics.clip_scale_fp64_downward
    assert not torch.equal(result.output, torch.tensor([1.5, 2.0], dtype=torch.float32))
    assert _exact_disk_margin(result.output, signal) > 0


@pytest.mark.parametrize(
    "candidate",
    (
        torch.tensor([math.nan, 1.0], dtype=torch.float32),
        torch.tensor([math.inf, -math.inf], dtype=torch.float32),
    ),
)
def test_nonfinite_candidates_use_safe_half_fallback(candidate: torch.Tensor) -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([3.0, 4.0], dtype=torch.float32)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.active
    assert not result.diagnostics.candidate_clipped
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fail_closed
    assert result.diagnostics.action is ShieldAction.HALF_FALLBACK
    assert result.diagnostics.reason == "nonfinite_candidate_half_fallback"
    assert torch.equal(result.output, torch.tensor([1.5, 2.0], dtype=torch.float32))
    assert _exact_disk_margin(result.output, signal) > 0


def test_displacement_overflow_falls_back_without_emitting_nonfinite_output() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    maximum = torch.finfo(torch.float32).max
    signal = torch.tensor([-maximum], dtype=torch.float32)
    candidate = torch.tensor([maximum], dtype=torch.float32)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.active
    assert not result.diagnostics.candidate_clipped
    assert result.diagnostics.used_fallback
    assert result.diagnostics.fail_closed
    assert result.diagnostics.action is ShieldAction.HALF_FALLBACK
    assert result.diagnostics.reason == "displacement_overflow_half_fallback"
    assert bool(torch.isfinite(result.output).all())
    assert _exact_disk_margin(result.output, signal) > 0


def test_zero_signal_disk_is_singleton_for_finite_and_nonfinite_candidates() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    zero = torch.zeros(2, dtype=torch.float32)
    accepted = shield_reduced_spectrum_mixed_precision(zero, zero.clone(), config)
    finite = shield_reduced_spectrum_mixed_precision(
        zero, torch.tensor([7.0, -9.0], dtype=torch.float32), config
    )
    nonfinite = shield_reduced_spectrum_mixed_precision(
        zero, torch.tensor([math.nan, math.inf], dtype=torch.float32), config
    )

    assert accepted.diagnostics.candidate_accepted
    assert not accepted.diagnostics.active
    assert finite.diagnostics.active
    assert nonfinite.diagnostics.active
    assert nonfinite.diagnostics.fail_closed
    assert torch.equal(accepted.output, zero)
    assert torch.equal(finite.output, zero)
    assert torch.equal(nonfinite.output, zero)


def test_all_subnormal_signal_uses_only_exact_half_or_fails_closed() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    even = _from_bits(0x0000_0002).reshape(1)
    odd = _from_bits(0x0000_0003).reshape(1)
    candidate = torch.zeros(1, dtype=torch.float32)

    result = shield_reduced_spectrum_mixed_precision(even, candidate, config)

    assert result.diagnostics.signal_guard_class is SignalGuardClass.SUBNORMAL_EXACT_HALVING
    assert result.diagnostics.exact_halving_guard
    assert result.diagnostics.action is ShieldAction.HALF_FALLBACK
    assert _bits(result.output) == 0x0000_0001
    assert _exact_disk_margin(result.output, even) > 0
    with pytest.raises(NearZeroUnrepresentable, match="all nonzero signal entries") as raised:
        shield_reduced_spectrum_mixed_precision(odd, candidate, config)
    assert raised.value.matrix_shape == (768, 768)
    assert raised.value.signal_max_abs < FP32_MIN_NORMAL
    assert raised.value.signal_frobenius_strict_upper == config.near_zero_frobenius_upper


def test_exact_halving_bit_guard_is_blocked_and_checks_the_final_partial_block() -> None:
    even = _from_bits(0x0000_0002)
    odd = _from_bits(0x0000_0001)
    signal = torch.full(
        (LOCKED_REDUCTION_BLOCK_SIZE + 1,),
        even,
        dtype=torch.float32,
    )

    assert _all_entries_halved_exactly_fp32(signal)
    signal[-1] = odd
    assert not _all_entries_halved_exactly_fp32(signal)


def test_widened_bf16_subnormal_has_an_exact_fp32_half() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.tensor([2.0**-133], dtype=torch.float32).to(torch.bfloat16)
    candidate = torch.zeros(1, dtype=torch.bfloat16)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.signal_guard_class is SignalGuardClass.SUBNORMAL_EXACT_HALVING
    assert result.diagnostics.exact_halving_guard
    assert _bits(result.output) == 0x0000_8000
    assert _exact_disk_margin(result.output, signal.float()) > 0


def test_normal_anchor_absorbs_rounded_halves_of_odd_subnormal_entries() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    maximum = torch.finfo(torch.float32).max
    signal = torch.stack((torch.tensor(-maximum), _from_bits(0x0000_0001)))
    candidate = torch.tensor([maximum, 0.0], dtype=torch.float32)

    result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.normal_anchor
    assert not result.diagnostics.exact_halving_guard
    assert result.diagnostics.used_fallback
    assert result.diagnostics.action is ShieldAction.HALF_FALLBACK
    assert _bits(result.output[0]) == 0xFEFF_FFFF
    assert _bits(result.output[1]) == 0x0000_0000
    assert _exact_disk_margin(result.output, signal) > 0


def test_original_boundary_and_one_ulp_escape_are_rejected_by_inward_screen() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    signal = torch.ones(1, dtype=torch.float32)
    boundary = torch.tensor([509.0 / 512.0], dtype=torch.float32)
    escaped = torch.nextafter(boundary, torch.tensor([math.inf], dtype=torch.float32))

    boundary_result = shield_reduced_spectrum_mixed_precision(signal, boundary, config)
    escaped_result = shield_reduced_spectrum_mixed_precision(signal, escaped, config)

    assert _exact_disk_margin(boundary, signal) == 0
    assert _exact_disk_margin(escaped, signal) < 0
    assert boundary_result.diagnostics.active
    assert escaped_result.diagnostics.active
    assert boundary_result.diagnostics.candidate_clipped
    assert escaped_result.diagnostics.candidate_clipped
    assert not boundary_result.diagnostics.used_fallback
    assert not escaped_result.diagnostics.used_fallback
    assert boundary_result.diagnostics.action is ShieldAction.RADIAL_CLIP
    assert escaped_result.diagnostics.action is ShieldAction.RADIAL_CLIP
    assert float(boundary_result.output) < float(boundary)
    assert float(escaped_result.output) < float(escaped)
    assert _exact_disk_margin(boundary_result.output, signal) > 0
    assert _exact_disk_margin(escaped_result.output, signal) > 0


def test_seeded_finite_adversarial_clips_all_pass_offline_exact_disk_check() -> None:
    config = ScalableSectorShieldConfig((768, 768))
    generator = np.random.default_rng(20_260_905)
    clipped = 0
    for exponent in (-100, -40, 0, 40, 100):
        for gain in (2.0, 17.0, 1_024.0):
            signal = torch.from_numpy(
                np.asarray(generator.normal(size=9) * 2.0**exponent, dtype=np.float32)
            )
            candidate = torch.from_numpy(
                np.asarray(
                    generator.normal(size=9) * 2.0**exponent * gain,
                    dtype=np.float32,
                )
            )
            result = shield_reduced_spectrum_mixed_precision(signal, candidate, config)
            assert _exact_disk_margin(result.output, signal) >= 0
            assert not result.diagnostics.used_fallback
            clipped += int(result.diagnostics.candidate_clipped)
    assert clipped >= 12


def test_chunked_balanced_tree_matches_global_adjacent_tree_bitwise() -> None:
    length = LOCKED_REDUCTION_BLOCK_SIZE + 5
    values = torch.zeros(length, dtype=torch.float32)
    values[0] = 1.0
    values[LOCKED_REDUCTION_BLOCK_SIZE - 1] = 2.0**-20
    values[LOCKED_REDUCTION_BLOCK_SIZE] = 2.0**-21
    values[-1] = 0.25

    chunked = _balanced_sum_fp32(values)
    global_tree = _balanced_sum_one_block(values)

    assert _bits(chunked) == _bits(global_tree)


def test_manifest_freezes_cast_reduction_ftz_and_scope() -> None:
    config = ScalableSectorShieldConfig((4_096, 11_008))
    manifest = scalable_sector_shield_manifest(config)

    assert manifest["schema_version"] == SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION
    assert manifest["matrix_domain"]["shape"] == [4_096, 11_008]
    assert manifest["matrix_domain"]["input_dtypes"] == [
        "torch.float32",
        "torch.bfloat16",
    ]
    assert manifest["matrix_domain"]["output_dtype"] == "torch.float32"
    assert manifest["disk"]["acceptance_radius"] == "891/2048"
    assert manifest["disk"]["clip_coefficient"] == "890/2048 = 445/1024"
    assert "ratio64=nextafter" in manifest["operation_graph"]["radial_clip"]
    assert manifest["operation_graph"]["fma_allowed"] is False
    assert manifest["underflow_contract"]["ftz_daz_allowed"] is False
    assert manifest["backend"]["per_output_fraction_or_big_integer_postcheck"] is False
    assert manifest["backend"]["runtime_uses_generated_shape_table"] is True
    assert manifest["backend"]["name"] == LOCKED_SCALABLE_SHIELD_BACKEND
    exclusions = " ".join(manifest["excluded_claims"])
    assert "GPU" in exclusions
    assert "throughput" in exclusions
    assert "unmodified upstream Muon" in exclusions
    assert len(manifest["provenance"]["source_sha256"]) == 64


def test_input_validation_and_near_zero_bound() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        ScalableSectorShieldConfig((0, 2))
    with pytest.raises(ValueError, match="positive integers"):
        ScalableSectorShieldConfig((True, 2))
    with pytest.raises(ValueError, match="requires backend"):
        ScalableSectorShieldConfig((768, 768), backend="native-cuda")
    with pytest.raises(ValueError, match="generated P20 certified shape table"):
        ScalableSectorShieldConfig((3, 3))

    config = ScalableSectorShieldConfig((768, 768))
    assert config.near_zero_frobenius_upper == 768.0 * FP32_MIN_NORMAL
    signal = torch.ones(4, dtype=torch.float32)
    candidate = torch.ones(4, dtype=torch.float32)
    with pytest.raises(TypeError, match=r"float32 or torch\.bfloat16"):
        shield_reduced_spectrum_mixed_precision(signal.double(), candidate, config)
    with pytest.raises(ValueError, match="shape"):
        shield_sector_candidate_mixed_precision(signal, candidate, config)
    noncontiguous = torch.ones(8, dtype=torch.float32)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        shield_reduced_spectrum_mixed_precision(noncontiguous, candidate, config)
    too_many = torch.ones(config.entry_count + 1, dtype=torch.float32)
    with pytest.raises(ValueError, match="may not exceed"):
        shield_reduced_spectrum_mixed_precision(too_many, too_many, config)
    bad_signal = signal.clone()
    bad_signal[0] = math.nan
    with pytest.raises(ScalableSectorShieldFailure, match="nonfinite signal"):
        shield_reduced_spectrum_mixed_precision(bad_signal, candidate, config)


def test_full_matrix_entrypoint_enforces_shape_and_handles_zero_without_allocation_tricks() -> None:
    config = ScalableSectorShieldConfig((2, 2))
    signal = torch.zeros(config.matrix_shape, dtype=torch.float32)
    candidate = torch.zeros(config.matrix_shape, dtype=torch.bfloat16)

    result = shield_sector_candidate_mixed_precision(signal, candidate, config)

    assert result.diagnostics.signal_guard_class is SignalGuardClass.ZERO
    assert result.diagnostics.candidate_accepted
    assert not result.diagnostics.reduced_spectrum_diagnostic
    assert result.output.shape == config.matrix_shape
    assert result.output.dtype == torch.float32
    assert not bool(torch.count_nonzero(result.output))
