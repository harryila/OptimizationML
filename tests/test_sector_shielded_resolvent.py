from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest
from sympy import Rational, simplify

from passive_muon.sector_shielded_resolvent import (
    LOCKED_FP64_FALLBACK_GAIN,
    LOCKED_FP64_INWARD_ULPS,
    LOCKED_SHIELD_CENTER,
    LOCKED_SHIELD_RADIUS,
    SectorShieldFailure,
    evaluate_sector_shielded_resolvent_fp64,
    evaluate_sector_shielded_singular_values_fp64,
    fp64_output_in_exact_sector,
    fp64_sector_disk_margin_exact,
    project_onto_sector_disk_exact,
    shield_sector_candidate_fp64,
)


def _fractions(*values: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in values)


def test_locked_shield_is_exactly_the_p18_sector_disk() -> None:
    assert Fraction(1_143, 2_048) == LOCKED_SHIELD_CENTER
    assert Fraction(893, 2_048) == LOCKED_SHIELD_RADIUS
    assert Fraction(125, 1_024) == LOCKED_SHIELD_CENTER - LOCKED_SHIELD_RADIUS
    assert Fraction(509, 512) == LOCKED_SHIELD_CENTER + LOCKED_SHIELD_RADIUS
    assert Fraction(1, 2) == LOCKED_FP64_FALLBACK_GAIN
    assert LOCKED_FP64_INWARD_ULPS == 32


def test_exact_shield_is_identity_inside_and_total_at_zero() -> None:
    inside = project_onto_sector_disk_exact(_fractions(1, 2), _fractions(2, 4))
    zero = project_onto_sector_disk_exact(_fractions(7, -9), _fractions(0, 0))

    assert inside.candidate_inside
    assert not inside.active
    assert inside.scale == 1
    assert inside.output == (Rational(1), Rational(2))
    assert inside.sector_margin > 0
    assert inside.certified

    assert not zero.candidate_inside
    assert zero.active
    assert zero.scale == 0
    assert zero.output == (Rational(0), Rational(0))
    assert zero.sector_margin == 0
    assert zero.certified


def test_exact_shield_projects_an_arbitrary_candidate_to_the_boundary() -> None:
    result = project_onto_sector_disk_exact(_fractions(2, 1), _fractions(1, 0))
    gamma = Rational(1_143, 2_048)
    radius = Rational(893, 2_048)
    distance_squared = sum(
        (value - gamma * source) ** 2
        for value, source in zip(
            result.output,
            (Rational(1), Rational(0)),
            strict=True,
        )
    )

    assert not result.candidate_inside
    assert result.active
    assert 0 < result.scale < 1
    assert simplify(distance_squared - radius**2) == 0
    assert result.sector_margin == 0
    assert result.certified


def test_exact_shield_validates_types_shapes_and_disk_parameters() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        project_onto_sector_disk_exact(_fractions(1), _fractions(1, 2))
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        project_onto_sector_disk_exact((1,), _fractions(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="center >= radius"):
        project_onto_sector_disk_exact(
            _fractions(1),
            _fractions(1),
            center=Fraction(1, 4),
            radius=Fraction(1, 2),
        )


def test_fp64_shield_preserves_a_safely_interior_candidate_bit_for_bit() -> None:
    signal = np.asarray([[3.0, -4.0], [0.5, -0.25]], dtype=np.float64)
    candidate = np.ascontiguousarray(0.5 * signal)
    candidate_bits = candidate.view(np.uint64).copy()

    result = shield_sector_candidate_fp64(signal, candidate)

    assert result.diagnostics.candidate_inside
    assert not result.diagnostics.active
    assert not result.diagnostics.fail_closed
    assert not result.diagnostics.used_interior_fallback
    assert result.output.flags.c_contiguous
    np.testing.assert_array_equal(result.output.view(np.uint64), candidate_bits)
    assert result.diagnostics.exact_sector_margin > 0
    assert fp64_output_in_exact_sector(result.output, signal)


@pytest.mark.parametrize(
    "candidate",
    (
        np.asarray([2.0, 1.0], dtype=np.float64),
        np.asarray([-3.0, 8.0], dtype=np.float64),
        np.asarray([1.0e300, -1.0e300], dtype=np.float64),
        np.asarray([1.0e-300, -1.0e-300], dtype=np.float64),
    ),
)
def test_fp64_shield_projects_finite_corruptions_and_exactly_postchecks(
    candidate: np.ndarray,
) -> None:
    signal = np.asarray([1.0, -0.25], dtype=np.float64)

    result = shield_sector_candidate_fp64(signal, candidate)

    assert result.diagnostics.active
    assert not result.diagnostics.candidate_inside
    assert result.diagnostics.certified
    assert result.diagnostics.exact_sector_margin >= 0
    assert fp64_sector_disk_margin_exact(result.output, signal) >= 0
    assert np.isfinite(result.output).all()
    assert result.diagnostics.inward_radius < float(LOCKED_SHIELD_RADIUS)


def test_fp64_shield_scaled_projection_handles_near_maximum_signal() -> None:
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    signal = np.asarray([maximum, -maximum], dtype=np.float64)
    candidate = np.asarray([-maximum, maximum], dtype=np.float64)

    result = shield_sector_candidate_fp64(signal, candidate)

    assert result.diagnostics.active
    assert result.diagnostics.signal_scale == maximum
    assert math.isfinite(result.diagnostics.signal_scaled_norm)
    assert np.isfinite(result.output).all()
    assert fp64_output_in_exact_sector(result.output, signal)


def test_fp64_shield_is_total_at_zero_even_for_nonfinite_candidate() -> None:
    signal = np.zeros((2, 2), dtype=np.float64)
    finite = shield_sector_candidate_fp64(signal, np.ones((2, 2), dtype=np.float64))
    nonfinite = shield_sector_candidate_fp64(
        signal,
        np.asarray([[math.nan, math.inf], [-math.inf, 0.0]], dtype=np.float64),
    )

    assert finite.diagnostics.active
    assert not finite.diagnostics.fail_closed
    assert nonfinite.diagnostics.active
    assert nonfinite.diagnostics.fail_closed
    assert nonfinite.diagnostics.reason == "nonfinite_candidate_zero_signal"
    np.testing.assert_array_equal(finite.output, signal)
    np.testing.assert_array_equal(nonfinite.output, signal)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_candidate_falls_back_to_certified_interior_point(bad: float) -> None:
    signal = np.asarray([3.0, 4.0], dtype=np.float64)
    candidate = np.asarray([bad, 1.0], dtype=np.float64)

    result = shield_sector_candidate_fp64(signal, candidate)

    assert result.diagnostics.active
    assert result.diagnostics.fail_closed
    assert result.diagnostics.used_interior_fallback
    assert result.diagnostics.reason == "nonfinite_candidate_interior_fallback"
    np.testing.assert_array_equal(result.output, 0.5 * signal)
    assert fp64_output_in_exact_sector(result.output, signal)


def test_nonfinite_signal_and_invalid_runtime_inputs_fail_closed() -> None:
    finite = np.ones(2, dtype=np.float64)
    with pytest.raises(SectorShieldFailure, match="nonfinite signal"):
        shield_sector_candidate_fp64(
            np.asarray([math.nan, 1.0], dtype=np.float64),
            finite,
        )
    with pytest.raises(TypeError, match=r"numpy\.float64"):
        shield_sector_candidate_fp64(
            np.ones(2, dtype=np.float32),  # type: ignore[arg-type]
            finite,
        )
    with pytest.raises(ValueError, match="equal shapes"):
        shield_sector_candidate_fp64(finite, np.ones(3, dtype=np.float64))
    with pytest.raises(ValueError, match="positive"):
        shield_sector_candidate_fp64(finite, finite, inward_ulps=0)


def test_unrepresentable_subnormal_update_stops_without_emitting_output() -> None:
    least_subnormal = math.ulp(0.0)
    signal = np.asarray([least_subnormal], dtype=np.float64)
    candidate = np.zeros(1, dtype=np.float64)

    assert fp64_sector_disk_margin_exact(candidate, signal) < 0
    with pytest.raises(SectorShieldFailure, match="no certified dyadic interior"):
        shield_sector_candidate_fp64(signal, candidate)


def test_exact_binary64_postcheck_detects_one_ulp_escape() -> None:
    signal = np.asarray([1.0], dtype=np.float64)
    exact_upper = float(Fraction(509, 512))
    boundary = np.asarray([exact_upper], dtype=np.float64)
    escaped = np.asarray([math.nextafter(exact_upper, math.inf)], dtype=np.float64)

    assert fp64_sector_disk_margin_exact(boundary, signal) == 0
    assert fp64_output_in_exact_sector(boundary, signal)
    assert fp64_sector_disk_margin_exact(escaped, signal) < 0
    assert not fp64_output_in_exact_sector(escaped, signal)
    repaired = shield_sector_candidate_fp64(signal, escaped)
    assert repaired.diagnostics.active
    assert repaired.diagnostics.exact_sector_margin > 0


def test_guarded_p18_solver_output_normally_passes_through_shield_unchanged() -> None:
    values = np.asarray([3.0, 4.0], dtype=np.float64)
    singular = evaluate_sector_shielded_singular_values_fp64(values)
    matrix = evaluate_sector_shielded_resolvent_fp64(np.diag(values))

    assert singular.p18_evaluation.solver_diagnostics.certified
    assert matrix.p18_evaluation.resolvent_result.diagnostics.certified
    assert singular.shield_diagnostics.candidate_inside
    assert matrix.shield_diagnostics.candidate_inside
    assert not singular.shield_diagnostics.active
    assert not matrix.shield_diagnostics.active
    np.testing.assert_array_equal(singular.output, singular.p18_evaluation.output)
    np.testing.assert_array_equal(matrix.output, matrix.p18_evaluation.output)
    np.testing.assert_allclose(np.diag(matrix.output), singular.output, rtol=3e-13, atol=5e-13)
