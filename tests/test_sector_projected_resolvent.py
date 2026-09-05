from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from passive_muon.sector_projected_resolvent import (
    LOCKED_GATE_CAP,
    LOCKED_PASSIVE_DIVISOR,
    LOCKED_POINTWISE_SECTOR,
    LOCKED_PROJECTION_GAIN,
    LOCKED_SECTOR_PROJECTED_DESIGN,
    SectorProjectedDesign,
    blend_sector_projected_exact,
    blend_sector_projected_fp64,
    evaluate_sector_projected_resolvent_fp64,
    evaluate_sector_projected_singular_values_fp64,
    project_onto_origin_sector_exact,
    project_onto_origin_sector_fp64,
    projected_blend_pointwise_sector,
    sector_disk_margin_exact,
    sector_projection_scale_exact,
)


def _fractions(*values: int) -> tuple[Fraction, ...]:
    return tuple(Fraction(value) for value in values)


def test_locked_projected_sector_endpoints_are_exact() -> None:
    sector = projected_blend_pointwise_sector()

    assert LOCKED_PROJECTION_GAIN == 1
    assert LOCKED_PASSIVE_DIVISOR == 1_024
    assert Fraction(3, 4) == LOCKED_GATE_CAP
    assert sector is not LOCKED_POINTWISE_SECTOR
    assert sector == LOCKED_POINTWISE_SECTOR
    assert sector.lower == Fraction(125, 1_024)
    assert sector.upper == Fraction(509, 512)
    assert sector.center == Fraction(1_143, 2_048)
    assert sector.radius == Fraction(893, 2_048)
    assert sector.condition_ratio == Fraction(1_018, 125)


def test_projection_is_inactive_when_shape_output_already_fits() -> None:
    signal = _fractions(3, 4)
    shape = (Fraction(3, 2), Fraction(2))
    projected = project_onto_origin_sector_exact(shape, signal)

    assert projected.scale == 1
    assert projected.output == shape
    assert projected.sector_margin == Fraction(25, 4)
    assert projected.certified


def test_projection_active_branch_saturates_sector_boundary() -> None:
    signal = _fractions(1, 0)
    shape = _fractions(1, 1)
    projected = project_onto_origin_sector_exact(shape, signal)

    assert projected.scale == Fraction(1, 2)
    assert projected.output == (Fraction(1, 2), Fraction(1, 2))
    assert projected.projected_squared_norm == Fraction(1, 2)
    assert projected.projected_signal_inner_product == Fraction(1, 2)
    assert projected.sector_margin == 0
    assert projected.certified


def test_projection_definition_is_total_at_zero_and_zero_alignment() -> None:
    zero_shape = project_onto_origin_sector_exact(_fractions(0, 0), _fractions(3, 4))
    orthogonal_shape = project_onto_origin_sector_exact(_fractions(0, 2), _fractions(3, 0))

    assert sector_projection_scale_exact(Fraction(0), Fraction(0)) == 0
    assert zero_shape.scale == 0
    assert zero_shape.output == _fractions(0, 0)
    assert zero_shape.sector_margin == 0
    assert orthogonal_shape.scale == 0
    assert orthogonal_shape.output == _fractions(0, 0)
    assert orthogonal_shape.sector_margin == 0


def test_projection_rejects_data_outside_the_nonnegative_alignment_hypothesis() -> None:
    with pytest.raises(ValueError, match="nonnegative inner product"):
        project_onto_origin_sector_exact(_fractions(-1, 0), _fractions(1, 0))
    with pytest.raises(ValueError, match="zero shape norm"):
        sector_projection_scale_exact(Fraction(1), Fraction(0))
    with pytest.raises(TypeError, match=r"fractions\.Fraction"):
        sector_projection_scale_exact(1, Fraction(1))  # type: ignore[arg-type]


def test_exact_blend_attains_both_locked_uniform_sector_endpoints() -> None:
    signal = _fractions(1, 0)

    lower = blend_sector_projected_exact(
        signal,
        _fractions(500, 0),
        _fractions(0, 0),
        LOCKED_GATE_CAP,
    )
    upper = blend_sector_projected_exact(
        signal,
        _fractions(1_000, 0),
        _fractions(1, 0),
        LOCKED_GATE_CAP,
    )

    assert lower.output == (LOCKED_POINTWISE_SECTOR.lower, Fraction(0))
    assert lower.sector_margin == 0
    assert lower.certified
    assert upper.output == (LOCKED_POINTWISE_SECTOR.upper, Fraction(0))
    assert upper.sector_margin == 0
    assert upper.certified


def test_exact_blend_handles_zero_signal_and_gate_edges() -> None:
    zero = _fractions(0, 0, 0)
    at_zero_gate = blend_sector_projected_exact(
        _fractions(2, 0),
        _fractions(1_500, 0),
        _fractions(100, 0),
        Fraction(0),
    )
    at_zero_signal = blend_sector_projected_exact(zero, zero, _fractions(2, 3, 4), Fraction(1, 2))

    assert at_zero_gate.output == (Fraction(375, 256), Fraction(0))
    assert at_zero_gate.projection.scale == Fraction(1, 50)
    assert at_zero_gate.certified
    assert at_zero_signal.output == zero
    assert at_zero_signal.projection.scale == 0
    assert at_zero_signal.certified


def test_general_sector_helper_selects_the_correct_upper_hull_endpoint() -> None:
    passive_dominates = SectorProjectedDesign(
        projection_gain=Fraction(1, 2),
        passive_divisor=Fraction(1_024),
        gate_cap=Fraction(3, 4),
    )
    projected_dominates = SectorProjectedDesign(
        projection_gain=Fraction(1),
        passive_divisor=Fraction(1_024),
        gate_cap=Fraction(1, 2),
    )

    first = projected_blend_pointwise_sector(passive_dominates)
    second = projected_blend_pointwise_sector(projected_dominates)

    assert first.lower == Fraction(125, 1_024)
    assert first.upper == Fraction(125, 128)
    assert second.lower == Fraction(125, 512)
    assert second.upper == Fraction(253, 256)
    assert (
        projected_blend_pointwise_sector(
            SectorProjectedDesign(Fraction(1), Fraction(1_024), Fraction(1))
        ).condition_ratio
        is None
    )


def test_sector_disk_margin_uses_the_full_vector_tangent_space() -> None:
    output = (Fraction(1, 2), Fraction(1, 2))
    signal = _fractions(1, 0)

    assert sector_disk_margin_exact(output, signal, Fraction(0), Fraction(1)) == 0
    assert sector_disk_margin_exact(_fractions(2, 0), signal, Fraction(0), Fraction(1)) < 0


def test_fp64_projection_and_blend_match_the_exact_simple_case() -> None:
    signal = np.asarray([1.0, 0.0], dtype=np.float64)
    shape = np.asarray([1.0, 1.0], dtype=np.float64)
    yosida = np.asarray([1_000.0, 0.0], dtype=np.float64)

    scale, projected = project_onto_origin_sector_fp64(shape, signal)
    blend_scale, blend_projected, output = blend_sector_projected_fp64(
        signal,
        yosida,
        shape,
        0.75,
    )

    assert scale == blend_scale == 0.5
    np.testing.assert_array_equal(projected, np.asarray([0.5, 0.5]))
    np.testing.assert_array_equal(blend_projected, projected)
    np.testing.assert_array_equal(output, np.asarray([317.0 / 512.0, 3.0 / 8.0]))
    assert (
        sector_disk_margin_exact(
            tuple(Fraction(value) for value in output),
            _fractions(1, 0),
            LOCKED_POINTWISE_SECTOR.lower,
            LOCKED_POINTWISE_SECTOR.upper,
        )
        >= 0
    )


def test_design_and_numerical_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="gate_cap"):
        SectorProjectedDesign(Fraction(1), Fraction(1_024), Fraction(5, 4))
    with pytest.raises(ValueError, match="projection_gain"):
        SectorProjectedDesign(Fraction(-1), Fraction(1_024), Fraction(1, 2))
    with pytest.raises(ValueError, match="equal shapes"):
        project_onto_origin_sector_fp64(
            np.ones(2, dtype=np.float64),
            np.ones(3, dtype=np.float64),
        )
    with pytest.raises(TypeError, match=r"numpy\.float64"):
        project_onto_origin_sector_fp64(
            np.ones(2, dtype=np.float32),
            np.ones(2, dtype=np.float64),
        )
    with pytest.raises(ValueError, match="nonnegative inner product"):
        project_onto_origin_sector_fp64(
            np.asarray([-1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        )
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        blend_sector_projected_fp64(
            np.ones(1, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            np.ones(1, dtype=np.float64),
            1.1,
        )


def test_guarded_singular_wrapper_returns_projection_diagnostics_and_locked_output() -> None:
    signal = np.asarray([3.0, 4.0], dtype=np.float64)
    evaluated = evaluate_sector_projected_singular_values_fp64(signal)
    expected = (
        1.0 - evaluated.gate_value
    ) * evaluated.yosida_output / 1_024.0 + evaluated.gate_value * evaluated.projected_shape_output

    assert evaluated.solver_diagnostics.certified
    assert evaluated.gate_value == 0.75
    assert 0.0 <= evaluated.projection_alpha <= 1.0
    assert evaluated.shape_signal_inner_product >= 0.0
    assert evaluated.shape_squared_norm >= 0.0
    assert evaluated.projection_active == (
        evaluated.shape_squared_norm > 0.0 and evaluated.projection_alpha < 1.0
    )
    np.testing.assert_array_equal(evaluated.output, expected)


def test_guarded_matrix_wrapper_matches_singular_diagonal_and_preserves_postcondition() -> None:
    signal_values = np.asarray([3.0, 4.0], dtype=np.float64)
    singular = evaluate_sector_projected_singular_values_fp64(signal_values)
    matrix = evaluate_sector_projected_resolvent_fp64(np.diag(signal_values))

    assert matrix.resolvent_result.diagnostics.certified
    assert matrix.projection_alpha == pytest.approx(singular.projection_alpha, abs=2e-15)
    assert matrix.projection_active == singular.projection_active
    assert matrix.shape_signal_inner_product == pytest.approx(
        singular.shape_signal_inner_product,
        rel=3e-15,
    )
    np.testing.assert_allclose(np.diag(matrix.output), singular.output, rtol=3e-13, atol=5e-13)
    np.testing.assert_allclose(matrix.output - np.diag(np.diag(matrix.output)), 0.0, atol=1e-14)


def test_guarded_wrappers_define_zero_projection_without_division() -> None:
    singular = evaluate_sector_projected_singular_values_fp64(np.zeros(3, dtype=np.float64))
    matrix = evaluate_sector_projected_resolvent_fp64(np.zeros((2, 3), dtype=np.float64))

    assert singular.solver_diagnostics.certified
    assert matrix.resolvent_result.diagnostics.certified
    assert singular.projection_alpha == matrix.projection_alpha == 0.0
    assert not singular.projection_active
    assert not matrix.projection_active
    assert singular.shape_signal_inner_product == matrix.shape_signal_inner_product == 0.0
    assert np.array_equal(singular.output, np.zeros(3))
    assert np.array_equal(matrix.output, np.zeros((2, 3)))


def test_locked_design_object_is_the_default_helper_input() -> None:
    assert (
        SectorProjectedDesign(
            projection_gain=Fraction(1),
            passive_divisor=Fraction(1_024),
            gate_cap=Fraction(3, 4),
        )
        == LOCKED_SECTOR_PROJECTED_DESIGN
    )
