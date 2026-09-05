from __future__ import annotations

from fractions import Fraction

from flint import arb, fmpq

from passive_muon.equivariant_resolvent_certificate import (
    BEST_SCALAR_DEPARTURE_GATE,
    CANONICAL_DEPARTURE_GUARD,
    CANONICAL_RETENTION_GUARD,
    CANONICAL_UPSTREAM_DEPARTURE_GUARD,
    JACOBIAN_DIAGONAL_MARGINS,
    LOCKED_BEST_SCALAR_DEPARTURE_GUARD,
    LOCKED_RETENTION_GUARD,
    LOCKED_UPSTREAM_DEPARTURE_GUARD,
    UPSTREAM_SHAPING_RETENTION_GATE,
    evaluate_canonical_shaping_enclosure,
    evaluate_shaping_witness,
    exact_certificate_checks,
    forward_shaping_witness,
)


def _arb_fraction(value: Fraction) -> arb:
    return arb(fmpq(value.numerator, value.denominator))


def test_exact_reduction_and_witness_checks_close() -> None:
    checks = exact_certificate_checks()
    assert checks
    assert all(checks.values())
    assert all(value > 0 for value in JACOBIAN_DIAGONAL_MARGINS)


def test_forward_witness_is_an_exact_noncollapse_control() -> None:
    witness = forward_shaping_witness()
    assert sum(value**2 for value in witness.resolvent_singular_values) == 1
    assert witness.pre_resolvent_modal_gain_difference > 0
    assert witness.algebraically_distinct
    assert witness.radial_primitive.logarithm_coefficient > 0


def test_arb_witness_cross_precision_proves_distinct_but_fails_fidelity_gate() -> None:
    evaluations = [evaluate_shaping_witness(precision_bits=bits) for bits in (160, 224)]
    for evaluation in evaluations:
        assert evaluation.modal_gain_gap > 0
        assert evaluation.modal_gains[1] > evaluation.modal_gains[0]
        assert not evaluation.absolute_gate_passes
        assert not evaluation.retention_gate_passes
        assert evaluation.absolute_gate_fails
        assert evaluation.retention_gate_fails
        assert evaluation.gate_decisions_are_definite
        assert not evaluation.meaningful_fidelity_passes
        assert evaluation.best_scalar_departure < _arb_fraction(BEST_SCALAR_DEPARTURE_GATE)
        assert evaluation.upstream_shaping_retention < _arb_fraction(
            UPSTREAM_SHAPING_RETENTION_GATE
        )

        departure_lower, departure_upper = LOCKED_BEST_SCALAR_DEPARTURE_GUARD
        upstream_lower, upstream_upper = LOCKED_UPSTREAM_DEPARTURE_GUARD
        retention_lower, retention_upper = LOCKED_RETENTION_GUARD
        assert evaluation.best_scalar_departure > _arb_fraction(departure_lower)
        assert evaluation.best_scalar_departure < _arb_fraction(departure_upper)
        assert evaluation.upstream_best_scalar_departure > _arb_fraction(upstream_lower)
        assert evaluation.upstream_best_scalar_departure < _arb_fraction(upstream_upper)
        assert evaluation.upstream_shaping_retention > _arb_fraction(retention_lower)
        assert evaluation.upstream_shaping_retention < _arb_fraction(retention_upper)


def test_canonical_diag_3_4_output_is_rigorously_distinct_but_effectively_scalar() -> None:
    for bits in (160, 224):
        enclosure = evaluate_canonical_shaping_enclosure(precision_bits=bits)
        assert enclosure.graph_residual_norm > 0
        assert enclosure.exact_output_error_norm_upper < _arb_fraction(Fraction(1, 10**10))
        assert enclosure.modal_gain_gap > 0
        assert enclosure.modal_gains[1] > enclosure.modal_gains[0]
        assert enclosure.absolute_gate_fails
        assert enclosure.retention_gate_fails
        assert enclosure.gate_decisions_are_definite
        assert not enclosure.meaningful_fidelity_passes

        for value, bounds in (
            (enclosure.best_scalar_departure, CANONICAL_DEPARTURE_GUARD),
            (enclosure.upstream_best_scalar_departure, CANONICAL_UPSTREAM_DEPARTURE_GUARD),
            (enclosure.upstream_shaping_retention, CANONICAL_RETENTION_GUARD),
        ):
            assert value > _arb_fraction(bounds[0])
            assert value < _arb_fraction(bounds[1])


def test_witness_validation_rejects_invalid_exact_parameters() -> None:
    for kwargs in (
        {"epsilon": Fraction(0)},
        {"resolvent_parameter": Fraction(-1)},
        {"shunt": Fraction(0)},
    ):
        try:
            forward_shaping_witness(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid witness parameters accepted: {kwargs}")
