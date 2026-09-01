from __future__ import annotations

import hashlib
from decimal import Decimal, localcontext
from fractions import Fraction

import torch

from passive_muon.normalizers import FixedScaleNormalizer
from passive_muon.operator import NormalizedMap
from passive_muon.orthogonalizers import jordan_ns
from passive_muon.passivity import jacobian_matrix
from passive_muon.specs import JORDAN_QUINTIC, scalar_response_and_derivative_exact
from passive_muon.witness import canonical_jordan_witness


def _fraction_digest(value: Fraction) -> str:
    canonical = f"{value.numerator}/{value.denominator}".encode()
    return hashlib.sha256(canonical).hexdigest()


def test_canonical_witness_is_exact_and_causally_clean() -> None:
    payload = canonical_jordan_witness()
    exact = payload["local_exact_certificate"]
    exact_pair = payload["finite_pair_exact_certificate"]
    pair = payload["finite_pair_high_precision"]

    assert exact["certified_indefinite"] is True
    assert exact["derivatives_positive"] is True
    assert exact["derivatives_unequal"] is True
    assert exact["determinant_sign"] == "negative"
    assert exact_pair["certified_normalized_gap_negative"] is True
    assert exact_pair["certified_fixed_scale_gap_positive"] is True
    assert Decimal(pair["normalized_gap"]) < 0
    assert Decimal(pair["epsilon_regularized_gap"]) < 0
    assert Decimal(pair["fixed_scale_gap"]) > 0


def test_witness_payload_is_deterministic() -> None:
    assert canonical_jordan_witness() == canonical_jordan_witness()


def test_fixed_scale_full_2x2_local_modes_have_exact_positive_certificates() -> None:
    payload = canonical_jordan_witness()
    exact = payload["fixed_scale_local_exact_certificate"]
    diagonal = exact["diagonal_modes"]
    difference = exact["off_diagonal_difference_mode"]
    sum_mode = exact["off_diagonal_sum_mode"]

    u1, u2 = Fraction(3, 5), Fraction(4, 5)
    h1, d1 = scalar_response_and_derivative_exact(u1, JORDAN_QUINTIC, steps=5)
    h2, d2 = scalar_response_and_derivative_exact(u2, JORDAN_QUINTIC, steps=5)
    exact_difference = (h1 - h2) / (u1 - u2)
    exact_sum = (h1 + h2) / (u1 + u2)

    assert exact["claim_scope"] == "full_2x2_local_jacobian_at_M0_only_not_global"
    assert exact["global_fixed_scale_monotonicity_claim"] == "none"
    assert exact["certified_all_four_jacobian_modes_positive"] is True
    assert exact["certified_full_2x2_jacobian_positive_definite_at_witness"] is True
    assert exact["certified_full_2x2_local_monotonicity_at_witness"] is True
    assert "continuity" in exact["local_monotonicity_inference"]

    assert diagonal["eigendirections"] == ["E11", "E22"]
    assert diagonal["formula"] == "h'(ui)/5"
    assert diagonal["mode_1_sign"] == "positive"
    assert diagonal["mode_2_sign"] == "positive"
    assert difference["eigendirection"] == "E12+E21"
    assert difference["raw_mode_sign"] == "positive"
    assert difference["fixed_scale_jacobian_mode_sign"] == "positive"
    assert sum_mode["eigendirection"] == "E12-E21"
    assert sum_mode["raw_mode_sign"] == "positive"
    assert sum_mode["fixed_scale_jacobian_mode_sign"] == "positive"

    assert d1 / 5 > 0
    assert d2 / 5 > 0
    assert h1 - h2 < 0
    assert u1 - u2 < 0
    assert exact_difference > 0
    assert h1 + h2 > 0
    assert u1 + u2 > 0
    assert exact_sum > 0

    assert diagonal["mode_1_sha256"] == _fraction_digest(d1 / 5)
    assert diagonal["mode_2_sha256"] == _fraction_digest(d2 / 5)
    assert difference["raw_mode_sha256"] == _fraction_digest(exact_difference)
    assert difference["fixed_scale_jacobian_mode_sha256"] == _fraction_digest(exact_difference / 5)
    assert sum_mode["raw_mode_sha256"] == _fraction_digest(exact_sum)
    assert sum_mode["fixed_scale_jacobian_mode_sha256"] == _fraction_digest(exact_sum / 5)

    assert diagonal["mode_1_sha256"] == (
        "12f5c5e276ee93752a7354daf244817588c440a00dc5ce1939cb2fe7415f898e"
    )
    assert diagonal["mode_2_sha256"] == (
        "e0d5d99fea685afac485e1be62350a4e0de5fbe11b961852db92f72d5807e507"
    )
    assert difference["raw_mode_sha256"] == (
        "410777682b0be6444735df18f27a5b30e1e0e872686bb78f13c9b7949f8627bb"
    )
    assert difference["fixed_scale_jacobian_mode_sha256"] == (
        "589fdc814b93dac6b40f77b8cad28d58a80c44b21e5a4acec05ff4d7aac3f57b"
    )
    assert sum_mode["raw_mode_sha256"] == (
        "5a4e684a9f26d89545a041409b2d92d3686a5bd04f8c4254caee69f76cfa08e3"
    )
    assert sum_mode["fixed_scale_jacobian_mode_sha256"] == (
        "d85116006d60910eb399ab34860209828bac3ab52907e6b660adddf0450b8e6e"
    )


def test_fixed_scale_full_2x2_local_modes_have_high_precision_values() -> None:
    payload = canonical_jordan_witness()
    values = payload["fixed_scale_local_high_precision"]
    u1, u2 = Fraction(3, 5), Fraction(4, 5)
    _, d1 = scalar_response_and_derivative_exact(u1, JORDAN_QUINTIC, steps=5)
    _, d2 = scalar_response_and_derivative_exact(u2, JORDAN_QUINTIC, steps=5)

    raw_difference = Decimal(values["difference_mode_before_fixed_scale_chain_rule"])
    raw_sum = Decimal(values["sum_mode_before_fixed_scale_chain_rule"])
    assert raw_difference == Decimal(
        "1.981638806494635392914417114139463695267549193799806270787446882805430165100935163506872338923018884"
    )
    assert raw_sum == Decimal(
        "1.315771498952256913894785401896255533533977925930683497122527116112550449961610057470909915868273717"
    )
    with localcontext() as context:
        context.prec = values["decimal_digits"]
        expected_difference = raw_difference / 5
        expected_sum = raw_sum / 5
        expected_diagonal_1 = Decimal(d1.numerator) / Decimal(d1.denominator) / 5
        expected_diagonal_2 = Decimal(d2.numerator) / Decimal(d2.denominator) / 5
    assert Decimal(values["fixed_scale_difference_jacobian_mode"]) == expected_difference
    assert Decimal(values["fixed_scale_sum_jacobian_mode"]) == expected_sum
    assert Decimal(values["diagonal_mode_1"]) == expected_diagonal_1
    assert Decimal(values["diagonal_mode_2"]) == expected_diagonal_2


def test_fixed_scale_full_jacobian_acts_by_the_four_certified_modes() -> None:
    matrix = torch.diag(torch.tensor([3.0, 4.0], dtype=torch.float64))
    operator = NormalizedMap(
        FixedScaleNormalizer(5.0),
        lambda value: jordan_ns(value, steps=5),
    )
    jacobian = jacobian_matrix(operator, matrix)
    u1, u2 = Fraction(3, 5), Fraction(4, 5)
    h1, d1 = scalar_response_and_derivative_exact(u1, JORDAN_QUINTIC, steps=5)
    h2, d2 = scalar_response_and_derivative_exact(u2, JORDAN_QUINTIC, steps=5)
    modes = (
        d1 / 5,
        d2 / 5,
        ((h1 - h2) / (u1 - u2)) / 5,
        ((h1 + h2) / (u1 + u2)) / 5,
    )
    directions = (
        torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64),
    )
    for direction, mode in zip(directions, modes, strict=True):
        action = (jacobian @ direction.reshape(-1)).reshape_as(direction)
        torch.testing.assert_close(action, float(mode) * direction, rtol=2e-12, atol=2e-12)
