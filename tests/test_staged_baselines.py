from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.operator import orthogonalize
from passive_muon.orthogonalizers import ORTHOGONALIZERS, cans_5x4, polar_express
from passive_muon.polynomials import (
    CANS_5X4,
    POLAR_EXPRESS_5,
    iterate_staged_quintic,
    quintic_step,
    staged_scalar_response,
    staged_scalar_response_and_derivative_exact,
)


def test_polar_express_first_five_coefficients_are_frozen() -> None:
    actual = tuple((stage.a, stage.b, stage.c) for stage in POLAR_EXPRESS_5.stages)
    assert actual == (
        ("8.237312490495558", "-23.15774741455821", "16.68056841144592"),
        ("4.082441999064829", "-2.8930477353325825", "0.525284925697564"),
        ("3.926347992254658", "-2.854746803476532", "0.5318022422894996"),
        ("3.2982187133085143", "-2.424541981026707", "0.48632008358844103"),
        ("2.297036943455258", "-1.636625581259032", "0.4002628455953631"),
    )
    assert POLAR_EXPRESS_5.default_steps == 5
    assert "71cc37943d99cae780024c1d198977f2f8795407" in POLAR_EXPRESS_5.source


def test_cans_degree5_four_stage_coefficients_are_frozen() -> None:
    actual = tuple((stage.a, stage.b, stage.c) for stage in CANS_5X4.stages)
    assert actual == (
        ("8.420293602126344", "-24.910491192120688", "18.472094206318726"),
        ("4.101228661246281", "-3.0518555467946813", "0.5741241025302702"),
        ("3.6809819251109155", "-2.75396502307162", "0.5401902781108926"),
        ("2.7280916801566666", "-2.0315492757300913", "0.45866431681858805"),
    )
    assert CANS_5X4.default_steps == 4
    assert "arXiv:2506.10935v2" in CANS_5X4.source
    assert "delta=0.3" in CANS_5X4.source


@pytest.mark.parametrize(
    ("operator", "specification", "steps"),
    [(polar_express, POLAR_EXPRESS_5, 5), (cans_5x4, CANS_5X4, 4)],
)
def test_staged_matrix_map_matches_scalar_composition_on_diagonal(
    operator, specification, steps: int
) -> None:
    diagonal = torch.tensor([0.03, 0.2], dtype=torch.float64)
    expected = torch.diag(staged_scalar_response(diagonal, specification, steps=steps))
    torch.testing.assert_close(
        operator(torch.diag(diagonal), steps=steps), expected, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("operator", [polar_express, cans_5x4])
def test_staged_orthogonalizers_are_odd_and_transpose_equivariant(operator) -> None:
    matrix = torch.tensor([[0.03, -0.02, 0.01], [0.04, 0.05, -0.01]], dtype=torch.float64)
    torch.testing.assert_close(operator(-matrix), -operator(matrix))
    torch.testing.assert_close(operator(matrix.mT), operator(matrix).mT)


@pytest.mark.parametrize("operator", [polar_express, cans_5x4])
def test_staged_orthogonalizers_are_left_right_equivariant(operator) -> None:
    matrix = torch.tensor([[0.03, -0.02, 0.01], [0.04, 0.05, -0.01]], dtype=torch.float64)
    left, _ = torch.linalg.qr(torch.tensor([[1.0, 2.0], [3.0, -1.0]], dtype=torch.float64))
    right, _ = torch.linalg.qr(
        torch.tensor(
            [[1.0, 2.0, 0.5], [-1.0, 0.2, 1.5], [0.7, -0.4, 2.0]],
            dtype=torch.float64,
        )
    )
    transformed = left @ matrix @ right.mT
    expected = left @ operator(matrix) @ right.mT
    torch.testing.assert_close(operator(transformed), expected, rtol=1e-10, atol=1e-10)


def test_staged_iteration_selects_an_explicit_prefix() -> None:
    matrix = torch.tensor([[0.1, 0.02], [-0.03, 0.2]], dtype=torch.float64)
    manual = quintic_step(quintic_step(matrix, CANS_5X4.stages[0]), CANS_5X4.stages[1])
    actual = iterate_staged_quintic(matrix, CANS_5X4, steps=2)
    torch.testing.assert_close(actual, manual)

    with pytest.raises(ValueError, match="has 4 frozen stages; requested 5"):
        iterate_staged_quintic(matrix, CANS_5X4, steps=5)
    with pytest.raises(ValueError, match="steps must be nonnegative"):
        iterate_staged_quintic(matrix, CANS_5X4, steps=-1)


def test_exact_staged_derivative_matches_torch_autograd() -> None:
    exact_value, exact_derivative = staged_scalar_response_and_derivative_exact(
        Fraction(1, 10), CANS_5X4
    )
    value = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    response = staged_scalar_response(value, CANS_5X4)
    (derivative,) = torch.autograd.grad(response, value)
    assert response.item() == pytest.approx(float(exact_value), rel=1e-13, abs=1e-13)
    assert derivative.item() == pytest.approx(float(exact_derivative), rel=1e-12, abs=1e-12)


def test_staged_baselines_are_registered_under_explicit_names() -> None:
    assert ORTHOGONALIZERS["polar_express"] is polar_express
    assert ORTHOGONALIZERS["cans_5x4"] is cans_5x4


def test_public_interface_uses_cans_default_but_rejects_an_explicit_fifth_stage() -> None:
    matrix = torch.tensor([[0.03, -0.02], [0.04, 0.05]], dtype=torch.float64)
    expected = orthogonalize(matrix, polynomial="cans_5x4", steps=4)
    torch.testing.assert_close(orthogonalize(matrix, polynomial="cans_5x4"), expected)

    with pytest.raises(ValueError, match="has 4 frozen stages; requested 5"):
        orthogonalize(matrix, polynomial="cans_5x4", steps=5)
