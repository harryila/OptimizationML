from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from passive_muon.orthogonalizers import (
    ORTHOGONALIZERS,
    cans_5x4,
    classical_ns,
    exact_polar,
    jordan_ns,
    polar_express,
    taylor_ns5,
)
from passive_muon.polynomials import (
    CLASSICAL_CUBIC,
    TAYLOR_QUINTIC,
    QuinticCoefficients,
    scalar_response,
)

MatrixOperator = Callable[[torch.Tensor], torch.Tensor]


def _full_row_rank_matrix() -> torch.Tensor:
    # Its separated singular values (about 0.52 and 0.36) stay in the
    # classical/Taylor Newton--Schulz convergence regime.
    return torch.tensor(
        [[0.35, -0.08, 0.12], [0.04, 0.50, -0.10]],
        dtype=torch.float64,
    )


def _orthogonal_matrix(size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn((size, size), generator=generator, dtype=torch.float64)
    factor, _ = torch.linalg.qr(matrix)
    return factor


def test_taylor_is_odd_and_transpose_equivariant() -> None:
    matrix = _full_row_rank_matrix()
    torch.testing.assert_close(taylor_ns5(-matrix, steps=3), -taylor_ns5(matrix, steps=3))
    torch.testing.assert_close(
        taylor_ns5(matrix.mT, steps=3),
        taylor_ns5(matrix, steps=3).mT,
    )


def test_taylor_is_left_right_orthogonally_equivariant() -> None:
    matrix = _full_row_rank_matrix()
    left = _orthogonal_matrix(2, seed=101)
    right = _orthogonal_matrix(3, seed=102)
    transformed = left @ matrix @ right.mT
    expected = left @ taylor_ns5(matrix, steps=3) @ right.mT
    torch.testing.assert_close(
        taylor_ns5(transformed, steps=3),
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


def test_exact_polar_is_odd_and_transpose_equivariant_at_full_rank() -> None:
    matrix = _full_row_rank_matrix()
    torch.testing.assert_close(exact_polar(-matrix), -exact_polar(matrix))
    torch.testing.assert_close(exact_polar(matrix.mT), exact_polar(matrix).mT)


@pytest.mark.parametrize(
    ("operator", "coefficients"),
    [
        pytest.param(classical_ns, CLASSICAL_CUBIC, id="classical"),
        pytest.param(taylor_ns5, TAYLOR_QUINTIC, id="taylor"),
    ],
)
def test_stationary_ns_maps_match_scalar_iteration_on_diagonal(
    operator: Callable[..., torch.Tensor],
    coefficients: QuinticCoefficients,
) -> None:
    diagonal = torch.tensor([0.3, 0.7], dtype=torch.float64)
    expected = torch.diag(scalar_response(diagonal, coefficients, steps=5))
    torch.testing.assert_close(operator(torch.diag(diagonal), steps=5), expected)


@pytest.mark.parametrize(
    ("operator", "maximum_final_error"),
    [
        pytest.param(classical_ns, 1e-3, id="classical"),
        pytest.param(taylor_ns5, 1e-10, id="taylor"),
    ],
)
def test_convergent_ns_prefixes_approach_exact_polar(
    operator: Callable[..., torch.Tensor],
    maximum_final_error: float,
) -> None:
    matrix = _full_row_rank_matrix()
    reference = exact_polar(matrix)
    errors = [
        torch.linalg.vector_norm(operator(matrix, steps=steps) - reference).item()
        for steps in (1, 3, 5)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < maximum_final_error


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(jordan_ns, id="jordan"),
        pytest.param(classical_ns, id="classical"),
        pytest.param(taylor_ns5, id="taylor"),
        pytest.param(polar_express, id="polar-express"),
        pytest.param(cans_5x4, id="cans"),
        pytest.param(exact_polar, id="exact-polar"),
    ],
)
def test_raw_orthogonalizer_autograd_matches_central_difference(
    operator: MatrixOperator,
) -> None:
    matrix = _full_row_rank_matrix()
    direction = torch.tensor(
        [[0.11, -0.17, 0.07], [0.13, 0.09, -0.05]],
        dtype=torch.float64,
    )
    _, autodiff = torch.autograd.functional.jvp(
        operator,
        matrix,
        direction,
        strict=True,
    )
    step = 1e-6
    forward = operator(matrix + step * direction)
    backward = operator(matrix - step * direction)
    finite_difference = (forward - backward) / (2 * step)
    torch.testing.assert_close(autodiff, finite_difference, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize(
    "operator",
    [
        pytest.param(jordan_ns, id="jordan"),
        pytest.param(classical_ns, id="classical"),
        pytest.param(taylor_ns5, id="taylor"),
        pytest.param(polar_express, id="polar-express"),
        pytest.param(cans_5x4, id="cans"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_polynomial_orthogonalizers_preserve_supported_dtype(
    operator: MatrixOperator,
    dtype: torch.dtype,
) -> None:
    matrix = torch.tensor(
        [[0.08, -0.02, 0.03], [0.01, 0.12, -0.04]],
        dtype=dtype,
    )
    result = operator(matrix)
    assert result.dtype == dtype
    assert result.shape == matrix.shape
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_polar_preserves_svd_supported_dtype(dtype: torch.dtype) -> None:
    result = exact_polar(_full_row_rank_matrix().to(dtype))
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_taylor_is_registered_under_its_public_name() -> None:
    assert ORTHOGONALIZERS["taylor_ns5"] is taylor_ns5
