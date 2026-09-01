from __future__ import annotations

import pytest
import torch

from passive_muon.orthogonalizers import classical_ns, exact_polar, jordan_ns
from passive_muon.polynomials import JORDAN_QUINTIC, scalar_response


def _orthogonal_matrix(size: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn((size, size), generator=generator, dtype=torch.float64))
    return q


@pytest.mark.parametrize("operator", [jordan_ns, classical_ns])
def test_polynomial_orthogonalizers_are_odd_and_transpose_equivariant(operator) -> None:
    matrix = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]], dtype=torch.float64)
    torch.testing.assert_close(operator(-matrix, steps=3), -operator(matrix, steps=3))
    torch.testing.assert_close(operator(matrix.mT, steps=3), operator(matrix, steps=3).mT)


@pytest.mark.parametrize("operator", [jordan_ns, classical_ns])
def test_polynomial_orthogonalizers_are_left_right_equivariant(operator) -> None:
    matrix = torch.tensor([[0.2, -0.1, 0.3], [0.4, 0.5, -0.2]], dtype=torch.float64)
    left = _orthogonal_matrix(2, seed=1)
    right = _orthogonal_matrix(3, seed=2)
    transformed = left @ matrix @ right.mT
    expected = left @ operator(matrix, steps=2) @ right.mT
    torch.testing.assert_close(operator(transformed, steps=2), expected, rtol=1e-11, atol=1e-11)


def test_jordan_matches_scalar_iteration_on_diagonal() -> None:
    diagonal = torch.tensor([0.3, 0.7], dtype=torch.float64)
    matrix = torch.diag(diagonal)
    expected = torch.diag(scalar_response(diagonal, JORDAN_QUINTIC, steps=5))
    torch.testing.assert_close(jordan_ns(matrix, steps=5), expected, rtol=1e-12, atol=1e-12)


def test_exact_polar_matches_compact_svd_and_is_equivariant() -> None:
    matrix = torch.tensor([[2.0, -0.3, 0.1], [0.2, 1.2, 0.4]], dtype=torch.float64)
    u, _, vh = torch.linalg.svd(matrix, full_matrices=False)
    torch.testing.assert_close(exact_polar(matrix), u @ vh)
    left = _orthogonal_matrix(2, seed=3)
    right = _orthogonal_matrix(3, seed=4)
    torch.testing.assert_close(
        exact_polar(left @ matrix @ right.mT),
        left @ exact_polar(matrix) @ right.mT,
        rtol=1e-11,
        atol=1e-11,
    )
