from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from passive_muon.normalizers import (
    BF16FrobeniusPlusEpsNormalizer,
    ExactFrobeniusNormalizer,
    FixedScaleNormalizer,
    FrobeniusPlusEpsNormalizer,
)


def test_exact_frobenius_has_unit_norm_and_rejects_zero() -> None:
    normalizer = ExactFrobeniusNormalizer()
    matrix = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    assert torch.linalg.vector_norm(normalizer(matrix)).item() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="undefined at zero"):
        normalizer(torch.zeros((2, 2), dtype=torch.float64))


@given(scale=st.floats(min_value=1e-4, max_value=1e4, allow_nan=False))
@settings(max_examples=20, deadline=None)
def test_exact_frobenius_is_positive_scale_invariant(scale: float) -> None:
    matrix = torch.tensor([[0.3, -0.1], [0.2, 0.7]], dtype=torch.float64)
    normalizer = ExactFrobeniusNormalizer()
    torch.testing.assert_close(normalizer(scale * matrix), normalizer(matrix))


def test_epsilon_and_fixed_normalizers_are_distinct_policies() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    epsilon = FrobeniusPlusEpsNormalizer(eps=0.5)(matrix)
    fixed = FixedScaleNormalizer(scale=5.0)(matrix)
    assert torch.linalg.vector_norm(epsilon).item() == pytest.approx(5 / 5.5)
    torch.testing.assert_close(fixed, matrix / 5)


def test_deployed_normalizer_casts_before_measuring_norm() -> None:
    matrix = torch.tensor([[3.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    normalized = BF16FrobeniusPlusEpsNormalizer()(matrix)
    assert normalized.dtype == torch.bfloat16
    assert torch.linalg.vector_norm(normalized).float().item() == pytest.approx(1.0, abs=5e-3)
