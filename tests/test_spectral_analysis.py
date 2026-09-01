from __future__ import annotations

import numpy as np
import pytest
import torch

from passive_muon.normalizers import ExactFrobeniusNormalizer, FrobeniusPlusEpsNormalizer
from passive_muon.operator import NormalizedMap
from passive_muon.orthogonalizers import jordan_ns, polar_express
from passive_muon.passivity import local_deficit_at
from passive_muon.specs import JORDAN_QUINTIC, POLAR_EXPRESS_5
from passive_muon.spectral_analysis import (
    epsilon_normalized_local_minimum,
    epsilon_normalized_local_spectrum,
    exact_normalized_local_spectrum,
)


def test_spectral_reduction_matches_full_torch_jacobian() -> None:
    angle = np.array([np.arctan2(0.8, 0.6)], dtype=np.float64)
    reduced = exact_normalized_local_spectrum(angle, JORDAN_QUINTIC, steps=5)
    matrix = torch.diag(torch.tensor([0.6, 0.8], dtype=torch.float64))
    operator = NormalizedMap(ExactFrobeniusNormalizer(), lambda value: jordan_ns(value, steps=5))
    full = local_deficit_at(operator, matrix)
    assert reduced.minimum_eigenvalue[0] == pytest.approx(full.minimum_eigenvalue.item())


def test_epsilon_deficit_obeys_exact_scaling_identity() -> None:
    angles = np.array([[0.4, 0.8]], dtype=np.float64)
    radii = np.array([[0.2], [1.7]], dtype=np.float64)
    reference = epsilon_normalized_local_minimum(angles, radii, JORDAN_QUINTIC, steps=5, eps=1.0)
    eps = 0.25
    scaled = epsilon_normalized_local_minimum(angles, eps * radii, JORDAN_QUINTIC, steps=5, eps=eps)
    np.testing.assert_allclose(scaled, reference / eps, rtol=1e-12, atol=1e-12)


def test_staged_epsilon_reduction_matches_full_torch_jacobian() -> None:
    angle = 0.63
    radius = 0.8
    reduced = epsilon_normalized_local_spectrum(
        np.array([[angle]]),
        np.array([[radius]]),
        POLAR_EXPRESS_5,
        steps=5,
        eps=1.0,
    )
    matrix = torch.diag(radius * torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float64))
    operator = NormalizedMap(
        FrobeniusPlusEpsNormalizer(eps=1.0),
        lambda value: polar_express(value, steps=5),
    )
    full = local_deficit_at(operator, matrix)
    assert reduced.minimum_eigenvalue.item() == pytest.approx(
        full.minimum_eigenvalue.item(), rel=1e-10, abs=1e-10
    )
