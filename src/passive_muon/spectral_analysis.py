"""Fast spectral reductions for deficit experiments on 2x2 diagonal inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from passive_muon.specs import PolynomialCoefficients, coefficient_stages

FloatArray = NDArray[np.float64]


def scalar_response_and_derivative(
    values: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    steps: int,
) -> tuple[FloatArray, FloatArray]:
    """Vectorized composed response and derivative in float64."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    response = np.asarray(values, dtype=np.float64).copy()
    derivative = np.ones_like(response)
    for stage in coefficient_stages(coefficients, steps=steps):
        a, b, c = (float(value) for value in stage.fractions())
        derivative *= a + 3 * b * response**2 + 5 * c * response**4
        response = a * response + b * response**3 + c * response**5
    return response, derivative


def _divided_difference(
    left_x: FloatArray,
    right_x: FloatArray,
    left_y: FloatArray,
    right_y: FloatArray,
    derivative_at_tie: FloatArray,
) -> FloatArray:
    denominator = left_x - right_x
    output = np.empty_like(denominator)
    tied = np.abs(denominator) <= 1e-12
    np.divide(left_y - right_y, denominator, out=output, where=~tied)
    output[tied] = derivative_at_tie[tied]
    return output


def fixed_scale_spectral_minimum(
    singular_1: FloatArray,
    singular_2: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    steps: int,
) -> FloatArray:
    """Minimum full Jacobian eigenvalue of a 2x2 spectral polynomial map."""

    h1, d1 = scalar_response_and_derivative(singular_1, coefficients, steps=steps)
    h2, d2 = scalar_response_and_derivative(singular_2, coefficients, steps=steps)
    difference_mode = _divided_difference(singular_1, singular_2, h1, h2, d1)
    sum_mode = (h1 + h2) / (singular_1 + singular_2)
    return np.minimum.reduce([d1, d2, difference_mode, sum_mode])


@dataclass(frozen=True)
class LocalSpectrum:
    minimum_eigenvalue: FloatArray
    diagonal_minimum: FloatArray
    fixed_scale_minimum: FloatArray


@dataclass(frozen=True)
class EpsilonLocalSpectrum:
    """Local minima for epsilon normalization, including the diagonal subspace."""

    minimum_eigenvalue: FloatArray
    diagonal_minimum: FloatArray


def exact_normalized_local_spectrum(
    angles: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    steps: int,
    radius: float = 1.0,
) -> LocalSpectrum:
    """Full local spectrum minimum for exact current Frobenius normalization."""

    if radius <= 0:
        raise ValueError("radius must be positive")
    u1, u2 = np.cos(angles), np.sin(angles)
    h1, d1 = scalar_response_and_derivative(u1, coefficients, steps=steps)
    h2, d2 = scalar_response_and_derivative(u2, coefficients, steps=steps)

    s11 = d1 * u2**2 / radius
    s22 = d2 * u1**2 / radius
    s12 = -(d1 + d2) * u1 * u2 / (2 * radius)
    trace = s11 + s22
    diagonal_minimum = (trace - np.sqrt((s11 - s22) ** 2 + 4 * s12**2)) / 2

    difference_raw = _divided_difference(u1, u2, h1, h2, d1)
    sum_raw = (h1 + h2) / (u1 + u2)
    difference_mode = difference_raw / radius
    sum_mode = sum_raw / radius
    fixed_minimum = np.minimum.reduce([d1, d2, difference_raw, sum_raw]) / radius
    full_minimum = np.minimum.reduce([diagonal_minimum, difference_mode, sum_mode])
    return LocalSpectrum(full_minimum, diagonal_minimum, fixed_minimum)


def epsilon_normalized_local_minimum(
    angles: FloatArray,
    radii: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    steps: int,
    eps: float,
) -> FloatArray:
    """Full minimum for ``H(M / (||M||_F + eps))`` on a polar grid.

    ``angles`` and ``radii`` must be broadcast-compatible arrays.
    """

    return epsilon_normalized_local_spectrum(
        angles,
        radii,
        coefficients,
        steps=steps,
        eps=eps,
    ).minimum_eigenvalue


def epsilon_normalized_local_spectrum(
    angles: FloatArray,
    radii: FloatArray,
    coefficients: PolynomialCoefficients,
    *,
    steps: int,
    eps: float,
) -> EpsilonLocalSpectrum:
    """Full and diagonal-subspace minima for current-plus-epsilon normalization."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if np.any(radii <= 0):
        raise ValueError("radii must be positive")

    u1, u2 = np.cos(angles), np.sin(angles)
    denominator = radii + eps
    alpha = radii / denominator
    z1, z2 = alpha * u1, alpha * u2
    h1, d1 = scalar_response_and_derivative(z1, coefficients, steps=steps)
    h2, d2 = scalar_response_and_derivative(z2, coefficients, steps=steps)

    b11 = (1 - alpha * u1**2) / denominator
    b22 = (1 - alpha * u2**2) / denominator
    b12 = -alpha * u1 * u2 / denominator
    s11 = d1 * b11
    s22 = d2 * b22
    s12 = (d1 + d2) * b12 / 2
    trace = s11 + s22
    diagonal_minimum = (trace - np.sqrt((s11 - s22) ** 2 + 4 * s12**2)) / 2

    difference_mode = _divided_difference(z1, z2, h1, h2, d1) / denominator
    sum_mode = (h1 + h2) / (z1 + z2) / denominator
    full_minimum = np.minimum.reduce([diagonal_minimum, difference_mode, sum_mode])
    return EpsilonLocalSpectrum(full_minimum, diagonal_minimum)
