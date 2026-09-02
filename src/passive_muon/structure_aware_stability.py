"""Exact scalar-mode small-gain certificate for the EMA/Nesterov loop.

This module keeps the structure that is discarded by the generic sector IQC:
the derivative of the repaired operator is split into a positive scalar center
``g I`` and a residual with certified Lipschitz bound ``l``.  For each Hessian
eigenvalue ``lambda`` the centered EMA/Nesterov loop has the scalar transfer

``G_lambda(z) = -x*(1-beta)*((1+beta)*z-beta) / D(z, x)``,

where ``x=eta*lambda`` and

``D(z,x)=z^2+(1+beta)*(g*(1-beta)*x-1)*z+beta*(1-g*(1-beta)*x)``.

The proof replay below is exact rational arithmetic.  Scaled Jury inequalities
place the roots of ``D`` inside ``|z| < tau``.  On ``z=tau*exp(i*omega)``, the
small-gain gap is a quadratic ``Q(t,x)`` in ``t=1-cos(omega)``.  Exact positive
Bernstein coefficients for its leading coefficient and negative discriminant
prove ``Q>0`` throughout the locked curvature interval.

This is a certificate for the stated centered residual model.  The spectral
center and residual bounds are supplied by the full-matrix floored certificate;
they are constants and are never fitted from the current input.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb
from typing import TypeAlias

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_LOWER,
    JORDAN_DERIVATIVE_UPPER,
    certified_dimension_uniform_deficit,
)

Polynomial: TypeAlias = tuple[Fraction, ...]
Rational: TypeAlias = int | Fraction


def _trim_polynomial(coefficients: tuple[Fraction, ...]) -> Polynomial:
    if not coefficients:
        return (Fraction(0),)
    last = len(coefficients) - 1
    while last > 0 and coefficients[last] == 0:
        last -= 1
    return coefficients[: last + 1]


def polynomial(coefficients: tuple[Rational, ...] | list[Rational]) -> Polynomial:
    """Return canonical low-to-high exact power-basis coefficients."""

    return _trim_polynomial(tuple(Fraction(value) for value in coefficients))


def polynomial_add(left: Polynomial, right: Polynomial) -> Polynomial:
    """Add two exact power-basis polynomials."""

    size = max(len(left), len(right))
    return _trim_polynomial(
        tuple(
            (left[index] if index < len(left) else Fraction(0))
            + (right[index] if index < len(right) else Fraction(0))
            for index in range(size)
        )
    )


def polynomial_scale(scale: Rational, coefficients: Polynomial) -> Polynomial:
    """Multiply an exact polynomial by a scalar."""

    factor = Fraction(scale)
    return _trim_polynomial(tuple(factor * value for value in coefficients))


def polynomial_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    """Multiply two exact power-basis polynomials."""

    output = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output[left_index + right_index] += left_value * right_value
    return _trim_polynomial(tuple(output))


def polynomial_evaluate(coefficients: Polynomial, value: Rational) -> Fraction:
    """Evaluate an exact power-basis polynomial by Horner's rule."""

    point = Fraction(value)
    result = Fraction(0)
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


def polynomial_compose_affine(
    coefficients: Polynomial,
    *,
    offset: Rational,
    scale: Rational,
) -> Polynomial:
    """Return the power coefficients of ``p(offset + scale*u)``."""

    affine = (Fraction(offset), Fraction(scale))
    result = (Fraction(0),)
    for coefficient in reversed(coefficients):
        result = polynomial_add(
            (coefficient,),
            polynomial_multiply(result, affine),
        )
    return result


def bernstein_coefficients(
    coefficients: Polynomial,
    *,
    interval_lower: Rational,
    interval_upper: Rational,
    degree: int | None = None,
) -> tuple[Fraction, ...]:
    """Convert an exact power polynomial to Bernstein form on an interval.

    The returned tuple ``b_0,...,b_n`` satisfies

    ``p(lower + (upper-lower)*u) = sum_i b_i B_i^n(u)``.

    Strict positivity of every returned coefficient therefore proves strict
    positivity of ``p`` on the entire closed interval.
    """

    lower = Fraction(interval_lower)
    upper = Fraction(interval_upper)
    if lower >= upper:
        raise ValueError("Bernstein interval must have positive width")
    canonical = _trim_polynomial(tuple(Fraction(value) for value in coefficients))
    polynomial_degree = len(canonical) - 1
    selected_degree = polynomial_degree if degree is None else degree
    if selected_degree < polynomial_degree:
        raise ValueError("Bernstein degree cannot be below the polynomial degree")

    mapped = polynomial_compose_affine(
        canonical,
        offset=lower,
        scale=upper - lower,
    )
    mapped_power = mapped + (Fraction(0),) * (selected_degree + 1 - len(mapped))
    return tuple(
        sum(
            (
                mapped_power[power] * Fraction(comb(index, power), comb(selected_degree, power))
                for power in range(index + 1)
            ),
            Fraction(0),
        )
        for index in range(selected_degree + 1)
    )


@dataclass(frozen=True)
class CenteredSmallGainParameters:
    """Exact parameters for one centered EMA/Nesterov small-gain audit."""

    curvature_lower: Fraction
    curvature_upper: Fraction
    beta: Fraction
    center_gain: Fraction
    residual_lipschitz: Fraction
    learning_rate: Fraction
    tau: Fraction

    def __post_init__(self) -> None:
        if self.curvature_lower <= 0 or self.curvature_upper < self.curvature_lower:
            raise ValueError("curvatures must satisfy 0 < ell <= L")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if self.center_gain <= 0:
            raise ValueError("center gain must be positive")
        if self.residual_lipschitz < 0:
            raise ValueError("residual Lipschitz bound must be nonnegative")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        if not 0 < self.tau < 1:
            raise ValueError("tau must lie in (0, 1)")

    @property
    def scaled_curvature_lower(self) -> Fraction:
        return self.learning_rate * self.curvature_lower

    @property
    def scaled_curvature_upper(self) -> Fraction:
        return self.learning_rate * self.curvature_upper


@dataclass(frozen=True)
class ScaledJuryPolynomials:
    """The three exact Jury polynomials after scaling ``z=tau*w``."""

    positive_rate: Polynomial
    negative_rate: Polynomial
    constant_radius: Polynomial


@dataclass(frozen=True)
class FrequencyPolynomials:
    """Coefficients of ``Q(t,x)=q0(x)+q1(x)t+q2(x)t^2``."""

    q0: Polynomial
    q1: Polynomial
    q2: Polynomial
    vertex: Polynomial


@dataclass(frozen=True)
class BernsteinPositivityCertificate:
    """Exact Bernstein replay for one polynomial on one closed interval."""

    name: str
    interval_lower: Fraction
    interval_upper: Fraction
    power_coefficients: Polynomial
    coefficients: tuple[Fraction, ...]

    @property
    def strictly_positive(self) -> bool:
        return all(value > 0 for value in self.coefficients)

    @property
    def minimum_coefficient(self) -> Fraction:
        return min(self.coefficients)


@dataclass(frozen=True)
class StructureAwareStabilityAudit:
    """Complete exact replay of the centered scalar-mode rate certificate."""

    parameters: CenteredSmallGainParameters
    denominator_linear: Polynomial
    denominator_constant: Polynomial
    jury_polynomials: ScaledJuryPolynomials
    jury_certificates: tuple[BernsteinPositivityCertificate, ...]
    frequency_polynomials: FrequencyPolynomials
    q2_certificate: BernsteinPositivityCertificate
    vertex_certificate: BernsteinPositivityCertificate

    @property
    def jury_certified(self) -> bool:
        return all(certificate.strictly_positive for certificate in self.jury_certificates)

    @property
    def frequency_certified(self) -> bool:
        return self.q2_certificate.strictly_positive and self.vertex_certificate.strictly_positive

    @property
    def certified(self) -> bool:
        return self.jury_certified and self.frequency_certified


def denominator_polynomials(
    parameters: CenteredSmallGainParameters,
) -> tuple[Polynomial, Polynomial]:
    """Return ``A(x), B(x)`` for ``D(z,x)=z^2+A(x)z+B(x)``."""

    beta = parameters.beta
    one_minus_beta = 1 - beta
    centered_scale = parameters.center_gain * one_minus_beta
    linear = (
        -(1 + beta),
        (1 + beta) * centered_scale,
    )
    constant = (
        beta,
        -beta * centered_scale,
    )
    return linear, constant


def scaled_jury_polynomials(
    parameters: CenteredSmallGainParameters,
) -> ScaledJuryPolynomials:
    """Build the strict Jury conditions for roots inside ``|z|<rate``."""

    linear, constant = denominator_polynomials(parameters)
    tau = parameters.tau
    tau_squared = tau**2
    positive_rate = polynomial_add(
        (tau_squared,),
        polynomial_add(polynomial_scale(tau, linear), constant),
    )
    negative_rate = polynomial_add(
        (tau_squared,),
        polynomial_add(polynomial_scale(-tau, linear), constant),
    )
    constant_radius = polynomial_add((tau_squared,), polynomial_scale(-1, constant))
    return ScaledJuryPolynomials(
        positive_rate=positive_rate,
        negative_rate=negative_rate,
        constant_radius=constant_radius,
    )


def frequency_polynomials(
    parameters: CenteredSmallGainParameters,
) -> FrequencyPolynomials:
    """Build the exact circle small-gain polynomial and vertex numerator.

    With ``t=1-cos(omega)``, the gap on ``z=rate*exp(i*omega)`` is
    ``Q=q0+q1*t+q2*t^2``.  The polynomial named ``vertex`` is
    ``4*q2*q0-q1^2``.  Strict positivity of ``q2`` and ``vertex`` makes this
    quadratic positive for every real ``t``, hence in particular for
    ``0 <= t <= 2``.
    """

    linear, constant = denominator_polynomials(parameters)
    beta = parameters.beta
    tau = parameters.tau
    residual = parameters.residual_lipschitz
    one_minus_beta = 1 - beta
    x_squared = (Fraction(0), Fraction(0), Fraction(1))

    jury_at_rate = polynomial_add(
        (tau**2,),
        polynomial_add(polynomial_scale(tau, linear), constant),
    )
    numerator_at_rate_squared = (residual * one_minus_beta * ((1 + beta) * tau - beta)) ** 2
    q0 = polynomial_add(
        polynomial_multiply(jury_at_rate, jury_at_rate),
        polynomial_scale(-numerator_at_rate_squared, x_squared),
    )

    rate_squared_plus_constant = polynomial_add((tau**2,), constant)
    q1 = polynomial_add(
        polynomial_scale(
            -2 * tau,
            polynomial_multiply(linear, rate_squared_plus_constant),
        ),
        polynomial_add(
            polynomial_scale(-8 * tau**2, constant),
            polynomial_scale(
                -2 * residual**2 * one_minus_beta**2 * beta * (1 + beta) * tau,
                x_squared,
            ),
        ),
    )
    q2 = polynomial_scale(4 * tau**2, constant)
    vertex = polynomial_add(
        polynomial_scale(4, polynomial_multiply(q2, q0)),
        polynomial_scale(-1, polynomial_multiply(q1, q1)),
    )
    return FrequencyPolynomials(q0=q0, q1=q1, q2=q2, vertex=vertex)


def _positivity_certificate(
    name: str,
    coefficients: Polynomial,
    parameters: CenteredSmallGainParameters,
) -> BernsteinPositivityCertificate:
    lower = parameters.scaled_curvature_lower
    upper = parameters.scaled_curvature_upper
    return BernsteinPositivityCertificate(
        name=name,
        interval_lower=lower,
        interval_upper=upper,
        power_coefficients=coefficients,
        coefficients=bernstein_coefficients(
            coefficients,
            interval_lower=lower,
            interval_upper=upper,
        ),
    )


def audit_structure_aware_stability(
    parameters: CenteredSmallGainParameters,
) -> StructureAwareStabilityAudit:
    """Replay every exact Jury and circle-gap proof obligation."""

    linear, constant = denominator_polynomials(parameters)
    jury = scaled_jury_polynomials(parameters)
    jury_certificates = (
        _positivity_certificate("D(+tau)", jury.positive_rate, parameters),
        _positivity_certificate("D(-tau)", jury.negative_rate, parameters),
        _positivity_certificate("tau^2-B", jury.constant_radius, parameters),
    )
    frequency = frequency_polynomials(parameters)
    return StructureAwareStabilityAudit(
        parameters=parameters,
        denominator_linear=linear,
        denominator_constant=constant,
        jury_polynomials=jury,
        jury_certificates=jury_certificates,
        frequency_polynomials=frequency,
        q2_certificate=_positivity_certificate("q2", frequency.q2, parameters),
        vertex_certificate=_positivity_certificate(
            "4*q2*q0-q1^2",
            frequency.vertex,
            parameters,
        ),
    )


def transfer_numerator(
    z: complex,
    *,
    scaled_curvature: Rational,
    beta: Rational,
) -> complex:
    """Evaluate the centered-loop transfer numerator."""

    x = Fraction(scaled_curvature)
    beta_fraction = Fraction(beta)
    return -x * (1 - beta_fraction) * ((1 + beta_fraction) * z - beta_fraction)


def transfer_denominator(
    z: complex,
    *,
    scaled_curvature: Rational,
    beta: Rational,
    center_gain: Rational,
) -> complex:
    """Evaluate the centered-loop transfer denominator."""

    x = Fraction(scaled_curvature)
    beta_fraction = Fraction(beta)
    gain = Fraction(center_gain)
    centered_step = gain * (1 - beta_fraction) * x
    return (
        z**2 + (1 + beta_fraction) * (centered_step - 1) * z + beta_fraction * (1 - centered_step)
    )


LOCKED_FLOOR = Fraction(1)
LOCKED_HESSIAN_LOWER = Fraction(1)
LOCKED_HESSIAN_UPPER = Fraction(10)
LOCKED_BETA = Fraction(19, 20)
LOCKED_REPAIR_MARGIN = Fraction(648)
LOCKED_DEFICIT_UPPER = certified_dimension_uniform_deficit()
LOCKED_REPAIR_RHO = LOCKED_DEFICIT_UPPER / LOCKED_FLOOR + LOCKED_REPAIR_MARGIN
LOCKED_SYMMETRIC_GAIN_LOWER = LOCKED_REPAIR_MARGIN
LOCKED_SYMMETRIC_GAIN_UPPER = LOCKED_REPAIR_RHO + JORDAN_DERIVATIVE_UPPER / LOCKED_FLOOR
LOCKED_CENTER_GAIN = (LOCKED_SYMMETRIC_GAIN_LOWER + LOCKED_SYMMETRIC_GAIN_UPPER) / 2
LOCKED_SKEW_NORM_UPPER = (JORDAN_DERIVATIVE_UPPER - JORDAN_DERIVATIVE_LOWER) / (4 * LOCKED_FLOOR)
LOCKED_RESIDUAL_LIPSCHITZ = (
    LOCKED_SYMMETRIC_GAIN_UPPER - LOCKED_SYMMETRIC_GAIN_LOWER
) / 2 + LOCKED_SKEW_NORM_UPPER
LOCKED_LEARNING_RATE = Fraction(1, 32_000)
LOCKED_TAU = Fraction(99_999, 100_000)


def locked_structure_aware_parameters() -> CenteredSmallGainParameters:
    """Return the exact representative selected for the p4 proof replay."""

    return CenteredSmallGainParameters(
        curvature_lower=LOCKED_HESSIAN_LOWER,
        curvature_upper=LOCKED_HESSIAN_UPPER,
        beta=LOCKED_BETA,
        center_gain=LOCKED_CENTER_GAIN,
        residual_lipschitz=LOCKED_RESIDUAL_LIPSCHITZ,
        learning_rate=LOCKED_LEARNING_RATE,
        tau=LOCKED_TAU,
    )


def locked_structure_aware_certificate() -> StructureAwareStabilityAudit:
    """Return the exact locked structure-aware rate certificate audit."""

    return audit_structure_aware_stability(locked_structure_aware_parameters())
