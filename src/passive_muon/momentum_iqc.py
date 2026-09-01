"""Exact IQC certificates for a stylized repaired-floored momentum loop.

The main result in this module is deliberately sector based.  It applies to
any finite-dimensional map that is strongly monotone and Lipschitz; the
floored five-step Jordan map inherits those properties from the rigorous
certificate in :mod:`passive_muon.floored_certificate`.

All proof-replay arithmetic below uses :class:`fractions.Fraction`.  Numerical
SDP solvers were used only to discover the locked rational certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_UPPER,
    certified_dimension_uniform_deficit,
)
from passive_muon.specs import JORDAN_QUINTIC, zero_slope_gain_exact

RationalMatrix = tuple[tuple[Fraction, ...], ...]


def _transpose(matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(row[index] for row in matrix) for index in range(len(matrix[0])))


def _matrix_multiply(left: RationalMatrix, right: RationalMatrix) -> RationalMatrix:
    if len(left[0]) != len(right):
        raise ValueError("matrix dimensions are not aligned")
    return tuple(
        tuple(
            sum(
                (left[row][inner] * right[inner][column] for inner in range(len(right))),
                Fraction(0),
            )
            for column in range(len(right[0]))
        )
        for row in range(len(left))
    )


def _matrix_add(*matrices: RationalMatrix) -> RationalMatrix:
    shape = (len(matrices[0]), len(matrices[0][0]))
    if any((len(matrix), len(matrix[0])) != shape for matrix in matrices):
        raise ValueError("matrix dimensions do not agree")
    return tuple(
        tuple(
            sum((matrix[row][column] for matrix in matrices), Fraction(0))
            for column in range(shape[1])
        )
        for row in range(shape[0])
    )


def _matrix_scale(scale: Fraction, matrix: RationalMatrix) -> RationalMatrix:
    return tuple(tuple(scale * value for value in row) for row in matrix)


def _outer(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> RationalMatrix:
    return tuple(tuple(left_value * right_value for right_value in right) for left_value in left)


def _determinant(matrix: RationalMatrix) -> Fraction:
    if len(matrix) != len(matrix[0]):
        raise ValueError("determinants require a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    return sum(
        (
            Fraction((-1) ** column)
            * matrix[0][column]
            * _determinant(tuple(row[:column] + row[column + 1 :] for row in matrix[1:]))
            for column in range(len(matrix))
        ),
        Fraction(0),
    )


def leading_principal_minors(matrix: RationalMatrix) -> tuple[Fraction, ...]:
    """Return exact leading principal minors in increasing order."""

    if not matrix or len(matrix) != len(matrix[0]):
        raise ValueError("leading principal minors require a nonempty square matrix")
    return tuple(
        _determinant(tuple(row[:order] for row in matrix[:order]))
        for order in range(1, len(matrix) + 1)
    )


def _symmetric_outer(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> RationalMatrix:
    return _matrix_scale(
        Fraction(1, 2),
        _matrix_add(_outer(left, right), _outer(right, left)),
    )


def sector_alpha_limit(beta: Fraction, normalized_strongness: Fraction) -> Fraction:
    """Sharp step limit for the normalized monotone--Lipschitz sector class.

    The canonical loop uses a ``nu``-strongly monotone, one-Lipschitz map and
    dimensionless step ``alpha``.  Strict exponential contraction holds for
    ``0 < alpha < sector_alpha_limit(beta, nu)``.  Equality is the unit-circle
    boundary of an admissible two-dimensional constant skew slope.
    """

    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if not 0 < normalized_strongness <= 1:
        raise ValueError("normalized strong monotonicity must lie in (0, 1]")
    one = Fraction(1)
    numerator = 2 * (one - beta) ** 2 * (one + beta) * normalized_strongness
    denominator = (one + beta) ** 2 - 4 * beta * normalized_strongness**2
    return numerator / denominator


@dataclass(frozen=True)
class MomentumLmiCertificate:
    """A rational certificate for one stylized repaired-Jordan configuration."""

    floor: Fraction
    repair_margin: Fraction
    hessian_lower: Fraction
    hessian_upper: Fraction
    beta: Fraction
    dimensionless_step: Fraction
    conditioning_scale: Fraction
    rate_squared: Fraction
    storage: RationalMatrix
    lambda_strong: Fraction
    lambda_lipschitz: Fraction

    def __post_init__(self) -> None:
        if self.floor <= 0 or self.repair_margin <= 0:
            raise ValueError("floor and repair margin must be positive")
        if self.hessian_lower <= 0 or self.hessian_upper < self.hessian_lower:
            raise ValueError("Hessian bounds must satisfy 0 < ell <= L")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if self.dimensionless_step <= 0 or self.conditioning_scale <= 0:
            raise ValueError("dimensionless step and conditioning scale must be positive")
        if not 0 < self.rate_squared <= 1:
            raise ValueError("rate squared must lie in (0, 1]")
        if len(self.storage) != 2 or any(len(row) != 2 for row in self.storage):
            raise ValueError("storage must be a 2-by-2 matrix")
        if self.storage[0][1] != self.storage[1][0]:
            raise ValueError("storage must be symmetric")
        if self.lambda_strong < 0 or self.lambda_lipschitz < 0:
            raise ValueError("IQC multipliers must be nonnegative")

    @property
    def deficit_upper(self) -> Fraction:
        return certified_dimension_uniform_deficit()

    @property
    def repair_rho(self) -> Fraction:
        return self.deficit_upper / self.floor + self.repair_margin

    @property
    def lipschitz_bound(self) -> Fraction:
        return self.repair_rho + JORDAN_DERIVATIVE_UPPER / self.floor

    @property
    def normalized_strongness(self) -> Fraction:
        return self.repair_margin * self.hessian_lower / (self.lipschitz_bound * self.hessian_upper)

    @property
    def learning_rate(self) -> Fraction:
        return self.dimensionless_step / (self.lipschitz_bound * self.hessian_upper)

    @property
    def certified_learning_rate_supremum(self) -> Fraction:
        return sector_alpha_limit(self.beta, self.normalized_strongness) / (
            self.lipschitz_bound * self.hessian_upper
        )


@dataclass(frozen=True)
class ExactLmiAudit:
    """Exact Sylvester audit of a locked rational momentum certificate."""

    lmi: RationalMatrix
    storage_leading_minors: tuple[Fraction, ...]
    negative_lmi_leading_minors: tuple[Fraction, ...]

    @property
    def storage_positive_definite(self) -> bool:
        return all(value > 0 for value in self.storage_leading_minors)

    @property
    def lmi_negative_definite(self) -> bool:
        return all(value > 0 for value in self.negative_lmi_leading_minors)

    @property
    def certified(self) -> bool:
        return self.storage_positive_definite and self.lmi_negative_definite


@dataclass(frozen=True)
class ClosedFormSectorCertificate:
    """Exact closed-form strict LMI for a point inside the sector region."""

    beta: Fraction
    normalized_strongness: Fraction
    dimensionless_step: Fraction
    epsilon: Fraction
    storage: RationalMatrix
    lambda_strong: Fraction
    lambda_lipschitz: Fraction
    factor_gamma: Fraction
    factor_vector: tuple[Fraction, Fraction, Fraction]


def closed_form_sector_certificate(
    *,
    beta: Fraction,
    normalized_strongness: Fraction,
    dimensionless_step: Fraction,
    epsilon: Fraction = Fraction(1),
) -> ClosedFormSectorCertificate:
    """Construct the rational strict LMI proving the full sector region.

    The lifted state is ``(v_t, e_t)`` with
    ``e_t=v_t-v_(t-1)`` and nonlinear output ``u_t``.  Its transition is
    ``v_next=v+beta*e-alpha*u`` and ``e_next=beta*e-alpha*u``.

    Direct exact expansion of the returned quantities gives

    ``M = -epsilon*I - factor_gamma*n*n.T``.

    This identity makes strict feasibility solver independent.  It also keeps
    the strong-monotonicity and Lipschitz IQCs separate.
    """

    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if not 0 < normalized_strongness <= 1:
        raise ValueError("normalized strong monotonicity must lie in (0, 1]")
    if not 0 < dimensionless_step < sector_alpha_limit(beta, normalized_strongness):
        raise ValueError("dimensionless step must lie strictly inside the sector region")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    alpha = dimensionless_step
    nu = normalized_strongness
    one = Fraction(1)
    d = one - beta
    s = one + beta
    denominator = s**2 - 4 * beta * nu**2
    gap = 2 * nu * d**2 * s - alpha * denominator

    a_p = alpha**2 * (s**2 + 4 * beta * nu**2) + 8 * alpha * beta * nu * s + 2 * d**2 * s**2
    a_q = alpha**2 * (s**2 - 4 * nu**2) - 4 * alpha * nu * d * s + 2 * d**2 * s**2
    a_r = 2 * beta * d * s**2 + 2 * alpha * nu * d * s - alpha**2 * denominator
    a_l = 2 * alpha**2 * nu * s + alpha * (s**2 + 4 * beta * nu**2) + 2 * nu * d**2 * s

    common = epsilon / (alpha * gap)
    storage = (
        (common * d * a_p / s, common * beta * a_q / s),
        (common * beta * a_q / s, common * a_r),
    )
    lambda_strong = 2 * epsilon * a_p / (s * gap)
    lambda_lipschitz = epsilon * a_l / gap
    factor_gamma = 2 * beta * epsilon * d**2 * s * (alpha * nu + s) / (alpha * gap)
    factor_vector = (2 * alpha * nu / (d * s), -one, -alpha / d)
    return ClosedFormSectorCertificate(
        beta=beta,
        normalized_strongness=nu,
        dimensionless_step=alpha,
        epsilon=epsilon,
        storage=storage,
        lambda_strong=lambda_strong,
        lambda_lipschitz=lambda_lipschitz,
        factor_gamma=factor_gamma,
        factor_vector=factor_vector,
    )


def closed_form_sector_lmi_matrix(
    certificate: ClosedFormSectorCertificate,
) -> RationalMatrix:
    """Build the exact rate-one LMI in ``(v,e,u)`` coordinates."""

    alpha = certificate.dimensionless_step
    beta = certificate.beta
    zero = Fraction(0)
    one = Fraction(1)
    transition = (
        (one, beta, -alpha),
        (zero, beta, -alpha),
    )
    selection = ((one, zero, zero), (zero, one, zero))
    nonlinear_input = (one, zero, zero)
    nonlinear_output = (zero, zero, one)
    strong_iqc = _matrix_add(
        _symmetric_outer(nonlinear_input, nonlinear_output),
        _matrix_scale(
            -certificate.normalized_strongness,
            _outer(nonlinear_input, nonlinear_input),
        ),
    )
    lipschitz_iqc = _matrix_add(
        _outer(nonlinear_input, nonlinear_input),
        _matrix_scale(-one, _outer(nonlinear_output, nonlinear_output)),
    )
    return _matrix_add(
        _matrix_multiply(
            _transpose(transition),
            _matrix_multiply(certificate.storage, transition),
        ),
        _matrix_scale(
            -one,
            _matrix_multiply(
                _transpose(selection),
                _matrix_multiply(certificate.storage, selection),
            ),
        ),
        _matrix_scale(certificate.lambda_strong, strong_iqc),
        _matrix_scale(certificate.lambda_lipschitz, lipschitz_iqc),
    )


def closed_form_factor_matrix(certificate: ClosedFormSectorCertificate) -> RationalMatrix:
    """Return ``-epsilon*I-gamma*n*n.T`` for identity replay."""

    identity = tuple(tuple(Fraction(int(row == column)) for column in range(3)) for row in range(3))
    return _matrix_add(
        _matrix_scale(-certificate.epsilon, identity),
        _matrix_scale(
            -certificate.factor_gamma,
            _outer(certificate.factor_vector, certificate.factor_vector),
        ),
    )


def locked_momentum_certificate() -> MomentumLmiCertificate:
    """Return the exact representative certificate used by the artifact."""

    deficit = certified_dimension_uniform_deficit()
    return MomentumLmiCertificate(
        floor=Fraction(1),
        repair_margin=deficit,
        hessian_lower=Fraction(1),
        hessian_upper=Fraction(10),
        beta=Fraction(9, 10),
        dimensionless_step=Fraction(1, 10_000),
        conditioning_scale=Fraction(1, 10),
        rate_squared=Fraction(99_999, 100_000),
        storage=(
            (Fraction(739_113, 1_000_000), Fraction(-260_843, 1_000_000)),
            (Fraction(-260_843, 1_000_000), Fraction(260_887, 1_000_000)),
        ),
        lambda_strong=Fraction(957, 1_000_000),
        lambda_lipschitz=Fraction(11, 1_000_000),
    )


def momentum_lmi_matrix(certificate: MomentumLmiCertificate) -> RationalMatrix:
    """Build the exact 3-by-3 conditioned IQC/Lyapunov LMI.

    With ``gamma=conditioning_scale``, the lifted difference is
    ``chi=(y, r, u_hat)`` where ``r=gamma*z``, the incremental nonlinear
    input is ``v_hat=gamma*y+beta*r``, and ``u_hat=gamma*u`` is the
    corresponding scaled output difference. The transition is

    ``(y_next, r_next)=(y-alpha*u_hat/gamma, v_hat)``.

    The two nonnegative IQCs are kept separate: strong monotonicity
    ``<v,u>-nu*||v||^2`` and Lipschitzness ``||v||^2-||u||^2``.  Combining
    them into the cocoercive gradient-sector IQC would be invalid for the
    generally nonsymmetric repaired map.
    """

    alpha = certificate.dimensionless_step
    beta = certificate.beta
    gamma = certificate.conditioning_scale
    nu = certificate.normalized_strongness
    if alpha <= 0:
        raise ValueError("dimensionless step must be positive")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if gamma <= 0:
        raise ValueError("conditioning scale must be positive")
    if not 0 < certificate.rate_squared <= 1:
        raise ValueError("rate squared must lie in (0, 1]")
    if certificate.lambda_strong < 0 or certificate.lambda_lipschitz < 0:
        raise ValueError("IQC multipliers must be nonnegative")

    zero = Fraction(0)
    one = Fraction(1)
    transition = (
        (one, zero, -alpha / gamma),
        (gamma, beta, zero),
    )
    selection = ((one, zero, zero), (zero, one, zero))
    nonlinear_input = (gamma, beta, zero)
    nonlinear_output = (zero, zero, one)
    strong_iqc = _matrix_add(
        _symmetric_outer(nonlinear_input, nonlinear_output),
        _matrix_scale(-nu, _outer(nonlinear_input, nonlinear_input)),
    )
    lipschitz_iqc = _matrix_add(
        _outer(nonlinear_input, nonlinear_input),
        _matrix_scale(-one, _outer(nonlinear_output, nonlinear_output)),
    )
    return _matrix_add(
        _matrix_multiply(
            _transpose(transition),
            _matrix_multiply(certificate.storage, transition),
        ),
        _matrix_scale(
            -certificate.rate_squared,
            _matrix_multiply(
                _transpose(selection),
                _matrix_multiply(certificate.storage, selection),
            ),
        ),
        _matrix_scale(certificate.lambda_strong, strong_iqc),
        _matrix_scale(certificate.lambda_lipschitz, lipschitz_iqc),
    )


def audit_momentum_certificate(
    certificate: MomentumLmiCertificate | None = None,
) -> ExactLmiAudit:
    """Replay the exact rational proof for ``certificate``."""

    selected = locked_momentum_certificate() if certificate is None else certificate
    lmi = momentum_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return ExactLmiAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(selected.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
    )


def repaired_zero_input_gain(*, floor: Fraction, repair_rho: Fraction) -> Fraction:
    """Derivative of the repaired five-step Jordan map at zero."""

    if floor <= 0:
        raise ValueError("floor must be positive")
    if repair_rho < 0:
        raise ValueError("repair conductance must be nonnegative")
    return zero_slope_gain_exact(JORDAN_QUINTIC, steps=5) / floor + repair_rho


def scalar_momentum_jury_margin(
    *,
    learning_rate: Fraction,
    beta: Fraction,
    curvature: Fraction,
    repaired_gain: Fraction,
) -> Fraction:
    """Return the exact upper Jury margin at the optimizer.

    For the scalar invariant quadratic mode, local Schur stability requires
    ``learning_rate*curvature*repaired_gain < 2*(1+beta)``.  A negative return
    value therefore proves local instability of the actual nonlinear loop.
    """

    if learning_rate <= 0 or curvature <= 0 or repaired_gain <= 0:
        raise ValueError("learning rate, curvature, and gain must be positive")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    return 2 * (1 + beta) - learning_rate * curvature * repaired_gain


def fraction_matrix_to_strings(matrix: RationalMatrix) -> list[list[str]]:
    """Serialize an exact matrix without losing rational structure."""

    return [[str(value) for value in row] for row in matrix]
