"""Exact IQC replay for the pinned Muon EMA/Nesterov signal ordering.

The theorem in this module concerns the deterministic real-arithmetic loop

``m_next = beta*m + (1-beta)*gradient``
``signal = beta*m_next + (1-beta)*gradient``
``W_next = W - eta*R(signal)``.

It matches the pinned upstream state and signal ordering after replacing the
upstream orthogonalizer by the repaired floored operator.  It does not model
BF16 arithmetic, weight decay, or the rest of a training system.

All proof-replay arithmetic uses :class:`fractions.Fraction`.  A floating-point
SDP was used only to discover the committed rational storage and multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_UPPER,
    certified_dimension_uniform_deficit,
)
from passive_muon.momentum_iqc import ExactLmiAudit, RationalMatrix, leading_principal_minors


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


def _symmetric_outer(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> RationalMatrix:
    return _matrix_scale(
        Fraction(1, 2),
        _matrix_add(_outer(left, right), _outer(right, left)),
    )


@dataclass(frozen=True)
class EmaNesterovLmiCertificate:
    """One exact rational certificate for the repaired EMA/Nesterov loop."""

    floor: Fraction
    repair_margin: Fraction
    hessian_lower: Fraction
    hessian_upper: Fraction
    beta: Fraction
    dimensionless_step: Fraction
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
        if self.dimensionless_step <= 0:
            raise ValueError("dimensionless step must be positive")
        if not 0 < self.rate_squared < 1:
            raise ValueError("rate squared must lie in (0, 1)")
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


def locked_ema_nesterov_certificate() -> EmaNesterovLmiCertificate:
    """Return the exact beta=0.95 representative selected after margin search."""

    return EmaNesterovLmiCertificate(
        floor=Fraction(1),
        repair_margin=Fraction(648),
        hessian_lower=Fraction(1),
        hessian_upper=Fraction(10),
        beta=Fraction(19, 20),
        dimensionless_step=Fraction(1, 400),
        rate_squared=Fraction(99_999, 100_000),
        storage=(
            (Fraction(68_309, 100_000), Fraction(-315_367, 1_000_000)),
            (Fraction(-315_367, 1_000_000), Fraction(31_691, 100_000)),
        ),
        lambda_strong=Fraction(1_847, 1_000_000),
        lambda_lipschitz=Fraction(43, 500_000),
    )


def ema_nesterov_lmi_matrix(certificate: EmaNesterovLmiCertificate) -> RationalMatrix:
    """Build the exact conditioned ``3 x 3`` EMA/Nesterov IQC LMI.

    In the lifted coordinates ``chi=(y,z,u)``, the nonlinear input is
    ``p=(1-beta**2)*y+beta**2*z`` and the state transition is
    ``y_next=y-alpha*u``, ``z_next=(1-beta)*y+beta*z``.
    """

    alpha = certificate.dimensionless_step
    beta = certificate.beta
    nu = certificate.normalized_strongness
    zero = Fraction(0)
    one = Fraction(1)
    transition = (
        (one, zero, -alpha),
        (one - beta, beta, zero),
    )
    selection = ((one, zero, zero), (zero, one, zero))
    nonlinear_input = (one - beta**2, beta**2, zero)
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


def audit_ema_nesterov_certificate(
    certificate: EmaNesterovLmiCertificate | None = None,
) -> ExactLmiAudit:
    """Replay positive storage and negative LMI by exact Sylvester checks."""

    selected = locked_ema_nesterov_certificate() if certificate is None else certificate
    lmi = ema_nesterov_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return ExactLmiAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(selected.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
    )


def ema_nesterov_jury_margin(
    *,
    learning_rate: Fraction,
    beta: Fraction,
    curvature: Fraction,
    repaired_gain: Fraction,
) -> Fraction:
    """Return the only nontrivial upper Jury margin at the optimizer.

    For the scalar EMA/Nesterov invariant mode, the other two strict Jury
    inequalities are positive for every positive step.  A negative result here
    therefore proves local instability of the actual nonlinear loop.
    """

    if learning_rate <= 0 or curvature <= 0 or repaired_gain <= 0:
        raise ValueError("learning rate, curvature, and gain must be positive")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    return 2 * (1 + beta) - (
        (1 - beta) * (1 + 2 * beta) * learning_rate * curvature * repaired_gain
    )


def ema_nesterov_local_learning_rate_threshold(
    *, beta: Fraction, curvature: Fraction, repaired_gain: Fraction
) -> Fraction:
    """Return the exact positive-gain local Schur threshold."""

    if curvature <= 0 or repaired_gain <= 0:
        raise ValueError("curvature and gain must be positive")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    return 2 * (1 + beta) / ((1 - beta) * (1 + 2 * beta) * curvature * repaired_gain)


def skew_boundary_polynomial(
    *, alpha: Fraction, beta: Fraction, normalized_strongness: Fraction
) -> Fraction:
    """Evaluate the exact Schur-boundary cubic for an admissible skew slope.

    The complex slope is ``nu + i*sqrt(1-nu**2)``.  The smallest positive
    zero supplies a necessary boundary for any theorem using only the two
    separate strong-monotonicity and Lipschitz IQCs.
    """

    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if not 0 < normalized_strongness <= 1:
        raise ValueError("normalized strongness must lie in (0, 1]")
    nu = normalized_strongness
    return (
        beta**2 * (1 - beta) ** 2 * (2 * beta + 1) * alpha**3
        - 2 * (1 - beta) * beta * (3 * beta**2 - 1) * nu * alpha**2
        + ((1 + beta) * (2 * beta**2 - beta + 1) + 4 * beta * (beta**2 - beta - 1) * nu**2) * alpha
        - 2 * (1 - beta**2) * nu
    )
