"""Exact robust dissipation certificate for the repaired Muon loop.

This module extends the smooth-PL value-storage certificate with two arbitrary
additive inputs.  The measured gradient is

``g_hat_t = grad(f(W_t)) + xi_t``

and the same measured gradient is used in both occurrences of the pinned
EMA/Nesterov update.  A second error ``e_t`` is added after evaluating the
repaired orthogonalizer:

``W_(t+1) = W_t - eta * (R(s_(t+1)) + e_t)``.

The lifted coordinates are

``chi = (z, u, v, u_next, w, h)``

with ``z=m/L``, ``u=grad(f(W))/L``, ``v=E(s)/(K_E L)``,
``w=xi/L``, and ``h=e/(gamma L)``.  The locked exact certificate proves the
pathwise input--output inequality

``V_(t+1) <= q*V_t + gamma_g*||xi_t||_F^2 + gamma_R*||e_t||_F^2``.

As in :mod:`passive_muon.pl_convergence`, the objective may be nonconvex and
need not have a unique minimizer.  Every authoritative calculation below uses
:class:`fractions.Fraction`; numerical solvers are not part of proof replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.momentum_iqc import RationalMatrix, leading_principal_minors
from passive_muon.pl_convergence import (
    PlConvergenceCertificate,
    locked_pl_convergence_certificate,
    pl_supply_function_coefficients,
)

RationalVector = tuple[Fraction, ...]


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


def _outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return tuple(tuple(left_value * right_value for right_value in right) for left_value in left)


def _symmetric_outer(left: RationalVector, right: RationalVector) -> RationalMatrix:
    return _matrix_scale(
        Fraction(1, 2),
        _matrix_add(_outer(left, right), _outer(right, left)),
    )


def _vector_add(*vectors: RationalVector) -> RationalVector:
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("vector dimensions do not agree")
    return tuple(
        sum((vector[index] for vector in vectors), Fraction(0)) for index in range(len(vectors[0]))
    )


def _vector_scale(scale: Fraction, vector: RationalVector) -> RationalVector:
    return tuple(scale * value for value in vector)


def _nonconvex_smooth_interpolation_quadratic(
    *,
    input_i: RationalVector,
    gradient_i: RationalVector,
    input_j: RationalVector,
    gradient_j: RationalVector,
) -> RationalMatrix:
    """Return the quadratic part of directed ``(-1,1)`` interpolation.

    If ``dx=input_i-input_j`` and ``du=gradient_i-gradient_j``, the complete
    normalized one-smooth supply is

    ``F_i-F_j-<gradient_j,dx>``
    ``- (||du||^2+2<dx,du>-||dx||^2)/4 >= 0``.
    """

    dx = _vector_add(input_i, _vector_scale(Fraction(-1), input_j))
    du = _vector_add(gradient_i, _vector_scale(Fraction(-1), gradient_j))
    interpolation_model = _matrix_scale(
        Fraction(1, 4),
        _matrix_add(
            _outer(du, du),
            _matrix_scale(Fraction(2), _symmetric_outer(dx, du)),
            _matrix_scale(Fraction(-1), _outer(dx, dx)),
        ),
    )
    return _matrix_add(
        _matrix_scale(Fraction(-1), _symmetric_outer(gradient_j, dx)),
        _matrix_scale(Fraction(-1), interpolation_model),
    )


@dataclass(frozen=True)
class RobustDissipativityCertificate:
    """One exact disturbed-loop extension of a smooth-PL certificate."""

    pl_certificate: PlConvergenceCertificate
    gradient_noise_gain: Fraction
    implementation_error_gain: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.gradient_noise_gain, Fraction) or not isinstance(
            self.implementation_error_gain, Fraction
        ):
            raise TypeError("disturbance gains must be fractions.Fraction values")
        if self.gradient_noise_gain <= 0 or self.implementation_error_gain <= 0:
            raise ValueError("disturbance gains must be positive")

    @property
    def rate(self) -> Fraction:
        """Return the storage contraction factor ``q=tau^2``."""

        return self.pl_certificate.tau**2

    @property
    def normalized_gradient_noise_penalty(self) -> Fraction:
        """Return the coefficient on ``||w||^2`` for ``w=xi/L``."""

        return self.gradient_noise_gain * self.pl_certificate.smoothness**2

    @property
    def normalized_implementation_error_penalty(self) -> Fraction:
        """Return the coefficient on ``||h||^2`` for ``h=e/(gamma*L)``."""

        scale = self.pl_certificate.center_gain * self.pl_certificate.smoothness
        return self.implementation_error_gain * scale**2


@dataclass(frozen=True)
class RobustDissipativityMatrices:
    """Exact disturbed dynamics, selectors, and static supplies."""

    transition: RationalMatrix
    selection: RationalMatrix
    noisy_gradient_selector: RationalVector
    signal_selector: RationalVector
    step_selector: RationalVector
    interpolation_12: RationalMatrix
    interpolation_21: RationalMatrix
    pl_next: RationalMatrix
    residual_norm: RationalMatrix
    gradient_noise_norm: RationalMatrix
    implementation_error_norm: RationalMatrix


@dataclass(frozen=True)
class RobustDissipativityAudit:
    """Exact Sylvester and objective-value audit of the robust LMI."""

    lmi: RationalMatrix
    storage_leading_minors: tuple[Fraction, ...]
    negative_lmi_leading_minors: tuple[Fraction, ...]
    supply_function_coefficients: tuple[Fraction, Fraction]
    expected_function_coefficients: tuple[Fraction, Fraction]

    @property
    def storage_positive_definite(self) -> bool:
        return all(value > 0 for value in self.storage_leading_minors)

    @property
    def lmi_negative_definite(self) -> bool:
        return all(value > 0 for value in self.negative_lmi_leading_minors)

    @property
    def function_values_cancel(self) -> bool:
        return self.supply_function_coefficients == self.expected_function_coefficients

    @property
    def certified(self) -> bool:
        return (
            self.storage_positive_definite
            and self.lmi_negative_definite
            and self.function_values_cancel
        )


def robust_dissipativity_matrices(
    certificate: RobustDissipativityCertificate,
) -> RobustDissipativityMatrices:
    """Build the lift in ``chi=(z,u,v,u_next,w,h)``.

    Here ``w=xi/L`` enters both the EMA and Nesterov occurrences of the noisy
    gradient.  The normalized post-operator error ``h=e/(gamma*L)`` enters only
    the parameter update.
    """

    base = certificate.pl_certificate
    zero = Fraction(0)
    one = Fraction(1)
    beta = base.beta
    alpha = base.dimensionless_step
    residual_ratio = base.residual_ratio
    basis = tuple(tuple(one if row == column else zero for row in range(6)) for column in range(6))
    momentum, gradient, residual, gradient_next, gradient_noise, implementation_error = basis
    noisy_gradient = _vector_add(gradient, gradient_noise)
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(one - beta**2, noisy_gradient),
    )
    step = _vector_add(
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * residual_ratio, residual),
        _vector_scale(-alpha, implementation_error),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(one - beta, noisy_gradient),
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    null = (zero,) * 6
    points = {
        1: (null, gradient),
        2: (step, gradient_next),
    }

    def interpolation(edge: tuple[int, int]) -> RationalMatrix:
        index_i, index_j = edge
        return _nonconvex_smooth_interpolation_quadratic(
            input_i=points[index_i][0],
            gradient_i=points[index_i][1],
            input_j=points[index_j][0],
            gradient_j=points[index_j][1],
        )

    return RobustDissipativityMatrices(
        transition=transition,
        selection=selection,
        noisy_gradient_selector=noisy_gradient,
        signal_selector=signal,
        step_selector=step,
        interpolation_12=interpolation((1, 2)),
        interpolation_21=interpolation((2, 1)),
        pl_next=_outer(gradient_next, gradient_next),
        residual_norm=_matrix_add(
            _outer(signal, signal),
            _matrix_scale(Fraction(-1), _outer(residual, residual)),
        ),
        gradient_noise_norm=_outer(gradient_noise, gradient_noise),
        implementation_error_norm=_outer(implementation_error, implementation_error),
    )


def robust_dissipativity_lmi_matrix(
    certificate: RobustDissipativityCertificate,
) -> RationalMatrix:
    """Build the exact ``6 x 6`` pathwise robust dissipation LMI."""

    base = certificate.pl_certificate
    matrices = robust_dissipativity_matrices(certificate)
    return _matrix_add(
        _matrix_multiply(
            _transpose(matrices.transition),
            _matrix_multiply(base.storage, matrices.transition),
        ),
        _matrix_scale(
            -certificate.rate,
            _matrix_multiply(
                _transpose(matrices.selection),
                _matrix_multiply(base.storage, matrices.selection),
            ),
        ),
        _matrix_scale(base.lambda_interpolation_12, matrices.interpolation_12),
        _matrix_scale(base.lambda_interpolation_21, matrices.interpolation_21),
        _matrix_scale(base.lambda_pl_next, matrices.pl_next),
        _matrix_scale(base.lambda_residual_norm, matrices.residual_norm),
        _matrix_scale(
            -certificate.normalized_gradient_noise_penalty,
            matrices.gradient_noise_norm,
        ),
        _matrix_scale(
            -certificate.normalized_implementation_error_penalty,
            matrices.implementation_error_norm,
        ),
    )


def audit_robust_dissipativity(
    certificate: RobustDissipativityCertificate | None = None,
) -> RobustDissipativityAudit:
    """Replay exact storage positivity, LMI negativity, and value flow."""

    selected = locked_robust_dissipativity_certificate() if certificate is None else certificate
    base = selected.pl_certificate
    lmi = robust_dissipativity_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return RobustDissipativityAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(base.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
        supply_function_coefficients=pl_supply_function_coefficients(base),
        expected_function_coefficients=(
            base.function_storage * selected.rate,
            -base.function_storage,
        ),
    )


LOCKED_GRADIENT_NOISE_GAIN = Fraction(1, 2)
LOCKED_IMPLEMENTATION_ERROR_GAIN = Fraction(1, 2_000_000)


def locked_robust_dissipativity_certificate() -> RobustDissipativityCertificate:
    """Return the exact full-P6-step robust dissipation certificate."""

    return RobustDissipativityCertificate(
        pl_certificate=locked_pl_convergence_certificate(),
        gradient_noise_gain=LOCKED_GRADIENT_NOISE_GAIN,
        implementation_error_gain=LOCKED_IMPLEMENTATION_ERROR_GAIN,
    )
