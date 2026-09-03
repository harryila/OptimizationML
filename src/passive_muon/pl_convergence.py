"""Exact function-value convergence certificate for smooth PL objectives.

The certificate in this module applies to the deterministic repaired
max-floored EMA/Nesterov loop.  Unlike the strongly-convex p5 certificate, it
does not store displacement from a distinguished minimizer.  Its storage uses
only the normalized momentum, current gradient, and objective gap, so the
minimizer set may be non-singleton and the objective may be nonconvex.

The directed quadratic supplies are the exact ``(-1, 1)`` interpolation
inequalities obeyed by every differentiable one-smooth scalar potential.  A
pointwise Polyak--Lojasiewicz supply at the next iterate cancels the remaining
objective-gap flow.  All authoritative replay arithmetic uses
:class:`fractions.Fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.momentum_iqc import RationalMatrix, leading_principal_minors
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_CENTER_GAIN,
    LOCKED_HESSIAN_LOWER,
    LOCKED_HESSIAN_UPPER,
    LOCKED_RESIDUAL_LIPSCHITZ,
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
    """Return the non-function part of a directed one-smooth inequality.

    For ``dx=input_i-input_j`` and ``du=gradient_i-gradient_j``, the complete
    valid supply is

    ``F_i-F_j-<gradient_j,dx>``
    ``- (||du||^2 + 2<dx,du> - ||dx||^2)/4 >= 0``.

    This is the ``mu=-1, L=1`` smooth interpolation inequality.  It permits
    negative curvature and does not assume convexity or gradient monotonicity.
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
class PlConvergenceCertificate:
    """One exact value--momentum storage certificate for a smooth PL class."""

    pl_constant: Fraction
    smoothness: Fraction
    beta: Fraction
    center_gain: Fraction
    residual_lipschitz: Fraction
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    function_storage: Fraction
    interpolation_reverse_weight: Fraction
    lambda_residual_norm: Fraction

    def __post_init__(self) -> None:
        exact_scalars = (
            self.pl_constant,
            self.smoothness,
            self.beta,
            self.center_gain,
            self.residual_lipschitz,
            self.learning_rate,
            self.tau,
            self.function_storage,
            self.interpolation_reverse_weight,
            self.lambda_residual_norm,
        )
        if any(not isinstance(value, Fraction) for value in exact_scalars):
            raise TypeError("certificate scalars must be fractions.Fraction values")
        if any(not isinstance(value, Fraction) for row in self.storage for value in row):
            raise TypeError("storage entries must be fractions.Fraction values")
        if not 0 < self.pl_constant <= self.smoothness:
            raise ValueError("bounds must satisfy 0 < PL constant <= smoothness")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if self.center_gain <= 0 or self.residual_lipschitz <= 0:
            raise ValueError("center and residual gains must be positive")
        if self.learning_rate <= 0 or not 0 < self.tau < 1:
            raise ValueError("learning rate must be positive and tau must lie in (0, 1)")
        if len(self.storage) != 2 or any(len(row) != 2 for row in self.storage):
            raise ValueError("storage must be a 2-by-2 matrix")
        if self.storage[0][1] != self.storage[1][0]:
            raise ValueError("storage must be symmetric")
        if self.function_storage <= 0:
            raise ValueError("function storage must be positive")
        if self.interpolation_reverse_weight < 0 or self.lambda_residual_norm < 0:
            raise ValueError("supply multipliers must be nonnegative")

    @property
    def condition_ratio(self) -> Fraction:
        """Return the normalized PL constant ``k=ell_PL/L``."""

        return self.pl_constant / self.smoothness

    @property
    def dimensionless_step(self) -> Fraction:
        """Return ``alpha=eta*gamma*L`` in the normalized transition."""

        return self.learning_rate * self.center_gain * self.smoothness

    @property
    def residual_ratio(self) -> Fraction:
        """Return ``r=K_E/gamma`` in the normalized transition."""

        return self.residual_lipschitz / self.center_gain

    @property
    def lambda_interpolation_12(self) -> Fraction:
        """Weight on the current-to-next directed interpolation supply."""

        return self.interpolation_reverse_weight + self.function_storage * self.tau**2

    @property
    def lambda_interpolation_21(self) -> Fraction:
        """Weight on the next-to-current directed interpolation supply."""

        return self.interpolation_reverse_weight

    @property
    def lambda_pl_next(self) -> Fraction:
        """Weight on ``||u_next||^2-2*k*F_next >= 0``."""

        return self.function_storage * (1 - self.tau**2) / (2 * self.condition_ratio)


@dataclass(frozen=True)
class PlConvergenceMatrices:
    """Exact lifted dynamics and static quadratic supplies."""

    transition: RationalMatrix
    selection: RationalMatrix
    signal_selector: RationalVector
    step_selector: RationalVector
    interpolation_12: RationalMatrix
    interpolation_21: RationalMatrix
    pl_next: RationalMatrix
    residual_norm: RationalMatrix


@dataclass(frozen=True)
class PlConvergenceAudit:
    """Exact Sylvester and function-flow audit of one PL certificate."""

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


def pl_convergence_matrices(certificate: PlConvergenceCertificate) -> PlConvergenceMatrices:
    """Return matrices in ``chi=(m/L, grad/L, E/(K_E*L), grad_next/L)``."""

    zero = Fraction(0)
    one = Fraction(1)
    beta = certificate.beta
    alpha = certificate.dimensionless_step
    residual_ratio = certificate.residual_ratio
    basis = tuple(tuple(one if row == column else zero for row in range(4)) for column in range(4))
    momentum, gradient, residual, gradient_next = basis
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(one - beta**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * residual_ratio, residual),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(one - beta, gradient),
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    null = (zero,) * 4
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

    return PlConvergenceMatrices(
        transition=transition,
        selection=selection,
        signal_selector=signal,
        step_selector=step,
        interpolation_12=interpolation((1, 2)),
        interpolation_21=interpolation((2, 1)),
        pl_next=_outer(gradient_next, gradient_next),
        residual_norm=_matrix_add(
            _outer(signal, signal),
            _matrix_scale(Fraction(-1), _outer(residual, residual)),
        ),
    )


def pl_supply_function_coefficients(
    certificate: PlConvergenceCertificate,
) -> tuple[Fraction, Fraction]:
    """Return coefficients of ``(F_current,F_next)`` in all supplies."""

    return (
        certificate.lambda_interpolation_12 - certificate.lambda_interpolation_21,
        -certificate.lambda_interpolation_12
        + certificate.lambda_interpolation_21
        - 2 * certificate.condition_ratio * certificate.lambda_pl_next,
    )


def pl_convergence_lmi_matrix(certificate: PlConvergenceCertificate) -> RationalMatrix:
    """Build the exact ``4 x 4`` value--momentum dissipation LMI."""

    matrices = pl_convergence_matrices(certificate)
    return _matrix_add(
        _matrix_multiply(
            _transpose(matrices.transition),
            _matrix_multiply(certificate.storage, matrices.transition),
        ),
        _matrix_scale(
            -(certificate.tau**2),
            _matrix_multiply(
                _transpose(matrices.selection),
                _matrix_multiply(certificate.storage, matrices.selection),
            ),
        ),
        _matrix_scale(
            certificate.lambda_interpolation_12,
            matrices.interpolation_12,
        ),
        _matrix_scale(
            certificate.lambda_interpolation_21,
            matrices.interpolation_21,
        ),
        _matrix_scale(certificate.lambda_pl_next, matrices.pl_next),
        _matrix_scale(certificate.lambda_residual_norm, matrices.residual_norm),
    )


def audit_pl_convergence(
    certificate: PlConvergenceCertificate | None = None,
) -> PlConvergenceAudit:
    """Replay exact storage positivity, LMI negativity, and value flow."""

    selected = locked_pl_convergence_certificate() if certificate is None else certificate
    lmi = pl_convergence_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return PlConvergenceAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(selected.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
        supply_function_coefficients=pl_supply_function_coefficients(selected),
        expected_function_coefficients=(
            selected.function_storage * selected.tau**2,
            -selected.function_storage,
        ),
    )


LOCKED_PL_CONVERGENCE_LEARNING_RATE = Fraction(1, 32_000)
LOCKED_PL_CONVERGENCE_TAU = Fraction(19_999, 20_000)


def locked_pl_convergence_certificate() -> PlConvergenceCertificate:
    """Return the exact full-p5-step smooth-PL convergence certificate."""

    denominator = 100_000
    return PlConvergenceCertificate(
        pl_constant=LOCKED_HESSIAN_LOWER,
        smoothness=LOCKED_HESSIAN_UPPER,
        beta=LOCKED_BETA,
        center_gain=LOCKED_CENTER_GAIN,
        residual_lipschitz=LOCKED_RESIDUAL_LIPSCHITZ,
        learning_rate=LOCKED_PL_CONVERGENCE_LEARNING_RATE,
        tau=LOCKED_PL_CONVERGENCE_TAU,
        storage=(
            (Fraction(72_435, denominator), Fraction(-3_185, denominator)),
            (Fraction(-3_185, denominator), Fraction(499, denominator)),
        ),
        function_storage=Fraction(27_066, denominator),
        interpolation_reverse_weight=Fraction(20_753, denominator),
        lambda_residual_norm=Fraction(5_052, denominator),
    )
