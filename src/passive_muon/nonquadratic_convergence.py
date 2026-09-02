"""Exact full-step convergence certificate using objective interpolation.

This module proves trajectory-to-minimizer convergence for the deterministic
pinned EMA/Nesterov loop.  It is deliberately separate from
``nonquadratic_stability``: the latter is an arbitrary-pair incremental result
at five percent of the p4 step, whereas this certificate reaches the full p4
step by adding the normalized objective gap to the storage.

The quadratic interpolation supply constructed here is not a standalone hard
IQC.  Its function-value boundary term is cancelled exactly by the objective
gap in the storage.  All authoritative replay arithmetic uses
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
    return tuple(tuple(a * b for b in right) for a in left)


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


def _interpolation_quadratic(
    *,
    input_i: RationalVector,
    gradient_i: RationalVector,
    input_j: RationalVector,
    gradient_j: RationalVector,
    condition_ratio: Fraction,
) -> RationalMatrix:
    """Return the non-function part of the normalized ``F_{k,1}`` inequality.

    For ``dx=input_i-input_j`` and ``du=gradient_i-gradient_j``, this is the
    quadratic part of

    ``I_ij = F_i-F_j-<u_j,dx>-Phi(dx,du) >= 0``.
    """

    dx = _vector_add(input_i, _vector_scale(Fraction(-1), input_j))
    du = _vector_add(gradient_i, _vector_scale(Fraction(-1), gradient_j))
    interpolation_model = _matrix_scale(
        Fraction(1, 2) / (1 - condition_ratio),
        _matrix_add(
            _outer(du, du),
            _matrix_scale(-2 * condition_ratio, _symmetric_outer(dx, du)),
            _matrix_scale(condition_ratio, _outer(dx, dx)),
        ),
    )
    return _matrix_add(
        _matrix_scale(Fraction(-1), _symmetric_outer(gradient_j, dx)),
        _matrix_scale(Fraction(-1), interpolation_model),
    )


@dataclass(frozen=True)
class NonquadraticConvergenceCertificate:
    """One exact interpolation-storage certificate for a fixed objective."""

    strong_convexity: Fraction
    smoothness: Fraction
    beta: Fraction
    center_gain: Fraction
    residual_lipschitz: Fraction
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    function_storage: Fraction
    interpolation_cycle_weight: Fraction
    interpolation_reverse_weight: Fraction
    lambda_residual_norm: Fraction

    def __post_init__(self) -> None:
        exact_scalars = (
            self.strong_convexity,
            self.smoothness,
            self.beta,
            self.center_gain,
            self.residual_lipschitz,
            self.learning_rate,
            self.tau,
            self.function_storage,
            self.interpolation_cycle_weight,
            self.interpolation_reverse_weight,
            self.lambda_residual_norm,
        )
        if any(not isinstance(value, Fraction) for value in exact_scalars):
            raise TypeError("certificate scalars must be fractions.Fraction values")
        if any(not isinstance(value, Fraction) for row in self.storage for value in row):
            raise TypeError("storage entries must be fractions.Fraction values")
        if not 0 < self.strong_convexity < self.smoothness:
            raise ValueError("objective bounds must satisfy 0 < ell < L")
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
        if self.function_storage < 0:
            raise ValueError("function storage must be nonnegative")
        if self.interpolation_cycle_weight < self.function_storage * (1 - self.tau**2):
            raise ValueError("interpolation cycle weight does not make lambda_20 nonnegative")
        if self.interpolation_reverse_weight < 0 or self.lambda_residual_norm < 0:
            raise ValueError("IQC weights must be nonnegative")

    @property
    def condition_ratio(self) -> Fraction:
        return self.strong_convexity / self.smoothness

    @property
    def dimensionless_step(self) -> Fraction:
        return self.learning_rate * self.center_gain * self.smoothness

    @property
    def residual_ratio(self) -> Fraction:
        return self.residual_lipschitz / self.center_gain

    @property
    def lambda_interpolation_01(self) -> Fraction:
        return self.interpolation_cycle_weight

    @property
    def lambda_interpolation_12(self) -> Fraction:
        return (
            self.interpolation_reverse_weight
            + self.interpolation_cycle_weight
            + self.function_storage * self.tau**2
        )

    @property
    def lambda_interpolation_20(self) -> Fraction:
        return self.interpolation_cycle_weight - self.function_storage * (1 - self.tau**2)

    @property
    def lambda_interpolation_21(self) -> Fraction:
        return self.interpolation_reverse_weight

    @property
    def interpolation_multipliers(self) -> dict[tuple[int, int], Fraction]:
        return {
            (0, 1): self.lambda_interpolation_01,
            (1, 2): self.lambda_interpolation_12,
            (2, 0): self.lambda_interpolation_20,
            (2, 1): self.lambda_interpolation_21,
        }


@dataclass(frozen=True)
class NonquadraticConvergenceMatrices:
    """Exact lifted dynamics and interpolation quadratic supplies."""

    transition: RationalMatrix
    selection: RationalMatrix
    signal_selector: RationalVector
    interpolation_01: RationalMatrix
    interpolation_12: RationalMatrix
    interpolation_20: RationalMatrix
    interpolation_21: RationalMatrix
    residual_norm: RationalMatrix


@dataclass(frozen=True)
class NonquadraticConvergenceAudit:
    """Exact Sylvester and function-flow audit of one certificate."""

    lmi: RationalMatrix
    storage_leading_minors: tuple[Fraction, ...]
    negative_lmi_leading_minors: tuple[Fraction, ...]
    interpolation_function_coefficients: tuple[Fraction, Fraction]
    expected_function_coefficients: tuple[Fraction, Fraction]

    @property
    def storage_positive_definite(self) -> bool:
        return all(value > 0 for value in self.storage_leading_minors)

    @property
    def lmi_negative_definite(self) -> bool:
        return all(value > 0 for value in self.negative_lmi_leading_minors)

    @property
    def function_values_cancel(self) -> bool:
        return self.interpolation_function_coefficients == self.expected_function_coefficients

    @property
    def certified(self) -> bool:
        return (
            self.storage_positive_definite
            and self.lmi_negative_definite
            and self.function_values_cancel
        )


def nonquadratic_convergence_matrices(
    certificate: NonquadraticConvergenceCertificate,
) -> NonquadraticConvergenceMatrices:
    """Return exact matrices in ``chi=(w,m/L,grad/L,E/(K_E L),grad_next/L)``."""

    zero = Fraction(0)
    one = Fraction(1)
    beta = certificate.beta
    alpha = certificate.dimensionless_step
    residual_ratio = certificate.residual_ratio
    basis = tuple(tuple(one if row == column else zero for row in range(5)) for column in range(5))
    w, momentum, gradient, residual, gradient_next = basis
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(one - beta**2, gradient),
    )
    w_next = _vector_add(
        w,
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * residual_ratio, residual),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(one - beta, gradient),
    )
    transition = (w_next, momentum_next)
    selection = (w, momentum)
    null = (zero,) * 5
    points = {
        0: (null, null),
        1: (w, gradient),
        2: (w_next, gradient_next),
    }

    def interpolation(edge: tuple[int, int]) -> RationalMatrix:
        index_i, index_j = edge
        return _interpolation_quadratic(
            input_i=points[index_i][0],
            gradient_i=points[index_i][1],
            input_j=points[index_j][0],
            gradient_j=points[index_j][1],
            condition_ratio=certificate.condition_ratio,
        )

    residual_norm = _matrix_add(
        _outer(signal, signal),
        _matrix_scale(-one, _outer(residual, residual)),
    )
    return NonquadraticConvergenceMatrices(
        transition=transition,
        selection=selection,
        signal_selector=signal,
        interpolation_01=interpolation((0, 1)),
        interpolation_12=interpolation((1, 2)),
        interpolation_20=interpolation((2, 0)),
        interpolation_21=interpolation((2, 1)),
        residual_norm=residual_norm,
    )


def interpolation_function_coefficients(
    certificate: NonquadraticConvergenceCertificate,
) -> tuple[Fraction, Fraction]:
    """Return coefficients of ``(F_current,F_next)`` in the weighted edges."""

    multipliers = certificate.interpolation_multipliers
    return (
        -multipliers[(0, 1)] + multipliers[(1, 2)] - multipliers[(2, 1)],
        -multipliers[(1, 2)] + multipliers[(2, 0)] + multipliers[(2, 1)],
    )


def nonquadratic_convergence_lmi_matrix(
    certificate: NonquadraticConvergenceCertificate,
) -> RationalMatrix:
    """Build the exact ``5 x 5`` interpolation-storage dissipation LMI."""

    matrices = nonquadratic_convergence_matrices(certificate)
    multipliers = certificate.interpolation_multipliers
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
        _matrix_scale(multipliers[(0, 1)], matrices.interpolation_01),
        _matrix_scale(multipliers[(1, 2)], matrices.interpolation_12),
        _matrix_scale(multipliers[(2, 0)], matrices.interpolation_20),
        _matrix_scale(multipliers[(2, 1)], matrices.interpolation_21),
        _matrix_scale(certificate.lambda_residual_norm, matrices.residual_norm),
    )


def audit_nonquadratic_convergence(
    certificate: NonquadraticConvergenceCertificate | None = None,
) -> NonquadraticConvergenceAudit:
    """Replay strict definiteness and exact objective-value cancellation."""

    selected = locked_nonquadratic_convergence_certificate() if certificate is None else certificate
    lmi = nonquadratic_convergence_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return NonquadraticConvergenceAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(selected.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
        interpolation_function_coefficients=interpolation_function_coefficients(selected),
        expected_function_coefficients=(
            selected.function_storage * selected.tau**2,
            -selected.function_storage,
        ),
    )


LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE = Fraction(1, 32_000)
LOCKED_NONQUADRATIC_CONVERGENCE_TAU = Fraction(2_499, 2_500)


def locked_nonquadratic_convergence_certificate() -> NonquadraticConvergenceCertificate:
    """Return the exact full-p4 trajectory-to-minimizer certificate."""

    denominator = 100_000_000
    return NonquadraticConvergenceCertificate(
        strong_convexity=LOCKED_HESSIAN_LOWER,
        smoothness=LOCKED_HESSIAN_UPPER,
        beta=LOCKED_BETA,
        center_gain=LOCKED_CENTER_GAIN,
        residual_lipschitz=LOCKED_RESIDUAL_LIPSCHITZ,
        learning_rate=LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE,
        tau=LOCKED_NONQUADRATIC_CONVERGENCE_TAU,
        storage=(
            (Fraction(495_723, denominator), Fraction(-3_085_119, denominator)),
            (Fraction(-3_085_119, denominator), Fraction(72_422_647, denominator)),
        ),
        function_storage=Fraction(27_081_630, denominator),
        interpolation_cycle_weight=Fraction(270_356, denominator),
        interpolation_reverse_weight=Fraction(48_120, denominator),
        lambda_residual_norm=Fraction(5_066_256, denominator),
    )
