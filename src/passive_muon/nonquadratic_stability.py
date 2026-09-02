"""Exact IQC certificate for nonlinear strongly-convex objectives.

The deterministic real-arithmetic loop is the pinned EMA/Nesterov ordering

``m_next = beta*m + (1-beta)*gradient``
``signal = beta*m_next + (1-beta)*gradient``
``W_next = W - eta*R(signal)``.

Unlike the p4 scalar-mode argument, this module never diagonalizes an
objective Hessian.  It combines the pointwise interpolation IQC for every
``ell``-strongly convex, ``L``-smooth objective with the global p4 split
``R(s)=gamma*s+E(s)``, ``Lip(E)<=K_E``.  A common ``P tensor I`` storage then
covers arbitrary changes of local Hessian orientation over time and every
finite matrix shape.

All authoritative proof replay uses :class:`fractions.Fraction`.  A floating
point SDP was used only to discover the committed rational storage and
multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.momentum_iqc import ExactLmiAudit, RationalMatrix, leading_principal_minors
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_CENTER_GAIN,
    LOCKED_HESSIAN_LOWER,
    LOCKED_HESSIAN_UPPER,
    LOCKED_RESIDUAL_LIPSCHITZ,
    LOCKED_SYMMETRIC_GAIN_LOWER,
    LOCKED_SYMMETRIC_GAIN_UPPER,
)


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
    return tuple(tuple(a * b for b in right) for a in left)


def _symmetric_outer(left: tuple[Fraction, ...], right: tuple[Fraction, ...]) -> RationalMatrix:
    return _matrix_scale(
        Fraction(1, 2),
        _matrix_add(_outer(left, right), _outer(right, left)),
    )


@dataclass(frozen=True)
class NonquadraticLmiCertificate:
    """One exact common-storage certificate for the nonlinear objective class."""

    strong_convexity: Fraction
    smoothness: Fraction
    beta: Fraction
    center_gain: Fraction
    residual_lipschitz: Fraction
    centered_residual_radius: Fraction
    learning_rate: Fraction
    tau: Fraction
    storage: RationalMatrix
    lambda_gradient: Fraction
    lambda_residual_norm: Fraction
    lambda_residual_lower: Fraction = Fraction(0)
    lambda_residual_upper: Fraction = Fraction(0)

    def __post_init__(self) -> None:
        exact_scalars = (
            self.strong_convexity,
            self.smoothness,
            self.beta,
            self.center_gain,
            self.residual_lipschitz,
            self.centered_residual_radius,
            self.learning_rate,
            self.tau,
            self.lambda_gradient,
            self.lambda_residual_norm,
            self.lambda_residual_lower,
            self.lambda_residual_upper,
        )
        if any(not isinstance(value, Fraction) for value in exact_scalars):
            raise TypeError("certificate scalars must be fractions.Fraction values")
        if any(not isinstance(value, Fraction) for row in self.storage for value in row):
            raise TypeError("storage entries must be fractions.Fraction values")
        if self.strong_convexity <= 0 or self.smoothness < self.strong_convexity:
            raise ValueError("objective bounds must satisfy 0 < ell <= L")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must lie in [0, 1)")
        if self.center_gain <= 0 or self.residual_lipschitz <= 0:
            raise ValueError("center gain and residual Lipschitz bound must be positive")
        if not 0 <= self.centered_residual_radius <= self.residual_lipschitz:
            raise ValueError("centered residual radius must lie in [0, residual Lipschitz]")
        if self.learning_rate <= 0 or not 0 < self.tau < 1:
            raise ValueError("learning rate must be positive and tau must lie in (0, 1)")
        if len(self.storage) != 2 or any(len(row) != 2 for row in self.storage):
            raise ValueError("storage must be a 2-by-2 matrix")
        if self.storage[0][1] != self.storage[1][0]:
            raise ValueError("storage must be symmetric")
        multipliers = (
            self.lambda_gradient,
            self.lambda_residual_norm,
            self.lambda_residual_lower,
            self.lambda_residual_upper,
        )
        if any(value < 0 for value in multipliers):
            raise ValueError("IQC multipliers must be nonnegative")

    @property
    def condition_ratio(self) -> Fraction:
        """Return ``ell/L`` in the scaled gradient IQC."""

        return self.strong_convexity / self.smoothness

    @property
    def dimensionless_step(self) -> Fraction:
        """Return ``alpha=eta*gamma*L`` used by the lifted transition."""

        return self.learning_rate * self.center_gain * self.smoothness

    @property
    def residual_ratio(self) -> Fraction:
        """Return ``K_E/gamma`` used by the lifted transition."""

        return self.residual_lipschitz / self.center_gain

    @property
    def centered_residual_ratio(self) -> Fraction:
        """Return ``d/K_E`` for the optional centered residual IQCs."""

        return self.centered_residual_radius / self.residual_lipschitz


def nonquadratic_iqc_matrices(
    certificate: NonquadraticLmiCertificate,
) -> dict[str, RationalMatrix]:
    """Return the exact IQCs in ``chi=(w,m/L,g/L,e/(K_E*L))``.

    Write ``q=s/L=beta**2*m/L+(1-beta**2)*g/L``.  The matrices encode

    ``<g/L-k*w, w-g/L> >= 0``, ``||q||^2-||v||^2 >= 0``, and
    ``sigma*||q||^2 +/- <q,v> >= 0``.
    """

    zero = Fraction(0)
    one = Fraction(1)
    beta = certificate.beta
    kappa = certificate.condition_ratio
    q = (zero, beta**2, one - beta**2, zero)
    w = (one, zero, zero, zero)
    gradient = (zero, zero, one, zero)
    residual = (zero, zero, zero, one)

    gradient_iqc = _matrix_add(
        _matrix_scale(1 + kappa, _symmetric_outer(w, gradient)),
        _matrix_scale(-kappa, _outer(w, w)),
        _matrix_scale(-one, _outer(gradient, gradient)),
    )
    residual_norm_iqc = _matrix_add(
        _outer(q, q),
        _matrix_scale(-one, _outer(residual, residual)),
    )
    centered_base = _matrix_scale(certificate.centered_residual_ratio, _outer(q, q))
    centered_cross = _symmetric_outer(q, residual)
    return {
        "gradient_interpolation": gradient_iqc,
        "residual_lipschitz": residual_norm_iqc,
        "residual_lower_inner_product": _matrix_add(centered_base, centered_cross),
        "residual_upper_inner_product": _matrix_add(
            centered_base,
            _matrix_scale(-one, centered_cross),
        ),
    }


def nonquadratic_lmi_matrix(certificate: NonquadraticLmiCertificate) -> RationalMatrix:
    """Build the exact dimension-independent ``4 x 4`` nonlinear IQC LMI."""

    zero = Fraction(0)
    one = Fraction(1)
    beta = certificate.beta
    alpha = certificate.dimensionless_step
    residual_ratio = certificate.residual_ratio
    transition = (
        (
            one,
            -alpha * beta**2,
            -alpha * (one - beta**2),
            -alpha * residual_ratio,
        ),
        (zero, beta, one - beta, zero),
    )
    selection = ((one, zero, zero, zero), (zero, one, zero, zero))
    iqcs = nonquadratic_iqc_matrices(certificate)
    return _matrix_add(
        _matrix_multiply(
            _transpose(transition),
            _matrix_multiply(certificate.storage, transition),
        ),
        _matrix_scale(
            -(certificate.tau**2),
            _matrix_multiply(
                _transpose(selection),
                _matrix_multiply(certificate.storage, selection),
            ),
        ),
        _matrix_scale(certificate.lambda_gradient, iqcs["gradient_interpolation"]),
        _matrix_scale(certificate.lambda_residual_norm, iqcs["residual_lipschitz"]),
        _matrix_scale(
            certificate.lambda_residual_lower,
            iqcs["residual_lower_inner_product"],
        ),
        _matrix_scale(
            certificate.lambda_residual_upper,
            iqcs["residual_upper_inner_product"],
        ),
    )


def audit_nonquadratic_certificate(
    certificate: NonquadraticLmiCertificate | None = None,
) -> ExactLmiAudit:
    """Replay storage positivity and strict LMI negativity exactly."""

    selected = locked_nonquadratic_certificate() if certificate is None else certificate
    lmi = nonquadratic_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    return ExactLmiAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(selected.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
    )


LOCKED_CENTERED_RESIDUAL_RADIUS = (LOCKED_SYMMETRIC_GAIN_UPPER - LOCKED_SYMMETRIC_GAIN_LOWER) / 2
LOCKED_NONQUADRATIC_LEARNING_RATE = Fraction(1, 640_000)
LOCKED_NONQUADRATIC_TAU = Fraction(99_999, 100_000)


def locked_nonquadratic_certificate() -> NonquadraticLmiCertificate:
    """Return the exact 5%-of-p4 representative selected after SDP search."""

    return NonquadraticLmiCertificate(
        strong_convexity=LOCKED_HESSIAN_LOWER,
        smoothness=LOCKED_HESSIAN_UPPER,
        beta=LOCKED_BETA,
        center_gain=LOCKED_CENTER_GAIN,
        residual_lipschitz=LOCKED_RESIDUAL_LIPSCHITZ,
        centered_residual_radius=LOCKED_CENTERED_RESIDUAL_RADIUS,
        learning_rate=LOCKED_NONQUADRATIC_LEARNING_RATE,
        tau=LOCKED_NONQUADRATIC_TAU,
        storage=(
            (Fraction(6_682_146_216, 10_000_000_000), Fraction(-1_788_501_566, 10_000_000_000)),
            (Fraction(-1_788_501_566, 10_000_000_000), Fraction(3_317_853_784, 10_000_000_000)),
        ),
        lambda_gradient=Fraction(19_472_227, 1_000_000_000),
        lambda_residual_norm=Fraction(136_225_077, 10_000_000_000),
    )
