"""Exact stored-signal outer-loop composition certificates for P21.

P20 certifies a *pointwise* sector at the stored FP32 signal.  It does not
provide an incremental Lipschitz bound with which to compare a rounded signal
to an ideal one.  The correct P21 lift therefore puts the actual stored signal
directly in the P18/P19 supply.

Normalize ``z=m/L`` and ``u=grad f(W)/L`` and define the total arithmetic
residuals relative to the true gradient by

``a_m = (m_next-beta*m-(1-beta)*grad f(W))/L`` and
``a_s = (s_hat-beta*m_next-(1-beta)*grad f(W))/L``.

If ``p=s_hat/L``, the exact stored recurrence is

``z_next = beta*z + (1-beta)*u + a_m`` and
``p = beta**2*z + (1-beta**2)*u + beta*a_m + a_s``.

Write a successful shield output as
``U/L = gamma*p + K_T*v``, where ``||v||_F <= ||p||_F``.  Finally define the
aggregate output-equivalent port ``h`` by

``W_next-W = -eta*gamma*L*(p + (K_T/gamma)*v + h)``.

For example, a logical master-update residual ``r_W`` contributes
``h=-r_W/(eta*gamma*L)``.  A decoupled update ``-eta*d_W`` contributes
``h=d_W/(gamma*L)``.  This sign convention is part of the certificate.

The lifted variable order is exactly
``(z, u, v, u_next, a_m, a_s, h)``.  Setting the last three variables to zero
recovers the frozen P18/P19 4-by-4 LMI entry for entry.  All authoritative
arithmetic uses :class:`fractions.Fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.momentum_iqc import RationalMatrix, leading_principal_minors
from passive_muon.pl_convergence import (
    PlConvergenceCertificate,
    audit_pl_convergence,
    pl_convergence_lmi_matrix,
)
from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    MAXIMUM_STEP_POINT,
    ShieldOperatingPoint,
    make_pl_certificate,
)

CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION = (
    "passive-muon-certified-outer-loop-composition-certificate-v1"
)

PORT_VARIABLE_ORDER = (
    "normalized_momentum",
    "normalized_gradient",
    "normalized_sector_residual",
    "normalized_next_gradient",
    "normalized_momentum_roundoff",
    "normalized_signal_roundoff",
    "normalized_output_equivalent_error",
)

PRIMARY_PORT_GAINS = (Fraction(16_384), Fraction(1_024), Fraction(512))
SECONDARY_PORT_GAINS = (Fraction(32_768), Fraction(2_048), Fraction(512))

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
    return tuple(tuple(x * y for y in right) for x in left)


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


def _directed_smooth_interpolation_quadratic(
    *,
    input_i: RationalVector,
    gradient_i: RationalVector,
    input_j: RationalVector,
    gradient_j: RationalVector,
) -> RationalMatrix:
    """Return the quadratic part of the exact one-smooth interpolation supply."""

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
class StoredSignalPortCertificate:
    """One exact 7-by-7 stored-signal dissipation certificate."""

    point: ShieldOperatingPoint
    pl_certificate: PlConvergenceCertificate
    port_gains: tuple[Fraction, Fraction, Fraction]
    raw_lmi: RationalMatrix
    lmi: RationalMatrix

    @property
    def rate(self) -> Fraction:
        return self.point.rate_squared


@dataclass(frozen=True)
class StoredSignalPortAudit:
    """Exact recovery and Sylvester checks for one P21 operating point."""

    certificate: StoredSignalPortCertificate
    storage_leading_minors: tuple[Fraction, ...]
    negative_lmi_leading_minors: tuple[Fraction, ...]
    recovered_zero_port_lmi: RationalMatrix
    frozen_zero_port_lmi: RationalMatrix
    function_values_cancel: bool

    @property
    def storage_positive_definite(self) -> bool:
        return all(value > 0 for value in self.storage_leading_minors)

    @property
    def lmi_negative_definite(self) -> bool:
        return all(value > 0 for value in self.negative_lmi_leading_minors)

    @property
    def zero_port_recovers_p18_exactly(self) -> bool:
        return self.recovered_zero_port_lmi == self.frozen_zero_port_lmi

    @property
    def certified(self) -> bool:
        return (
            self.storage_positive_definite
            and self.lmi_negative_definite
            and self.zero_port_recovers_p18_exactly
            and self.function_values_cancel
        )


def _raw_stored_signal_lmi(certificate: PlConvergenceCertificate) -> RationalMatrix:
    """Build the unpenalized 7-by-7 LMI in ``PORT_VARIABLE_ORDER``."""

    dimension = len(PORT_VARIABLE_ORDER)
    zero = Fraction(0)
    one = Fraction(1)
    basis = tuple(
        tuple(one if row == column else zero for row in range(dimension))
        for column in range(dimension)
    )
    momentum, gradient, residual, gradient_next, momentum_error, signal_error, output_error = basis
    beta = certificate.beta
    alpha = certificate.dimensionless_step
    residual_ratio = certificate.residual_ratio

    stored_signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(one - beta**2, gradient),
        _vector_scale(beta, momentum_error),
        signal_error,
    )
    step = _vector_add(
        _vector_scale(-alpha, stored_signal),
        _vector_scale(-alpha * residual_ratio, residual),
        _vector_scale(-alpha, output_error),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(one - beta, gradient),
        momentum_error,
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    null = (zero,) * dimension
    current = (null, gradient)
    following = (step, gradient_next)

    interpolation_12 = _directed_smooth_interpolation_quadratic(
        input_i=current[0],
        gradient_i=current[1],
        input_j=following[0],
        gradient_j=following[1],
    )
    interpolation_21 = _directed_smooth_interpolation_quadratic(
        input_i=following[0],
        gradient_i=following[1],
        input_j=current[0],
        gradient_j=current[1],
    )
    residual_norm = _matrix_add(
        _outer(stored_signal, stored_signal),
        _matrix_scale(Fraction(-1), _outer(residual, residual)),
    )

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
        _matrix_scale(certificate.lambda_interpolation_12, interpolation_12),
        _matrix_scale(certificate.lambda_interpolation_21, interpolation_21),
        _matrix_scale(certificate.lambda_pl_next, _outer(gradient_next, gradient_next)),
        _matrix_scale(certificate.lambda_residual_norm, residual_norm),
    )


def _port_gains_for_point(
    point: ShieldOperatingPoint,
) -> tuple[Fraction, Fraction, Fraction]:
    if point == FASTER_RATE_POINT:
        return PRIMARY_PORT_GAINS
    if point == MAXIMUM_STEP_POINT:
        return SECONDARY_PORT_GAINS
    raise ValueError("point must be one of the two frozen P19 operating points")


def build_stored_signal_port_certificate(
    point: ShieldOperatingPoint,
) -> StoredSignalPortCertificate:
    """Return a strict exact certificate for one frozen P18/P19 point."""

    pl_certificate = make_pl_certificate(point)
    raw_lmi = _raw_stored_signal_lmi(pl_certificate)
    port_gains = _port_gains_for_point(point)
    lmi = tuple(
        tuple(
            value - (port_gains[row - 4] if row == column and row >= 4 else 0)
            for column, value in enumerate(values)
        )
        for row, values in enumerate(raw_lmi)
    )
    return StoredSignalPortCertificate(
        point=point,
        pl_certificate=pl_certificate,
        port_gains=port_gains,
        raw_lmi=raw_lmi,
        lmi=lmi,
    )


def audit_stored_signal_port_certificate(
    point: ShieldOperatingPoint,
) -> StoredSignalPortAudit:
    """Replay strict definiteness and literal zero-port P18 recovery."""

    certificate = build_stored_signal_port_certificate(point)
    pl_audit = audit_pl_convergence(certificate.pl_certificate)
    recovered = tuple(tuple(row[column] for column in range(4)) for row in certificate.raw_lmi[:4])
    frozen = pl_convergence_lmi_matrix(certificate.pl_certificate)
    return StoredSignalPortAudit(
        certificate=certificate,
        storage_leading_minors=leading_principal_minors(certificate.pl_certificate.storage),
        negative_lmi_leading_minors=leading_principal_minors(
            _matrix_scale(Fraction(-1), certificate.lmi)
        ),
        recovered_zero_port_lmi=recovered,
        frozen_zero_port_lmi=frozen,
        function_values_cancel=pl_audit.function_values_cancel,
    )


@dataclass(frozen=True)
class AffinePortEnvelope:
    """A pathwise bound ``||port||_F <= slope*sqrt(V)+intercept``."""

    slope: Fraction
    intercept: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.slope, Fraction) or not isinstance(self.intercept, Fraction):
            raise TypeError("affine port coefficients must be Fraction values")
        if self.slope < 0 or self.intercept < 0:
            raise ValueError("affine port coefficients must be nonnegative")


@dataclass(frozen=True)
class AffinePortAbsorption:
    """Exact Young absorption of the three normalized P21 ports."""

    base_rate: Fraction
    absorbed_rate: Fraction
    constant_forcing: Fraction
    storage_radius: Fraction
    objective_gap_ultimate_bound: Fraction | None
    forward_invariant: bool

    @property
    def contractive(self) -> bool:
        return self.absorbed_rate < 1


def absorb_affine_ports(
    certificate: StoredSignalPortCertificate,
    envelopes: tuple[AffinePortEnvelope, AffinePortEnvelope, AffinePortEnvelope],
    young_parameters: tuple[Fraction, Fraction, Fraction],
    *,
    storage_radius: Fraction = Fraction(1),
) -> AffinePortAbsorption:
    """Absorb affine port envelopes into the exact stored-signal inequality.

    For each port, Young's inequality gives
    ``(a*sqrt(V)+b)^2 <= (1+theta)*a^2*V +
    (1+1/theta)*b^2``.  The result is exact and intentionally performs no
    theorem-facing decimal or floating-point rounding.
    """

    if len(envelopes) != 3 or len(young_parameters) != 3:
        raise ValueError("P21 has exactly three normalized ports")
    if any(not isinstance(theta, Fraction) or theta <= 0 for theta in young_parameters):
        raise ValueError("Young parameters must be positive Fraction values")
    if not isinstance(storage_radius, Fraction) or storage_radius <= 0:
        raise ValueError("storage_radius must be a positive Fraction")

    rate_increment = sum(
        (gain * (1 + theta) * envelope.slope**2)
        for gain, envelope, theta in zip(
            certificate.port_gains, envelopes, young_parameters, strict=True
        )
    )
    forcing = sum(
        (gain * (1 + 1 / theta) * envelope.intercept**2)
        for gain, envelope, theta in zip(
            certificate.port_gains, envelopes, young_parameters, strict=True
        )
    )
    rate = certificate.rate + rate_increment
    forward_invariant = rate < 1 and forcing <= (1 - rate) * storage_radius
    objective_bound = None
    if rate < 1:
        objective_bound = (
            certificate.pl_certificate.smoothness
            / certificate.pl_certificate.function_storage
            * forcing
            / (1 - rate)
        )
    return AffinePortAbsorption(
        base_rate=certificate.rate,
        absorbed_rate=rate,
        constant_forcing=forcing,
        storage_radius=storage_radius,
        objective_gap_ultimate_bound=objective_bound,
        forward_invariant=forward_invariant,
    )


@dataclass(frozen=True)
class CenteredDecayCorollary:
    """Exact strong-convexity bound for decay centred at the true minimizer."""

    weight_decay: Fraction
    strong_convexity: Fraction
    normalized_output_squared_slope: Fraction
    certified_rate: Fraction

    @property
    def contractive(self) -> bool:
        return self.certified_rate < 1


def centered_decay_corollary(
    certificate: StoredSignalPortCertificate,
    *,
    weight_decay: Fraction,
    strong_convexity: Fraction,
) -> CenteredDecayCorollary:
    """Bound ``-eta*wd*(W-W_star)`` as a zero-intercept output port.

    Strong convexity gives ``||W-W_star|| <= ||grad f(W)||/ell``.  Since the
    quadratic part of ``V`` stores ``(m/L, grad/L)`` with matrix ``P``, the
    exact inverse-storage coefficient bounds the normalized gradient.  This
    is a centred-decay theorem; ordinary decay towards zero is covered only
    when the objective minimizer is zero.
    """

    if not isinstance(weight_decay, Fraction) or weight_decay < 0:
        raise ValueError("weight_decay must be a nonnegative Fraction")
    if not isinstance(strong_convexity, Fraction) or strong_convexity <= 0:
        raise ValueError("strong_convexity must be a positive Fraction")
    storage = certificate.pl_certificate.storage
    determinant = storage[0][0] * storage[1][1] - storage[0][1] ** 2
    inverse_gradient_diagonal = storage[0][0] / determinant
    squared_slope = (
        weight_decay / (certificate.pl_certificate.center_gain * strong_convexity)
    ) ** 2 * inverse_gradient_diagonal
    rate = certificate.rate + certificate.port_gains[2] * squared_slope
    return CenteredDecayCorollary(
        weight_decay=weight_decay,
        strong_convexity=strong_convexity,
        normalized_output_squared_slope=squared_slope,
        certified_rate=rate,
    )


@dataclass(frozen=True)
class ZeroCenteredDecayCounterexample:
    """Scalar quadratic witness that ordinary decay moves a nonzero minimizer."""

    learning_rate: Fraction
    weight_decay: Fraction
    minimizer: Fraction
    next_iterate: Fraction
    displacement: Fraction
    next_objective_gap: Fraction

    @property
    def original_minimizer_is_not_an_equilibrium(self) -> bool:
        return self.displacement != 0 and self.next_objective_gap > 0


def zero_centered_decay_counterexample(
    *,
    learning_rate: Fraction = Fraction(1, 120),
    weight_decay: Fraction = Fraction(1, 100),
    minimizer: Fraction = Fraction(1),
    curvature: Fraction = Fraction(1),
) -> ZeroCenteredDecayCounterexample:
    """Return the exact control for ``f(w)=curvature*(w-w_star)^2/2``."""

    values = (learning_rate, weight_decay, minimizer, curvature)
    if any(not isinstance(value, Fraction) for value in values):
        raise TypeError("counterexample inputs must be Fraction values")
    if learning_rate <= 0 or weight_decay <= 0 or minimizer == 0 or curvature <= 0:
        raise ValueError("the control requires eta, decay, curvature > 0 and a nonzero minimizer")
    next_iterate = (1 - learning_rate * weight_decay) * minimizer
    displacement = next_iterate - minimizer
    objective_gap = curvature * displacement**2 / 2
    return ZeroCenteredDecayCounterexample(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        minimizer=minimizer,
        next_iterate=next_iterate,
        displacement=displacement,
        next_objective_gap=objective_gap,
    )


def exact_certificate_checks() -> dict[str, bool]:
    """Replay the two exact P21 LMIs and the weight-decay scope controls."""

    primary = audit_stored_signal_port_certificate(FASTER_RATE_POINT)
    secondary = audit_stored_signal_port_certificate(MAXIMUM_STEP_POINT)
    centered = centered_decay_corollary(
        primary.certificate,
        weight_decay=Fraction(1, 1_000_000),
        strong_convexity=Fraction(1),
    )
    control = zero_centered_decay_counterexample()
    return {
        "variable_order_is_locked_to_seven_channels": len(PORT_VARIABLE_ORDER) == 7,
        "primary_eta_and_rate_are_exact": (
            primary.certificate.point.learning_rate == Fraction(1, 120)
            and primary.certificate.rate == Fraction(624_350_169, 625_000_000)
        ),
        "secondary_eta_and_rate_are_exact": (
            secondary.certificate.point.learning_rate == Fraction(1, 83)
            and secondary.certificate.rate == Fraction(999_598_040_401, 10**12)
        ),
        "primary_stored_signal_lmi_is_certified": primary.certified,
        "secondary_stored_signal_lmi_is_certified": secondary.certified,
        "both_zero_port_blocks_recover_P18_exactly": (
            primary.zero_port_recovers_p18_exactly and secondary.zero_port_recovers_p18_exactly
        ),
        "small_centered_decay_retains_exact_contraction": centered.contractive,
        "ordinary_decay_nonzero_minimizer_control_moves": (
            control.original_minimizer_is_not_an_equilibrium
        ),
    }


__all__ = [
    "CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION",
    "PORT_VARIABLE_ORDER",
    "PRIMARY_PORT_GAINS",
    "SECONDARY_PORT_GAINS",
    "AffinePortAbsorption",
    "AffinePortEnvelope",
    "CenteredDecayCorollary",
    "StoredSignalPortAudit",
    "StoredSignalPortCertificate",
    "ZeroCenteredDecayCounterexample",
    "absorb_affine_ports",
    "audit_stored_signal_port_certificate",
    "build_stored_signal_port_certificate",
    "centered_decay_corollary",
    "exact_certificate_checks",
    "zero_centered_decay_counterexample",
]
