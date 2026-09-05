"""Exact residual robustness for the P14 mathematical Yosida step.

Let ``A`` be the continuous full-domain monotone P13 operator,
``B=A+mu*I``, ``F=I+lambda*B``, and ``u=J(s)=F^{-1}(s)``.  For an arbitrary
approximate solution ``u_hat``, define its equation residual by

``r = s-u_hat-lambda*B(u_hat)``.

Then ``u_hat=J(s-r)``.  Strong monotonicity of ``F`` and the P14 resolvent
sector give the sharp class-uniform bounds

``||u_hat-u|| <= ||r||/(1+lambda*mu)``

and, for the *resolvent-form* approximate output

``Y_hat(s)=(s-u_hat)/lambda``,

``||Y_hat(s)-Y(s)|| <= ||r||/[lambda*(1+lambda*mu)]``.

At the locked ``lambda=1/1000`` and ``mu=1000``, the gains are ``1/2`` and
``500``.  This output convention matters: ``Y_hat`` differs from the graph
value ``B(u_hat)`` by ``r/lambda``.  Directly using ``B(u_hat)=Y(s-r)`` has
only the larger class-uniform output-error bound ``||r||/lambda`` and is not
covered by the locked P15 storage certificate.

If ``||r|| <= kappa*||s||+r_bar``, the output error can be split into a
relative part bounded by ``500*kappa*||s||`` and an absolute part bounded by
``500*r_bar``.  With ``kappa=1/250``, the relative part enlarges P14's
centered residual radius from ``250`` to ``252``.  The frozen P14 storage and
multipliers remain strict at radius ``252`` but fail at radius ``253``.  A
fifth disturbance coordinate proves

``V_(t+1) <= (249001/250000)*V_t + (5/2)*r_bar^2``.

This is a dimension-independent exact-real theorem for every differentiable,
globally 10-smooth objective with finite infimum satisfying the global PL
inequality with constant 1.  It assumes that the displayed residual bound is
verified at every solve.  It does not specify a numerical solver, iteration
count, stopping-cost guarantee, BF16 implementation, or upstream Muon kernel.
All authoritative calculations use :class:`fractions.Fraction`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction

from passive_muon.momentum_iqc import RationalMatrix, leading_principal_minors
from passive_muon.pl_convergence import (
    PlConvergenceAudit,
    PlConvergenceCertificate,
    audit_pl_convergence,
    pl_supply_function_coefficients,
)
from passive_muon.yosida_stability import (
    LOCKED_BASE_STRONG_MONOTONICITY,
    LOCKED_BETA,
    LOCKED_LEARNING_RATE,
    LOCKED_RATE_SQUARED,
    LOCKED_RESOLVENT_PARAMETER,
    YosidaSector,
    locked_yosida_pl_certificate,
    locked_yosida_sector,
)

INEXACT_YOSIDA_ROBUSTNESS_SCHEMA_VERSION = "passive-muon-inexact-yosida-robustness-certificate-v1"

LOCKED_SOLVER_RELATIVE_RESIDUAL = Fraction(1, 250)
LOCKED_SOLUTION_ERROR_GAIN = Fraction(1, 2)
LOCKED_RESOLVENT_FORM_OUTPUT_ERROR_GAIN = Fraction(500)
LOCKED_GRAPH_FORM_OUTPUT_ERROR_GAIN = Fraction(1_000)
LOCKED_EFFECTIVE_RESIDUAL_LIPSCHITZ = Fraction(252)
LOCKED_ABSOLUTE_OUTPUT_PENALTY = Fraction(1, 100_000)
LOCKED_ABSOLUTE_FORCING_GAIN = Fraction(5, 2)
LOCKED_PASSING_FROZEN_RADIUS = Fraction(252)
LOCKED_FAILING_FROZEN_RADIUS = Fraction(253)

# At radius 252, the fifth leading minor of ``-LMI`` is affine in the
# physical output-error penalty.  This is its exact zero.  The selected clean
# penalty 1/100000 is strictly larger.
FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD = Fraction(
    7_167_838_551_819_102_967_459_883_001_403_922_668_449_707_353,
    1_283_935_131_617_993_795_632_989_901_464_334_924_254_261_960_000_000,
)

RationalVector = tuple[Fraction, ...]


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")


def _require_nonnegative_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


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
    """Return the quadratic part of directed normalized smooth interpolation."""

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
class InexactResolventBounds:
    """Sharp class-uniform residual gains for one strongly monotone resolvent."""

    resolvent_parameter: Fraction
    base_strong_monotonicity: Fraction

    def __post_init__(self) -> None:
        _require_fraction(self.resolvent_parameter, "resolvent_parameter")
        _require_fraction(self.base_strong_monotonicity, "base_strong_monotonicity")
        if self.resolvent_parameter <= 0 or self.base_strong_monotonicity <= 0:
            raise ValueError(
                "resolvent parameter and strong-monotonicity constant must be positive"
            )

    @property
    def denominator(self) -> Fraction:
        """Return ``1+lambda*mu``."""

        return 1 + self.resolvent_parameter * self.base_strong_monotonicity

    @property
    def solution_error_gain(self) -> Fraction:
        """Return the sharp gain from equation residual to solution error."""

        return 1 / self.denominator

    @property
    def resolvent_form_output_error_gain(self) -> Fraction:
        """Return the sharp residual gain for ``(s-u_hat)/lambda``."""

        return 1 / (self.resolvent_parameter * self.denominator)

    @property
    def graph_form_output_error_gain(self) -> Fraction:
        """Return the class-uniform residual gain for ``B(u_hat)``."""

        return 1 / self.resolvent_parameter


@dataclass(frozen=True)
class ResidualErrorEstimate:
    """Exact norm bounds induced by a supplied equation-residual magnitude."""

    residual_norm: Fraction
    solution_error_bound: Fraction
    resolvent_form_output_error_bound: Fraction
    graph_form_output_error_bound: Fraction


@dataclass(frozen=True)
class ScalarResidualSharpnessControl:
    """Lower-endpoint scalar map attaining the solution and output gains."""

    input_value: Fraction
    residual_value: Fraction
    exact_solution: Fraction
    approximate_solution: Fraction
    exact_output: Fraction
    approximate_resolvent_output: Fraction
    solution_error_ratio: Fraction
    output_error_ratio: Fraction


@dataclass(frozen=True)
class RelativeToleranceControl:
    """Adversarial relative-residual control for the scalar map ``A=0``."""

    relative_tolerance: Fraction
    residual_multiplier: Fraction
    approximate_output_slope: Fraction
    scalar_curvature: Fraction
    p_at_one: Fraction

    @property
    def stalls(self) -> bool:
        return self.approximate_output_slope == 0

    @property
    def violates_schur_condition(self) -> bool:
        return self.p_at_one < 0


@dataclass(frozen=True)
class InexactYosidaCertificate:
    """Exact PL storage certificate with a relative/absolute solve residual."""

    sector: YosidaSector
    pl_certificate: PlConvergenceCertificate
    relative_residual_slope: Fraction
    absolute_output_penalty: Fraction

    def __post_init__(self) -> None:
        _require_nonnegative_fraction(self.relative_residual_slope, "relative_residual_slope")
        _require_fraction(self.absolute_output_penalty, "absolute_output_penalty")
        if self.absolute_output_penalty <= 0:
            raise ValueError("absolute_output_penalty must be positive")
        if self.pl_certificate.center_gain != self.sector.center_gain:
            raise ValueError("PL center gain must equal the Yosida sector center")
        if self.pl_certificate.residual_lipschitz != self.effective_residual_lipschitz:
            raise ValueError("PL residual radius must include the relative solve error")

    @property
    def residual_bounds(self) -> InexactResolventBounds:
        return InexactResolventBounds(
            resolvent_parameter=self.sector.resolvent_parameter,
            base_strong_monotonicity=self.sector.base_strong_monotonicity,
        )

    @property
    def output_error_gain(self) -> Fraction:
        """Gain for the resolvent-form output used by this certificate."""

        return self.residual_bounds.resolvent_form_output_error_gain

    @property
    def effective_residual_lipschitz(self) -> Fraction:
        """Return ``P14 radius + output_gain*kappa``."""

        return (
            self.sector.residual_lipschitz + self.output_error_gain * self.relative_residual_slope
        )

    @property
    def rate(self) -> Fraction:
        return self.pl_certificate.tau**2

    @property
    def normalized_absolute_output_penalty(self) -> Fraction:
        """Penalty on ``h=d/(gamma*L)`` in the exact lifted LMI."""

        scale = self.pl_certificate.center_gain * self.pl_certificate.smoothness
        return self.absolute_output_penalty * scale**2

    @property
    def absolute_forcing_gain(self) -> Fraction:
        """Return ``C`` in ``V+ <= q*V+C*r_bar^2``."""

        return self.absolute_output_penalty * self.output_error_gain**2

    @property
    def ultimate_storage_gain(self) -> Fraction:
        """Return ``C/(1-q)`` for a uniform absolute residual floor."""

        return self.absolute_forcing_gain / (1 - self.rate)


@dataclass(frozen=True)
class InexactYosidaMatrices:
    """Exact lift in ``chi=(m/L,g/L,v,g_next/L,h)``."""

    transition: RationalMatrix
    selection: RationalMatrix
    signal_selector: RationalVector
    step_selector: RationalVector
    interpolation_12: RationalMatrix
    interpolation_21: RationalMatrix
    pl_next: RationalMatrix
    residual_norm: RationalMatrix
    absolute_output_error_norm: RationalMatrix


@dataclass(frozen=True)
class InexactYosidaAudit:
    """Exact Sylvester and function-flow replay for P15."""

    lmi: RationalMatrix
    storage_leading_minors: tuple[Fraction, ...]
    negative_lmi_leading_minors: tuple[Fraction, ...]
    supply_function_coefficients: tuple[Fraction, Fraction]
    expected_function_coefficients: tuple[Fraction, Fraction]
    checks: tuple[tuple[str, bool], ...]

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
    def all_checks_pass(self) -> bool:
        return all(value for _name, value in self.checks)

    @property
    def certified(self) -> bool:
        return (
            self.storage_positive_definite
            and self.lmi_negative_definite
            and self.function_values_cancel
            and self.all_checks_pass
        )


@dataclass(frozen=True)
class FrozenStorageRadiusControl:
    """Audit of the unchanged P14 storage/multipliers at one residual radius.

    Failure means only that this frozen certificate is rejected.  It is not an
    impossibility theorem for all storages or all IQC multipliers.
    """

    residual_radius: Fraction
    audit: PlConvergenceAudit

    @property
    def certificate_passes(self) -> bool:
        return self.audit.certified


def locked_inexact_resolvent_bounds() -> InexactResolventBounds:
    """Return the locked ``lambda=1/1000, mu=1000`` residual gains."""

    return InexactResolventBounds(
        resolvent_parameter=LOCKED_RESOLVENT_PARAMETER,
        base_strong_monotonicity=LOCKED_BASE_STRONG_MONOTONICITY,
    )


def residual_error_estimate(
    residual_norm: Fraction,
    bounds: InexactResolventBounds | None = None,
) -> ResidualErrorEstimate:
    """Evaluate all three exact class-uniform residual bounds."""

    _require_nonnegative_fraction(residual_norm, "residual_norm")
    selected = locked_inexact_resolvent_bounds() if bounds is None else bounds
    return ResidualErrorEstimate(
        residual_norm=residual_norm,
        solution_error_bound=selected.solution_error_gain * residual_norm,
        resolvent_form_output_error_bound=(
            selected.resolvent_form_output_error_gain * residual_norm
        ),
        graph_form_output_error_bound=selected.graph_form_output_error_gain * residual_norm,
    )


def scalar_residual_sharpness_control(
    input_value: Fraction = Fraction(7, 5),
    residual_value: Fraction = Fraction(3, 11),
) -> ScalarResidualSharpnessControl:
    """Return an exact equality witness using ``B=mu*I``."""

    _require_fraction(input_value, "input_value")
    _require_fraction(residual_value, "residual_value")
    if residual_value == 0:
        raise ValueError("residual_value must be nonzero for a ratio witness")
    bounds = locked_inexact_resolvent_bounds()
    exact_solution = input_value / bounds.denominator
    approximate_solution = (input_value - residual_value) / bounds.denominator
    exact_output = (input_value - exact_solution) / bounds.resolvent_parameter
    approximate_output = (input_value - approximate_solution) / bounds.resolvent_parameter
    return ScalarResidualSharpnessControl(
        input_value=input_value,
        residual_value=residual_value,
        exact_solution=exact_solution,
        approximate_solution=approximate_solution,
        exact_output=exact_output,
        approximate_resolvent_output=approximate_output,
        solution_error_ratio=abs(approximate_solution - exact_solution) / abs(residual_value),
        output_error_ratio=abs(approximate_output - exact_output) / abs(residual_value),
    )


def relative_tolerance_control(relative_tolerance: Fraction) -> RelativeToleranceControl:
    """Return an exact adversarial control for ``A=0`` and ``r=-kappa*s``."""

    _require_nonnegative_fraction(relative_tolerance, "relative_tolerance")
    bounds = locked_inexact_resolvent_bounds()
    output_slope = (1 - relative_tolerance) / (bounds.resolvent_parameter * bounds.denominator)
    curvature = Fraction(1)
    p_at_one = LOCKED_LEARNING_RATE * (1 - LOCKED_BETA) * curvature * output_slope
    return RelativeToleranceControl(
        relative_tolerance=relative_tolerance,
        residual_multiplier=-relative_tolerance,
        approximate_output_slope=output_slope,
        scalar_curvature=curvature,
        p_at_one=p_at_one,
    )


def unit_relative_residual_stall_control() -> RelativeToleranceControl:
    """Return the ``kappa=1`` admissible residual that makes the output zero."""

    return relative_tolerance_control(Fraction(1))


def double_relative_residual_instability_control() -> RelativeToleranceControl:
    """Return the ``kappa=2`` control with negative Jury value ``p(1)``."""

    return relative_tolerance_control(Fraction(2))


def locked_inexact_yosida_pl_certificate() -> PlConvergenceCertificate:
    """Return the frozen P14 certificate with effective radius ``252``."""

    return replace(
        locked_yosida_pl_certificate(),
        residual_lipschitz=LOCKED_EFFECTIVE_RESIDUAL_LIPSCHITZ,
    )


def locked_inexact_yosida_certificate() -> InexactYosidaCertificate:
    """Return the selected full-step P15 robust solve certificate."""

    return InexactYosidaCertificate(
        sector=locked_yosida_sector(),
        pl_certificate=locked_inexact_yosida_pl_certificate(),
        relative_residual_slope=LOCKED_SOLVER_RELATIVE_RESIDUAL,
        absolute_output_penalty=LOCKED_ABSOLUTE_OUTPUT_PENALTY,
    )


def inexact_yosida_matrices(
    certificate: InexactYosidaCertificate | None = None,
) -> InexactYosidaMatrices:
    """Build the exact five-coordinate robust smooth-PL lift.

    ``v`` normalizes the combined P14 and relative-solve residual at radius
    ``K=252``.  ``h=d/(gamma*L)`` is the absolute output-error part.
    """

    selected = locked_inexact_yosida_certificate() if certificate is None else certificate
    base = selected.pl_certificate
    zero = Fraction(0)
    one = Fraction(1)
    basis = tuple(tuple(one if row == column else zero for row in range(5)) for column in range(5))
    momentum, gradient, residual, gradient_next, absolute_error = basis
    beta = base.beta
    alpha = base.dimensionless_step
    signal = _vector_add(
        _vector_scale(beta**2, momentum),
        _vector_scale(one - beta**2, gradient),
    )
    step = _vector_add(
        _vector_scale(-alpha, signal),
        _vector_scale(-alpha * base.residual_ratio, residual),
        _vector_scale(-alpha, absolute_error),
    )
    momentum_next = _vector_add(
        _vector_scale(beta, momentum),
        _vector_scale(one - beta, gradient),
    )
    transition = (momentum_next, gradient_next)
    selection = (momentum, gradient)
    null = (zero,) * 5
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

    return InexactYosidaMatrices(
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
        absolute_output_error_norm=_outer(absolute_error, absolute_error),
    )


def inexact_yosida_lmi_matrix(
    certificate: InexactYosidaCertificate | None = None,
) -> RationalMatrix:
    """Return the exact ``5 x 5`` pathwise inexact-solve LMI."""

    selected = locked_inexact_yosida_certificate() if certificate is None else certificate
    base = selected.pl_certificate
    matrices = inexact_yosida_matrices(selected)
    return _matrix_add(
        _matrix_multiply(
            _transpose(matrices.transition),
            _matrix_multiply(base.storage, matrices.transition),
        ),
        _matrix_scale(
            -selected.rate,
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
            -selected.normalized_absolute_output_penalty,
            matrices.absolute_output_error_norm,
        ),
    )


def frozen_storage_radius_control(residual_radius: Fraction) -> FrozenStorageRadiusControl:
    """Replay the unchanged P14 storage and multipliers at one radius."""

    _require_nonnegative_fraction(residual_radius, "residual_radius")
    certificate = replace(
        locked_yosida_pl_certificate(),
        residual_lipschitz=residual_radius,
    )
    return FrozenStorageRadiusControl(
        residual_radius=residual_radius,
        audit=audit_pl_convergence(certificate),
    )


def exact_certificate_checks() -> dict[str, bool]:
    """Replay the residual identities, robust LMI, and tolerance controls."""

    bounds = locked_inexact_resolvent_bounds()
    certificate = locked_inexact_yosida_certificate()
    lmi = inexact_yosida_lmi_matrix(certificate)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    witness = scalar_residual_sharpness_control()
    stall = unit_relative_residual_stall_control()
    unstable = double_relative_residual_instability_control()
    radius_pass = frozen_storage_radius_control(LOCKED_PASSING_FROZEN_RADIUS)
    radius_fail = frozen_storage_radius_control(LOCKED_FAILING_FROZEN_RADIUS)
    return {
        "locked_residual_gains": (
            bounds.denominator == 2
            and bounds.solution_error_gain == LOCKED_SOLUTION_ERROR_GAIN
            and bounds.resolvent_form_output_error_gain == LOCKED_RESOLVENT_FORM_OUTPUT_ERROR_GAIN
            and bounds.graph_form_output_error_gain == LOCKED_GRAPH_FORM_OUTPUT_ERROR_GAIN
        ),
        "lower_endpoint_attains_both_locked_gains": (
            witness.solution_error_ratio == LOCKED_SOLUTION_ERROR_GAIN
            and witness.output_error_ratio == LOCKED_RESOLVENT_FORM_OUTPUT_ERROR_GAIN
        ),
        "relative_budget_gives_radius_252": (
            certificate.effective_residual_lipschitz == LOCKED_EFFECTIVE_RESIDUAL_LIPSCHITZ
        ),
        "selected_rate_is_p14_rate": certificate.rate == LOCKED_RATE_SQUARED,
        "selected_forcing_gain_is_5_over_2": (
            certificate.absolute_forcing_gain == LOCKED_ABSOLUTE_FORCING_GAIN
        ),
        "selected_penalty_is_above_strict_threshold": (
            certificate.absolute_output_penalty > FROZEN_ABSOLUTE_OUTPUT_PENALTY_THRESHOLD
        ),
        "five_by_five_lmi_is_strict": all(
            value > 0 for value in leading_principal_minors(negative_lmi)
        ),
        "frozen_radius_252_passes": radius_pass.certificate_passes,
        "frozen_radius_253_is_rejected": not radius_fail.certificate_passes,
        "unit_relative_residual_can_stall": stall.stalls and stall.p_at_one == 0,
        "double_relative_residual_is_unstable": (
            unstable.approximate_output_slope == -500
            and unstable.p_at_one == Fraction(-1, 1_280)
            and unstable.violates_schur_condition
        ),
    }


def audit_inexact_yosida_robustness(
    certificate: InexactYosidaCertificate | None = None,
) -> InexactYosidaAudit:
    """Return the authoritative exact P15 core audit."""

    selected = locked_inexact_yosida_certificate() if certificate is None else certificate
    base = selected.pl_certificate
    lmi = inexact_yosida_lmi_matrix(selected)
    negative_lmi = _matrix_scale(Fraction(-1), lmi)
    checks = (
        exact_certificate_checks()
        if certificate is None
        else {
            "effective_radius_matches": (
                base.residual_lipschitz == selected.effective_residual_lipschitz
            ),
            "lmi_is_strict": all(value > 0 for value in leading_principal_minors(negative_lmi)),
        }
    )
    return InexactYosidaAudit(
        lmi=lmi,
        storage_leading_minors=leading_principal_minors(base.storage),
        negative_lmi_leading_minors=leading_principal_minors(negative_lmi),
        supply_function_coefficients=pl_supply_function_coefficients(base),
        expected_function_coefficients=(
            base.function_storage * selected.rate,
            -base.function_storage,
        ),
        checks=tuple(checks.items()),
    )
