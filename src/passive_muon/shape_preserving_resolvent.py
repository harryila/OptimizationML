"""Pointwise-sector certificates for a gated, shape-preserving resolvent map.

This module studies the exact-real mathematical interface

``T(S) = (1-theta(||S||_F**2))*Y(S)/c + theta(||S||_F**2)*X(S)``,

where ``Y=(I-J)/lambda`` is the P14 Yosida map, ``J`` is the resolvent of
the shifted P13 operator, and ``X(S)=E_h,epsilon(J(S))`` retains the unshifted
five-stage Jordan response.  The gate is a locked C2 quintic smootherstep.

The exact result is an origin-centred *pointwise* gain sector.  Since ``Y``,
``J``, and ``E_h,epsilon`` preserve the singular vectors of ``S``, every
singular-mode gain of ``T`` lies in a common interval ``[m,M]``.  Therefore

``||T(S)-gamma*S||_F <= K*||S||_F``,

with ``gamma=(m+M)/2`` and ``K=(M-m)/2``.  This supply is sufficient for the
global smooth-PL trajectory certificate in :mod:`passive_muon.pl_convergence`.
It is not an incremental sector: differentiating the radial gate introduces
an additional rank-one term, and no incremental claim is made here.

All theorem constants and LMI replays use :class:`fractions.Fraction`.  The
FP64 evaluator is a guarded reference implementation which calls the P16
solver and checks its computed P15 residual.  It is neither an IEEE-754 error
certificate nor a theorem for a BF16 or upstream Muon implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

import numpy as np
from numpy.typing import NDArray

from passive_muon.additive_epsilon_deficit import DEPLOYED_EPSILON
from passive_muon.equivariant_resolvent_solver import (
    LOCKED_FP64_ABSOLUTE_RESIDUAL,
    EquivariantResolventSolverConfig,
    MatrixSolveResult,
    SolverDiagnostics,
    jordan_response_and_derivative_fp64,
    solve_resolvent_fp64,
    solve_resolvent_singular_values,
)
from passive_muon.floored_certificate import JORDAN_DERIVATIVE_UPPER
from passive_muon.pl_convergence import (
    PlConvergenceAudit,
    PlConvergenceCertificate,
    audit_pl_convergence,
)
from passive_muon.radial_passivation_tradeoff import (
    TAIL_DERIVATIVE_LOWER,
    TAIL_PROJECTION_LOWER,
    UNIT_DEFICIT_UPPER,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    FULL_STEP_DESIGN,
    GATE_INNER_SQUARED_RADIUS,
    GATE_OUTER_SQUARED_RADIUS,
    PRIMARY_DESIGN,
    RAW_SHAPE_GAIN_UPPER,
    GateDesign,
    smootherstep_gate_derivatives,
)
from passive_muon.shape_preserving_resolvent_certificate import (
    smootherstep_gate as _smootherstep_gate_exact,
)
from passive_muon.yosida_stability import (
    LOCKED_BASE_STRONG_MONOTONICITY,
    LOCKED_BETA,
    LOCKED_PL_CONSTANT,
    LOCKED_RESOLVENT_PARAMETER,
    LOCKED_SMOOTHNESS,
    LOCKED_YOSIDA_LIPSCHITZ,
    LOCKED_YOSIDA_STRONG_MONOTONICITY,
)

FloatArray = NDArray[np.float64]

SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION = "passive-muon-shape-preserving-resolvent-certificate-v1"

LOCKED_GATE_Q0: Final = GATE_INNER_SQUARED_RADIUS
LOCKED_GATE_Q1: Final = GATE_OUTER_SQUARED_RADIUS

# The exact gain of X(S)=E_h,epsilon(J(S)) relative to S is derived below,
# rather than inferred from sampled singular values.
LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER: Final = RAW_SHAPE_GAIN_UPPER


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")


def _require_positive_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_gate_design(design: GateDesign) -> None:
    if not isinstance(design, GateDesign):
        raise TypeError("design must be a GateDesign")
    if not isinstance(design.name, str) or not design.name:
        raise ValueError("design name must be a nonempty string")
    _require_positive_fraction(design.cap, "design.cap")
    _require_positive_fraction(design.passive_divisor, "design.passive_divisor")
    _require_positive_fraction(design.learning_rate, "design.learning_rate")
    if design.cap >= 1:
        raise ValueError("design cap must lie strictly below one")


@dataclass(frozen=True)
class QuinticGateJet:
    """Exact value and first two ``q`` derivatives of the locked gate."""

    value: Fraction
    first_derivative: Fraction
    second_derivative: Fraction


@dataclass(frozen=True)
class PointwiseGainSector:
    """Dimension-uniform origin-centred gain bounds for the exact-real map.

    This is a pointwise modal sector, not an incremental slope restriction.
    """

    lower: Fraction
    upper: Fraction
    center: Fraction
    radius: Fraction
    jordan_after_resolvent_upper: Fraction

    def __post_init__(self) -> None:
        for name in (
            "lower",
            "upper",
            "center",
            "radius",
            "jordan_after_resolvent_upper",
        ):
            _require_fraction(getattr(self, name), name)
        if not 0 < self.lower <= self.upper:
            raise ValueError("pointwise sector must satisfy 0 < lower <= upper")
        if self.center != (self.lower + self.upper) / 2:
            raise ValueError("center must be the midpoint of the sector")
        if self.radius != (self.upper - self.lower) / 2:
            raise ValueError("radius must be half the sector width")

    @property
    def condition_ratio(self) -> Fraction:
        """Return ``upper/lower`` exactly."""

        return self.upper / self.lower


@dataclass(frozen=True)
class ShapePreservingResolventAudit:
    """Exact gate, gain-sector, and smooth-PL audit for one locked design."""

    config: GateDesign
    sector: PointwiseGainSector
    pl_certificate: PlConvergenceCertificate
    pl_audit: PlConvergenceAudit
    checks: tuple[tuple[str, bool], ...]

    @property
    def all_checks_pass(self) -> bool:
        return all(value for _name, value in self.checks)

    @property
    def certified(self) -> bool:
        """Whether every exact algebraic and LMI check passes."""

        return self.all_checks_pass and self.pl_audit.certified


@dataclass(frozen=True)
class ShapePreservingResolventEvaluation:
    """One guarded FP64 evaluation of the proposed gated interface.

    The arrays are numerical evidence only.  The exact pointwise theorem is
    about the mathematical resolvent, while this evaluator uses P16's
    approximate solution accepted by its computed-residual postcondition.
    """

    output: FloatArray
    scaled_yosida_output: FloatArray
    jordan_output: FloatArray
    gate_weight: float
    squared_signal_norm: float
    resolvent_result: MatrixSolveResult


@dataclass(frozen=True)
class ShapePreservingSingularValueEvaluation:
    """Guarded reduced-coordinate FP64 evaluation of the gated interface.

    ``yosida_output`` is the raw P16 resolvent-form output, before division
    by the design's ``passive_divisor``.  ``shape_output`` is the five-stage
    Jordan component evaluated at the approximate resolvent singular values.
    ``output`` is their gated blend.  The result is numerical evidence and
    does not promote the computed-residual check into an FP64 error theorem.
    """

    solution_singular_values: FloatArray
    yosida_output: FloatArray
    shape_output: FloatArray
    output: FloatArray
    gate_value: float
    squared_signal_norm: float
    solver_diagnostics: SolverDiagnostics


LOCKED_FULL_STEP_CONFIG: Final = FULL_STEP_DESIGN
LOCKED_HIGH_FIDELITY_CONFIG: Final = PRIMARY_DESIGN
LOCKED_FULL_STEP_TAU: Final = Fraction(16_777_209, 16_777_216)
LOCKED_HIGH_FIDELITY_TAU: Final = Fraction(16_777_215, 16_777_216)


def quintic_gate_jet_exact(
    squared_norm: Fraction,
    design: GateDesign,
) -> QuinticGateJet:
    """Return the exact C2 gate value and its first two ``q`` derivatives."""

    _require_nonnegative_fraction(squared_norm, "squared_norm")
    _validate_gate_design(design)
    value, first, second = smootherstep_gate_derivatives(squared_norm, design.cap)
    return QuinticGateJet(value, first, second)


def quintic_gate_weight_fp64(
    squared_norm: float,
    design: GateDesign,
) -> float:
    """Evaluate the locked quintic gate in FP64."""

    _validate_gate_design(design)
    if not math.isfinite(squared_norm) or squared_norm < 0.0:
        raise ValueError("squared_norm must be finite and nonnegative")
    q0 = float(LOCKED_GATE_Q0)
    q1 = float(LOCKED_GATE_Q1)
    if squared_norm <= q0:
        return 0.0
    if squared_norm >= q1:
        return float(design.cap)
    coordinate = (squared_norm - q0) / (q1 - q0)
    smootherstep = coordinate**3 * (10.0 + coordinate * (-15.0 + 6.0 * coordinate))
    # Horner evaluation can overshoot an endpoint by a few ulps even when the
    # exact smootherstep lies in [0, 1].  Admit only that rounding-scale
    # excursion, then clamp to the theorem's exact range.
    endpoint_tolerance = 16.0 * math.ulp(1.0)
    if (
        not math.isfinite(smootherstep)
        or smootherstep < -endpoint_tolerance
        or smootherstep > 1.0 + endpoint_tolerance
    ):
        raise FloatingPointError("invalid FP64 smootherstep value")
    smootherstep = min(1.0, max(0.0, smootherstep))
    result = float(design.cap) * smootherstep
    if not math.isfinite(result) or not 0.0 <= result <= float(design.cap):
        raise FloatingPointError("invalid FP64 gate value")
    return result


def smootherstep_gate(
    squared_norm: Fraction | float,
    design: GateDesign = PRIMARY_DESIGN,
) -> Fraction | float:
    """Evaluate the C2 gate exactly for ``Fraction`` input, otherwise in FP64."""

    _validate_gate_design(design)
    if isinstance(squared_norm, Fraction):
        return _smootherstep_gate_exact(squared_norm, design.cap)
    if not isinstance(squared_norm, (float, np.floating)):
        raise TypeError("squared_norm must be a Fraction or floating-point scalar")
    return quintic_gate_weight_fp64(float(squared_norm), design)


def jordan_after_resolvent_gain_upper() -> Fraction:
    """Derive the exact dimension-uniform bound on ``X(S)/S``.

    In a common singular basis, the exact resolvent equation is

    ``sigma_i = (1+lambda*(mu+g))*x_i + lambda*e_i``,

    where ``e_i=h(w_i)``, ``w_i=x_i/(epsilon*(1+z))``, and
    ``g=(p(z)/z)/epsilon``.  For a positive mode put ``H_i=e_i/x_i``.
    The bounds ``0<=h(w)/w<=b`` and ``p(z)/z>=d_hat(z)`` imply

    ``H_i<=b/(epsilon*(1+z))`` and ``g>=d_hat(z)/epsilon``.

    Moreover ``(1+z)*d_hat(z)>=U``: it is immediate on the constant
    branch, while on the tail the left side is a convex combination of
    ``-a`` and ``-Gamma``, both strictly above ``U``.  Monotonicity of
    ``H/(lambda**-1+mu+g+H)`` then yields the returned bound.  Zero modes
    follow by rank preservation and continuity; repeated modes do not depend
    on a choice of singular basis.
    """

    b = JORDAN_DERIVATIVE_UPPER
    epsilon = DEPLOYED_EPSILON
    lam = LOCKED_RESOLVENT_PARAMETER
    mu = LOCKED_BASE_STRONG_MONOTONICITY
    return (b / epsilon) / (1 + lam * (mu + (UNIT_DEFICIT_UPPER + b) / epsilon))


def pointwise_gain_sector(
    config: GateDesign,
) -> PointwiseGainSector:
    """Return the exact full-matrix pointwise sector for one gate design.

    The singular graph directly gives a Yosida modal gain in ``[500,1000)``,
    which is enclosed by ``[500,1000]``.  After division by ``c`` its
    contribution lies in ``[500/c,1000/c]``.  The unshifted Jordan component
    has modal gain in ``[0,M_X]`` relative to the original input.
    Taking the interval hull over ``0<=theta<=theta_bar`` gives the bounds
    below in every fixed finite rectangular matrix space.
    """

    _validate_gate_design(config)
    ceiling = config.cap
    divisor = config.passive_divisor
    jordan_upper = jordan_after_resolvent_gain_upper()
    lower = (1 - ceiling) * LOCKED_YOSIDA_STRONG_MONOTONICITY / divisor
    pure_yosida_upper = LOCKED_YOSIDA_LIPSCHITZ / divisor
    endpoint_upper = (1 - ceiling) * LOCKED_YOSIDA_LIPSCHITZ / divisor + ceiling * jordan_upper
    upper = max(pure_yosida_upper, endpoint_upper)
    return PointwiseGainSector(
        lower=lower,
        upper=upper,
        center=(lower + upper) / 2,
        radius=(upper - lower) / 2,
        jordan_after_resolvent_upper=jordan_upper,
    )


LOCKED_FULL_STEP_PL_CERTIFICATE: Final = PlConvergenceCertificate(
    pl_constant=LOCKED_PL_CONSTANT,
    smoothness=LOCKED_SMOOTHNESS,
    beta=LOCKED_BETA,
    center_gain=Fraction(
        41_421_767_353_260_183_227_140_375,
        877_960_272_438_948_654_710_784,
    ),
    residual_lipschitz=Fraction(
        41_374_879_216_151_779_902_380_125,
        877_960_272_438_948_654_710_784,
    ),
    learning_rate=LOCKED_FULL_STEP_CONFIG.learning_rate,
    tau=LOCKED_FULL_STEP_TAU,
    storage=(
        (Fraction(13_151, 100_000), Fraction(-6_559, 50_000)),
        (Fraction(-6_559, 50_000), Fraction(6_547, 50_000)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(24_999, 5_000),
    lambda_residual_norm=Fraction(737, 100_000),
)

LOCKED_HIGH_FIDELITY_PL_CERTIFICATE: Final = PlConvergenceCertificate(
    pl_constant=LOCKED_PL_CONSTANT,
    smoothness=LOCKED_SMOOTHNESS,
    beta=LOCKED_BETA,
    center_gain=Fraction(
        20_679_066_726_449_389_357_482_875,
        73_163_356_036_579_054_559_232,
    ),
    residual_lipschitz=Fraction(
        20_676_833_958_015_655_865_827_625,
        73_163_356_036_579_054_559_232,
    ),
    learning_rate=LOCKED_HIGH_FIDELITY_CONFIG.learning_rate,
    tau=LOCKED_HIGH_FIDELITY_TAU,
    storage=(
        (Fraction(1_792, 7_757), Fraction(-2_143, 9_279)),
        (Fraction(-2_143, 9_279), Fraction(2_168, 9_389)),
    ),
    function_storage=Fraction(1),
    interpolation_reverse_weight=Fraction(1_831, 200),
    lambda_residual_norm=Fraction(110, 9_963),
)


def locked_pl_certificate(
    config: GateDesign,
) -> PlConvergenceCertificate:
    """Return the exact PL certificate associated with a locked design."""

    _validate_gate_design(config)
    if config == LOCKED_FULL_STEP_CONFIG:
        return LOCKED_FULL_STEP_PL_CERTIFICATE
    if config == LOCKED_HIGH_FIDELITY_CONFIG:
        return LOCKED_HIGH_FIDELITY_PL_CERTIFICATE
    raise ValueError("no exact PL certificate is locked for this configuration")


def audit_shape_preserving_resolvent(
    config: GateDesign = PRIMARY_DESIGN,
) -> ShapePreservingResolventAudit:
    """Replay the exact gate identities, pointwise sector, and PL LMI."""

    certificate = locked_pl_certificate(config)
    sector = pointwise_gain_sector(config)
    gain_upper = jordan_after_resolvent_gain_upper()
    q0_jet = quintic_gate_jet_exact(LOCKED_GATE_Q0, config)
    q1_jet = quintic_gate_jet_exact(LOCKED_GATE_Q1, config)
    midpoint = (LOCKED_GATE_Q0 + LOCKED_GATE_Q1) / 2
    midpoint_jet = quintic_gate_jet_exact(midpoint, config)
    checks = (
        (
            "jordan_after_resolvent_gain_identity",
            gain_upper == LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER,
        ),
        ("tail_a_endpoint_dominates_U", -TAIL_DERIVATIVE_LOWER > UNIT_DEFICIT_UPPER),
        (
            "tail_Gamma_endpoint_dominates_U",
            -TAIL_PROJECTION_LOWER > UNIT_DEFICIT_UPPER,
        ),
        (
            "gate_left_C2_jet",
            q0_jet == QuinticGateJet(Fraction(0), Fraction(0), Fraction(0)),
        ),
        (
            "gate_right_C2_jet",
            q1_jet == QuinticGateJet(config.cap, Fraction(0), Fraction(0)),
        ),
        ("gate_midpoint_value", midpoint_jet.value == config.cap / 2),
        ("sector_center_matches_certificate", sector.center == certificate.center_gain),
        (
            "sector_radius_matches_certificate",
            sector.radius == certificate.residual_lipschitz,
        ),
        ("learning_rate_matches_certificate", config.learning_rate == certificate.learning_rate),
        (
            "tau_matches_locked_design",
            certificate.tau
            == (
                LOCKED_FULL_STEP_TAU
                if config == LOCKED_FULL_STEP_CONFIG
                else LOCKED_HIGH_FIDELITY_TAU
            ),
        ),
        ("locked_epsilon", Fraction(1, 10_000_000) == DEPLOYED_EPSILON),
        ("locked_resolvent_parameter", Fraction(1, 1_000) == LOCKED_RESOLVENT_PARAMETER),
        (
            "locked_base_strong_monotonicity",
            Fraction(1_000) == LOCKED_BASE_STRONG_MONOTONICITY,
        ),
    )
    return ShapePreservingResolventAudit(
        config=config,
        sector=sector,
        pl_certificate=certificate,
        pl_audit=audit_pl_convergence(certificate),
        checks=checks,
    )


def _require_fp64_matrix(matrix: FloatArray, name: str) -> FloatArray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if matrix.dtype != np.float64:
        raise TypeError(f"{name} must have dtype numpy.float64")
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite entries")
    return np.ascontiguousarray(matrix)


def _scaled_frobenius_norm(matrix: FloatArray) -> float:
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0.0:
        return 0.0
    scaled = matrix / maximum
    return maximum * math.sqrt(float(np.dot(scaled.ravel(), scaled.ravel())))


def additive_epsilon_jordan_fp64(
    matrix: FloatArray,
    *,
    epsilon: float = float(DEPLOYED_EPSILON),
) -> FloatArray:
    """Evaluate the exact-formula five-stage Jordan component in FP64.

    This routine fixes the arithmetic order but does not certify its rounding
    error.  The additive epsilon is outside the Frobenius norm.
    """

    selected = _require_fp64_matrix(matrix, "matrix")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    radius = _scaled_frobenius_norm(selected)
    if radius == 0.0:
        return np.zeros_like(selected)
    try:
        left, singular_values, right_transpose = np.linalg.svd(selected, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("FP64 Jordan-component SVD did not converge") from error
    normalized = np.ascontiguousarray(singular_values / (radius + epsilon), dtype=np.float64)
    response, _derivative = jordan_response_and_derivative_fp64(normalized)
    result = (left * response) @ right_transpose
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("nonfinite FP64 Jordan-component output")
    return np.ascontiguousarray(result, dtype=np.float64)


def _validate_solver_architecture(config: EquivariantResolventSolverConfig) -> None:
    if not isinstance(config, EquivariantResolventSolverConfig):
        raise TypeError("solver_config must be an EquivariantResolventSolverConfig")
    locked = (
        (config.epsilon, float(DEPLOYED_EPSILON), "epsilon"),
        (
            config.resolvent_parameter,
            float(LOCKED_RESOLVENT_PARAMETER),
            "resolvent_parameter",
        ),
        (config.shunt, float(LOCKED_BASE_STRONG_MONOTONICITY), "shunt"),
        (config.p15_relative_tolerance, 1.0 / 250.0, "p15_relative_tolerance"),
        (
            config.p15_absolute_tolerance,
            LOCKED_FP64_ABSOLUTE_RESIDUAL,
            "p15_absolute_tolerance",
        ),
    )
    for actual, expected, name in locked:
        if actual != expected:
            raise ValueError(f"solver {name} does not match the locked exact-real architecture")


def evaluate_gated_singular_values_fp64(
    signal_values: FloatArray,
    design: GateDesign = PRIMARY_DESIGN,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
) -> ShapePreservingSingularValueEvaluation:
    """Evaluate the gate in P16's nonnegative singular-value coordinates.

    Every returned call has passed the configured computed P15 residual rule.
    The exact PL theorem instead concerns the mathematical resolvent and the
    pointwise sector returned by :func:`pointwise_gain_sector`.
    """

    if not isinstance(signal_values, np.ndarray):
        raise TypeError("signal_values must be a numpy.ndarray")
    if signal_values.dtype != np.float64:
        raise TypeError("signal_values must have dtype numpy.float64")
    if signal_values.ndim != 1 or signal_values.size == 0:
        raise ValueError("signal_values must be a nonempty vector")
    if not bool(np.isfinite(signal_values).all()):
        raise ValueError("signal_values must contain only finite entries")
    if bool(np.any(signal_values < 0.0)):
        raise ValueError("signal_values must be nonnegative")
    _validate_gate_design(design)
    locked_pl_certificate(design)
    selected_solver = EquivariantResolventSolverConfig() if solver_config is None else solver_config
    _validate_solver_architecture(selected_solver)
    selected_values = np.ascontiguousarray(signal_values)
    solved = solve_resolvent_singular_values(selected_values, config=selected_solver)
    if not solved.diagnostics.certified or solved.output_singular_values is None:
        raise RuntimeError(
            "P16 reduced solver did not satisfy the configured computed-residual rule: "
            f"{solved.diagnostics.status}"
        )

    signal_norm = _scaled_frobenius_norm(selected_values)
    squared_norm = signal_norm**2
    if not math.isfinite(squared_norm):
        raise FloatingPointError("nonfinite squared signal norm")
    gate_value = quintic_gate_weight_fp64(squared_norm, design)
    solution = solved.solution_singular_values
    solution_norm = _scaled_frobenius_norm(solution)
    if solution_norm == 0.0:
        shape_output = np.zeros_like(solution)
    else:
        normalized = np.ascontiguousarray(
            solution / (solution_norm + selected_solver.epsilon),
            dtype=np.float64,
        )
        shape_output, _derivative = jordan_response_and_derivative_fp64(normalized)
    yosida_output = solved.output_singular_values
    output = (1.0 - gate_value) * yosida_output / float(
        design.passive_divisor
    ) + gate_value * shape_output
    if not bool(
        np.isfinite(yosida_output).all()
        and np.isfinite(shape_output).all()
        and np.isfinite(output).all()
    ):
        raise FloatingPointError("nonfinite FP64 gated singular-value output")
    return ShapePreservingSingularValueEvaluation(
        solution_singular_values=np.ascontiguousarray(solution, dtype=np.float64),
        yosida_output=np.ascontiguousarray(yosida_output, dtype=np.float64),
        shape_output=np.ascontiguousarray(shape_output, dtype=np.float64),
        output=np.ascontiguousarray(output, dtype=np.float64),
        gate_value=gate_value,
        squared_signal_norm=squared_norm,
        solver_diagnostics=solved.diagnostics,
    )


def evaluate_shape_preserving_resolvent_fp64(
    signal: FloatArray,
    config: GateDesign = PRIMARY_DESIGN,
    *,
    solver_config: EquivariantResolventSolverConfig | None = None,
) -> ShapePreservingResolventEvaluation:
    """Evaluate the gated interface using P16's guarded FP64 solver.

    A result is returned only if P16's recomputed matrix-coordinate graph
    residual passes its configured P15 stopping rule.  That fail-closed check
    is still a computed FP64 residual, not a floating-point error certificate.
    """

    selected_signal = _require_fp64_matrix(signal, "signal")
    _validate_gate_design(config)
    # Refuse an unlocked exact-certificate configuration even though the
    # numerical blend itself would be syntactically evaluable.
    locked_pl_certificate(config)
    selected_solver = EquivariantResolventSolverConfig() if solver_config is None else solver_config
    _validate_solver_architecture(selected_solver)
    solved = solve_resolvent_fp64(selected_signal, config=selected_solver)
    if not solved.diagnostics.certified or solved.output is None:
        raise RuntimeError(
            "P16 solver did not satisfy the configured computed-residual rule: "
            f"{solved.diagnostics.status}"
        )

    signal_norm = _scaled_frobenius_norm(selected_signal)
    squared_norm = signal_norm**2
    if not math.isfinite(squared_norm):
        raise FloatingPointError("nonfinite squared signal norm")
    weight = quintic_gate_weight_fp64(squared_norm, config)
    scaled_yosida = solved.output / float(config.passive_divisor)
    jordan = additive_epsilon_jordan_fp64(
        solved.approximate_solution,
        epsilon=selected_solver.epsilon,
    )
    output = (1.0 - weight) * scaled_yosida + weight * jordan
    if not bool(
        np.isfinite(scaled_yosida).all() and np.isfinite(jordan).all() and np.isfinite(output).all()
    ):
        raise FloatingPointError("nonfinite FP64 shape-preserving interface output")
    return ShapePreservingResolventEvaluation(
        output=np.ascontiguousarray(output, dtype=np.float64),
        scaled_yosida_output=np.ascontiguousarray(scaled_yosida, dtype=np.float64),
        jordan_output=jordan,
        gate_weight=weight,
        squared_signal_norm=squared_norm,
        resolvent_result=solved,
    )


__all__ = [
    "FULL_STEP_DESIGN",
    "LOCKED_FULL_STEP_CONFIG",
    "LOCKED_FULL_STEP_PL_CERTIFICATE",
    "LOCKED_FULL_STEP_TAU",
    "LOCKED_GATE_Q0",
    "LOCKED_GATE_Q1",
    "LOCKED_HIGH_FIDELITY_CONFIG",
    "LOCKED_HIGH_FIDELITY_PL_CERTIFICATE",
    "LOCKED_HIGH_FIDELITY_TAU",
    "LOCKED_JORDAN_AFTER_RESOLVENT_GAIN_UPPER",
    "PRIMARY_DESIGN",
    "SHAPE_PRESERVING_RESOLVENT_SCHEMA_VERSION",
    "GateDesign",
    "PointwiseGainSector",
    "QuinticGateJet",
    "ShapePreservingResolventAudit",
    "ShapePreservingResolventEvaluation",
    "ShapePreservingSingularValueEvaluation",
    "additive_epsilon_jordan_fp64",
    "audit_shape_preserving_resolvent",
    "evaluate_gated_singular_values_fp64",
    "evaluate_shape_preserving_resolvent_fp64",
    "jordan_after_resolvent_gain_upper",
    "locked_pl_certificate",
    "pointwise_gain_sector",
    "quintic_gate_jet_exact",
    "quintic_gate_weight_fp64",
    "smootherstep_gate",
]
