"""Safeguarded FP64 reference solver for the P14--P15 resolvent.

This module implements the proposed exact-real architecture numerically; it
does not certify floating-point rounding.  For a matrix ``S`` it computes a
candidate for

``S = U + lambda * (A_epsilon(U) + mu * U)``,

where ``A_epsilon`` is P13's additive-epsilon Jordan map plus its radial
passivator.  Bi-orthogonal equivariance reduces the equation to the singular
values.  The reduced Jacobian is diagonal plus rank one, so every Newton step
uses Sherman--Morrison rather than a dense solve.

Successful full-matrix calls are accepted only after ``B`` is reevaluated on
the stored reconstructed candidate and the resulting P15 graph residual passes

``||r||_F <= ||S||_F / 250 + rbar_fp64``.

The returned operator value is always ``(S-U)/lambda``.  In particular it is
not silently replaced by ``A_epsilon(U)+mu*U`` away from an exact solution.

This literal postcheck requires a second SVD in the reference implementation.
The locked absolute tolerance below is an operational solver budget, not an
IEEE-754 error theorem.  P17 is reserved for certifying finite-precision
residual evaluation and solve errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from passive_muon.additive_epsilon_deficit import DEPLOYED_EPSILON
from passive_muon.radial_passivation_tradeoff import (
    P11_SIGNAL_NORM_BOUND,
    SWITCH_RAW_RATIO,
    TAIL_DERIVATIVE_LOWER,
    TAIL_PROJECTION_LOWER,
    UNIT_DEFICIT_UPPER,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.yosida_stability import (
    LOCKED_BASE_STRONG_MONOTONICITY,
    LOCKED_RESOLVENT_PARAMETER,
)

FloatArray = NDArray[np.float64]

LOCKED_FP64_ABSOLUTE_RESIDUAL: Final = 2.0**-40
LOCKED_STRICT_RELATIVE_TOLERANCE: Final = 2.0**-40
LOCKED_STRICT_ABSOLUTE_TOLERANCE: Final = 2.0**-48
LOCKED_MAXIMUM_ITERATIONS: Final = 64
LOCKED_MAXIMUM_BACKTRACKS: Final = 64
LOCKED_ARMIJO_CONSTANT: Final = 1.0e-4
LOCKED_MINIMUM_STEP: Final = 2.0**-60
LOCKED_MINIMUM_SHERMAN_MORRISON_DENOMINATOR: Final = 2.0**-30
LOCKED_NEGATIVE_SOLUTION_TOLERANCE: Final = 64.0 * np.finfo(np.float64).eps

_JORDAN_A, _JORDAN_B, _JORDAN_C = (float(value) for value in JORDAN_QUINTIC.fractions())
_SWITCH_RAW_RATIO = float(SWITCH_RAW_RATIO)
_TAIL_DERIVATIVE_LOWER = float(TAIL_DERIVATIVE_LOWER)
_TAIL_PROJECTION_LOWER = float(TAIL_PROJECTION_LOWER)
_UNIT_DEFICIT_UPPER = float(UNIT_DEFICIT_UPPER)


class SolverStatus(StrEnum):
    """Terminal status of one reduced or full-matrix solve."""

    ZERO_INPUT = "zero_input"
    STRICT_RESIDUAL = "strict_residual"
    P15_RESIDUAL_ONLY = "p15_residual_only"
    MAXIMUM_ITERATIONS = "maximum_iterations"
    LINE_SEARCH_FAILED = "line_search_failed"
    UNSAFE_JACOBIAN = "unsafe_jacobian"
    NONFINITE_ITERATE = "nonfinite_iterate"
    NEGATIVE_SOLUTION = "negative_solution"
    ACTUAL_RESIDUAL_FAILED = "actual_residual_failed"


@dataclass(frozen=True)
class EquivariantResolventSolverConfig:
    """Locked numerical policy for the safeguarded reference solver."""

    epsilon: float = float(DEPLOYED_EPSILON)
    resolvent_parameter: float = float(LOCKED_RESOLVENT_PARAMETER)
    shunt: float = float(LOCKED_BASE_STRONG_MONOTONICITY)
    p15_relative_tolerance: float = 1.0 / 250.0
    p15_absolute_tolerance: float = LOCKED_FP64_ABSOLUTE_RESIDUAL
    strict_relative_tolerance: float = LOCKED_STRICT_RELATIVE_TOLERANCE
    strict_absolute_tolerance: float = LOCKED_STRICT_ABSOLUTE_TOLERANCE
    maximum_iterations: int = LOCKED_MAXIMUM_ITERATIONS
    maximum_backtracks: int = LOCKED_MAXIMUM_BACKTRACKS
    armijo_constant: float = LOCKED_ARMIJO_CONSTANT
    minimum_step: float = LOCKED_MINIMUM_STEP
    minimum_sherman_morrison_denominator: float = LOCKED_MINIMUM_SHERMAN_MORRISON_DENOMINATOR
    negative_solution_tolerance: float = LOCKED_NEGATIVE_SOLUTION_TOLERANCE
    signal_norm_guard: float = float(P11_SIGNAL_NORM_BOUND)

    def __post_init__(self) -> None:
        positive = {
            "epsilon": self.epsilon,
            "resolvent_parameter": self.resolvent_parameter,
            "shunt": self.shunt,
            "minimum_step": self.minimum_step,
            "minimum_sherman_morrison_denominator": (self.minimum_sherman_morrison_denominator),
            "negative_solution_tolerance": self.negative_solution_tolerance,
            "signal_norm_guard": self.signal_norm_guard,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and strictly positive")
        tolerances = {
            "p15_relative_tolerance": self.p15_relative_tolerance,
            "p15_absolute_tolerance": self.p15_absolute_tolerance,
            "strict_relative_tolerance": self.strict_relative_tolerance,
            "strict_absolute_tolerance": self.strict_absolute_tolerance,
        }
        for name, value in tolerances.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.strict_relative_tolerance > self.p15_relative_tolerance:
            raise ValueError("strict_relative_tolerance must not exceed the P15 tolerance")
        if self.strict_absolute_tolerance > self.p15_absolute_tolerance:
            raise ValueError("strict_absolute_tolerance must not exceed the P15 tolerance")
        if self.maximum_iterations < 0:
            raise ValueError("maximum_iterations must be nonnegative")
        if self.maximum_backtracks < 0:
            raise ValueError("maximum_backtracks must be nonnegative")
        if not 0.0 < self.armijo_constant < 0.5:
            raise ValueError("armijo_constant must lie strictly between zero and one half")
        if self.minimum_step > 1.0:
            raise ValueError("minimum_step must not exceed the unit Newton step")


@dataclass(frozen=True)
class SolverDiagnostics:
    """Auditable stopping result and operation counters."""

    status: SolverStatus
    iterations: int
    backtracks: int
    initial_residual_norm: float
    reduced_residual_norm: float
    residual_norm: float
    p15_threshold: float
    strict_target: float
    operator_evaluations: int
    jacobian_evaluations: int
    sherman_morrison_solves: int
    svd_count: int
    reconstruction_matmuls: int
    minimum_abs_sherman_morrison_denominator: float

    @property
    def certified(self) -> bool:
        """Whether the actual recorded residual passes the P15 rule."""

        return (
            self.status
            in {
                SolverStatus.ZERO_INPUT,
                SolverStatus.STRICT_RESIDUAL,
                SolverStatus.P15_RESIDUAL_ONLY,
            }
            and self.residual_norm <= self.p15_threshold
        )


@dataclass(frozen=True)
class SingularValueSolveResult:
    """Result of the reduced solve in a fixed singular-vector basis."""

    solution_singular_values: FloatArray
    output_singular_values: FloatArray | None
    graph_residual: FloatArray
    diagnostics: SolverDiagnostics


@dataclass(frozen=True)
class MatrixSolveResult:
    """Full-matrix result with the required resolvent-form output."""

    approximate_solution: FloatArray
    output: FloatArray | None
    graph_residual: FloatArray
    input_singular_values: FloatArray
    solution_singular_values: FloatArray
    diagnostics: SolverDiagnostics


def _require_float64_vector(values: FloatArray, name: str) -> FloatArray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.dtype != np.float64:
        raise TypeError(f"{name} must have dtype numpy.float64")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not bool(np.isfinite(values).all()):
        raise ValueError(f"{name} must contain only finite entries")
    return np.ascontiguousarray(values)


def _require_float64_matrix(matrix: FloatArray, name: str) -> FloatArray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if matrix.dtype != np.float64:
        raise TypeError(f"{name} must have dtype numpy.float64")
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not bool(np.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite entries")
    return np.ascontiguousarray(matrix)


def _scaled_euclidean_norm(values: FloatArray) -> float:
    """Compute a scale-safe Euclidean/Frobenius norm."""

    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return 0.0
    scaled = values / maximum
    return maximum * math.sqrt(float(np.dot(scaled.ravel(), scaled.ravel())))


def jordan_response_and_derivative_fp64(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Evaluate the exact-coefficient five-stage scalar response in FP64."""

    current = np.asarray(values, dtype=np.float64).copy()
    derivative = np.ones_like(current)
    for _ in range(5):
        square = current * current
        stage_derivative = _JORDAN_A + 3.0 * _JORDAN_B * square + 5.0 * _JORDAN_C * square**2
        derivative *= stage_derivative
        current = current * (_JORDAN_A + _JORDAN_B * square + _JORDAN_C * square**2)
    return current, derivative


def radial_deficit_majorant_fp64(raw_ratio: float) -> float:
    """Evaluate P13's ``d_hat`` using its frozen exact coefficients."""

    if not math.isfinite(raw_ratio) or raw_ratio < 0.0:
        raise ValueError("raw_ratio must be finite and nonnegative")
    if raw_ratio <= _SWITCH_RAW_RATIO:
        return _UNIT_DEFICIT_UPPER
    reciprocal = 1.0 / (1.0 + raw_ratio)
    return -(
        _TAIL_DERIVATIVE_LOWER * reciprocal**2
        + _TAIL_PROJECTION_LOWER * (raw_ratio * reciprocal) * reciprocal
    )


def radial_integral_fp64(raw_ratio: float) -> float:
    """Evaluate P13's closed-form radial primitive with ``log1p``."""

    if not math.isfinite(raw_ratio) or raw_ratio < 0.0:
        raise ValueError("raw_ratio must be finite and nonnegative")
    if raw_ratio <= _SWITCH_RAW_RATIO:
        return _UNIT_DEFICIT_UPPER * raw_ratio
    return (
        _UNIT_DEFICIT_UPPER * _SWITCH_RAW_RATIO
        - _TAIL_PROJECTION_LOWER * (math.log1p(raw_ratio) - math.log1p(_SWITCH_RAW_RATIO))
        + (_TAIL_DERIVATIVE_LOWER - _TAIL_PROJECTION_LOWER)
        * (1.0 / (1.0 + raw_ratio) - 1.0 / (1.0 + _SWITCH_RAW_RATIO))
    )


def p13_singular_operator_fp64(values: FloatArray, *, epsilon: float) -> FloatArray:
    """Evaluate ``A_epsilon`` on signed diagonal spectral coordinates."""

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if not bool(np.isfinite(values).all()):
        raise ValueError("values must contain only finite entries")
    radius = _scaled_euclidean_norm(values)
    if radius == 0.0:
        return np.zeros_like(values)
    normalized = values / (radius + epsilon)
    jordan, _derivative = jordan_response_and_derivative_fp64(normalized)
    raw_ratio = radius / epsilon
    # In the inner P13 branch the radial repair is exactly linear.  Use that
    # identity directly so subnormal nonzero inputs never form p(z) / ||x||
    # after p(z) has underflowed.
    if raw_ratio <= _SWITCH_RAW_RATIO:
        radial_gain = _UNIT_DEFICIT_UPPER / epsilon
    else:
        radial_gain = radial_integral_fp64(raw_ratio) / radius
    result = jordan + radial_gain * values
    if not bool(np.isfinite(result).all()):
        raise FloatingPointError("nonfinite P13 singular operator evaluation")
    return result


def shifted_p13_singular_operator_fp64(
    values: FloatArray,
    *,
    epsilon: float,
    shunt: float,
) -> FloatArray:
    """Evaluate ``B=A_epsilon+mu*I`` on singular values."""

    if not math.isfinite(shunt) or shunt <= 0.0:
        raise ValueError("shunt must be finite and strictly positive")
    return p13_singular_operator_fp64(values, epsilon=epsilon) + shunt * values


def resolvent_graph_singular_values_fp64(
    values: FloatArray,
    *,
    config: EquivariantResolventSolverConfig,
) -> FloatArray:
    """Evaluate ``u+lambda*B(u)`` in the reduced coordinates."""

    return values + config.resolvent_parameter * shifted_p13_singular_operator_fp64(
        values,
        epsilon=config.epsilon,
        shunt=config.shunt,
    )


def _reduced_jacobian_diagonal_plus_rank_one(
    values: FloatArray,
    config: EquivariantResolventSolverConfig,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``D,v,x`` such that ``DF=diag(D)+v*x.T``."""

    radius = _scaled_euclidean_norm(values)
    lam = config.resolvent_parameter
    base_diagonal = 1.0 + lam * config.shunt
    if radius == 0.0:
        origin_slope = (_JORDAN_A**5 + _UNIT_DEFICIT_UPPER) / config.epsilon
        diagonal = np.full_like(values, base_diagonal + lam * origin_slope)
        return diagonal, np.zeros_like(values), np.zeros_like(values)

    denominator = radius + config.epsilon
    normalized = values / denominator
    _response, response_derivative = jordan_response_and_derivative_fp64(normalized)
    raw_ratio = radius / config.epsilon
    if raw_ratio <= _SWITCH_RAW_RATIO:
        # Here G(x)=(U/epsilon)x exactly, so its radial and tangential
        # derivatives agree.  Besides being exact, this avoids radius**2
        # underflow for tiny but nonzero FP64 inputs.
        tangential_gain = _UNIT_DEFICIT_UPPER / config.epsilon
        radial_gain = tangential_gain
    else:
        tangential_gain = radial_integral_fp64(raw_ratio) / radius
        radial_gain = radial_deficit_majorant_fp64(raw_ratio) / config.epsilon

    diagonal = base_diagonal + lam * (response_derivative / denominator + tangential_gain)
    # Write the outer-product column using x/||x||.  This is algebraically
    # identical to the radius-squared formula but remains finite throughout
    # the subnormal range.
    direction = values / radius
    left = (
        lam
        * direction
        * ((radial_gain - tangential_gain) / radius - response_derivative / denominator**2)
    )
    if not bool(np.isfinite(diagonal).all() and np.isfinite(left).all()):
        raise FloatingPointError("nonfinite reduced Jacobian")
    return diagonal, left, values


def _sherman_morrison_solve(
    diagonal: FloatArray,
    left: FloatArray,
    right: FloatArray,
    rhs: FloatArray,
    *,
    minimum_denominator: float,
) -> tuple[FloatArray, float]:
    if bool(np.any(~np.isfinite(diagonal))) or bool(np.any(diagonal == 0.0)):
        raise ArithmeticError("unsafe diagonal in reduced Jacobian")
    inverse_rhs = rhs / diagonal
    inverse_left = left / diagonal
    denominator = 1.0 + float(np.dot(right, inverse_left))
    if not math.isfinite(denominator) or denominator < minimum_denominator:
        raise ArithmeticError("unsafe Sherman--Morrison denominator")
    solution = inverse_rhs - inverse_left * (float(np.dot(right, inverse_rhs)) / denominator)
    if not bool(np.isfinite(solution).all()):
        raise FloatingPointError("nonfinite Sherman--Morrison solution")
    return solution, denominator


def _terminal_status(
    residual_norm: float,
    strict_target: float,
    p15_threshold: float,
) -> SolverStatus | None:
    if residual_norm <= strict_target:
        return SolverStatus.STRICT_RESIDUAL
    if residual_norm <= p15_threshold:
        return SolverStatus.P15_RESIDUAL_ONLY
    return None


def solve_resolvent_singular_values(
    singular_values: FloatArray,
    config: EquivariantResolventSolverConfig | None = None,
) -> SingularValueSolveResult:
    """Solve the locked resolvent equation in singular-value coordinates.

    Newton iteration targets the stricter numerical tolerance.  If progress
    stalls after the P15 rule has already passed, the result is returned with
    status :attr:`SolverStatus.P15_RESIDUAL_ONLY`; otherwise every numerical
    guard failure remains explicit.
    """

    selected = EquivariantResolventSolverConfig() if config is None else config
    sigma = _require_float64_vector(singular_values, "singular_values")
    if bool(np.any(sigma < 0.0)):
        raise ValueError("singular_values must be nonnegative")
    sigma_norm = _scaled_euclidean_norm(sigma)
    if sigma_norm > selected.signal_norm_guard:
        raise ValueError("input exceeds the locked P11 Frobenius signal guard")

    p15_threshold = selected.p15_relative_tolerance * sigma_norm + selected.p15_absolute_tolerance
    strict_target = (
        selected.strict_relative_tolerance * sigma_norm + selected.strict_absolute_tolerance
    )
    if sigma_norm == 0.0:
        zeros = np.zeros_like(sigma)
        diagnostics = SolverDiagnostics(
            status=SolverStatus.ZERO_INPUT,
            iterations=0,
            backtracks=0,
            initial_residual_norm=0.0,
            reduced_residual_norm=0.0,
            residual_norm=0.0,
            p15_threshold=p15_threshold,
            strict_target=strict_target,
            operator_evaluations=0,
            jacobian_evaluations=0,
            sherman_morrison_solves=0,
            svd_count=0,
            reconstruction_matmuls=0,
            minimum_abs_sherman_morrison_denominator=1.0,
        )
        return SingularValueSolveResult(zeros.copy(), zeros.copy(), zeros, diagnostics)

    origin_slope = (_JORDAN_A**5 + _UNIT_DEFICIT_UPPER) / selected.epsilon
    origin_denominator = (
        1.0
        + selected.resolvent_parameter * selected.shunt
        + selected.resolvent_parameter * origin_slope
    )
    if not math.isfinite(origin_denominator) or origin_denominator <= 0.0:
        raise FloatingPointError("invalid FP64 origin linearization")
    values = sigma / origin_denominator

    operator_evaluations = 1
    graph = resolvent_graph_singular_values_fp64(values, config=selected)
    residual = sigma - graph
    residual_norm = _scaled_euclidean_norm(residual)
    initial_residual_norm = residual_norm
    iterations = 0
    backtracks = 0
    jacobian_evaluations = 0
    sherman_morrison_solves = 0
    minimum_abs_sm_denominator = math.inf
    failure_status = SolverStatus.MAXIMUM_ITERATIONS

    while residual_norm > strict_target and iterations < selected.maximum_iterations:
        try:
            diagonal, left, right = _reduced_jacobian_diagonal_plus_rank_one(
                values,
                selected,
            )
            jacobian_evaluations += 1
            step, sm_denominator = _sherman_morrison_solve(
                diagonal,
                left,
                right,
                residual,
                minimum_denominator=selected.minimum_sherman_morrison_denominator,
            )
            sherman_morrison_solves += 1
            minimum_abs_sm_denominator = min(
                minimum_abs_sm_denominator,
                abs(sm_denominator),
            )
        except ArithmeticError:
            failure_status = SolverStatus.UNSAFE_JACOBIAN
            break
        except FloatingPointError:
            failure_status = SolverStatus.NONFINITE_ITERATE
            break

        if not bool(np.isfinite(step).all()):
            failure_status = SolverStatus.NONFINITE_ITERATE
            break

        # The exact Armijo argument is global on the signed diagonal
        # coordinate space.  Do not project or truncate intermediate iterates;
        # the unique exact root is nonnegative by sign preservation.
        step_scale = 1.0
        merit = 0.5 * residual_norm**2
        accepted = False
        for attempt in range(selected.maximum_backtracks + 1):
            if step_scale < selected.minimum_step:
                break
            candidate = values + step_scale * step
            if not bool(np.isfinite(candidate).all()):
                if attempt == selected.maximum_backtracks:
                    break
                step_scale *= 0.5
                backtracks += 1
                continue
            operator_evaluations += 1
            try:
                candidate_graph = resolvent_graph_singular_values_fp64(candidate, config=selected)
            except (FloatingPointError, OverflowError, ValueError):
                if attempt == selected.maximum_backtracks:
                    break
                step_scale *= 0.5
                backtracks += 1
                continue
            candidate_residual = sigma - candidate_graph
            candidate_norm = _scaled_euclidean_norm(candidate_residual)
            armijo_right = merit * (1.0 - 2.0 * selected.armijo_constant * step_scale)
            if math.isfinite(candidate_norm) and 0.5 * candidate_norm**2 <= armijo_right:
                values = candidate
                residual = candidate_residual
                residual_norm = candidate_norm
                iterations += 1
                accepted = True
                break
            if attempt == selected.maximum_backtracks:
                break
            step_scale *= 0.5
            backtracks += 1
        if not accepted:
            failure_status = SolverStatus.LINE_SEARCH_FAILED
            break

    negative_limit = selected.negative_solution_tolerance * max(1.0, sigma_norm)
    negative_solution = bool(np.any(values < -negative_limit))
    if not negative_solution and bool(np.any(values < 0.0)):
        # SVD singular values are nonnegative at the exact root.  Clamp only
        # roundoff-sized negative components and then repeat the graph-residual
        # test; the clamp never inherits the pre-clamp success status.
        values = np.maximum(values, 0.0)
        graph = resolvent_graph_singular_values_fp64(values, config=selected)
        operator_evaluations += 1
        residual = sigma - graph
        residual_norm = _scaled_euclidean_norm(residual)

    terminal = _terminal_status(residual_norm, strict_target, p15_threshold)
    if negative_solution:
        status = SolverStatus.NEGATIVE_SOLUTION
    else:
        status = terminal if terminal is not None else failure_status
        if iterations >= selected.maximum_iterations and terminal is None:
            status = SolverStatus.MAXIMUM_ITERATIONS
    if not math.isfinite(minimum_abs_sm_denominator):
        minimum_abs_sm_denominator = 1.0

    diagnostics = SolverDiagnostics(
        status=status,
        iterations=iterations,
        backtracks=backtracks,
        initial_residual_norm=initial_residual_norm,
        reduced_residual_norm=residual_norm,
        residual_norm=residual_norm,
        p15_threshold=p15_threshold,
        strict_target=strict_target,
        operator_evaluations=operator_evaluations,
        jacobian_evaluations=jacobian_evaluations,
        sherman_morrison_solves=sherman_morrison_solves,
        svd_count=0,
        reconstruction_matmuls=0,
        minimum_abs_sherman_morrison_denominator=minimum_abs_sm_denominator,
    )
    output = (sigma - values) / selected.resolvent_parameter if diagnostics.certified else None
    return SingularValueSolveResult(
        solution_singular_values=values,
        output_singular_values=output,
        graph_residual=residual,
        diagnostics=diagnostics,
    )


def solve_resolvent_fp64(
    signal: FloatArray,
    config: EquivariantResolventSolverConfig | None = None,
) -> MatrixSolveResult:
    """Return a guarded full-matrix approximate Yosida evaluation.

    The final residual is recomputed in the original matrix coordinates from
    the reduced graph value.  A singular-space success is downgraded if this
    actual postcondition fails after SVD reconstruction.
    """

    selected = EquivariantResolventSolverConfig() if config is None else config
    matrix = _require_float64_matrix(signal, "signal")
    matrix_norm = _scaled_euclidean_norm(matrix)
    if matrix_norm > selected.signal_norm_guard:
        raise ValueError("input exceeds the locked P11 Frobenius signal guard")
    if matrix_norm == 0.0:
        zeros = np.zeros_like(matrix)
        singular_zeros = np.zeros(min(matrix.shape), dtype=np.float64)
        reduced = solve_resolvent_singular_values(singular_zeros, selected)
        diagnostics = replace(reduced.diagnostics, svd_count=0)
        return MatrixSolveResult(
            approximate_solution=zeros.copy(),
            output=zeros.copy(),
            graph_residual=zeros,
            input_singular_values=singular_zeros,
            solution_singular_values=singular_zeros.copy(),
            diagnostics=diagnostics,
        )

    try:
        left, singular_values, right_transpose = np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError as error:
        raise RuntimeError("FP64 singular-value decomposition did not converge") from error
    singular_values = np.ascontiguousarray(singular_values, dtype=np.float64)
    reduced = solve_resolvent_singular_values(singular_values, selected)

    approximate_solution = (left * reduced.solution_singular_values) @ right_transpose
    # Reevaluate B on the *stored reconstructed matrix*.  Reconstructing the
    # reduced graph in the original rounded SVD basis is mathematically
    # equivalent in exact arithmetic, but is not literally the computable
    # residual S-U_hat-lambda*B(U_hat) in FP64.  This second SVD makes the
    # fail-closed postcondition match P15's residual definition exactly at the
    # implementation level (without yet certifying IEEE-754 rounding error).
    try:
        candidate_p13 = p13_operator_fp64(
            approximate_solution,
            epsilon=selected.epsilon,
        )
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "FP64 graph-check singular-value decomposition did not converge"
        ) from error
    candidate_shifted = candidate_p13 + selected.shunt * approximate_solution
    graph_residual = (
        matrix - approximate_solution - selected.resolvent_parameter * candidate_shifted
    )
    actual_residual_norm = _scaled_euclidean_norm(graph_residual)
    p15_threshold = selected.p15_relative_tolerance * matrix_norm + selected.p15_absolute_tolerance
    strict_target = (
        selected.strict_relative_tolerance * matrix_norm + selected.strict_absolute_tolerance
    )
    status = reduced.diagnostics.status
    if reduced.diagnostics.certified:
        if actual_residual_norm <= strict_target:
            status = SolverStatus.STRICT_RESIDUAL
        elif actual_residual_norm <= p15_threshold:
            status = SolverStatus.P15_RESIDUAL_ONLY
        else:
            status = SolverStatus.ACTUAL_RESIDUAL_FAILED
    diagnostics = replace(
        reduced.diagnostics,
        status=status,
        residual_norm=actual_residual_norm,
        p15_threshold=p15_threshold,
        strict_target=strict_target,
        operator_evaluations=reduced.diagnostics.operator_evaluations + 1,
        svd_count=2,
        reconstruction_matmuls=2,
    )
    output = (
        (matrix - approximate_solution) / selected.resolvent_parameter
        if diagnostics.certified
        else None
    )
    return MatrixSolveResult(
        approximate_solution=approximate_solution,
        output=output,
        graph_residual=graph_residual,
        input_singular_values=singular_values,
        solution_singular_values=reduced.solution_singular_values,
        diagnostics=diagnostics,
    )


def p13_operator_fp64(
    matrix: FloatArray, *, epsilon: float = float(DEPLOYED_EPSILON)
) -> FloatArray:
    """Reference spectral evaluation of P13's full-matrix ``A_epsilon``."""

    selected = _require_float64_matrix(matrix, "matrix")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and strictly positive")
    if _scaled_euclidean_norm(selected) == 0.0:
        return np.zeros_like(selected)
    left, singular_values, right_transpose = np.linalg.svd(selected, full_matrices=False)
    output_values = p13_singular_operator_fp64(
        np.ascontiguousarray(singular_values, dtype=np.float64),
        epsilon=epsilon,
    )
    return (left * output_values) @ right_transpose


__all__ = [
    "LOCKED_FP64_ABSOLUTE_RESIDUAL",
    "LOCKED_MAXIMUM_BACKTRACKS",
    "LOCKED_MAXIMUM_ITERATIONS",
    "LOCKED_NEGATIVE_SOLUTION_TOLERANCE",
    "EquivariantResolventSolverConfig",
    "MatrixSolveResult",
    "SingularValueSolveResult",
    "SolverDiagnostics",
    "SolverStatus",
    "jordan_response_and_derivative_fp64",
    "p13_operator_fp64",
    "p13_singular_operator_fp64",
    "radial_deficit_majorant_fp64",
    "radial_integral_fp64",
    "resolvent_graph_singular_values_fp64",
    "shifted_p13_singular_operator_fp64",
    "solve_resolvent_fp64",
    "solve_resolvent_singular_values",
]
