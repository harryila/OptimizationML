"""Deterministic nonlinear falsification probes for the repaired Muon loop.

The objective family in this module is analytically ``ell``-strongly convex
and ``L``-smooth, while its Hessian eigenspaces change with the iterate.  The
trajectory runs are deliberately qualified as sampling diagnostics: they may
expose a concrete implementation-level failure, but passing samples do not
certify stability for nonlinear objectives.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from passive_muon.nonquadratic_stability import (
    LOCKED_NONQUADRATIC_LEARNING_RATE,
    LOCKED_NONQUADRATIC_TAU,
    locked_nonquadratic_certificate,
)
from passive_muon.operator import orthogonalize
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_FLOOR,
    LOCKED_HESSIAN_LOWER,
    LOCKED_HESSIAN_UPPER,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
)

RATIONAL_GIVENS_TRIPLES = ((3, 4, 5), (5, 12, 13), (8, 15, 17))


def rational_givens_frame(dimension: int, *, variant: int = 0) -> np.ndarray:
    """Return a deterministic orthogonal frame made from rational rotations.

    In exact real arithmetic each plane rotation uses one of the Pythagorean
    triples in :data:`RATIONAL_GIVENS_TRIPLES`.  ``variant`` cyclically shifts
    both the plane order and coefficient schedule.  The returned realization
    is float64, and callers record its numerical orthogonality residual.
    """

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if variant < 0:
        raise ValueError("variant must be nonnegative")

    frame = np.eye(dimension, dtype=np.float64)
    pairs = list(itertools.combinations(range(dimension), 2))
    if not pairs:
        return frame
    offset = variant % len(pairs)
    pairs = pairs[offset:] + pairs[:offset]
    for index, (left, right) in enumerate(pairs):
        numerator_cosine, numerator_sine, denominator = RATIONAL_GIVENS_TRIPLES[
            (index + variant) % len(RATIONAL_GIVENS_TRIPLES)
        ]
        cosine = numerator_cosine / denominator
        sine = numerator_sine / denominator
        if (index + variant) % 2:
            sine = -sine
        left_column = frame[:, left].copy()
        right_column = frame[:, right].copy()
        frame[:, left] = cosine * left_column + sine * right_column
        frame[:, right] = -sine * left_column + cosine * right_column
    return frame


def _sech_squared(value: np.ndarray) -> np.ndarray:
    tangent = np.tanh(value)
    return np.clip(1.0 - tangent * tangent, 0.0, 1.0)


def _tanh_secant(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Evaluate the tanh divided difference without close-pair cancellation."""

    difference = first - second
    output = np.empty_like(difference)
    equal = difference == 0.0
    output[equal] = _sech_squared(first[equal])
    unequal = ~equal
    if np.any(unequal):
        first_tanh = np.tanh(first[unequal])
        second_tanh = np.tanh(second[unequal])
        # tanh(a)-tanh(b) = tanh(a-b) * (1-tanh(a)*tanh(b)).
        output[unequal] = (
            np.tanh(difference[unequal]) / difference[unequal] * (1.0 - first_tanh * second_tanh)
        )
    return np.clip(output, 0.0, 1.0)


@dataclass(frozen=True)
class RotatingLogCoshObjective:
    """Two-frame log-cosh objective with exact global Hessian bounds.

    For ``a=(L-ell)/2`` and orthogonal ``Q``, the objective is

    ``ell*||x||^2/2 + a*s^2*sum_{U in {I,Q},i} log(cosh((U.T*x)_i/s))``.

    Its Hessian lies in ``[ell*I, L*I]`` globally.  The upper endpoint is
    attained at zero and the lower endpoint is approached on generic rays.
    """

    dimension: int
    transition_scale: float
    frame_variant: int = 0
    curvature_lower: float = float(LOCKED_HESSIAN_LOWER)
    curvature_upper: float = float(LOCKED_HESSIAN_UPPER)
    frame: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.transition_scale <= 0:
            raise ValueError("transition scale must be positive")
        if self.frame_variant < 0:
            raise ValueError("frame variant must be nonnegative")
        if self.curvature_lower <= 0 or self.curvature_upper < self.curvature_lower:
            raise ValueError("curvatures must satisfy 0 < ell <= L")
        object.__setattr__(
            self,
            "frame",
            rational_givens_frame(self.dimension, variant=self.frame_variant),
        )

    @property
    def nonlinear_weight(self) -> float:
        return (self.curvature_upper - self.curvature_lower) / 2.0

    @property
    def frame_orthogonality_residual(self) -> float:
        identity = np.eye(self.dimension, dtype=np.float64)
        return float(np.linalg.norm(self.frame.T @ self.frame - identity, ord="fro"))

    @property
    def frame_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.frame).tobytes()).hexdigest()

    def _vector(self, value: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(value, dtype=np.float64)
        if array.size != self.dimension:
            raise ValueError(f"expected {self.dimension} entries, received {array.size}")
        return array.reshape(-1), array.shape

    def value(self, position: np.ndarray) -> float:
        """Evaluate the objective with a stable log-cosh formula."""

        vector, _shape = self._vector(position)
        scale = self.transition_scale
        result = 0.5 * self.curvature_lower * float(vector @ vector)
        for basis in (np.eye(self.dimension, dtype=np.float64), self.frame):
            coordinates = basis.T @ vector / scale
            absolute = np.abs(coordinates)
            log_cosh = absolute + np.log1p(np.exp(-2.0 * absolute)) - math.log(2.0)
            result += self.nonlinear_weight * scale**2 * float(np.sum(log_cosh))
        return result

    def gradient(self, position: np.ndarray) -> np.ndarray:
        """Return the gradient with the same shape as ``position``."""

        vector, shape = self._vector(position)
        scale = self.transition_scale
        gradient = self.curvature_lower * vector
        for basis in (np.eye(self.dimension, dtype=np.float64), self.frame):
            gradient = gradient + (
                self.nonlinear_weight * scale * basis @ np.tanh(basis.T @ vector / scale)
            )
        return gradient.reshape(shape)

    def hessian(self, position: np.ndarray) -> np.ndarray:
        """Return the analytic Hessian on the flattened coordinate space."""

        vector, _shape = self._vector(position)
        scale = self.transition_scale
        hessian = self.curvature_lower * np.eye(self.dimension, dtype=np.float64)
        for basis in (np.eye(self.dimension, dtype=np.float64), self.frame):
            weights = _sech_squared(basis.T @ vector / scale)
            hessian = hessian + self.nonlinear_weight * (basis * weights) @ basis.T
        return hessian

    def secant_hessian(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        """Return a symmetric secant mapping the position difference to gradient difference."""

        first_vector, _first_shape = self._vector(first)
        second_vector, _second_shape = self._vector(second)
        scale = self.transition_scale
        secant = self.curvature_lower * np.eye(self.dimension, dtype=np.float64)
        for basis in (np.eye(self.dimension, dtype=np.float64), self.frame):
            first_coordinates = basis.T @ first_vector / scale
            second_coordinates = basis.T @ second_vector / scale
            weights = _tanh_secant(first_coordinates, second_coordinates)
            secant = secant + self.nonlinear_weight * (basis * weights) @ basis.T
        return secant


@dataclass(frozen=True)
class NonquadraticProbeConfig:
    """Complete deterministic configuration for the nonlinear pair search."""

    seed: int = 2_026_090_2
    shapes: tuple[tuple[int, int], ...] = ((2, 2), (3, 3))
    transition_scales: tuple[float, ...] = (1e-4, 1.0, 100.0)
    radius_multipliers: tuple[float, ...] = (0.25, 1.0, 4.0)
    pair_modes: tuple[str, ...] = ("position_only", "full_state")
    perturbation_ratio: float = 2.0**-12
    iterations: int = 500
    diagnostic_stride: int = 5
    tail_window: int = 32
    learning_rate: float = float(LOCKED_NONQUADRATIC_LEARNING_RATE)
    beta: float = float(LOCKED_BETA)
    floor: float = float(LOCKED_FLOOR)
    repair_rho: float = float(LOCKED_REPAIR_RHO)
    curvature_lower: float = float(LOCKED_HESSIAN_LOWER)
    curvature_upper: float = float(LOCKED_HESSIAN_UPPER)
    orientation_tolerance: float = 1e-10
    resolution_multiplier: float = 256.0
    candidate_growth_tolerance: float = 1e-6
    storage_rate_tolerance: float = 1e-10
    divergence_norm: float = 1e100

    def __post_init__(self) -> None:
        if not self.shapes or any(rows <= 0 or columns <= 0 for rows, columns in self.shapes):
            raise ValueError("shapes must contain positive matrix dimensions")
        if not self.transition_scales or any(value <= 0 for value in self.transition_scales):
            raise ValueError("transition scales must be positive")
        if not self.radius_multipliers or any(value <= 0 for value in self.radius_multipliers):
            raise ValueError("radius multipliers must be positive")
        if not self.pair_modes or any(
            mode not in {"position_only", "full_state"} for mode in self.pair_modes
        ):
            raise ValueError("pair modes must be 'position_only' or 'full_state'")
        if self.perturbation_ratio <= 0:
            raise ValueError("perturbation ratio must be positive")
        if self.iterations <= 0 or self.diagnostic_stride <= 0 or self.tail_window < 2:
            raise ValueError("iteration, diagnostic stride, and tail window must be positive")
        if self.learning_rate <= 0 or not 0 <= self.beta < 1:
            raise ValueError("learning rate must be positive and beta must lie in [0, 1)")
        if self.floor <= 0 or self.repair_rho < 0:
            raise ValueError("floor must be positive and repair must be nonnegative")
        if self.curvature_lower <= 0 or self.curvature_upper < self.curvature_lower:
            raise ValueError("curvatures must satisfy 0 < ell <= L")
        if self.orientation_tolerance < 0 or self.resolution_multiplier <= 0:
            raise ValueError("diagnostic tolerances must be nonnegative")
        if (
            self.candidate_growth_tolerance < 0
            or self.storage_rate_tolerance < 0
            or self.divergence_norm <= 0
        ):
            raise ValueError("growth tolerance and divergence norm must be nonnegative")


@dataclass(frozen=True)
class NonquadraticTrial:
    """Measurements from one paired nonlinear trajectory."""

    shape: tuple[int, int]
    transition_scale: float
    radius_multiplier: float
    pair_mode: str
    case_index: int
    case_seed: int
    frame_variant: int
    frame_sha256: str
    frame_orthogonality_residual: float
    requested_iterations: int
    executed_iterations: int
    initial_first_position_norm: float
    initial_second_position_norm: float
    initial_state_separation: float
    final_first_position_norm: float
    final_second_position_norm: float
    final_state_separation: float
    maximum_position_norm: float
    maximum_state_separation_ratio: float
    final_state_separation_ratio: float
    maximum_storage_ratio: float
    maximum_normalized_storage_rate_excess: float
    storage_transition_count: int
    storage_rate_violation_count: int
    first_objective_ratio: float
    second_objective_ratio: float
    sampled_secant_eigenvalue_minimum: float
    sampled_secant_eigenvalue_maximum: float
    maximum_secant_identity_relative_residual: float
    maximum_normalized_secant_commutator: float
    orientation_change_count: int
    secant_sample_count: int
    resolved_tail_sample_count: int
    tail_geometric_state_factor: float | None
    nonfinite: bool
    exceeded_divergence_norm: bool
    candidate_instability: bool


@dataclass(frozen=True)
class NonquadraticProbeSummary:
    """Aggregate nonlinear probe result with explicit nonproof scope."""

    config: NonquadraticProbeConfig
    trials: tuple[NonquadraticTrial, ...]
    worst_state_separation_amplification: float
    worst_final_state_separation_ratio: float
    maximum_storage_ratio: float
    maximum_normalized_storage_rate_excess: float
    storage_rate_violating_case_count: int
    maximum_orientation_commutator: float
    sampled_secant_eigenvalue_minimum: float
    sampled_secant_eigenvalue_maximum: float
    orientation_changing_case_count: int
    candidate_instability_count: int
    divergence_count: int
    nonfinite_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a self-contained JSON-compatible experiment record."""

        coefficients = JORDAN_QUINTIC.fractions()
        return {
            "schema_version": "passive-muon-nonquadratic-falsification-v1",
            "evidence_kind": "deterministic_sampled_paired_nonlinear_trajectories",
            "claim_scope": (
                "sampling diagnostic only; a sampled violation can identify a concrete "
                "float64 candidate witness, but sampled passes do not certify global "
                "nonquadratic stability"
            ),
            "objective_family": {
                "name": "two_rationally_rotated_log_cosh_frames",
                "vectorization": "row-major flattening of each real matrix",
                "formula": (
                    "ell*||x||^2/2 + ((L-ell)/2)*s^2*sum_{U in {I,Q},i} log(cosh((U^T*x)_i/s))"
                ),
                "gradient_secant_identity": "grad_f(x)-grad_f(y)=S(x,y)*(x-y)",
                "analytic_hessian_bounds": {
                    "lower_exact": str(LOCKED_HESSIAN_LOWER),
                    "upper_exact": str(LOCKED_HESSIAN_UPPER),
                    "qualification": (
                        "global analytic real-arithmetic bounds; trajectory eigenvalue "
                        "samples only audit the float64 implementation"
                    ),
                },
                "frame_generator": {
                    "algorithm": (
                        "lexicographic coordinate pairs with cyclic order/coefficient shifts "
                        "and alternating signed rational Givens rotations"
                    ),
                    "pythagorean_triples": [list(value) for value in RATIONAL_GIVENS_TRIPLES],
                },
                "orientation_measure": (
                    "||S_t*S_previous-S_previous*S_t||_F/(||S_t||_F*||S_previous||_F)"
                ),
            },
            "operator": {
                "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
                "domain": "the fixed finite real matrix shapes listed in config.shapes",
                "normalization": "fixed_frobenius_floor",
                "floor_exact": str(LOCKED_FLOOR),
                "additive_epsilon": "not_applicable; denominator uses a max floor",
                "orthogonalizer": JORDAN_QUINTIC.name,
                "coefficients_exact": {
                    "a": str(coefficients[0]),
                    "b": str(coefficients[1]),
                    "c": str(coefficients[2]),
                },
                "iterations": 5,
                "repair_margin_exact": str(LOCKED_REPAIR_MARGIN),
                "repair_rho_exact": str(LOCKED_REPAIR_RHO),
            },
            "update_order": [
                "g_t=grad_f(W_t)",
                "m_(t+1)=beta*m_t+(1-beta)*g_t",
                "s_(t+1)=beta*m_(t+1)+(1-beta)*g_t",
                "W_(t+1)=W_t-eta*R(s_(t+1))",
            ],
            "locked_exact_parameters": {
                "learning_rate_eta": str(LOCKED_NONQUADRATIC_LEARNING_RATE),
                "beta": str(LOCKED_BETA),
                "reference_rate_tau": str(LOCKED_NONQUADRATIC_TAU),
                "curvature_lower": str(LOCKED_HESSIAN_LOWER),
                "curvature_upper": str(LOCKED_HESSIAN_UPPER),
                "storage": [
                    [str(value) for value in row]
                    for row in locked_nonquadratic_certificate().storage
                ],
            },
            "paired_trajectory_protocol": {
                "shared_items": "objective, operator, configuration, and arithmetic",
                "changed_item": "initial position/state perturbation only",
                "state_separation": "sqrt(||delta_W||_F^2+||delta_m||_F^2/L^2)",
                "storage_diagnostic": ("V=[delta_W,delta_m/L]^T*(P tensor I)*[delta_W,delta_m/L]"),
                "storage_rate_excess": "V_next/V_previous-tau^2 on resolved pairs",
                "candidate_rule": (
                    "nonfinite, divergence threshold, or a full resolved tail window with "
                    "geometric state-separation factor above 1+candidate_growth_tolerance, "
                    "or storage-rate excess above storage_rate_tolerance"
                ),
                "qualification": "candidate flags require independent replay before any claim",
            },
            "config": asdict(self.config),
            "summary": {
                "trial_count": len(self.trials),
                "worst_state_separation_amplification": (self.worst_state_separation_amplification),
                "worst_final_state_separation_ratio": self.worst_final_state_separation_ratio,
                "maximum_storage_ratio": self.maximum_storage_ratio,
                "maximum_normalized_storage_rate_excess": (
                    self.maximum_normalized_storage_rate_excess
                ),
                "storage_rate_violating_case_count": self.storage_rate_violating_case_count,
                "maximum_orientation_commutator": self.maximum_orientation_commutator,
                "sampled_secant_eigenvalue_minimum": self.sampled_secant_eigenvalue_minimum,
                "sampled_secant_eigenvalue_maximum": self.sampled_secant_eigenvalue_maximum,
                "orientation_changing_case_count": self.orientation_changing_case_count,
                "candidate_instability_count": self.candidate_instability_count,
                "divergence_count": self.divergence_count,
                "nonfinite_count": self.nonfinite_count,
            },
            "trials": [asdict(trial) for trial in self.trials],
            "experiment_provenance": {
                "seed": self.config.seed,
                "determinism": "seeded CPU float64 paired trajectories",
                "dtype": "float64",
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "cpu_count": __import__("os").cpu_count(),
                    "torch_device": "cpu",
                },
                "software": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
                "git": _git_state(),
            },
        }


def _git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]

    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD") or "unavailable",
        "branch": run("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _normalized_random_vector(
    *, dimension: int, norm: float, rng: np.random.Generator
) -> np.ndarray:
    vector = rng.standard_normal(dimension)
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        vector[0] = 1.0
        vector_norm = 1.0
    return vector * (norm / vector_norm)


def _state_separation(
    first_position: np.ndarray,
    second_position: np.ndarray,
    first_momentum: np.ndarray,
    second_momentum: np.ndarray,
    *,
    smoothness: float,
) -> float:
    position_difference = float(np.linalg.norm(first_position - second_position))
    momentum_difference = float(np.linalg.norm(first_momentum - second_momentum)) / smoothness
    return math.hypot(position_difference, momentum_difference)


def _storage_value(
    first_position: np.ndarray,
    second_position: np.ndarray,
    first_momentum: np.ndarray,
    second_momentum: np.ndarray,
    *,
    smoothness: float,
    storage: np.ndarray,
) -> float:
    position_difference = first_position - second_position
    momentum_difference = (first_momentum - second_momentum) / smoothness
    return float(
        storage[0, 0] * (position_difference @ position_difference)
        + 2.0 * storage[0, 1] * (position_difference @ momentum_difference)
        + storage[1, 1] * (momentum_difference @ momentum_difference)
    )


def _repaired_floored_jordan(
    signal: np.ndarray,
    *,
    shape: tuple[int, int],
    floor: float,
    repair_rho: float,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(signal.reshape(shape)))
    with torch.no_grad():
        update = orthogonalize(
            tensor,
            polynomial="jordan",
            normalization="floored_frobenius",
            repair_rho=repair_rho,
            steps=5,
            floor=floor,
        )
    return update.numpy().reshape(-1)


def _case_seed(master_seed: int, case_index: int, dimension: int) -> int:
    sequence = np.random.SeedSequence([master_seed, case_index, dimension])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def run_nonquadratic_trial(
    *,
    config: NonquadraticProbeConfig,
    shape: tuple[int, int],
    transition_scale: float,
    radius_multiplier: float,
    pair_mode: str,
    case_index: int,
) -> NonquadraticTrial:
    """Run one deterministic pair through the actual repaired floored-Jordan loop."""

    if shape not in config.shapes:
        raise ValueError("shape is not present in the probe configuration")
    if transition_scale not in config.transition_scales:
        raise ValueError("transition scale is not present in the probe configuration")
    if radius_multiplier not in config.radius_multipliers:
        raise ValueError("radius multiplier is not present in the probe configuration")
    if pair_mode not in config.pair_modes:
        raise ValueError("pair mode is not present in the probe configuration")

    dimension = math.prod(shape)
    seed = _case_seed(config.seed, case_index, dimension)
    rng = np.random.default_rng(seed)
    pair_count = max(1, dimension * (dimension - 1) // 2)
    frame_variant = seed % pair_count
    objective = RotatingLogCoshObjective(
        dimension=dimension,
        transition_scale=transition_scale,
        frame_variant=frame_variant,
        curvature_lower=config.curvature_lower,
        curvature_upper=config.curvature_upper,
    )

    initial_radius = transition_scale * radius_multiplier
    perturbation_norm = initial_radius * config.perturbation_ratio
    first_position = _normalized_random_vector(
        dimension=dimension,
        norm=initial_radius,
        rng=rng,
    )
    second_position = first_position + _normalized_random_vector(
        dimension=dimension,
        norm=perturbation_norm,
        rng=rng,
    )
    if pair_mode == "position_only":
        first_momentum = np.zeros(dimension, dtype=np.float64)
        second_momentum = np.zeros(dimension, dtype=np.float64)
    else:
        first_momentum = _normalized_random_vector(
            dimension=dimension,
            norm=config.curvature_upper * initial_radius,
            rng=rng,
        )
        second_momentum = first_momentum + _normalized_random_vector(
            dimension=dimension,
            norm=config.curvature_upper * perturbation_norm,
            rng=rng,
        )

    initial_first_position_norm = float(np.linalg.norm(first_position))
    initial_second_position_norm = float(np.linalg.norm(second_position))
    initial_first_objective = objective.value(first_position)
    initial_second_objective = objective.value(second_position)
    initial_separation = _state_separation(
        first_position,
        second_position,
        first_momentum,
        second_momentum,
        smoothness=config.curvature_upper,
    )
    if initial_separation == 0.0:
        raise AssertionError("the seeded trajectory perturbation vanished")

    exact_certificate = locked_nonquadratic_certificate()
    storage = np.asarray(
        [[float(value) for value in row] for row in exact_certificate.storage],
        dtype=np.float64,
    )
    tau_squared = float(LOCKED_NONQUADRATIC_TAU**2)
    current_storage = _storage_value(
        first_position,
        second_position,
        first_momentum,
        second_momentum,
        smoothness=config.curvature_upper,
        storage=storage,
    )
    if current_storage <= 0.0:
        raise AssertionError("the locked storage was not positive on the initial pair")
    storage_resolution = max(
        np.finfo(np.float64).tiny,
        current_storage * config.resolution_multiplier * np.finfo(np.float64).eps,
    )

    maximum_position_norm = max(initial_first_position_norm, initial_second_position_norm)
    maximum_separation_ratio = 1.0
    maximum_storage_ratio = -math.inf
    maximum_storage_excess = -math.inf
    storage_transition_count = 0
    storage_rate_violation_count = 0
    minimum_secant_eigenvalue = math.inf
    maximum_secant_eigenvalue = -math.inf
    maximum_identity_residual = 0.0
    maximum_commutator = 0.0
    orientation_change_count = 0
    secant_sample_count = 0
    previous_secant: np.ndarray | None = None
    resolved_separations = [initial_separation]
    machine_epsilon = np.finfo(np.float64).eps
    resolution = (
        config.resolution_multiplier
        * machine_epsilon
        * max(1.0, initial_first_position_norm, initial_second_position_norm)
    )
    current_separation = initial_separation
    nonfinite = False
    exceeded = False
    executed = 0
    one_minus_beta = 1.0 - config.beta

    def sample_secant() -> None:
        nonlocal maximum_commutator
        nonlocal maximum_identity_residual
        nonlocal maximum_secant_eigenvalue
        nonlocal minimum_secant_eigenvalue
        nonlocal orientation_change_count
        nonlocal previous_secant
        nonlocal secant_sample_count

        position_difference = first_position - second_position
        if float(np.linalg.norm(position_difference)) <= resolution:
            return
        first_gradient = objective.gradient(first_position).reshape(-1)
        second_gradient = objective.gradient(second_position).reshape(-1)
        gradient_difference = first_gradient - second_gradient
        secant = objective.secant_hessian(first_position, second_position)
        predicted = secant @ position_difference
        denominator = max(
            float(np.linalg.norm(gradient_difference)),
            float(np.linalg.norm(predicted)),
            np.finfo(np.float64).tiny,
        )
        maximum_identity_residual = max(
            maximum_identity_residual,
            float(np.linalg.norm(gradient_difference - predicted)) / denominator,
        )
        eigenvalues = np.linalg.eigvalsh(secant)
        minimum_secant_eigenvalue = min(minimum_secant_eigenvalue, float(eigenvalues[0]))
        maximum_secant_eigenvalue = max(maximum_secant_eigenvalue, float(eigenvalues[-1]))
        if previous_secant is not None:
            commutator = secant @ previous_secant - previous_secant @ secant
            commutator_denominator = float(
                np.linalg.norm(secant, ord="fro") * np.linalg.norm(previous_secant, ord="fro")
            )
            normalized = float(np.linalg.norm(commutator, ord="fro")) / commutator_denominator
            maximum_commutator = max(maximum_commutator, normalized)
            orientation_change_count += normalized > config.orientation_tolerance
        previous_secant = secant
        secant_sample_count += 1

    sample_secant()
    for iteration in range(1, config.iterations + 1):
        first_gradient = objective.gradient(first_position).reshape(-1)
        second_gradient = objective.gradient(second_position).reshape(-1)
        first_momentum = config.beta * first_momentum + one_minus_beta * first_gradient
        second_momentum = config.beta * second_momentum + one_minus_beta * second_gradient
        first_signal = config.beta * first_momentum + one_minus_beta * first_gradient
        second_signal = config.beta * second_momentum + one_minus_beta * second_gradient
        first_position = first_position - config.learning_rate * _repaired_floored_jordan(
            first_signal,
            shape=shape,
            floor=config.floor,
            repair_rho=config.repair_rho,
        )
        second_position = second_position - config.learning_rate * _repaired_floored_jordan(
            second_signal,
            shape=shape,
            floor=config.floor,
            repair_rho=config.repair_rho,
        )
        executed = iteration

        next_storage = _storage_value(
            first_position,
            second_position,
            first_momentum,
            second_momentum,
            smoothness=config.curvature_upper,
            storage=storage,
        )
        if (
            current_separation > resolution
            and current_storage > storage_resolution
            and next_storage >= 0.0
        ):
            storage_ratio = next_storage / current_storage
            storage_excess = storage_ratio - tau_squared
            maximum_storage_ratio = max(maximum_storage_ratio, storage_ratio)
            maximum_storage_excess = max(maximum_storage_excess, storage_excess)
            storage_transition_count += 1
            storage_rate_violation_count += storage_excess > config.storage_rate_tolerance
        current_storage = next_storage

        first_norm = float(np.linalg.norm(first_position))
        second_norm = float(np.linalg.norm(second_position))
        separation = _state_separation(
            first_position,
            second_position,
            first_momentum,
            second_momentum,
            smoothness=config.curvature_upper,
        )
        maximum_position_norm = max(maximum_position_norm, first_norm, second_norm)
        maximum_separation_ratio = max(maximum_separation_ratio, separation / initial_separation)
        finite_values = (
            first_norm,
            second_norm,
            separation,
            float(np.linalg.norm(first_momentum)),
            float(np.linalg.norm(second_momentum)),
            float(np.linalg.norm(first_signal)),
            float(np.linalg.norm(second_signal)),
        )
        if not all(math.isfinite(value) for value in finite_values):
            nonfinite = True
            exceeded = True
            break
        if max(finite_values) > config.divergence_norm:
            exceeded = True
            break
        if separation > resolution:
            resolved_separations.append(separation)
        current_separation = separation
        if iteration % config.diagnostic_stride == 0 or iteration == config.iterations:
            sample_secant()

    final_first_norm = float(np.linalg.norm(first_position))
    final_second_norm = float(np.linalg.norm(second_position))
    final_separation = _state_separation(
        first_position,
        second_position,
        first_momentum,
        second_momentum,
        smoothness=config.curvature_upper,
    )
    tail_count = min(config.tail_window, len(resolved_separations))
    tail_factor: float | None = None
    full_tail = tail_count == config.tail_window
    if tail_count >= 2:
        tail = resolved_separations[-tail_count:]
        tail_factor = float((tail[-1] / tail[0]) ** (1.0 / (tail_count - 1)))
    candidate = bool(
        nonfinite
        or exceeded
        or storage_rate_violation_count > 0
        or (
            full_tail
            and tail_factor is not None
            and tail_factor > 1.0 + config.candidate_growth_tolerance
        )
    )
    if secant_sample_count == 0:
        minimum_secant_eigenvalue = math.nan
        maximum_secant_eigenvalue = math.nan
    if storage_transition_count == 0:
        maximum_storage_ratio = math.nan
        maximum_storage_excess = math.nan

    return NonquadraticTrial(
        shape=shape,
        transition_scale=transition_scale,
        radius_multiplier=radius_multiplier,
        pair_mode=pair_mode,
        case_index=case_index,
        case_seed=seed,
        frame_variant=frame_variant,
        frame_sha256=objective.frame_sha256,
        frame_orthogonality_residual=objective.frame_orthogonality_residual,
        requested_iterations=config.iterations,
        executed_iterations=executed,
        initial_first_position_norm=initial_first_position_norm,
        initial_second_position_norm=initial_second_position_norm,
        initial_state_separation=initial_separation,
        final_first_position_norm=final_first_norm,
        final_second_position_norm=final_second_norm,
        final_state_separation=final_separation,
        maximum_position_norm=maximum_position_norm,
        maximum_state_separation_ratio=maximum_separation_ratio,
        final_state_separation_ratio=final_separation / initial_separation,
        maximum_storage_ratio=maximum_storage_ratio,
        maximum_normalized_storage_rate_excess=maximum_storage_excess,
        storage_transition_count=storage_transition_count,
        storage_rate_violation_count=storage_rate_violation_count,
        first_objective_ratio=objective.value(first_position) / initial_first_objective,
        second_objective_ratio=objective.value(second_position) / initial_second_objective,
        sampled_secant_eigenvalue_minimum=minimum_secant_eigenvalue,
        sampled_secant_eigenvalue_maximum=maximum_secant_eigenvalue,
        maximum_secant_identity_relative_residual=maximum_identity_residual,
        maximum_normalized_secant_commutator=maximum_commutator,
        orientation_change_count=orientation_change_count,
        secant_sample_count=secant_sample_count,
        resolved_tail_sample_count=tail_count,
        tail_geometric_state_factor=tail_factor,
        nonfinite=nonfinite,
        exceeded_divergence_norm=exceeded,
        candidate_instability=candidate,
    )


def run_nonquadratic_probe(
    config: NonquadraticProbeConfig | None = None,
) -> NonquadraticProbeSummary:
    """Run the complete deterministic nonlinear pair grid."""

    selected = NonquadraticProbeConfig() if config is None else config
    cases = itertools.product(
        selected.shapes,
        selected.transition_scales,
        selected.radius_multipliers,
        selected.pair_modes,
    )
    trials = tuple(
        run_nonquadratic_trial(
            config=selected,
            shape=shape,
            transition_scale=transition_scale,
            radius_multiplier=radius_multiplier,
            pair_mode=pair_mode,
            case_index=case_index,
        )
        for case_index, (shape, transition_scale, radius_multiplier, pair_mode) in enumerate(cases)
    )
    finite_minima = [
        trial.sampled_secant_eigenvalue_minimum
        for trial in trials
        if math.isfinite(trial.sampled_secant_eigenvalue_minimum)
    ]
    finite_maxima = [
        trial.sampled_secant_eigenvalue_maximum
        for trial in trials
        if math.isfinite(trial.sampled_secant_eigenvalue_maximum)
    ]
    finite_storage_ratios = [
        trial.maximum_storage_ratio
        for trial in trials
        if math.isfinite(trial.maximum_storage_ratio)
    ]
    finite_storage_excesses = [
        trial.maximum_normalized_storage_rate_excess
        for trial in trials
        if math.isfinite(trial.maximum_normalized_storage_rate_excess)
    ]
    return NonquadraticProbeSummary(
        config=selected,
        trials=trials,
        worst_state_separation_amplification=max(
            trial.maximum_state_separation_ratio for trial in trials
        ),
        worst_final_state_separation_ratio=max(
            trial.final_state_separation_ratio for trial in trials
        ),
        maximum_storage_ratio=max(finite_storage_ratios, default=math.nan),
        maximum_normalized_storage_rate_excess=max(
            finite_storage_excesses,
            default=math.nan,
        ),
        storage_rate_violating_case_count=sum(
            trial.storage_rate_violation_count > 0 for trial in trials
        ),
        maximum_orientation_commutator=max(
            trial.maximum_normalized_secant_commutator for trial in trials
        ),
        sampled_secant_eigenvalue_minimum=min(finite_minima, default=math.nan),
        sampled_secant_eigenvalue_maximum=max(finite_maxima, default=math.nan),
        orientation_changing_case_count=sum(trial.orientation_change_count > 0 for trial in trials),
        candidate_instability_count=sum(trial.candidate_instability for trial in trials),
        divergence_count=sum(trial.exceeded_divergence_norm for trial in trials),
        nonfinite_count=sum(trial.nonfinite for trial in trials),
    )


__all__ = [
    "RATIONAL_GIVENS_TRIPLES",
    "NonquadraticProbeConfig",
    "NonquadraticProbeSummary",
    "NonquadraticTrial",
    "RotatingLogCoshObjective",
    "rational_givens_frame",
    "run_nonquadratic_probe",
    "run_nonquadratic_trial",
]
