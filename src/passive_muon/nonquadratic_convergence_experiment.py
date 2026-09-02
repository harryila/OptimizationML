"""Falsification probes for the full-step nonquadratic convergence storage.

The exact theorem lives in :mod:`passive_muon.nonquadratic_convergence`.  This
module only exercises its Lyapunov function on deterministic float64
trajectories with changing Hessian orientations.  A sampled violation is a
candidate implementation witness; sampled passes are not a proof.
"""

from __future__ import annotations

import itertools
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from passive_muon.nonquadratic_convergence import (
    LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE,
    NonquadraticConvergenceCertificate,
    locked_nonquadratic_convergence_certificate,
)
from passive_muon.nonquadratic_experiment import (
    RATIONAL_GIVENS_TRIPLES,
    RotatingLogCoshObjective,
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


def _stable_log_cosh(value: np.ndarray) -> np.ndarray:
    """Evaluate log(cosh(value)) accurately near zero and without overflow."""

    absolute = np.abs(value)
    output = np.empty_like(absolute)
    small = absolute < 1e-3
    squared = value[small] ** 2
    output[small] = squared * (
        0.5 + squared * (-1.0 / 12.0 + squared * (1.0 / 45.0 - 17.0 * squared / 2520.0))
    )
    large = ~small
    output[large] = absolute[large] + np.log1p(np.exp(-2.0 * absolute[large])) - math.log(2.0)
    return output


def objective_gap(objective: RotatingLogCoshObjective, position: np.ndarray) -> float:
    """Return ``f(position)-f(0)`` for the locked log-cosh family."""

    vector = np.asarray(position, dtype=np.float64).reshape(-1)
    if vector.size != objective.dimension:
        raise ValueError(f"expected {objective.dimension} entries, received {vector.size}")
    scale = objective.transition_scale
    gap = 0.5 * objective.curvature_lower * float(vector @ vector)
    for basis in (np.eye(objective.dimension, dtype=np.float64), objective.frame):
        coordinates = basis.T @ vector / scale
        gap += objective.nonlinear_weight * scale**2 * float(np.sum(_stable_log_cosh(coordinates)))
    return gap


def convergence_lyapunov_value(
    objective: RotatingLogCoshObjective,
    position: np.ndarray,
    momentum: np.ndarray,
    certificate: NonquadraticConvergenceCertificate | None = None,
) -> float:
    """Evaluate the theorem's ``P`` state storage plus normalized function gap."""

    selected = locked_nonquadratic_convergence_certificate() if certificate is None else certificate
    position_vector = np.asarray(position, dtype=np.float64).reshape(-1)
    momentum_vector = np.asarray(momentum, dtype=np.float64).reshape(-1)
    if position_vector.size != objective.dimension or momentum_vector.size != objective.dimension:
        raise ValueError("position and momentum must match the objective dimension")
    normalized_momentum = momentum_vector / float(selected.smoothness)
    storage = np.asarray(
        [[float(value) for value in row] for row in selected.storage],
        dtype=np.float64,
    )
    state_storage = float(
        storage[0, 0] * (position_vector @ position_vector)
        + 2.0 * storage[0, 1] * (position_vector @ normalized_momentum)
        + storage[1, 1] * (normalized_momentum @ normalized_momentum)
    )
    normalized_gap = objective_gap(objective, position_vector) / float(selected.smoothness)
    return state_storage + float(selected.function_storage) * normalized_gap


@dataclass(frozen=True)
class NonquadraticConvergenceProbeConfig:
    """Locked full-step configuration and deterministic objective grid."""

    seed: int = 2_026_090_2
    shapes: tuple[tuple[int, int], ...] = ((2, 2), (3, 3))
    transition_scales: tuple[float, ...] = (1e-4, 1.0, 100.0)
    radius_multipliers: tuple[float, ...] = (0.25, 1.0, 4.0)
    momentum_modes: tuple[str, ...] = ("zero", "random")
    iterations: int = 500
    diagnostic_stride: int = 5
    learning_rate: float = float(LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE)
    beta: float = float(LOCKED_BETA)
    floor: float = float(LOCKED_FLOOR)
    repair_rho: float = float(LOCKED_REPAIR_RHO)
    curvature_lower: float = float(LOCKED_HESSIAN_LOWER)
    curvature_upper: float = float(LOCKED_HESSIAN_UPPER)
    orientation_tolerance: float = 1e-10
    resolution_multiplier: float = 256.0
    storage_rate_tolerance: float = 1e-10
    divergence_norm: float = 1e100

    def __post_init__(self) -> None:
        if not self.shapes or any(rows <= 0 or columns <= 0 for rows, columns in self.shapes):
            raise ValueError("shapes must contain positive matrix dimensions")
        if not self.transition_scales or any(scale <= 0 for scale in self.transition_scales):
            raise ValueError("transition scales must be positive")
        if not self.radius_multipliers or any(radius <= 0 for radius in self.radius_multipliers):
            raise ValueError("radius multipliers must be positive")
        if not self.momentum_modes or any(
            mode not in {"zero", "random"} for mode in self.momentum_modes
        ):
            raise ValueError("momentum modes must be 'zero' or 'random'")
        if self.iterations <= 0 or self.diagnostic_stride <= 0:
            raise ValueError("iterations and diagnostic stride must be positive")
        locked_values = {
            "learning_rate": float(LOCKED_NONQUADRATIC_CONVERGENCE_LEARNING_RATE),
            "beta": float(LOCKED_BETA),
            "floor": float(LOCKED_FLOOR),
            "repair_rho": float(LOCKED_REPAIR_RHO),
            "curvature_lower": float(LOCKED_HESSIAN_LOWER),
            "curvature_upper": float(LOCKED_HESSIAN_UPPER),
        }
        for name, expected in locked_values.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal the locked convergence-certificate value")
        if self.orientation_tolerance < 0 or self.storage_rate_tolerance < 0:
            raise ValueError("diagnostic tolerances must be nonnegative")
        if self.resolution_multiplier <= 0 or self.divergence_norm <= 0:
            raise ValueError("resolution multiplier and divergence norm must be positive")

    @property
    def case_count(self) -> int:
        return (
            len(self.shapes)
            * len(self.transition_scales)
            * len(self.radius_multipliers)
            * len(self.momentum_modes)
        )


@dataclass(frozen=True)
class NonquadraticConvergenceTrial:
    """One trajectory-to-minimizer storage replay."""

    shape: tuple[int, int]
    transition_scale: float
    radius_multiplier: float
    momentum_mode: str
    case_index: int
    case_seed: int
    frame_variant: int
    frame_sha256: str
    frame_orthogonality_residual: float
    requested_iterations: int
    executed_iterations: int
    initial_position_norm: float
    initial_momentum_norm: float
    initial_objective_gap: float
    initial_lyapunov_value: float
    final_position_norm: float
    final_momentum_norm: float
    final_signal_norm: float
    final_objective_gap: float
    final_lyapunov_value: float
    final_lyapunov_ratio: float
    maximum_position_norm: float
    maximum_momentum_norm: float
    maximum_signal_norm: float
    maximum_lyapunov_ratio: float
    maximum_normalized_lyapunov_rate_excess: float
    lyapunov_transition_count: int
    lyapunov_rate_violation_count: int
    nonpositive_lyapunov_count: int
    sampled_hessian_eigenvalue_minimum: float
    sampled_hessian_eigenvalue_maximum: float
    maximum_normalized_hessian_commutator: float
    orientation_change_count: int
    hessian_sample_count: int
    nonfinite: bool
    exceeded_divergence_norm: bool
    candidate_implementation_violation: bool


@dataclass(frozen=True)
class NonquadraticConvergenceProbeSummary:
    """Aggregate full-step convergence falsification result."""

    config: NonquadraticConvergenceProbeConfig
    trials: tuple[NonquadraticConvergenceTrial, ...]
    maximum_lyapunov_ratio: float
    maximum_normalized_lyapunov_rate_excess: float
    worst_final_lyapunov_ratio: float
    maximum_orientation_commutator: float
    sampled_hessian_eigenvalue_minimum: float
    sampled_hessian_eigenvalue_maximum: float
    orientation_changing_case_count: int
    lyapunov_rate_violating_case_count: int
    candidate_violation_count: int
    divergence_count: int
    nonfinite_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a self-contained JSON-compatible diagnostic record."""

        certificate = locked_nonquadratic_convergence_certificate()
        coefficients = JORDAN_QUINTIC.fractions()
        return {
            "schema_version": "passive-muon-nonquadratic-convergence-falsification-v1",
            "evidence_kind": "deterministic_sampled_trajectory_to_minimizer_replay",
            "claim_scope": {
                "role": (
                    "float64 falsification and implementation-parity diagnostic for the "
                    "separate exact convergence theorem"
                ),
                "objective_domain": (
                    "the recorded two-frame log-cosh objectives with analytic global "
                    "1-strong-convexity and 10-smoothness"
                ),
                "qualification": (
                    "a sampled violation is a candidate implementation witness requiring "
                    "independent replay; sampled passes do not prove the theorem"
                ),
                "not_claimed": [
                    "an arbitrary-pair incremental contraction result at the full p4 step",
                    "stochastic, BF16, additive-epsilon, or neural-network convergence",
                    "a theorem inferred from sampled Hessians or trajectories",
                ],
            },
            "objective_family": {
                "name": "two_rationally_rotated_log_cosh_frames",
                "implementation": "passive_muon.nonquadratic_experiment.RotatingLogCoshObjective",
                "vectorization": "row-major flattening of each real matrix",
                "minimizer": "W_star=0 and f_star=0",
                "analytic_hessian_bounds_exact": {
                    "lower": str(LOCKED_HESSIAN_LOWER),
                    "upper": str(LOCKED_HESSIAN_UPPER),
                },
                "pythagorean_givens_triples": [list(triple) for triple in RATIONAL_GIVENS_TRIPLES],
                "orientation_measure": (
                    "||H_t*H_previous-H_previous*H_t||_F/(||H_t||_F*||H_previous||_F)"
                ),
            },
            "locked_convergence_storage": {
                "state": "[W-W_star, m/L]",
                "formula": ("V=state^T*(P tensor I)*state+function_storage*(f(W)-f_star)/L"),
                "storage_P": [[str(value) for value in row] for row in certificate.storage],
                "function_storage": str(certificate.function_storage),
                "learning_rate_eta": str(certificate.learning_rate),
                "exponential_rate_tau": str(certificate.tau),
                "rate_squared": str(certificate.tau**2),
                "measured_rate_excess": "V_next/V_previous-tau^2 on resolved states",
                "resolution_rule": (
                    "previous state norm exceeds resolution_multiplier*float64_epsilon*"
                    "max(1,initial_state_norm), and previous V exceeds the corresponding "
                    "relative storage floor"
                ),
            },
            "operator": {
                "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
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
            "config": asdict(self.config),
            "summary": {
                "trial_count": len(self.trials),
                "maximum_lyapunov_ratio": self.maximum_lyapunov_ratio,
                "maximum_normalized_lyapunov_rate_excess": (
                    self.maximum_normalized_lyapunov_rate_excess
                ),
                "worst_final_lyapunov_ratio": self.worst_final_lyapunov_ratio,
                "maximum_orientation_commutator": self.maximum_orientation_commutator,
                "sampled_hessian_eigenvalue_minimum": (self.sampled_hessian_eigenvalue_minimum),
                "sampled_hessian_eigenvalue_maximum": (self.sampled_hessian_eigenvalue_maximum),
                "orientation_changing_case_count": self.orientation_changing_case_count,
                "lyapunov_rate_violating_case_count": (self.lyapunov_rate_violating_case_count),
                "candidate_violation_count": self.candidate_violation_count,
                "divergence_count": self.divergence_count,
                "nonfinite_count": self.nonfinite_count,
            },
            "trials": [asdict(trial) for trial in self.trials],
            "experiment_provenance": {
                "seed": self.config.seed,
                "determinism": "seeded CPU float64 trajectories",
                "dtype": "float64",
                "hardware": {
                    "platform": platform.platform(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "cpu_count": os.cpu_count(),
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


def _state_norm(position: np.ndarray, momentum: np.ndarray, *, smoothness: float) -> float:
    return math.hypot(
        float(np.linalg.norm(position)),
        float(np.linalg.norm(momentum)) / smoothness,
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


def run_nonquadratic_convergence_trial(
    *,
    config: NonquadraticConvergenceProbeConfig,
    shape: tuple[int, int],
    transition_scale: float,
    radius_multiplier: float,
    momentum_mode: str,
    case_index: int,
) -> NonquadraticConvergenceTrial:
    """Run one trajectory and replay the locked objective-gap storage each step."""

    if shape not in config.shapes:
        raise ValueError("shape is not present in the probe configuration")
    if transition_scale not in config.transition_scales:
        raise ValueError("transition scale is not present in the probe configuration")
    if radius_multiplier not in config.radius_multipliers:
        raise ValueError("radius multiplier is not present in the probe configuration")
    if momentum_mode not in config.momentum_modes:
        raise ValueError("momentum mode is not present in the probe configuration")

    dimension = math.prod(shape)
    seed = _case_seed(config.seed, case_index, dimension)
    rng = np.random.default_rng(seed)
    pair_count = max(1, dimension * (dimension - 1) // 2)
    objective = RotatingLogCoshObjective(
        dimension=dimension,
        transition_scale=transition_scale,
        frame_variant=seed % pair_count,
        curvature_lower=config.curvature_lower,
        curvature_upper=config.curvature_upper,
    )
    initial_radius = transition_scale * radius_multiplier
    position = _normalized_random_vector(dimension=dimension, norm=initial_radius, rng=rng)
    momentum = (
        np.zeros(dimension, dtype=np.float64)
        if momentum_mode == "zero"
        else _normalized_random_vector(
            dimension=dimension,
            norm=config.curvature_upper * initial_radius,
            rng=rng,
        )
    )
    certificate = locked_nonquadratic_convergence_certificate()
    tau_squared = float(certificate.tau**2)
    initial_position_norm = float(np.linalg.norm(position))
    initial_momentum_norm = float(np.linalg.norm(momentum))
    initial_gap = objective_gap(objective, position)
    initial_lyapunov = convergence_lyapunov_value(
        objective,
        position,
        momentum,
        certificate,
    )
    if initial_lyapunov <= 0.0:
        raise AssertionError("the locked Lyapunov storage was not initially positive")

    initial_state_norm = _state_norm(
        position,
        momentum,
        smoothness=config.curvature_upper,
    )
    machine_epsilon = np.finfo(np.float64).eps
    state_resolution = config.resolution_multiplier * machine_epsilon * max(1.0, initial_state_norm)
    lyapunov_resolution = max(
        np.finfo(np.float64).tiny,
        initial_lyapunov * config.resolution_multiplier * machine_epsilon,
    )
    current_lyapunov = initial_lyapunov
    current_state_norm = initial_state_norm
    maximum_lyapunov_ratio = -math.inf
    maximum_rate_excess = -math.inf
    lyapunov_transition_count = 0
    lyapunov_rate_violation_count = 0
    nonpositive_lyapunov_count = 0
    maximum_position_norm = initial_position_norm
    maximum_momentum_norm = initial_momentum_norm
    maximum_signal_norm = 0.0
    minimum_hessian_eigenvalue = math.inf
    maximum_hessian_eigenvalue = -math.inf
    maximum_commutator = 0.0
    orientation_change_count = 0
    hessian_sample_count = 0
    previous_hessian: np.ndarray | None = None
    signal = np.zeros(dimension, dtype=np.float64)
    nonfinite = False
    exceeded = False
    executed = 0
    one_minus_beta = 1.0 - config.beta

    def sample_hessian() -> None:
        nonlocal hessian_sample_count
        nonlocal maximum_commutator
        nonlocal maximum_hessian_eigenvalue
        nonlocal minimum_hessian_eigenvalue
        nonlocal orientation_change_count
        nonlocal previous_hessian

        hessian = objective.hessian(position)
        eigenvalues = np.linalg.eigvalsh(hessian)
        minimum_hessian_eigenvalue = min(minimum_hessian_eigenvalue, float(eigenvalues[0]))
        maximum_hessian_eigenvalue = max(maximum_hessian_eigenvalue, float(eigenvalues[-1]))
        if previous_hessian is not None:
            commutator = hessian @ previous_hessian - previous_hessian @ hessian
            denominator = float(
                np.linalg.norm(hessian, ord="fro") * np.linalg.norm(previous_hessian, ord="fro")
            )
            normalized = float(np.linalg.norm(commutator, ord="fro")) / denominator
            maximum_commutator = max(maximum_commutator, normalized)
            orientation_change_count += normalized > config.orientation_tolerance
        previous_hessian = hessian
        hessian_sample_count += 1

    sample_hessian()
    for iteration in range(1, config.iterations + 1):
        gradient = objective.gradient(position).reshape(-1)
        momentum = config.beta * momentum + one_minus_beta * gradient
        signal = config.beta * momentum + one_minus_beta * gradient
        position = position - config.learning_rate * _repaired_floored_jordan(
            signal,
            shape=shape,
            floor=config.floor,
            repair_rho=config.repair_rho,
        )
        executed = iteration

        next_lyapunov = convergence_lyapunov_value(
            objective,
            position,
            momentum,
            certificate,
        )
        next_state_norm = _state_norm(
            position,
            momentum,
            smoothness=config.curvature_upper,
        )
        if current_state_norm > state_resolution and current_lyapunov > lyapunov_resolution:
            if next_lyapunov <= 0.0:
                nonpositive_lyapunov_count += 1
            else:
                ratio = next_lyapunov / current_lyapunov
                excess = ratio - tau_squared
                maximum_lyapunov_ratio = max(maximum_lyapunov_ratio, ratio)
                maximum_rate_excess = max(maximum_rate_excess, excess)
                lyapunov_transition_count += 1
                lyapunov_rate_violation_count += excess > config.storage_rate_tolerance
        current_lyapunov = next_lyapunov
        current_state_norm = next_state_norm

        position_norm = float(np.linalg.norm(position))
        momentum_norm = float(np.linalg.norm(momentum))
        signal_norm = float(np.linalg.norm(signal))
        maximum_position_norm = max(maximum_position_norm, position_norm)
        maximum_momentum_norm = max(maximum_momentum_norm, momentum_norm)
        maximum_signal_norm = max(maximum_signal_norm, signal_norm)
        finite_values = (
            position_norm,
            momentum_norm,
            signal_norm,
            next_lyapunov,
            next_state_norm,
        )
        if not all(math.isfinite(value) for value in finite_values):
            nonfinite = True
            exceeded = True
            break
        if max(position_norm, momentum_norm, signal_norm) > config.divergence_norm:
            exceeded = True
            break
        if iteration % config.diagnostic_stride == 0 or iteration == config.iterations:
            sample_hessian()

    if lyapunov_transition_count == 0:
        maximum_lyapunov_ratio = math.nan
        maximum_rate_excess = math.nan
    final_position_norm = float(np.linalg.norm(position))
    final_momentum_norm = float(np.linalg.norm(momentum))
    final_signal_norm = float(np.linalg.norm(signal))
    final_gap = objective_gap(objective, position)
    final_lyapunov = convergence_lyapunov_value(objective, position, momentum, certificate)
    candidate = bool(
        nonfinite or exceeded or nonpositive_lyapunov_count > 0 or lyapunov_rate_violation_count > 0
    )
    return NonquadraticConvergenceTrial(
        shape=shape,
        transition_scale=transition_scale,
        radius_multiplier=radius_multiplier,
        momentum_mode=momentum_mode,
        case_index=case_index,
        case_seed=seed,
        frame_variant=objective.frame_variant,
        frame_sha256=objective.frame_sha256,
        frame_orthogonality_residual=objective.frame_orthogonality_residual,
        requested_iterations=config.iterations,
        executed_iterations=executed,
        initial_position_norm=initial_position_norm,
        initial_momentum_norm=initial_momentum_norm,
        initial_objective_gap=initial_gap,
        initial_lyapunov_value=initial_lyapunov,
        final_position_norm=final_position_norm,
        final_momentum_norm=final_momentum_norm,
        final_signal_norm=final_signal_norm,
        final_objective_gap=final_gap,
        final_lyapunov_value=final_lyapunov,
        final_lyapunov_ratio=final_lyapunov / initial_lyapunov,
        maximum_position_norm=maximum_position_norm,
        maximum_momentum_norm=maximum_momentum_norm,
        maximum_signal_norm=maximum_signal_norm,
        maximum_lyapunov_ratio=maximum_lyapunov_ratio,
        maximum_normalized_lyapunov_rate_excess=maximum_rate_excess,
        lyapunov_transition_count=lyapunov_transition_count,
        lyapunov_rate_violation_count=lyapunov_rate_violation_count,
        nonpositive_lyapunov_count=nonpositive_lyapunov_count,
        sampled_hessian_eigenvalue_minimum=minimum_hessian_eigenvalue,
        sampled_hessian_eigenvalue_maximum=maximum_hessian_eigenvalue,
        maximum_normalized_hessian_commutator=maximum_commutator,
        orientation_change_count=orientation_change_count,
        hessian_sample_count=hessian_sample_count,
        nonfinite=nonfinite,
        exceeded_divergence_norm=exceeded,
        candidate_implementation_violation=candidate,
    )


def run_nonquadratic_convergence_probe(
    config: NonquadraticConvergenceProbeConfig | None = None,
) -> NonquadraticConvergenceProbeSummary:
    """Run all locked trajectory-to-minimizer falsification cases."""

    selected = NonquadraticConvergenceProbeConfig() if config is None else config
    cases = itertools.product(
        selected.shapes,
        selected.transition_scales,
        selected.radius_multipliers,
        selected.momentum_modes,
    )
    trials = tuple(
        run_nonquadratic_convergence_trial(
            config=selected,
            shape=shape,
            transition_scale=transition_scale,
            radius_multiplier=radius_multiplier,
            momentum_mode=momentum_mode,
            case_index=case_index,
        )
        for case_index, (shape, transition_scale, radius_multiplier, momentum_mode) in enumerate(
            cases
        )
    )

    def finite_values(attribute: str) -> list[float]:
        values = [float(getattr(trial, attribute)) for trial in trials]
        return [value for value in values if math.isfinite(value)]

    return NonquadraticConvergenceProbeSummary(
        config=selected,
        trials=trials,
        maximum_lyapunov_ratio=max(finite_values("maximum_lyapunov_ratio"), default=math.nan),
        maximum_normalized_lyapunov_rate_excess=max(
            finite_values("maximum_normalized_lyapunov_rate_excess"),
            default=math.nan,
        ),
        worst_final_lyapunov_ratio=max(
            finite_values("final_lyapunov_ratio"),
            default=math.nan,
        ),
        maximum_orientation_commutator=max(
            finite_values("maximum_normalized_hessian_commutator"),
            default=math.nan,
        ),
        sampled_hessian_eigenvalue_minimum=min(
            finite_values("sampled_hessian_eigenvalue_minimum"),
            default=math.nan,
        ),
        sampled_hessian_eigenvalue_maximum=max(
            finite_values("sampled_hessian_eigenvalue_maximum"),
            default=math.nan,
        ),
        orientation_changing_case_count=sum(trial.orientation_change_count > 0 for trial in trials),
        lyapunov_rate_violating_case_count=sum(
            trial.lyapunov_rate_violation_count > 0 for trial in trials
        ),
        candidate_violation_count=sum(trial.candidate_implementation_violation for trial in trials),
        divergence_count=sum(trial.exceeded_divergence_norm for trial in trials),
        nonfinite_count=sum(trial.nonfinite for trial in trials),
    )


__all__ = [
    "NonquadraticConvergenceProbeConfig",
    "NonquadraticConvergenceProbeSummary",
    "NonquadraticConvergenceTrial",
    "convergence_lyapunov_value",
    "objective_gap",
    "run_nonquadratic_convergence_probe",
    "run_nonquadratic_convergence_trial",
]
