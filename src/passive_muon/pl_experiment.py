"""Deterministic falsification probes for the repaired Muon loop on PL objectives.

The objective family below is analytically nonconvex, globally smooth, and
satisfies a global Polyak--Lojasiewicz inequality.  It includes both a
full-rank instance and a rank-deficient instance with a nonunique minimizer
set.  The trajectory runs are CPU/float64 diagnostics only: a sampled failure
can falsify an implementation or a proposed certificate, while sampled passes
cannot prove a global PL convergence theorem.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from passive_muon.nonquadratic_experiment import (
    RATIONAL_GIVENS_TRIPLES,
    rational_givens_frame,
)
from passive_muon.operator import orthogonalize
from passive_muon.pl_convergence import (
    LOCKED_PL_CONVERGENCE_LEARNING_RATE,
    LOCKED_PL_CONVERGENCE_TAU,
    PlConvergenceCertificate,
    locked_pl_convergence_certificate,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_FLOOR,
    LOCKED_HESSIAN_UPPER,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
)

WARP_STRENGTH = 5.0
OBJECTIVE_SCALE = 1.5
ANALYTIC_PL_LOWER = 1.0
ANALYTIC_SMOOTHNESS_UPPER = 9.0
ANALYTIC_NEGATIVE_CURVATURE = -3.0 / 8.0


def pl_lyapunov_value(
    objective: WarpedRadialPLObjective,
    position: np.ndarray,
    momentum: np.ndarray,
    certificate: PlConvergenceCertificate | None = None,
) -> float:
    """Evaluate the exact P6 certificate storage in float64.

    The storage is ``[m/L, grad/L]^T(P tensor I)[m/L, grad/L]`` plus
    ``c*(f-f_star)/L``.  The exact proof and authoritative rational replay live
    in :mod:`passive_muon.pl_convergence`; this evaluation is diagnostic only.
    """

    selected = locked_pl_convergence_certificate() if certificate is None else certificate
    position_vector = np.asarray(position, dtype=np.float64).reshape(-1)
    momentum_vector = np.asarray(momentum, dtype=np.float64).reshape(-1)
    if position_vector.size != objective.dimension or momentum_vector.size != objective.dimension:
        raise ValueError("position and momentum must match the objective dimension")
    smoothness = float(selected.smoothness)
    normalized_momentum = momentum_vector / smoothness
    normalized_gradient = np.asarray(objective.gradient(position_vector)).reshape(-1) / smoothness
    storage = np.asarray(
        [[float(value) for value in row] for row in selected.storage],
        dtype=np.float64,
    )
    state_storage = float(
        storage[0, 0] * (normalized_momentum @ normalized_momentum)
        + 2.0 * storage[0, 1] * (normalized_momentum @ normalized_gradient)
        + storage[1, 1] * (normalized_gradient @ normalized_gradient)
    )
    return (
        state_storage
        + float(selected.function_storage) * objective.value(position_vector) / smoothness
    )


@dataclass(frozen=True)
class WarpedRadialPLObjective:
    """A smooth nonconvex PL objective over a possibly deficient subspace.

    Let ``P`` be an orthogonal projector, ``s>0``, and

    ``q(x)=||P x||^2/(2 s^2)``.

    The locked objective is

    ``f(x)=(3/2) s^2 [q(x)+5 q(x)/(1+q(x))]``.

    Its minimizer set is ``ker(P)``.  If ``P=I`` the minimizer is unique; if
    ``rank(P)<dimension`` it is nonunique.  In exact real arithmetic:

    * ``(1/2)||grad f||^2 >= f`` (global PL constant at least one),
    * ``||nabla^2 f||_2 <= 9`` (hence it is also 10-smooth), and
    * the active radial Hessian eigenvalue equals ``-3/8`` at ``q=1``.

    ``frame`` and ``projector`` are float64 realizations of products of
    rational Givens rotations.  Their numerical residuals are recorded by the
    experiment; the analytic claims refer to the exact-real construction.
    """

    dimension: int
    transition_scale: float
    active_rank: int
    frame_variant: int = 0
    frame: np.ndarray = field(init=False, repr=False, compare=False)
    projector: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("dimension must be positive")
        if self.transition_scale <= 0:
            raise ValueError("transition_scale must be positive")
        if not 1 <= self.active_rank <= self.dimension:
            raise ValueError("active_rank must lie in [1, dimension]")
        if self.frame_variant < 0:
            raise ValueError("frame_variant must be nonnegative")
        frame = rational_givens_frame(self.dimension, variant=self.frame_variant)
        if self.active_rank == self.dimension:
            projector = np.eye(self.dimension, dtype=np.float64)
        else:
            active_frame = frame[:, : self.active_rank]
            projector = active_frame @ active_frame.T
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "projector", projector)

    @property
    def is_rank_deficient(self) -> bool:
        return self.active_rank < self.dimension

    @property
    def minimizer_set_dimension(self) -> int:
        return self.dimension - self.active_rank

    @property
    def frame_orthogonality_residual(self) -> float:
        identity = np.eye(self.dimension, dtype=np.float64)
        return float(np.linalg.norm(self.frame.T @ self.frame - identity, ord="fro"))

    @property
    def projector_idempotence_residual(self) -> float:
        return float(np.linalg.norm(self.projector @ self.projector - self.projector, ord="fro"))

    @property
    def projector_symmetry_residual(self) -> float:
        return float(np.linalg.norm(self.projector.T - self.projector, ord="fro"))

    @property
    def projector_sha256(self) -> str:
        return hashlib.sha256(np.ascontiguousarray(self.projector).tobytes()).hexdigest()

    def _vector(self, value: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
        array = np.asarray(value, dtype=np.float64)
        if array.size != self.dimension:
            raise ValueError(f"expected {self.dimension} entries, received {array.size}")
        return array.reshape(-1), array.shape

    def active_component(self, position: np.ndarray) -> np.ndarray:
        vector, shape = self._vector(position)
        return (self.projector @ vector).reshape(shape)

    def null_component(self, position: np.ndarray) -> np.ndarray:
        vector, shape = self._vector(position)
        return (vector - self.projector @ vector).reshape(shape)

    def scaled_radius(self, position: np.ndarray) -> float:
        active = np.asarray(self.active_component(position)).reshape(-1)
        return 0.5 * float(active @ active) / self.transition_scale**2

    def value(self, position: np.ndarray) -> float:
        q = self.scaled_radius(position)
        warp = q + WARP_STRENGTH * q / (1.0 + q)
        return OBJECTIVE_SCALE * self.transition_scale**2 * warp

    def gradient(self, position: np.ndarray) -> np.ndarray:
        vector, shape = self._vector(position)
        active = self.projector @ vector
        q = 0.5 * float(active @ active) / self.transition_scale**2
        derivative = 1.0 + WARP_STRENGTH / (1.0 + q) ** 2
        return (OBJECTIVE_SCALE * derivative * active).reshape(shape)

    def hessian(self, position: np.ndarray) -> np.ndarray:
        vector, _shape = self._vector(position)
        active = self.projector @ vector
        scale_squared = self.transition_scale**2
        q = 0.5 * float(active @ active) / scale_squared
        derivative = 1.0 + WARP_STRENGTH / (1.0 + q) ** 2
        second_derivative = -2.0 * WARP_STRENGTH / (1.0 + q) ** 3
        return OBJECTIVE_SCALE * (
            derivative * self.projector
            + second_derivative * np.outer(active, active) / scale_squared
        )

    def pl_ratio(self, position: np.ndarray) -> float:
        """Return ``||grad f||^2/(2(f-f_star))``, or infinity at a minimizer."""

        gap = self.value(position)
        if gap == 0.0:
            return math.inf
        gradient = np.asarray(self.gradient(position)).reshape(-1)
        return 0.5 * float(gradient @ gradient) / gap


@dataclass(frozen=True)
class PLProbeConfig:
    """Locked repaired-loop parameters and deterministic nonconvex PL grid."""

    seed: int = 2_026_090_2
    shapes: tuple[tuple[int, int], ...] = ((2, 2), (3, 3))
    rank_modes: tuple[str, ...] = ("full_rank", "codimension_one")
    transition_scales: tuple[float, ...] = (1e-4, 1.0, 100.0)
    radius_multipliers: tuple[float, ...] = (0.5, math.sqrt(2.0), 4.0)
    momentum_modes: tuple[str, ...] = ("zero", "random")
    null_component_multiplier: float = 1.0
    iterations: int = 500
    diagnostic_stride: int = 5
    learning_rate: float = float(LOCKED_PL_CONVERGENCE_LEARNING_RATE)
    beta: float = float(LOCKED_BETA)
    floor: float = float(LOCKED_FLOOR)
    repair_rho: float = float(LOCKED_REPAIR_RHO)
    claimed_pl_lower: float = ANALYTIC_PL_LOWER
    claimed_smoothness_upper: float = float(LOCKED_HESSIAN_UPPER)
    hessian_tolerance: float = 2e-11
    pl_tolerance: float = 2e-12
    orientation_tolerance: float = 1e-10
    resolution_multiplier: float = 256.0
    storage_rate_tolerance: float = 2e-10
    divergence_norm: float = 1e100

    def __post_init__(self) -> None:
        if not self.shapes or any(rows <= 0 or columns <= 0 for rows, columns in self.shapes):
            raise ValueError("shapes must contain positive matrix dimensions")
        if not self.rank_modes or any(
            mode not in {"full_rank", "codimension_one"} for mode in self.rank_modes
        ):
            raise ValueError("rank_modes must contain only supported modes")
        if any(math.prod(shape) < 2 for shape in self.shapes) and (
            "codimension_one" in self.rank_modes
        ):
            raise ValueError("codimension_one cases require at least two coordinates")
        if not self.transition_scales or any(scale <= 0 for scale in self.transition_scales):
            raise ValueError("transition_scales must be positive")
        if not self.radius_multipliers or any(radius <= 0 for radius in self.radius_multipliers):
            raise ValueError("radius_multipliers must be positive")
        if not self.momentum_modes or any(
            mode not in {"zero", "random"} for mode in self.momentum_modes
        ):
            raise ValueError("momentum_modes must contain only 'zero' or 'random'")
        if self.null_component_multiplier < 0:
            raise ValueError("null_component_multiplier must be nonnegative")
        if self.iterations <= 0 or self.diagnostic_stride <= 0:
            raise ValueError("iterations and diagnostic_stride must be positive")
        locked_values = {
            "learning_rate": float(LOCKED_PL_CONVERGENCE_LEARNING_RATE),
            "beta": float(LOCKED_BETA),
            "floor": float(LOCKED_FLOOR),
            "repair_rho": float(LOCKED_REPAIR_RHO),
            "claimed_pl_lower": ANALYTIC_PL_LOWER,
            "claimed_smoothness_upper": float(LOCKED_HESSIAN_UPPER),
        }
        for name, expected in locked_values.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal the locked P6 diagnostic value")
        if (
            min(
                self.hessian_tolerance,
                self.pl_tolerance,
                self.orientation_tolerance,
                self.storage_rate_tolerance,
            )
            < 0
        ):
            raise ValueError("diagnostic tolerances must be nonnegative")
        if self.resolution_multiplier <= 0 or self.divergence_norm <= 0:
            raise ValueError("resolution_multiplier and divergence_norm must be positive")

    @property
    def case_count(self) -> int:
        return (
            len(self.shapes)
            * len(self.rank_modes)
            * len(self.transition_scales)
            * len(self.radius_multipliers)
            * len(self.momentum_modes)
        )


@dataclass(frozen=True)
class PLProbeTrial:
    """One sampled repaired-loop trajectory on a nonconvex PL objective."""

    shape: tuple[int, int]
    rank_mode: str
    active_rank: int
    minimizer_set_dimension: int
    transition_scale: float
    radius_multiplier: float
    momentum_mode: str
    case_index: int
    case_seed: int
    frame_variant: int
    projector_sha256: str
    frame_orthogonality_residual: float
    projector_idempotence_residual: float
    projector_symmetry_residual: float
    requested_iterations: int
    executed_iterations: int
    initial_position_norm: float
    initial_distance_to_minimizer_set: float
    initial_null_component_norm: float
    initial_momentum_norm: float
    initial_objective_gap: float
    initial_lyapunov_value: float
    final_position_norm: float
    final_distance_to_minimizer_set: float
    final_null_component_norm: float
    null_component_drift_norm: float
    final_momentum_norm: float
    final_signal_norm: float
    final_gradient_norm: float
    final_objective_gap: float
    final_lyapunov_value: float
    final_lyapunov_ratio: float
    final_objective_gap_ratio: float
    final_momentum_to_peak_ratio: float
    empirical_gap_geometric_factor: float
    maximum_position_norm: float
    maximum_momentum_norm: float
    maximum_signal_norm: float
    maximum_objective_gap: float
    maximum_resolved_one_step_gap_ratio: float
    objective_gap_increase_count: int
    resolved_gap_transition_count: int
    maximum_lyapunov_ratio: float
    maximum_normalized_lyapunov_rate_excess: float
    lyapunov_transition_count: int
    lyapunov_rate_violation_count: int
    nonpositive_lyapunov_count: int
    sampled_hessian_eigenvalue_minimum: float
    sampled_hessian_eigenvalue_maximum: float
    minimum_sampled_pl_ratio: float
    hessian_sample_count: int
    pl_ratio_sample_count: int
    negative_curvature_sample_count: int
    maximum_normalized_hessian_commutator: float
    orientation_change_count: int
    nonfinite: bool
    exceeded_divergence_norm: bool
    sampled_objective_bound_violation: bool
    candidate_implementation_violation: bool


@dataclass(frozen=True)
class PLProbeSummary:
    """Aggregate deterministic PL falsification result."""

    config: PLProbeConfig
    trials: tuple[PLProbeTrial, ...]
    worst_final_objective_gap_ratio: float
    worst_empirical_gap_geometric_factor: float
    maximum_resolved_one_step_gap_ratio: float
    worst_final_momentum_to_peak_ratio: float
    worst_final_lyapunov_ratio: float
    maximum_lyapunov_ratio: float
    maximum_normalized_lyapunov_rate_excess: float
    sampled_hessian_eigenvalue_minimum: float
    sampled_hessian_eigenvalue_maximum: float
    minimum_sampled_pl_ratio: float
    maximum_orientation_commutator: float
    rank_deficient_case_count: int
    negative_curvature_case_count: int
    orientation_changing_case_count: int
    objective_increasing_case_count: int
    lyapunov_rate_violating_case_count: int
    nonpositive_lyapunov_case_count: int
    objective_bound_violating_case_count: int
    candidate_violation_count: int
    divergence_count: int
    nonfinite_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a self-contained JSON-compatible diagnostic record."""

        coefficients = JORDAN_QUINTIC.fractions()
        return {
            "schema_version": "passive-muon-pl-falsification-v1",
            "evidence_kind": "deterministic_sampled_nonconvex_pl_trajectory_replay",
            "claim_scope": {
                "role": (
                    "CPU/float64 falsification and implementation diagnostic for the separate "
                    "established exact P6 certificate"
                ),
                "qualification": (
                    "sampled failures can expose candidate implementation or certificate "
                    "violations; sampled passes do not prove global PL convergence"
                ),
                "not_claimed": [
                    "a proof inferred from sampled objectives, Hessians, or trajectories",
                    "convergence of W_t to a unique minimizer in rank-deficient cases",
                    "stochastic, BF16, additive-epsilon, or neural-network convergence",
                    "parity with every upstream practical Muon feature",
                ],
            },
            "objective_family": {
                "name": "warped_radial_projected_pl",
                "formula": ("f(x)=(3/2)*s^2*(q+5*q/(1+q)), q=||P*x||^2/(2*s^2)"),
                "vectorization": "row-major flattening of each finite real matrix",
                "projector_modes": {
                    "full_rank": "P=I; unique minimizer x=0",
                    "codimension_one": (
                        "P=Q*diag(1,...,1,0)*Q^T; minimizer set ker(P) is nonunique"
                    ),
                },
                "analytic_exact_real_properties": {
                    "global_PL_lower": "1",
                    "global_gradient_Lipschitz_upper": "9 (therefore also 10-smooth)",
                    "hessian_spectral_interval": "[-3/8, 9]",
                    "nonconvex_witness": "active radial Hessian eigenvalue -3/8 at q=1",
                    "PL_reduction": ("||grad f||^2/(2f)=(3/2)*(1+5/t^2)^2/(1+5/t), t=1+q"),
                    "PL_lower_proof": (
                        "(1+5/t^2)^2 >= (2/3)*(1+5/t) for t>=1 because "
                        "t^4-10*t^3+30*t^2+75 is positive and increasing"
                    ),
                    "PL_polynomial_derivative": (
                        "d/dt[t^4-10*t^3+30*t^2+75]="
                        "2*t*(2*t^2-15*t+30)>0; the quadratic discriminant is -15"
                    ),
                    "active_hessian_eigenvalues": (
                        "tangential=(3/2)*(1+5/t^2); radial=(3/2)*(1-15/t^2+20/t^3), t=1+q"
                    ),
                    "hessian_bound_proof": (
                        "the radial factor has derivative 30*(t-2)/t^4, minimum -1/4 "
                        "at t=2, maximum 6 at t=1; the tangential factor lies in [1,6]"
                    ),
                },
                "rational_frame_generator": {
                    "pythagorean_givens_triples": [
                        list(triple) for triple in RATIONAL_GIVENS_TRIPLES
                    ],
                    "qualification": (
                        "analytic properties refer to exact-real orthogonal products; "
                        "the experiment records float64 realization residuals"
                    ),
                },
                "orientation_measure": (
                    "||H_t*H_previous-H_previous*H_t||_F/(||H_t||_F*||H_previous||_F)"
                ),
            },
            "operator": {
                "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
                "domain": "the finite real matrix shapes in config.shapes",
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
            "locked_parameters": {
                "learning_rate_eta": str(LOCKED_PL_CONVERGENCE_LEARNING_RATE),
                "beta": str(LOCKED_BETA),
                "floor": str(LOCKED_FLOOR),
                "repair_rho": str(LOCKED_REPAIR_RHO),
                "objective_PL_lower": "1",
                "objective_smoothness_upper_used_by_target": str(LOCKED_HESSIAN_UPPER),
                "objective_sharper_smoothness_upper": "9",
            },
            "locked_pl_storage": {
                "state": "[m/L, grad_f(W)/L]",
                "formula": ("V=state^T*(P tensor I)*state+function_storage*(f(W)-f_star)/L"),
                "storage_P": [
                    [str(value) for value in row]
                    for row in locked_pl_convergence_certificate().storage
                ],
                "function_storage": str(locked_pl_convergence_certificate().function_storage),
                "exponential_rate_tau": str(LOCKED_PL_CONVERGENCE_TAU),
                "rate_squared": str(LOCKED_PL_CONVERGENCE_TAU**2),
                "measured_rate_excess": "V_next/V_previous-tau^2 on resolved states",
            },
            "diagnostics": {
                "function_value_target": "f(W_t)-f_star tends to zero geometrically",
                "momentum_target": "||m_t|| tends to zero",
                "distance_target": "||P*W_t|| tends to zero",
                "nullspace_qualification": (
                    "W_t itself need not converge to a unique point because ker(P) may be "
                    "nontrivial"
                ),
                "candidate_rule": (
                    "nonfinite values, divergence threshold, or sampled violations of the "
                    "objective family's analytic Hessian/PL bounds, or a resolved violation "
                    "of the exact P6 Lyapunov rate/nonpositive storage"
                ),
                "objective_increase_rule": (
                    "individual objective-gap increases are recorded and allowed; the theorem "
                    "contracts the composite Lyapunov storage, not f pointwise"
                ),
            },
            "config": asdict(self.config),
            "summary": {
                "trial_count": len(self.trials),
                "worst_final_objective_gap_ratio": self.worst_final_objective_gap_ratio,
                "worst_empirical_gap_geometric_factor": (self.worst_empirical_gap_geometric_factor),
                "maximum_resolved_one_step_gap_ratio": (self.maximum_resolved_one_step_gap_ratio),
                "worst_final_momentum_to_peak_ratio": self.worst_final_momentum_to_peak_ratio,
                "worst_final_lyapunov_ratio": self.worst_final_lyapunov_ratio,
                "maximum_lyapunov_ratio": self.maximum_lyapunov_ratio,
                "maximum_normalized_lyapunov_rate_excess": (
                    self.maximum_normalized_lyapunov_rate_excess
                ),
                "sampled_hessian_eigenvalue_minimum": (self.sampled_hessian_eigenvalue_minimum),
                "sampled_hessian_eigenvalue_maximum": (self.sampled_hessian_eigenvalue_maximum),
                "minimum_sampled_pl_ratio": self.minimum_sampled_pl_ratio,
                "maximum_orientation_commutator": self.maximum_orientation_commutator,
                "rank_deficient_case_count": self.rank_deficient_case_count,
                "negative_curvature_case_count": self.negative_curvature_case_count,
                "orientation_changing_case_count": self.orientation_changing_case_count,
                "objective_increasing_case_count": self.objective_increasing_case_count,
                "lyapunov_rate_violating_case_count": (self.lyapunov_rate_violating_case_count),
                "nonpositive_lyapunov_case_count": self.nonpositive_lyapunov_case_count,
                "objective_bound_violating_case_count": (self.objective_bound_violating_case_count),
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


def _normalized_projected_vector(
    *, projector: np.ndarray, active: bool, norm: float, rng: np.random.Generator
) -> np.ndarray:
    dimension = projector.shape[0]
    identity = np.eye(dimension, dtype=np.float64)
    selected = projector if active else identity - projector
    for _attempt in range(8):
        vector = selected @ rng.standard_normal(dimension)
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > 1e-12:
            return vector * (norm / vector_norm)
    raise RuntimeError("failed to sample a resolved projected direction")


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


def _active_rank(dimension: int, rank_mode: str) -> int:
    if rank_mode == "full_rank":
        return dimension
    if rank_mode == "codimension_one":
        return dimension - 1
    raise ValueError(f"unsupported rank_mode: {rank_mode}")


def run_pl_trial(
    *,
    config: PLProbeConfig,
    shape: tuple[int, int],
    rank_mode: str,
    transition_scale: float,
    radius_multiplier: float,
    momentum_mode: str,
    case_index: int,
) -> PLProbeTrial:
    """Run one pinned repaired EMA/Nesterov trajectory."""

    if shape not in config.shapes:
        raise ValueError("shape is not present in the probe configuration")
    if rank_mode not in config.rank_modes:
        raise ValueError("rank_mode is not present in the probe configuration")
    if transition_scale not in config.transition_scales:
        raise ValueError("transition_scale is not present in the probe configuration")
    if radius_multiplier not in config.radius_multipliers:
        raise ValueError("radius_multiplier is not present in the probe configuration")
    if momentum_mode not in config.momentum_modes:
        raise ValueError("momentum_mode is not present in the probe configuration")

    dimension = math.prod(shape)
    seed = _case_seed(config.seed, case_index, dimension)
    rng = np.random.default_rng(seed)
    pair_count = max(1, dimension * (dimension - 1) // 2)
    objective = WarpedRadialPLObjective(
        dimension=dimension,
        transition_scale=transition_scale,
        active_rank=_active_rank(dimension, rank_mode),
        frame_variant=seed % pair_count,
    )
    active_radius = transition_scale * radius_multiplier
    position = _normalized_projected_vector(
        projector=objective.projector,
        active=True,
        norm=active_radius,
        rng=rng,
    )
    if objective.is_rank_deficient and config.null_component_multiplier > 0:
        position = position + _normalized_projected_vector(
            projector=objective.projector,
            active=False,
            norm=transition_scale * config.null_component_multiplier,
            rng=rng,
        )
    momentum = (
        np.zeros(dimension, dtype=np.float64)
        if momentum_mode == "zero"
        else _normalized_projected_vector(
            projector=np.eye(dimension, dtype=np.float64),
            active=True,
            norm=config.claimed_smoothness_upper * max(active_radius, transition_scale),
            rng=rng,
        )
    )

    initial_null = np.asarray(objective.null_component(position)).reshape(-1)
    initial_position_norm = float(np.linalg.norm(position))
    initial_active_norm = float(np.linalg.norm(objective.active_component(position)))
    initial_null_norm = float(np.linalg.norm(initial_null))
    initial_momentum_norm = float(np.linalg.norm(momentum))
    initial_gap = objective.value(position)
    if initial_gap <= 0:
        raise AssertionError("the initialized PL objective gap must be positive")
    certificate = locked_pl_convergence_certificate()
    tau_squared = float(certificate.tau**2)
    initial_lyapunov = pl_lyapunov_value(objective, position, momentum, certificate)
    if initial_lyapunov <= 0:
        raise AssertionError("the locked PL Lyapunov storage must initially be positive")

    maximum_position_norm = initial_position_norm
    maximum_momentum_norm = initial_momentum_norm
    maximum_signal_norm = 0.0
    maximum_gap = initial_gap
    maximum_gap_ratio = -math.inf
    gap_increase_count = 0
    resolved_gap_transitions = 0
    current_lyapunov = initial_lyapunov
    maximum_lyapunov_ratio = -math.inf
    maximum_lyapunov_rate_excess = -math.inf
    lyapunov_transition_count = 0
    lyapunov_rate_violation_count = 0
    nonpositive_lyapunov_count = 0
    minimum_hessian = math.inf
    maximum_hessian = -math.inf
    minimum_pl_ratio = math.inf
    hessian_sample_count = 0
    pl_ratio_sample_count = 0
    negative_curvature_sample_count = 0
    maximum_commutator = 0.0
    orientation_change_count = 0
    previous_hessian: np.ndarray | None = None
    current_gap = initial_gap
    gap_resolution = max(
        np.finfo(np.float64).tiny,
        initial_gap * config.resolution_multiplier * np.finfo(np.float64).eps,
    )
    lyapunov_resolution = max(
        np.finfo(np.float64).tiny,
        initial_lyapunov * config.resolution_multiplier * np.finfo(np.float64).eps,
    )
    signal = np.zeros(dimension, dtype=np.float64)
    nonfinite = False
    exceeded = False
    sampled_bound_violation = False
    executed = 0
    one_minus_beta = 1.0 - config.beta

    def sample_objective_diagnostics() -> None:
        nonlocal hessian_sample_count
        nonlocal maximum_commutator
        nonlocal maximum_hessian
        nonlocal minimum_hessian
        nonlocal minimum_pl_ratio
        nonlocal negative_curvature_sample_count
        nonlocal orientation_change_count
        nonlocal pl_ratio_sample_count
        nonlocal previous_hessian
        nonlocal sampled_bound_violation

        hessian = objective.hessian(position)
        eigenvalues = np.linalg.eigvalsh(hessian)
        eigenvalue_minimum = float(eigenvalues[0])
        eigenvalue_maximum = float(eigenvalues[-1])
        minimum_hessian = min(minimum_hessian, eigenvalue_minimum)
        maximum_hessian = max(maximum_hessian, eigenvalue_maximum)
        negative_curvature_sample_count += eigenvalue_minimum < -config.hessian_tolerance
        sampled_bound_violation |= (
            eigenvalue_minimum < ANALYTIC_NEGATIVE_CURVATURE - config.hessian_tolerance
            or eigenvalue_maximum > ANALYTIC_SMOOTHNESS_UPPER + config.hessian_tolerance
        )

        gap = objective.value(position)
        if gap > gap_resolution:
            ratio = objective.pl_ratio(position)
            minimum_pl_ratio = min(minimum_pl_ratio, ratio)
            pl_ratio_sample_count += 1
            sampled_bound_violation |= ratio < config.claimed_pl_lower - config.pl_tolerance

        if previous_hessian is not None:
            denominator = float(
                np.linalg.norm(hessian, ord="fro") * np.linalg.norm(previous_hessian, ord="fro")
            )
            normalized = (
                float(
                    np.linalg.norm(
                        hessian @ previous_hessian - previous_hessian @ hessian, ord="fro"
                    )
                )
                / denominator
                if denominator > 0
                else 0.0
            )
            maximum_commutator = max(maximum_commutator, normalized)
            orientation_change_count += normalized > config.orientation_tolerance
        previous_hessian = hessian
        hessian_sample_count += 1

    sample_objective_diagnostics()
    for iteration in range(1, config.iterations + 1):
        gradient = np.asarray(objective.gradient(position)).reshape(-1)
        momentum = config.beta * momentum + one_minus_beta * gradient
        signal = config.beta * momentum + one_minus_beta * gradient
        position = position - config.learning_rate * _repaired_floored_jordan(
            signal,
            shape=shape,
            floor=config.floor,
            repair_rho=config.repair_rho,
        )
        executed = iteration

        next_gap = objective.value(position)
        if current_gap > gap_resolution and math.isfinite(next_gap):
            gap_ratio = next_gap / current_gap
            maximum_gap_ratio = max(maximum_gap_ratio, gap_ratio)
            gap_increase_count += gap_ratio > 1.0 + 1e-12
            resolved_gap_transitions += 1
        current_gap = next_gap

        next_lyapunov = pl_lyapunov_value(objective, position, momentum, certificate)
        if current_lyapunov > lyapunov_resolution and math.isfinite(next_lyapunov):
            if next_lyapunov <= 0:
                nonpositive_lyapunov_count += 1
            else:
                lyapunov_ratio = next_lyapunov / current_lyapunov
                rate_excess = lyapunov_ratio - tau_squared
                maximum_lyapunov_ratio = max(maximum_lyapunov_ratio, lyapunov_ratio)
                maximum_lyapunov_rate_excess = max(
                    maximum_lyapunov_rate_excess,
                    rate_excess,
                )
                lyapunov_transition_count += 1
                lyapunov_rate_violation_count += rate_excess > config.storage_rate_tolerance
        current_lyapunov = next_lyapunov

        position_norm = float(np.linalg.norm(position))
        momentum_norm = float(np.linalg.norm(momentum))
        signal_norm = float(np.linalg.norm(signal))
        maximum_position_norm = max(maximum_position_norm, position_norm)
        maximum_momentum_norm = max(maximum_momentum_norm, momentum_norm)
        maximum_signal_norm = max(maximum_signal_norm, signal_norm)
        maximum_gap = max(maximum_gap, next_gap)
        finite_values = (position_norm, momentum_norm, signal_norm, next_gap, next_lyapunov)
        if not all(math.isfinite(value) for value in finite_values):
            nonfinite = True
            exceeded = True
            break
        if max(position_norm, momentum_norm, signal_norm) > config.divergence_norm:
            exceeded = True
            break
        if iteration % config.diagnostic_stride == 0 or iteration == config.iterations:
            sample_objective_diagnostics()

    if resolved_gap_transitions == 0:
        maximum_gap_ratio = math.nan
    if lyapunov_transition_count == 0:
        maximum_lyapunov_ratio = math.nan
        maximum_lyapunov_rate_excess = math.nan
    final_position_norm = float(np.linalg.norm(position))
    final_active_norm = float(np.linalg.norm(objective.active_component(position)))
    final_null = np.asarray(objective.null_component(position)).reshape(-1)
    final_null_norm = float(np.linalg.norm(final_null))
    final_momentum_norm = float(np.linalg.norm(momentum))
    final_signal_norm = float(np.linalg.norm(signal))
    final_gradient_norm = float(np.linalg.norm(objective.gradient(position)))
    final_gap = objective.value(position)
    final_lyapunov = pl_lyapunov_value(objective, position, momentum, certificate)
    final_gap_ratio = final_gap / initial_gap
    empirical_gap_factor = (
        final_gap_ratio ** (1.0 / executed) if executed > 0 and final_gap_ratio >= 0 else math.nan
    )
    momentum_peak = max(maximum_momentum_norm, np.finfo(np.float64).tiny)
    candidate = bool(
        nonfinite
        or exceeded
        or sampled_bound_violation
        or nonpositive_lyapunov_count > 0
        or lyapunov_rate_violation_count > 0
    )
    return PLProbeTrial(
        shape=shape,
        rank_mode=rank_mode,
        active_rank=objective.active_rank,
        minimizer_set_dimension=objective.minimizer_set_dimension,
        transition_scale=transition_scale,
        radius_multiplier=radius_multiplier,
        momentum_mode=momentum_mode,
        case_index=case_index,
        case_seed=seed,
        frame_variant=objective.frame_variant,
        projector_sha256=objective.projector_sha256,
        frame_orthogonality_residual=objective.frame_orthogonality_residual,
        projector_idempotence_residual=objective.projector_idempotence_residual,
        projector_symmetry_residual=objective.projector_symmetry_residual,
        requested_iterations=config.iterations,
        executed_iterations=executed,
        initial_position_norm=initial_position_norm,
        initial_distance_to_minimizer_set=initial_active_norm,
        initial_null_component_norm=initial_null_norm,
        initial_momentum_norm=initial_momentum_norm,
        initial_objective_gap=initial_gap,
        initial_lyapunov_value=initial_lyapunov,
        final_position_norm=final_position_norm,
        final_distance_to_minimizer_set=final_active_norm,
        final_null_component_norm=final_null_norm,
        null_component_drift_norm=float(np.linalg.norm(final_null - initial_null)),
        final_momentum_norm=final_momentum_norm,
        final_signal_norm=final_signal_norm,
        final_gradient_norm=final_gradient_norm,
        final_objective_gap=final_gap,
        final_lyapunov_value=final_lyapunov,
        final_lyapunov_ratio=final_lyapunov / initial_lyapunov,
        final_objective_gap_ratio=final_gap_ratio,
        final_momentum_to_peak_ratio=final_momentum_norm / momentum_peak,
        empirical_gap_geometric_factor=empirical_gap_factor,
        maximum_position_norm=maximum_position_norm,
        maximum_momentum_norm=maximum_momentum_norm,
        maximum_signal_norm=maximum_signal_norm,
        maximum_objective_gap=maximum_gap,
        maximum_resolved_one_step_gap_ratio=maximum_gap_ratio,
        objective_gap_increase_count=gap_increase_count,
        resolved_gap_transition_count=resolved_gap_transitions,
        maximum_lyapunov_ratio=maximum_lyapunov_ratio,
        maximum_normalized_lyapunov_rate_excess=maximum_lyapunov_rate_excess,
        lyapunov_transition_count=lyapunov_transition_count,
        lyapunov_rate_violation_count=lyapunov_rate_violation_count,
        nonpositive_lyapunov_count=nonpositive_lyapunov_count,
        sampled_hessian_eigenvalue_minimum=minimum_hessian,
        sampled_hessian_eigenvalue_maximum=maximum_hessian,
        minimum_sampled_pl_ratio=minimum_pl_ratio,
        hessian_sample_count=hessian_sample_count,
        pl_ratio_sample_count=pl_ratio_sample_count,
        negative_curvature_sample_count=negative_curvature_sample_count,
        maximum_normalized_hessian_commutator=maximum_commutator,
        orientation_change_count=orientation_change_count,
        nonfinite=nonfinite,
        exceeded_divergence_norm=exceeded,
        sampled_objective_bound_violation=sampled_bound_violation,
        candidate_implementation_violation=candidate,
    )


def run_pl_probe(config: PLProbeConfig | None = None) -> PLProbeSummary:
    """Run the complete deterministic nonconvex PL objective grid."""

    selected = PLProbeConfig() if config is None else config
    cases = itertools.product(
        selected.shapes,
        selected.rank_modes,
        selected.transition_scales,
        selected.radius_multipliers,
        selected.momentum_modes,
    )
    trials = tuple(
        run_pl_trial(
            config=selected,
            shape=shape,
            rank_mode=rank_mode,
            transition_scale=transition_scale,
            radius_multiplier=radius_multiplier,
            momentum_mode=momentum_mode,
            case_index=case_index,
        )
        for case_index, (
            shape,
            rank_mode,
            transition_scale,
            radius_multiplier,
            momentum_mode,
        ) in enumerate(cases)
    )

    def finite_values(attribute: str) -> list[float]:
        values = [float(getattr(trial, attribute)) for trial in trials]
        return [value for value in values if math.isfinite(value)]

    return PLProbeSummary(
        config=selected,
        trials=trials,
        worst_final_objective_gap_ratio=max(
            finite_values("final_objective_gap_ratio"), default=math.nan
        ),
        worst_empirical_gap_geometric_factor=max(
            finite_values("empirical_gap_geometric_factor"), default=math.nan
        ),
        maximum_resolved_one_step_gap_ratio=max(
            finite_values("maximum_resolved_one_step_gap_ratio"), default=math.nan
        ),
        worst_final_momentum_to_peak_ratio=max(
            finite_values("final_momentum_to_peak_ratio"), default=math.nan
        ),
        worst_final_lyapunov_ratio=max(finite_values("final_lyapunov_ratio"), default=math.nan),
        maximum_lyapunov_ratio=max(finite_values("maximum_lyapunov_ratio"), default=math.nan),
        maximum_normalized_lyapunov_rate_excess=max(
            finite_values("maximum_normalized_lyapunov_rate_excess"),
            default=math.nan,
        ),
        sampled_hessian_eigenvalue_minimum=min(
            finite_values("sampled_hessian_eigenvalue_minimum"), default=math.nan
        ),
        sampled_hessian_eigenvalue_maximum=max(
            finite_values("sampled_hessian_eigenvalue_maximum"), default=math.nan
        ),
        minimum_sampled_pl_ratio=min(finite_values("minimum_sampled_pl_ratio"), default=math.nan),
        maximum_orientation_commutator=max(
            finite_values("maximum_normalized_hessian_commutator"), default=math.nan
        ),
        rank_deficient_case_count=sum(trial.minimizer_set_dimension > 0 for trial in trials),
        negative_curvature_case_count=sum(
            trial.negative_curvature_sample_count > 0 for trial in trials
        ),
        orientation_changing_case_count=sum(trial.orientation_change_count > 0 for trial in trials),
        objective_increasing_case_count=sum(
            trial.objective_gap_increase_count > 0 for trial in trials
        ),
        lyapunov_rate_violating_case_count=sum(
            trial.lyapunov_rate_violation_count > 0 for trial in trials
        ),
        nonpositive_lyapunov_case_count=sum(
            trial.nonpositive_lyapunov_count > 0 for trial in trials
        ),
        objective_bound_violating_case_count=sum(
            trial.sampled_objective_bound_violation for trial in trials
        ),
        candidate_violation_count=sum(trial.candidate_implementation_violation for trial in trials),
        divergence_count=sum(trial.exceeded_divergence_norm for trial in trials),
        nonfinite_count=sum(trial.nonfinite for trial in trials),
    )


__all__ = [
    "ANALYTIC_NEGATIVE_CURVATURE",
    "ANALYTIC_PL_LOWER",
    "ANALYTIC_SMOOTHNESS_UPPER",
    "OBJECTIVE_SCALE",
    "WARP_STRENGTH",
    "PLProbeConfig",
    "PLProbeSummary",
    "PLProbeTrial",
    "WarpedRadialPLObjective",
    "pl_lyapunov_value",
    "run_pl_probe",
    "run_pl_trial",
]
