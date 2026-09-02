"""Deterministic sampling probes for the structure-aware stability branch.

These experiments use the actual repository five-step Jordan map, the
Frobenius floor, a constant certified repair, and the pinned EMA/Nesterov
ordering.  They are diagnostics only: sampled trajectories cannot certify a
global stability statement.
"""

from __future__ import annotations

import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from passive_muon.floored_certificate import certified_dimension_uniform_deficit
from passive_muon.operator import orthogonalize
from passive_muon.specs import JORDAN_QUINTIC

LOCKED_P4_LEARNING_RATE = Fraction(1, 32_000)
LOCKED_BETA = Fraction(19, 20)
LOCKED_FLOOR = Fraction(1)
LOCKED_REPAIR_MARGIN = Fraction(648)
LOCKED_REPAIR_RHO = certified_dimension_uniform_deficit() + LOCKED_REPAIR_MARGIN


@dataclass(frozen=True)
class StructureAwareProbeConfig:
    """Complete deterministic configuration for a trajectory search."""

    seed: int = 2_026_090_2
    shapes: tuple[tuple[int, int], ...] = ((2, 2), (3, 3))
    orientations: tuple[str, ...] = ("aligned", "random")
    trials_per_case: int = 12
    iterations: int = 1_500
    learning_rate: float = float(LOCKED_P4_LEARNING_RATE)
    beta: float = float(LOCKED_BETA)
    floor: float = float(LOCKED_FLOOR)
    repair_rho: float = float(LOCKED_REPAIR_RHO)
    hessian_lower: float = 1.0
    hessian_upper: float = 10.0
    initial_scales: tuple[float, ...] = (1e-8, 1e-4, 1.0, 100.0)
    divergence_norm: float = 1e100

    def __post_init__(self) -> None:
        if not self.shapes or any(rows <= 0 or columns <= 0 for rows, columns in self.shapes):
            raise ValueError("shapes must contain positive matrix dimensions")
        if not self.orientations or any(
            orientation not in {"aligned", "random"} for orientation in self.orientations
        ):
            raise ValueError("orientations must contain only 'aligned' or 'random'")
        if self.trials_per_case <= 0 or self.iterations <= 0:
            raise ValueError("trial and iteration counts must be positive")
        if self.learning_rate <= 0 or not 0 <= self.beta < 1:
            raise ValueError("learning rate must be positive and beta must lie in [0, 1)")
        if self.floor <= 0 or self.repair_rho < 0:
            raise ValueError("floor must be positive and repair must be nonnegative")
        if self.hessian_lower <= 0 or self.hessian_upper < self.hessian_lower:
            raise ValueError("Hessian bounds must satisfy 0 < lower <= upper")
        if not self.initial_scales or any(scale <= 0 for scale in self.initial_scales):
            raise ValueError("initial scales must be positive")
        if self.divergence_norm <= 0:
            raise ValueError("divergence norm must be positive")


@dataclass(frozen=True)
class StructureAwareTrial:
    """Result of one deterministic full-matrix quadratic trajectory."""

    shape: tuple[int, int]
    orientation: str
    trial_index: int
    initial_scale: float
    initial_momentum_is_zero: bool
    hessian_eigenvalues: tuple[float, ...]
    requested_iterations: int
    executed_iterations: int
    initial_position_norm: float
    final_position_norm: float
    final_momentum_norm: float
    final_signal_norm: float
    maximum_position_norm: float
    maximum_momentum_norm: float
    maximum_signal_norm: float
    maximum_position_amplification: float
    final_position_ratio: float
    exceeded_divergence_norm: bool


@dataclass(frozen=True)
class StructureAwareProbeSummary:
    """Aggregate and per-trial results for the deterministic search."""

    config: StructureAwareProbeConfig
    trials: tuple[StructureAwareTrial, ...]
    worst_position_amplification: float
    worst_final_position_ratio: float
    divergence_count: int
    nonfinite_count: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable record including claim qualifications."""

        return {
            "schema_version": "passive-muon-structure-aware-probe-v1",
            "evidence_kind": "deterministic_sampled_full_matrix_trajectories",
            "claim_scope": (
                "sampling diagnostic only; it is neither a global stability certificate "
                "nor a proof over arbitrary SPD Hessians"
            ),
            "operator": {
                "polynomial": JORDAN_QUINTIC.name,
                "coefficients": {
                    "a": JORDAN_QUINTIC.a,
                    "b": JORDAN_QUINTIC.b,
                    "c": JORDAN_QUINTIC.c,
                },
                "iterations": 5,
                "normalization": "M/max(floor, ||M||_F)",
                "epsilon": 0.0,
                "floor": self.config.floor,
                "repair_rho": self.config.repair_rho,
                "repair_rho_exact": str(LOCKED_REPAIR_RHO),
            },
            "update_order": {
                "momentum": "m_next=beta*m+(1-beta)*H*W",
                "signal": "s_next=beta*m_next+(1-beta)*H*W",
                "parameter": "W_next=W-eta*R(s_next)",
            },
            "locked_exact_parameters": {
                "learning_rate": str(LOCKED_P4_LEARNING_RATE),
                "beta": str(LOCKED_BETA),
                "floor": str(LOCKED_FLOOR),
                "repair_margin": str(LOCKED_REPAIR_MARGIN),
            },
            "experiment_provenance": {
                "seed": self.config.seed,
                "git_sha": _git_sha(),
                "hardware": {
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "platform": platform.platform(),
                    "torch_device": "cpu",
                },
                "software": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
                "dtype": "float64",
            },
            "config": asdict(self.config),
            "summary": {
                "trial_count": len(self.trials),
                "worst_position_amplification": self.worst_position_amplification,
                "worst_final_position_ratio": self.worst_final_position_ratio,
                "divergence_count": self.divergence_count,
                "nonfinite_count": self.nonfinite_count,
            },
            "trials": [asdict(trial) for trial in self.trials],
        }


def _git_sha() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _givens_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Build a seeded orthogonal matrix from deterministic plane rotations."""

    basis = np.eye(dimension, dtype=np.float64)
    for left in range(dimension - 1):
        for right in range(left + 1, dimension):
            angle = float(rng.uniform(-math.pi, math.pi))
            cosine, sine = math.cos(angle), math.sin(angle)
            left_column = basis[:, left].copy()
            right_column = basis[:, right].copy()
            basis[:, left] = cosine * left_column + sine * right_column
            basis[:, right] = -sine * left_column + cosine * right_column
    return basis


def sampled_spd_hessian(
    *,
    dimension: int,
    orientation: str,
    lower: float,
    upper: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one seeded SPD Hessian with exact requested endpoint eigenvalues."""

    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if orientation not in {"aligned", "random"}:
        raise ValueError("orientation must be 'aligned' or 'random'")
    if lower <= 0 or upper < lower:
        raise ValueError("bounds must satisfy 0 < lower <= upper")

    eigenvalues = np.geomspace(lower, upper, dimension, dtype=np.float64)
    if orientation == "aligned":
        eigenvalues = eigenvalues[rng.permutation(dimension)]
        return np.diag(eigenvalues)
    basis = _givens_orthogonal(dimension, rng)
    return (basis * eigenvalues) @ basis.T


def _normalized_random_matrix(
    *, shape: tuple[int, int], scale: float, rng: np.random.Generator
) -> np.ndarray:
    value = rng.standard_normal(shape)
    norm = float(np.linalg.norm(value))
    if norm == 0:
        value.flat[0] = 1.0
        norm = 1.0
    return value * (scale / norm)


def _repaired_floored_jordan(signal: np.ndarray, *, floor: float, repair_rho: float) -> np.ndarray:
    tensor = torch.from_numpy(signal)
    with torch.no_grad():
        update = orthogonalize(
            tensor,
            polynomial="jordan",
            normalization="floored_frobenius",
            repair_rho=repair_rho,
            steps=5,
            floor=floor,
        )
    return update.numpy()


def run_structure_aware_trial(
    *,
    config: StructureAwareProbeConfig,
    shape: tuple[int, int],
    orientation: str,
    trial_index: int,
    rng: np.random.Generator,
) -> StructureAwareTrial:
    """Run one sampled SPD quadratic with the exact repository operator."""

    dimension = shape[0] * shape[1]
    hessian = sampled_spd_hessian(
        dimension=dimension,
        orientation=orientation,
        lower=config.hessian_lower,
        upper=config.hessian_upper,
        rng=rng,
    )
    initial_scale = config.initial_scales[trial_index % len(config.initial_scales)]
    position = _normalized_random_matrix(shape=shape, scale=initial_scale, rng=rng)
    momentum_is_zero = trial_index % 2 == 0
    momentum = (
        np.zeros(shape, dtype=np.float64)
        if momentum_is_zero
        else _normalized_random_matrix(shape=shape, scale=initial_scale, rng=rng)
    )

    initial_position_norm = float(np.linalg.norm(position))
    maximum_position = initial_position_norm
    maximum_momentum = float(np.linalg.norm(momentum))
    maximum_signal = 0.0
    signal = np.zeros(shape, dtype=np.float64)
    exceeded = False
    executed = 0
    one_minus_beta = 1.0 - config.beta

    for iteration in range(1, config.iterations + 1):
        gradient = (hessian @ position.reshape(-1)).reshape(shape)
        momentum = config.beta * momentum + one_minus_beta * gradient
        signal = config.beta * momentum + one_minus_beta * gradient
        update = _repaired_floored_jordan(
            signal,
            floor=config.floor,
            repair_rho=config.repair_rho,
        )
        position = position - config.learning_rate * update
        executed = iteration

        position_norm = float(np.linalg.norm(position))
        momentum_norm = float(np.linalg.norm(momentum))
        signal_norm = float(np.linalg.norm(signal))
        maximum_position = max(maximum_position, position_norm)
        maximum_momentum = max(maximum_momentum, momentum_norm)
        maximum_signal = max(maximum_signal, signal_norm)
        if (
            not math.isfinite(position_norm)
            or not math.isfinite(momentum_norm)
            or not math.isfinite(signal_norm)
            or position_norm > config.divergence_norm
        ):
            exceeded = True
            break

    final_position = float(np.linalg.norm(position))
    final_momentum = float(np.linalg.norm(momentum))
    final_signal = float(np.linalg.norm(signal))
    return StructureAwareTrial(
        shape=shape,
        orientation=orientation,
        trial_index=trial_index,
        initial_scale=initial_scale,
        initial_momentum_is_zero=momentum_is_zero,
        hessian_eigenvalues=tuple(float(value) for value in np.linalg.eigvalsh(hessian)),
        requested_iterations=config.iterations,
        executed_iterations=executed,
        initial_position_norm=initial_position_norm,
        final_position_norm=final_position,
        final_momentum_norm=final_momentum,
        final_signal_norm=final_signal,
        maximum_position_norm=maximum_position,
        maximum_momentum_norm=maximum_momentum,
        maximum_signal_norm=maximum_signal,
        maximum_position_amplification=maximum_position / initial_position_norm,
        final_position_ratio=final_position / initial_position_norm,
        exceeded_divergence_norm=exceeded,
    )


def run_structure_aware_probe(
    config: StructureAwareProbeConfig | None = None,
) -> StructureAwareProbeSummary:
    """Run every seeded shape/orientation case and summarize worst outcomes."""

    selected = StructureAwareProbeConfig() if config is None else config
    rng = np.random.default_rng(selected.seed)
    trials = tuple(
        run_structure_aware_trial(
            config=selected,
            shape=shape,
            orientation=orientation,
            trial_index=trial_index,
            rng=rng,
        )
        for shape in selected.shapes
        for orientation in selected.orientations
        for trial_index in range(selected.trials_per_case)
    )
    nonfinite_count = sum(
        not all(
            math.isfinite(value)
            for value in (
                trial.final_position_norm,
                trial.final_momentum_norm,
                trial.final_signal_norm,
            )
        )
        for trial in trials
    )
    return StructureAwareProbeSummary(
        config=selected,
        trials=trials,
        worst_position_amplification=max(trial.maximum_position_amplification for trial in trials),
        worst_final_position_ratio=max(trial.final_position_ratio for trial in trials),
        divergence_count=sum(trial.exceeded_divergence_norm for trial in trials),
        nonfinite_count=nonfinite_count,
    )


__all__ = [
    "LOCKED_BETA",
    "LOCKED_FLOOR",
    "LOCKED_P4_LEARNING_RATE",
    "LOCKED_REPAIR_MARGIN",
    "LOCKED_REPAIR_RHO",
    "StructureAwareProbeConfig",
    "StructureAwareProbeSummary",
    "StructureAwareTrial",
    "run_structure_aware_probe",
    "run_structure_aware_trial",
    "sampled_spd_hessian",
]
