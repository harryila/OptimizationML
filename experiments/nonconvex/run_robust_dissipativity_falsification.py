#!/usr/bin/env python3
"""Falsify the P7 disturbed repaired-Muon dissipation inequality numerically.

This CPU/float64 experiment exercises the pinned EMA/Nesterov ordering on the
analytic nonconvex PL family introduced for P6.  It injects gradient noise and
post-orthogonalizer implementation error, then checks the *candidate* P7
one-step inequality using the unchanged P6 storage.  A sampled violation would
falsify the implementation or candidate certificate.  Sampled passes cannot
certify the full objective class or any matrix dimension.

The separate flat-direction construction makes an important scope boundary
executable: square-summable, non-absolutely-summable output errors can leave the
PL storage identically zero while the iterates drift harmonically along a
nonunique minimizer set.  Thus square-summable disturbances alone do not imply
iterate convergence without an additional argument or assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from passive_muon.operator import orthogonalize
from passive_muon.pl_convergence import (
    LOCKED_PL_CONVERGENCE_LEARNING_RATE,
    LOCKED_PL_CONVERGENCE_TAU,
)
from passive_muon.pl_experiment import WarpedRadialPLObjective, pl_lyapunov_value
from passive_muon.robust_dissipativity import (
    LOCKED_GRADIENT_NOISE_GAIN,
    LOCKED_IMPLEMENTATION_ERROR_GAIN,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_CENTER_GAIN,
    LOCKED_FLOOR,
    LOCKED_HESSIAN_UPPER,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P6_CHECKPOINT = "ef88d8f5b26148af0ec1ca70b506048938bf9bef"

# Candidate P7 physical-unit inequality:
# V_(t+1) <= q_bar*V_t + gamma_g*||xi_t||_F^2 + gamma_R*||e_t||_F^2.
ROBUST_RATE = Fraction(399_960_001, 400_000_000)
GRADIENT_NOISE_GAIN = LOCKED_GRADIENT_NOISE_GAIN
IMPLEMENTATION_ERROR_GAIN = LOCKED_IMPLEMENTATION_ERROR_GAIN

if ROBUST_RATE != LOCKED_PL_CONVERGENCE_TAU**2:
    raise AssertionError("the P7 candidate must retain the exact P6 rate")

SOURCE_PATHS = (
    "experiments/nonconvex/run_robust_dissipativity_falsification.py",
    "src/passive_muon/pl_experiment.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_robust_dissipativity_experiment.py",
    "theory/robust_dissipativity_certificate.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)


@dataclass(frozen=True)
class RobustProbeConfig:
    """Locked P7 candidate and a finite disturbed nonconvex-PL grid."""

    seed: int = 2_026_090_3
    shapes: tuple[tuple[int, int], ...] = ((2, 2), (2, 3))
    rank_modes: tuple[str, ...] = ("full_rank", "codimension_one")
    transition_scales: tuple[float, ...] = (1e-3, 1.0, 100.0)
    radius_multipliers: tuple[float, ...] = (math.sqrt(2.0), 4.0)
    momentum_modes: tuple[str, ...] = ("zero", "random")
    disturbance_profiles: tuple[str, ...] = (
        "deterministic_bounded",
        "seeded_stochastic",
        "implementation_only",
    )
    iterations: int = 120
    gradient_noise_relative_scale: float = 2e-4
    implementation_error_relative_scale: float = 1e-4
    learning_rate: float = float(LOCKED_PL_CONVERGENCE_LEARNING_RATE)
    beta: float = float(LOCKED_BETA)
    floor: float = float(LOCKED_FLOOR)
    repair_rho: float = float(LOCKED_REPAIR_RHO)
    smoothness: float = float(LOCKED_HESSIAN_UPPER)
    center_gain: float = float(LOCKED_CENTER_GAIN)
    robust_rate: float = float(ROBUST_RATE)
    gradient_noise_gain: float = float(GRADIENT_NOISE_GAIN)
    implementation_error_gain: float = float(IMPLEMENTATION_ERROR_GAIN)
    inequality_absolute_tolerance: float = 3e-10
    inequality_relative_tolerance: float = 2e-11
    divergence_norm: float = 1e100
    harmonic_horizon: int = 10_000

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
        supported_profiles = {
            "deterministic_bounded",
            "seeded_stochastic",
            "implementation_only",
        }
        if not self.disturbance_profiles or any(
            profile not in supported_profiles for profile in self.disturbance_profiles
        ):
            raise ValueError("disturbance_profiles contains an unsupported profile")
        if self.iterations <= 0 or self.harmonic_horizon <= 1:
            raise ValueError("iterations must be positive and harmonic_horizon must exceed one")
        if min(self.gradient_noise_relative_scale, self.implementation_error_relative_scale) < 0:
            raise ValueError("disturbance scales must be nonnegative")
        if min(self.inequality_absolute_tolerance, self.inequality_relative_tolerance) < 0:
            raise ValueError("inequality tolerances must be nonnegative")
        if self.divergence_norm <= 0:
            raise ValueError("divergence_norm must be positive")
        locked = {
            "learning_rate": float(LOCKED_PL_CONVERGENCE_LEARNING_RATE),
            "beta": float(LOCKED_BETA),
            "floor": float(LOCKED_FLOOR),
            "repair_rho": float(LOCKED_REPAIR_RHO),
            "smoothness": float(LOCKED_HESSIAN_UPPER),
            "center_gain": float(LOCKED_CENTER_GAIN),
            "robust_rate": float(ROBUST_RATE),
            "gradient_noise_gain": float(GRADIENT_NOISE_GAIN),
            "implementation_error_gain": float(IMPLEMENTATION_ERROR_GAIN),
        }
        for name, expected in locked.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must equal the locked P7 diagnostic value")

    @property
    def case_count(self) -> int:
        return (
            len(self.shapes)
            * len(self.rank_modes)
            * len(self.transition_scales)
            * len(self.radius_multipliers)
            * len(self.momentum_modes)
            * len(self.disturbance_profiles)
        )


@dataclass(frozen=True)
class RobustTrial:
    """One sampled disturbed repaired EMA/Nesterov trajectory."""

    shape: tuple[int, int]
    rank_mode: str
    transition_scale: float
    radius_multiplier: float
    momentum_mode: str
    disturbance_profile: str
    case_index: int
    case_seed: int
    active_rank: int
    minimizer_set_dimension: int
    frame_variant: int
    requested_iterations: int
    executed_iterations: int
    initial_lyapunov: float
    final_lyapunov: float
    initial_objective_gap: float
    final_objective_gap: float
    final_momentum_norm: float
    maximum_gradient_noise_norm: float
    maximum_implementation_error_norm: float
    total_gradient_noise_energy: float
    total_implementation_error_energy: float
    maximum_dissipation_excess: float
    maximum_normalized_dissipation_excess: float
    minimum_dissipation_margin: float
    inequality_violation_count: int
    nonfinite: bool
    exceeded_divergence_norm: bool
    candidate_implementation_violation: bool


@dataclass(frozen=True)
class HarmonicDriftDiagnostic:
    """Finite prefix of an exact l2-but-not-l1 flat-direction construction."""

    horizon: int
    learning_rate: float
    error_formula: str
    error_series_classification: str
    finite_prefix_error_energy: float
    infinite_error_energy_upper_bound: float
    finite_prefix_absolute_error_mass: float
    initial_position: tuple[float, ...]
    final_position: tuple[float, ...]
    final_iterate_displacement: float
    expected_harmonic_displacement: float
    relative_displacement_error: float
    maximum_objective_gap: float
    maximum_lyapunov_value: float
    zero_signal_operator_residual: float
    interpretation: str


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY_ROOT,
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


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(path) for path in SOURCE_PATHS}


def _case_seed(master_seed: int, case_index: int, dimension: int) -> int:
    sequence = np.random.SeedSequence([master_seed, case_index, dimension])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _unit_random(dimension: int, rng: np.random.Generator) -> np.ndarray:
    for _attempt in range(8):
        vector = rng.standard_normal(dimension)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return vector / norm
    raise RuntimeError("failed to sample a resolved random direction")


def _projected_unit(
    projector: np.ndarray,
    *,
    active: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    dimension = projector.shape[0]
    selected = projector if active else np.eye(dimension, dtype=np.float64) - projector
    for _attempt in range(8):
        vector = selected @ rng.standard_normal(dimension)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-12:
            return vector / norm
    raise RuntimeError("failed to sample a resolved projected direction")


def _repaired_floored_jordan(
    signal: np.ndarray,
    *,
    shape: tuple[int, int],
    config: RobustProbeConfig,
) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(signal.reshape(shape)))
    with torch.no_grad():
        update = orthogonalize(
            tensor,
            polynomial="jordan",
            normalization="floored_frobenius",
            repair_rho=config.repair_rho,
            steps=5,
            floor=config.floor,
        )
    return update.numpy().reshape(-1)


def _disturbances(
    *,
    profile: str,
    iteration: int,
    dimension: int,
    gradient_scale: float,
    implementation_scale: float,
    deterministic_directions: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first, second, third, fourth = deterministic_directions
    if profile == "deterministic_bounded":
        angle = 0.37 * (iteration + 1)
        gradient_noise = (
            gradient_scale * (math.sin(angle) * first + math.cos(angle) * second) / math.sqrt(2.0)
        )
        implementation_error = (
            implementation_scale
            * (math.sin(0.71 * angle) * third + math.cos(0.71 * angle) * fourth)
            / math.sqrt(2.0)
        )
    elif profile == "seeded_stochastic":
        gradient_noise = gradient_scale * rng.standard_normal(dimension) / math.sqrt(dimension)
        implementation_error = (
            implementation_scale * rng.standard_normal(dimension) / math.sqrt(dimension)
        )
    elif profile == "implementation_only":
        gradient_noise = np.zeros(dimension, dtype=np.float64)
        implementation_error = (
            implementation_scale
            * (math.sin(0.53 * (iteration + 1)) * third + math.cos(0.53 * (iteration + 1)) * fourth)
            / math.sqrt(2.0)
        )
    else:
        raise ValueError(f"unsupported disturbance profile: {profile}")
    return gradient_noise, implementation_error


def run_robust_trial(
    *,
    config: RobustProbeConfig,
    shape: tuple[int, int],
    rank_mode: str,
    transition_scale: float,
    radius_multiplier: float,
    momentum_mode: str,
    disturbance_profile: str,
    case_index: int,
) -> RobustTrial:
    """Run one sampled disturbed pinned EMA/Nesterov trajectory."""

    dimension = math.prod(shape)
    seed = _case_seed(config.seed, case_index, dimension)
    rng = np.random.default_rng(seed)
    active_rank = dimension if rank_mode == "full_rank" else dimension - 1
    pair_count = max(1, dimension * (dimension - 1) // 2)
    objective = WarpedRadialPLObjective(
        dimension=dimension,
        transition_scale=transition_scale,
        active_rank=active_rank,
        frame_variant=seed % pair_count,
    )
    position = (
        transition_scale
        * radius_multiplier
        * _projected_unit(objective.projector, active=True, rng=rng)
    )
    if objective.is_rank_deficient:
        position += transition_scale * _projected_unit(
            objective.projector,
            active=False,
            rng=rng,
        )
    momentum = (
        np.zeros(dimension, dtype=np.float64)
        if momentum_mode == "zero"
        else config.smoothness
        * max(transition_scale, transition_scale * radius_multiplier)
        * _unit_random(dimension, rng)
    )
    deterministic_directions = tuple(_unit_random(dimension, rng) for _index in range(4))
    reference = max(transition_scale, transition_scale * radius_multiplier)
    gradient_scale = config.gradient_noise_relative_scale * config.smoothness * reference
    implementation_scale = (
        config.implementation_error_relative_scale * config.center_gain * reference
    )

    initial_lyapunov = pl_lyapunov_value(objective, position, momentum)
    initial_gap = objective.value(position)
    current_lyapunov = initial_lyapunov
    maximum_gradient_noise_norm = 0.0
    maximum_implementation_error_norm = 0.0
    total_gradient_noise_energy = 0.0
    total_implementation_error_energy = 0.0
    maximum_excess = -math.inf
    maximum_normalized_excess = -math.inf
    minimum_margin = math.inf
    violation_count = 0
    nonfinite = False
    exceeded = False
    executed = 0
    one_minus_beta = 1.0 - config.beta

    for iteration in range(config.iterations):
        gradient_noise, implementation_error = _disturbances(
            profile=disturbance_profile,
            iteration=iteration,
            dimension=dimension,
            gradient_scale=gradient_scale,
            implementation_scale=implementation_scale,
            deterministic_directions=deterministic_directions,
            rng=rng,
        )
        true_gradient = np.asarray(objective.gradient(position)).reshape(-1)
        disturbed_gradient = true_gradient + gradient_noise
        next_momentum = config.beta * momentum + one_minus_beta * disturbed_gradient
        signal = config.beta * next_momentum + one_minus_beta * disturbed_gradient
        repaired_output = _repaired_floored_jordan(signal, shape=shape, config=config)
        next_position = position - config.learning_rate * (repaired_output + implementation_error)
        next_lyapunov = pl_lyapunov_value(objective, next_position, next_momentum)

        gradient_energy = float(gradient_noise @ gradient_noise)
        implementation_energy = float(implementation_error @ implementation_error)
        right_hand_side = (
            config.robust_rate * current_lyapunov
            + config.gradient_noise_gain * gradient_energy
            + config.implementation_error_gain * implementation_energy
        )
        excess = next_lyapunov - right_hand_side
        accounting_scale = max(
            1.0,
            abs(next_lyapunov),
            abs(config.robust_rate * current_lyapunov),
            config.gradient_noise_gain * gradient_energy,
            config.implementation_error_gain * implementation_energy,
        )
        tolerance = (
            config.inequality_absolute_tolerance
            + config.inequality_relative_tolerance * accounting_scale
        )
        normalized_excess = excess / accounting_scale

        maximum_gradient_noise_norm = max(
            maximum_gradient_noise_norm,
            math.sqrt(gradient_energy),
        )
        maximum_implementation_error_norm = max(
            maximum_implementation_error_norm,
            math.sqrt(implementation_energy),
        )
        total_gradient_noise_energy += gradient_energy
        total_implementation_error_energy += implementation_energy
        maximum_excess = max(maximum_excess, excess)
        maximum_normalized_excess = max(maximum_normalized_excess, normalized_excess)
        minimum_margin = min(minimum_margin, right_hand_side - next_lyapunov)
        violation_count += excess > tolerance
        executed = iteration + 1
        position = next_position
        momentum = next_momentum
        current_lyapunov = next_lyapunov

        values = (
            current_lyapunov,
            float(np.linalg.norm(position)),
            float(np.linalg.norm(momentum)),
            maximum_excess,
        )
        if not all(math.isfinite(value) for value in values):
            nonfinite = True
            exceeded = True
            break
        if max(values[1], values[2]) > config.divergence_norm:
            exceeded = True
            break

    candidate = nonfinite or exceeded or violation_count > 0
    return RobustTrial(
        shape=shape,
        rank_mode=rank_mode,
        transition_scale=transition_scale,
        radius_multiplier=radius_multiplier,
        momentum_mode=momentum_mode,
        disturbance_profile=disturbance_profile,
        case_index=case_index,
        case_seed=seed,
        active_rank=objective.active_rank,
        minimizer_set_dimension=objective.minimizer_set_dimension,
        frame_variant=objective.frame_variant,
        requested_iterations=config.iterations,
        executed_iterations=executed,
        initial_lyapunov=initial_lyapunov,
        final_lyapunov=current_lyapunov,
        initial_objective_gap=initial_gap,
        final_objective_gap=objective.value(position),
        final_momentum_norm=float(np.linalg.norm(momentum)),
        maximum_gradient_noise_norm=maximum_gradient_noise_norm,
        maximum_implementation_error_norm=maximum_implementation_error_norm,
        total_gradient_noise_energy=total_gradient_noise_energy,
        total_implementation_error_energy=total_implementation_error_energy,
        maximum_dissipation_excess=maximum_excess,
        maximum_normalized_dissipation_excess=maximum_normalized_excess,
        minimum_dissipation_margin=minimum_margin,
        inequality_violation_count=violation_count,
        nonfinite=nonfinite,
        exceeded_divergence_norm=exceeded,
        candidate_implementation_violation=candidate,
    )


def run_harmonic_drift(config: RobustProbeConfig) -> HarmonicDriftDiagnostic:
    """Replay a flat-minimizer l2-not-l1 output-error counterexample.

    Use ``f(x)=x_1^2/2`` on a 2-by-2 matrix, start at a minimizer, and set
    ``e_t=v/(t+1)`` for a unit vector ``v`` in a flat coordinate.  The gradient,
    momentum, signal, repaired output, objective gap, and P6 storage remain
    zero, but ``W_t=-eta*H_t*v`` has no finite limit.
    """

    shape = (2, 2)
    position = np.zeros(4, dtype=np.float64)
    flat_direction = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    zero_signal = np.zeros(4, dtype=np.float64)
    zero_output = _repaired_floored_jordan(zero_signal, shape=shape, config=config)
    zero_residual = float(np.linalg.norm(zero_output))
    absolute_mass = 0.0
    square_energy = 0.0
    maximum_gap = 0.0
    maximum_storage = 0.0

    for index in range(1, config.harmonic_horizon + 1):
        implementation_error = flat_direction / index
        position -= config.learning_rate * (zero_output + implementation_error)
        absolute_mass += 1.0 / index
        square_energy += 1.0 / index**2
        objective_gap = 0.5 * position[0] ** 2
        storage = 0.0  # momentum=gradient=gap=0 in the unchanged P6 storage.
        maximum_gap = max(maximum_gap, objective_gap)
        maximum_storage = max(maximum_storage, storage)

    displacement = float(np.linalg.norm(position))
    expected = config.learning_rate * absolute_mass
    relative_error = abs(displacement - expected) / max(expected, np.finfo(np.float64).tiny)
    return HarmonicDriftDiagnostic(
        horizon=config.harmonic_horizon,
        learning_rate=config.learning_rate,
        error_formula="e_t=v/(t+1), ||v||_F=1, v in a flat minimizer direction",
        error_series_classification=(
            "sum_t ||e_t||_F^2=pi^2/6 is finite; sum_t ||e_t||_F is the divergent harmonic series"
        ),
        finite_prefix_error_energy=square_energy,
        infinite_error_energy_upper_bound=math.pi**2 / 6.0,
        finite_prefix_absolute_error_mass=absolute_mass,
        initial_position=(0.0, 0.0, 0.0, 0.0),
        final_position=tuple(float(value) for value in position),
        final_iterate_displacement=displacement,
        expected_harmonic_displacement=expected,
        relative_displacement_error=relative_error,
        maximum_objective_gap=maximum_gap,
        maximum_lyapunov_value=maximum_storage,
        zero_signal_operator_residual=zero_residual,
        interpretation=(
            "square-summable output errors can preserve V_t=0 while W_t drifts along a "
            "flat minimizer set; l2 disturbances alone do not imply convergence of W_t, "
            "although they may support storage/output convergence"
        ),
    )


def run_probe(config: RobustProbeConfig | None = None) -> dict[str, Any]:
    """Run the finite disturbed grid and return a provenance-complete payload."""

    selected = RobustProbeConfig() if config is None else config
    cases = (
        (shape, rank_mode, scale, radius, momentum_mode, profile)
        for shape in selected.shapes
        for rank_mode in selected.rank_modes
        for scale in selected.transition_scales
        for radius in selected.radius_multipliers
        for momentum_mode in selected.momentum_modes
        for profile in selected.disturbance_profiles
    )
    trials = tuple(
        run_robust_trial(
            config=selected,
            shape=shape,
            rank_mode=rank_mode,
            transition_scale=scale,
            radius_multiplier=radius,
            momentum_mode=momentum_mode,
            disturbance_profile=profile,
            case_index=index,
        )
        for index, (shape, rank_mode, scale, radius, momentum_mode, profile) in enumerate(cases)
    )
    harmonic = run_harmonic_drift(selected)
    coefficients = JORDAN_QUINTIC.fractions()
    maximum_excess = max(trial.maximum_dissipation_excess for trial in trials)
    maximum_normalized_excess = max(trial.maximum_normalized_dissipation_excess for trial in trials)
    candidate_count = sum(trial.candidate_implementation_violation for trial in trials)
    return {
        "schema_version": "passive-muon-robust-dissipativity-falsification-v1",
        "evidence_kind": "sampled_disturbed_nonconvex_pl_trajectory_falsification",
        "claim_scope": {
            "role": (
                "CPU/float64 falsification diagnostic for a separate exact rational P7 "
                "certificate candidate"
            ),
            "qualification": (
                "a sampled violation can falsify the implementation or candidate inequality; "
                "sampled passes do not prove global dissipativity, ISS, or stochastic convergence"
            ),
            "harmonic_boundary": (
                "the exact flat-direction construction proves that square-summable errors "
                "alone do not imply convergence of W_t when minimizers are nonunique"
            ),
            "not_claimed": [
                "a certificate inferred from sampled trajectories",
                "almost-sure convergence under persistent bounded-variance noise",
                "iterate convergence under merely square-summable disturbances",
                "BF16 or neural-network convergence",
            ],
        },
        "candidate_inequality": {
            "formula": "V_(t+1)<=q_bar*V_t+gamma_g*||xi_t||_F^2+gamma_R*||e_t||_F^2",
            "q_bar_exact": str(ROBUST_RATE),
            "gamma_g_exact": str(GRADIENT_NOISE_GAIN),
            "gamma_R_exact": str(IMPLEMENTATION_ERROR_GAIN),
            "storage": "unchanged exact P6 [m/L,grad/L] plus function-gap storage",
            "disturbance_units": {
                "xi_t": "physical gradient noise, added before both EMA/Nesterov uses",
                "e_t": "physical output error, added after the repaired orthogonalizer",
            },
        },
        "disturbance_generation": {
            "reference_scale": "max(transition_scale, transition_scale*radius_multiplier)",
            "gradient_noise_scale": ("gradient_noise_relative_scale * L * reference_scale"),
            "implementation_error_scale": (
                "implementation_error_relative_scale * center_gain * reference_scale"
            ),
            "deterministic_bounded": (
                "fixed seeded unit directions with sin/cos phases; both xi_t and e_t nonzero"
            ),
            "seeded_stochastic": (
                "independent standard-normal coordinate draws divided by sqrt(dimension), "
                "using the recorded per-case numpy Generator seed"
            ),
            "implementation_only": (
                "xi_t=0 and a fixed seeded sin/cos implementation-error sequence"
            ),
        },
        "update_order": [
            "g_t=grad_f(W_t)",
            "g_hat_t=g_t+xi_t",
            "m_(t+1)=beta*m_t+(1-beta)*g_hat_t",
            "s_(t+1)=beta*m_(t+1)+(1-beta)*g_hat_t",
            "W_(t+1)=W_t-eta*(R(s_(t+1))+e_t)",
        ],
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
        "objective_family": {
            "name": "warped_radial_projected_pl",
            "formula": "f(x)=(3/2)*s^2*(q+5*q/(1+q)), q=||P*x||^2/(2*s^2)",
            "exact_real_properties": {
                "global_PL_lower": "1",
                "global_gradient_Lipschitz_upper": "9 (therefore also 10-smooth)",
                "hessian_spectral_interval": "[-3/8, 9]",
                "permits": "negative curvature, rotating Hessians, and nonunique minimizers",
            },
        },
        "config": asdict(selected),
        "summary": {
            "trial_count": len(trials),
            "candidate_violation_count": candidate_count,
            "inequality_violation_count": sum(trial.inequality_violation_count for trial in trials),
            "nonfinite_count": sum(trial.nonfinite for trial in trials),
            "divergence_count": sum(trial.exceeded_divergence_norm for trial in trials),
            "maximum_dissipation_excess": maximum_excess,
            "maximum_normalized_dissipation_excess": maximum_normalized_excess,
            "minimum_dissipation_margin": min(trial.minimum_dissipation_margin for trial in trials),
            "rank_deficient_case_count": sum(trial.minimizer_set_dimension > 0 for trial in trials),
            "deterministic_bounded_case_count": sum(
                trial.disturbance_profile == "deterministic_bounded" for trial in trials
            ),
            "seeded_stochastic_case_count": sum(
                trial.disturbance_profile == "seeded_stochastic" for trial in trials
            ),
            "implementation_only_case_count": sum(
                trial.disturbance_profile == "implementation_only" for trial in trials
            ),
        },
        "trials": [asdict(trial) for trial in trials],
        "harmonic_drift_counterexample": asdict(harmonic),
        "p6_reference": {
            "checkpoint": P6_CHECKPOINT,
            "checkpoint_tag": "p6-checkpoint",
            "storage_source": "src/passive_muon/pl_convergence.py",
            "scope_boundary": (
                "P6 is deterministic; P7 adds exogenous gradient and implementation errors"
            ),
        },
        "upstream_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "muon_py_sha256": PINNED_MUON_PY_SHA256,
        },
        "experiment_provenance": {
            "seed": selected.seed,
            "determinism": (
                "CPU float64; deterministic profiles plus pseudorandom streams fixed by "
                "SeedSequence(seed,case_index,dimension)"
            ),
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
            "source_snapshot": _source_snapshot(),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    parser.add_argument("--seed", type=int, default=RobustProbeConfig.seed)
    parser.add_argument("--iterations", type=int, help="override iterations per trajectory")
    parser.add_argument("--harmonic-horizon", type=int, help="override flat-drift prefix length")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one rank-deficient 2x2 state with all disturbance profiles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RobustProbeConfig(seed=args.seed)
    if args.quick:
        config = replace(
            config,
            shapes=((2, 2),),
            rank_modes=("codimension_one",),
            transition_scales=(1.0,),
            radius_multipliers=(math.sqrt(2.0),),
            momentum_modes=("random",),
            iterations=12,
            harmonic_horizon=128,
        )
    if args.iterations is not None:
        config = replace(config, iterations=args.iterations)
    if args.harmonic_horizon is not None:
        config = replace(config, harmonic_horizon=args.harmonic_horizon)
    payload = run_probe(config)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
