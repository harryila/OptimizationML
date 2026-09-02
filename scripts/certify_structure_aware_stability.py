#!/usr/bin/env python3
"""Replay the p4 structure-aware EMA/Nesterov stability certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.optimize import brentq, minimize_scalar

from passive_muon.ema_nesterov_experiment import run_rank_one_ema_nesterov_trajectory
from passive_muon.ema_nesterov_iqc import (
    ema_nesterov_local_learning_rate_threshold,
    locked_ema_nesterov_certificate,
)
from passive_muon.floored_certificate import (
    JORDAN_DERIVATIVE_LOWER,
    JORDAN_DERIVATIVE_UPPER,
)
from passive_muon.momentum_iqc import repaired_zero_input_gain
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_experiment import (
    StructureAwareProbeConfig,
    run_structure_aware_probe,
)
from passive_muon.structure_aware_stability import (
    LOCKED_CENTER_GAIN,
    LOCKED_DEFICIT_UPPER,
    LOCKED_FLOOR,
    LOCKED_HESSIAN_LOWER,
    LOCKED_HESSIAN_UPPER,
    LOCKED_LEARNING_RATE,
    LOCKED_REPAIR_MARGIN,
    LOCKED_REPAIR_RHO,
    LOCKED_RESIDUAL_LIPSCHITZ,
    LOCKED_SKEW_NORM_UPPER,
    LOCKED_SYMMETRIC_GAIN_LOWER,
    LOCKED_SYMMETRIC_GAIN_UPPER,
    LOCKED_TAU,
    BernsteinPositivityCertificate,
    Polynomial,
    locked_structure_aware_certificate,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
P3_COMPARISON_COMMIT = "5c34d2b449273034b32e295a3f0583a40c539300"
FROZEN_FALLBACK_COMMIT = "518cc9384a7a478f3c5956532fd26a6937f70d5f"
SOURCE_PATHS = (
    "scripts/certify_structure_aware_stability.py",
    "src/passive_muon/structure_aware_experiment.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/ema_nesterov_experiment.py",
    "src/passive_muon/ema_nesterov_iqc.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_experiment.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_structure_aware_experiment.py",
    "tests/test_structure_aware_stability.py",
    "theory/structure_aware_stability_certificate.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is always printed to stdout",
    )
    parser.add_argument(
        "--probe-trials-per-case",
        type=int,
        default=12,
        help="seeded trials for each shape/orientation case (default: 12)",
    )
    parser.add_argument(
        "--probe-iterations",
        type=int,
        default=1_500,
        help="iterations per sampled full-matrix trajectory (default: 1500)",
    )
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 50) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction_payload(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _polynomial_payload(coefficients: Polynomial) -> list[str]:
    return [str(value) for value in coefficients]


def _bernstein_payload(
    certificate: BernsteinPositivityCertificate,
) -> dict[str, Any]:
    return {
        "name": certificate.name,
        "interval": {
            "lower_exact": str(certificate.interval_lower),
            "upper_exact": str(certificate.interval_upper),
        },
        "power_coefficients_low_to_high": _polynomial_payload(
            certificate.power_coefficients
        ),
        "bernstein_coefficients": [str(value) for value in certificate.coefficients],
        "minimum_bernstein_coefficient": _fraction_payload(
            certificate.minimum_coefficient
        ),
        "strictly_positive": certificate.strictly_positive,
    }


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _source_snapshot() -> dict[str, str]:
    return {
        path: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _normalized_gain_at_frequency(
    *, scaled_curvature: float, omega: float
) -> float:
    beta = 0.95
    tau = float(LOCKED_TAU)
    center = float(LOCKED_CENTER_GAIN)
    residual = float(LOCKED_RESIDUAL_LIPSCHITZ)
    z = tau * np.exp(1j * omega)
    numerator = -scaled_curvature * (1 - beta) * ((1 + beta) * z - beta)
    denominator = (
        (z - 1) * (z - beta)
        + center * scaled_curvature * (1 - beta) * ((1 + beta) * z - beta)
    )
    return float(residual * abs(numerator / denominator))


def _peak_normalized_gain(scaled_curvature: float) -> tuple[float, float]:
    frequencies = np.linspace(0.0, np.pi, 8_193, dtype=np.float64)
    values = np.array(
        [
            _normalized_gain_at_frequency(
                scaled_curvature=scaled_curvature,
                omega=float(omega),
            )
            for omega in frequencies
        ]
    )
    index = int(np.argmax(values))
    lower = float(frequencies[max(0, index - 2)])
    upper = float(frequencies[min(len(frequencies) - 1, index + 2)])
    optimized = minimize_scalar(
        lambda omega: -_normalized_gain_at_frequency(
            scaled_curvature=scaled_curvature,
            omega=float(omega),
        ),
        bounds=(lower, upper),
        method="bounded",
        options={"xatol": 1e-15},
    )
    return -float(optimized.fun), float(optimized.x)


def _frequency_diagnostics() -> dict[str, Any]:
    scaled_lower = float(LOCKED_LEARNING_RATE * LOCKED_HESSIAN_LOWER)
    scaled_upper = float(LOCKED_LEARNING_RATE * LOCKED_HESSIAN_UPPER)
    grid = np.linspace(scaled_lower, scaled_upper, 257, dtype=np.float64)
    peaks = [_peak_normalized_gain(float(value)) for value in grid]
    index = max(range(len(peaks)), key=lambda item: peaks[item][0])
    locked_peak, locked_frequency = peaks[index]

    def boundary_residual(learning_rate: float) -> float:
        peak, _frequency = _peak_normalized_gain(
            learning_rate * float(LOCKED_HESSIAN_UPPER)
        )
        return peak - 1.0

    boundary = brentq(boundary_residual, 3.2e-5, 3.4e-5, xtol=1e-15, rtol=1e-13)
    boundary_peak, boundary_frequency = _peak_normalized_gain(
        boundary * float(LOCKED_HESSIAN_UPPER)
    )
    return {
        "role": "float64_diagnostic_only; the rational Bernstein replay is authoritative",
        "locked_grid": {
            "scaled_curvature_points": len(grid),
            "frequency_seed_points_per_scaled_curvature": 8_193,
            "maximum_normalized_gain": locked_peak,
            "worst_curvature": float(grid[index] / float(LOCKED_LEARNING_RATE)),
            "worst_angular_frequency": locked_frequency,
            "squared_gain_margin": 1.0 - locked_peak**2,
        },
        "numerical_first_boundary_at_upper_curvature": {
            "learning_rate": boundary,
            "normalized_gain": boundary_peak,
            "angular_frequency": boundary_frequency,
        },
    }


def build_payload(*, probe_trials_per_case: int, probe_iterations: int) -> dict[str, Any]:
    audit = locked_structure_aware_certificate()
    if not audit.certified:
        raise AssertionError("exact structure-aware stability audit failed")

    p3 = locked_ema_nesterov_certificate()
    zero_gain = repaired_zero_input_gain(
        floor=p3.floor,
        repair_rho=p3.repair_rho,
    )
    linked_local_threshold = ema_nesterov_local_learning_rate_threshold(
        beta=p3.beta,
        curvature=p3.hessian_upper,
        repaired_gain=zero_gain,
    )
    improvement = LOCKED_LEARNING_RATE / p3.learning_rate
    remaining_local_gap = linked_local_threshold / LOCKED_LEARNING_RATE
    if improvement < 100:
        raise AssertionError("the predeclared 100x p4 continuation gate did not pass")

    rank_one_common = {
        "beta": float(p3.beta),
        "curvature": float(p3.hessian_upper),
        "floor": float(p3.floor),
        "repair_rho": float(p3.repair_rho),
        "initial_position": 1.0,
        "initial_momentum": 0.0,
        "iterations": 5_000,
    }
    p3_trajectory = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(p3.learning_rate),
        **rank_one_common,
    )
    p4_trajectory = run_rank_one_ema_nesterov_trajectory(
        learning_rate=float(LOCKED_LEARNING_RATE),
        **rank_one_common,
    )
    full_matrix_probe = run_structure_aware_probe(
        StructureAwareProbeConfig(
            trials_per_case=probe_trials_per_case,
            iterations=probe_iterations,
        )
    )

    revision = _run_git("rev-parse", "HEAD")
    branch = _run_git("branch", "--show-current")
    status = _run_git("status", "--porcelain")
    frequency = audit.frequency_polynomials
    return {
        "schema_version": "passive-muon-structure-aware-stability-certificate-v1",
        "claim_scope": {
            "evidence_kind": (
                "exact_rational_dimension_uniform_centered_small_gain_certificate_"
                "plus_deterministic_sampled_full_matrix_trajectories"
            ),
            "guarantee": (
                "global incremental exponential stability at rate tau for the specified "
                "real-arithmetic repaired floored-Jordan EMA/Nesterov loop"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "quadratic_domain": (
                "every fixed finite-dimensional real quadratic with I <= H <= 10*I; "
                "H may mix arbitrary vectorized matrix directions"
            ),
            "not_claimed": [
                "stability of the upstream unrepaired exact-current Muon operator",
                "stability of BF16 or additive-epsilon arithmetic",
                "stochastic or nonquadratic objective stability",
                "neural-network training convergence",
                "necessity or optimality of the certified learning rate",
                "a proof supplied by the sampled trajectory probe",
            ],
        },
        "operator": {
            "formula": "R(M)=H_h(M/max(c,||M||_F))+rho*M",
            "domain": "every fixed finite real matrix shape",
            "normalization": "fixed_frobenius_floor",
            "floor_c": _fraction_payload(LOCKED_FLOOR),
            "additive_epsilon": "not_applicable; denominator uses a max floor",
            "arithmetic_model": "real arithmetic",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "coefficients_exact": {
                "a": str(JORDAN_QUINTIC.fractions()[0]),
                "b": str(JORDAN_QUINTIC.fractions()[1]),
                "c": str(JORDAN_QUINTIC.fractions()[2]),
            },
            "steps": 5,
            "scalar_derivative_lower": _fraction_payload(JORDAN_DERIVATIVE_LOWER),
            "scalar_derivative_upper": _fraction_payload(JORDAN_DERIVATIVE_UPPER),
            "deficit_upper": _fraction_payload(LOCKED_DEFICIT_UPPER),
            "repair_margin_mu": _fraction_payload(LOCKED_REPAIR_MARGIN),
            "constant_repair_rho": _fraction_payload(LOCKED_REPAIR_RHO),
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
        },
        "structure_aware_center": {
            "symmetric_jacobian_lower": _fraction_payload(
                LOCKED_SYMMETRIC_GAIN_LOWER
            ),
            "symmetric_jacobian_upper": _fraction_payload(
                LOCKED_SYMMETRIC_GAIN_UPPER
            ),
            "skew_jacobian_norm_upper": _fraction_payload(LOCKED_SKEW_NORM_UPPER),
            "center_gain_gamma": _fraction_payload(LOCKED_CENTER_GAIN),
            "residual_lipschitz_upper": _fraction_payload(
                LOCKED_RESIDUAL_LIPSCHITZ
            ),
            "decomposition": "R=gamma*I+E, with Lip(E)<=residual_lipschitz_upper",
            "skew_bound_formula": "(b-a)/(4*c)",
            "skew_bound_scope": (
                "full rectangular tangent space, including the exterior radial coupling "
                "and the Clarke hull at ||M||_F=c"
            ),
        },
        "pinned_ema_nesterov_loop": {
            "beta": _fraction_payload(audit.parameters.beta),
            "update_order": [
                "g_t=H*(W_t-W_star)",
                "m_(t+1)=beta*m_t+(1-beta)*g_t",
                "s_(t+1)=beta*m_(t+1)+(1-beta)*g_t",
                "W_(t+1)=W_t-eta*R(s_(t+1))",
            ],
            "difference_plant": [
                "m_next=beta*m+(1-beta)*H*w",
                "s=beta^2*m+(1-beta^2)*H*w",
                "w_next=w-eta*gamma*s-eta*e",
                "||e||_F<=residual_lipschitz_upper*||s||_F",
            ],
            "scalar_transfer": (
                "G_lambda(z)=-eta*(1-beta)*lambda*((1+beta)*z-beta)/"
                "((z-1)*(z-beta)+gamma*eta*(1-beta)*lambda*((1+beta)*z-beta))"
            ),
            "orientation_argument": (
                "orthogonally diagonalize fixed H; the nominal plant becomes a direct "
                "sum while the residual remains one full-block Lipschitz uncertainty"
            ),
        },
        "locked_exact_certificate": {
            "curvature_lower": _fraction_payload(audit.parameters.curvature_lower),
            "curvature_upper": _fraction_payload(audit.parameters.curvature_upper),
            "learning_rate_eta": _fraction_payload(audit.parameters.learning_rate),
            "exponential_rate_tau": _fraction_payload(audit.parameters.tau),
            "rate_squared": _fraction_payload(audit.parameters.tau**2),
            "scaled_curvature_interval": {
                "lower": _fraction_payload(audit.parameters.scaled_curvature_lower),
                "upper": _fraction_payload(audit.parameters.scaled_curvature_upper),
            },
            "scaled_jury": [
                _bernstein_payload(certificate)
                for certificate in audit.jury_certificates
            ],
            "frequency_gap": {
                "definition": (
                    "Q(t,x)=|D(tau*exp(i*omega),x)|^2-"
                    "residual_lipschitz^2*|N(tau*exp(i*omega),x)|^2; "
                    "t=1-cos(omega)"
                ),
                "q0_power_coefficients": _polynomial_payload(frequency.q0),
                "q1_power_coefficients": _polynomial_payload(frequency.q1),
                "q2_power_coefficients": _polynomial_payload(frequency.q2),
                "vertex_power_coefficients": _polynomial_payload(frequency.vertex),
                "q2_bernstein_certificate": _bernstein_payload(
                    audit.q2_certificate
                ),
                "negative_discriminant_bernstein_certificate": _bernstein_payload(
                    audit.vertex_certificate
                ),
            },
            "all_exact_checks_passed": audit.certified,
        },
        "comparison_to_p3_and_linked_control": {
            "p3_comparison_commit": P3_COMPARISON_COMMIT,
            "p3_certified_learning_rate": _fraction_payload(p3.learning_rate),
            "p3_rate_squared": _fraction_payload(p3.rate_squared),
            "p4_over_p3_learning_rate": _fraction_payload(improvement),
            "predeclared_at_least_100x_gate_passed": improvement >= 100,
            "linked_zero_mode_local_threshold": _fraction_payload(
                linked_local_threshold
            ),
            "linked_local_threshold_over_p4": _fraction_payload(
                remaining_local_gap
            ),
            "within_100x_of_linked_local_threshold": remaining_local_gap <= 100,
            "qualification": (
                "the linked zero-mode value is a local necessary control, not a global "
                "stability certificate or a claim that the p4 rate is minimal"
            ),
        },
        "float64_frequency_diagnostics": _frequency_diagnostics(),
        "matched_rank_one_float64": {
            "controlled_variable": "learning_rate_only",
            "shared_configuration": rank_one_common,
            "p3_certified_step": asdict(p3_trajectory),
            "p4_certified_step": asdict(p4_trajectory),
        },
        "sampled_full_matrix_adversarial_probe": full_matrix_probe.as_dict(),
        "review_status": {
            "independent_human_review_of_p4_complete": False,
            "independent_upstream_parity_audit_complete": False,
        },
        "experiment_provenance": {
            "analytic_seed": None,
            "trajectory_seed": full_matrix_probe.config.seed,
            "determinism": (
                "exact Fraction/Bernstein replay plus deterministic seeded CPU float64 "
                "frequency and trajectory diagnostics"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pytorch": torch.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine_architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "git": {
            "sha": revision.stdout.strip() if revision.returncode == 0 else "unknown",
            "branch": branch.stdout.strip() if branch.returncode == 0 else "unknown",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "frozen_p3_fallback_commit": FROZEN_FALLBACK_COMMIT,
        },
        "upstream_provenance": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "muon_py_sha256": PINNED_MUON_PY_SHA256,
        },
    }


def main() -> None:
    args = parse_args()
    payload = build_payload(
        probe_trials_per_case=args.probe_trials_per_case,
        probe_iterations=args.probe_iterations,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
