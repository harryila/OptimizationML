#!/usr/bin/env python3
"""Run the deterministic 2x2 spectral/local-deficit audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch

from passive_muon.metrics import polynomial_matmul_count
from passive_muon.specs import (
    CANS_5X4,
    CLASSICAL_CUBIC,
    JORDAN_QUINTIC,
    POLAR_EXPRESS_5,
    TAYLOR_QUINTIC,
    PolynomialCoefficients,
    StagedQuinticCoefficients,
    scalar_response_and_derivative_exact,
    staged_scalar_response_and_derivative_exact,
)
from passive_muon.spectral_analysis import (
    epsilon_normalized_local_spectrum,
    exact_normalized_local_spectrum,
)


@dataclass(frozen=True)
class AuditRow:
    baseline: str
    steps: int
    matrix_multiplications: int
    unit_exact_local_deficit: float
    worst_angle_radians: float
    unit_fixed_scale_deficit_grid: float
    fixed_scale_minimum_at_worst_angle: float
    clean_normalization_deficit: float | None
    clean_normalization_grid_fraction: float
    eps1_full_calibration_deficit_lower_bound: float
    eps1_full_validation_deficit_lower_bound: float
    eps1_full_combined_grid_deficit_lower_bound: float
    eps1_diagonal_calibration_deficit_lower_bound: float
    eps1_diagonal_validation_deficit_lower_bound: float
    eps1_diagonal_combined_grid_deficit_lower_bound: float
    deployed_eps_local_deficit_grid_lower_bound: float
    deployed_eps_diagonal_grid_deficit_lower_bound: float
    exact_rational_slope_mismatch_at_3_5_4_5: bool
    exact_global_deficit: str


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return {
        "sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
    }


def _polynomial_manifest(coefficients: PolynomialCoefficients) -> dict[str, Any]:
    if isinstance(coefficients, StagedQuinticCoefficients):
        stages = [{"a": stage.a, "b": stage.b, "c": stage.c} for stage in coefficients.stages]
        return {
            "name": coefficients.name,
            "kind": "finite_staged_quintic",
            "stages": stages,
            "source": coefficients.source,
        }
    return {
        "name": coefficients.name,
        "kind": "repeated_quintic",
        "a": coefficients.a,
        "b": coefficients.b,
        "c": coefficients.c,
        "source": coefficients.source,
    }


def _source_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _exact_derivative(
    value: Fraction, coefficients: PolynomialCoefficients, *, steps: int
) -> Fraction:
    if isinstance(coefficients, StagedQuinticCoefficients):
        _response, derivative = staged_scalar_response_and_derivative_exact(
            value, coefficients, steps=steps
        )
    else:
        _response, derivative = scalar_response_and_derivative_exact(
            value, coefficients, steps=steps
        )
    return derivative


def run_audit(
    *,
    angle_points: int,
    epsilon_angle_points: int,
    radius_points: int,
    deployed_eps: float,
) -> tuple[list[AuditRow], dict[str, Any]]:
    margin = 1e-6
    angles = np.linspace(margin, np.pi / 2 - margin, angle_points, dtype=np.float64)
    epsilon_angle_values = np.linspace(
        margin, np.pi / 2 - margin, epsilon_angle_points, dtype=np.float64
    )
    epsilon_radius_values = np.geomspace(1e-8, 1e3, radius_points, dtype=np.float64)
    epsilon_angles = epsilon_angle_values[None, :]
    epsilon_radii = epsilon_radius_values[:, None]
    validation_angles = ((epsilon_angle_values[:-1] + epsilon_angle_values[1:]) / 2)[None, :]
    validation_radii = np.sqrt(epsilon_radius_values[:-1] * epsilon_radius_values[1:])[:, None]

    rows: list[AuditRow] = []
    baselines = [
        (JORDAN_QUINTIC, range(1, 6)),
        (CLASSICAL_CUBIC, range(1, 6)),
        (TAYLOR_QUINTIC, range(1, 6)),
        (POLAR_EXPRESS_5, range(1, 6)),
        (CANS_5X4, range(1, 5)),
    ]
    for coefficients, step_values in baselines:
        for steps in step_values:
            exact_slope_mismatch = _exact_derivative(
                Fraction(3, 5), coefficients, steps=steps
            ) != _exact_derivative(Fraction(4, 5), coefficients, steps=steps)
            exact = exact_normalized_local_spectrum(angles, coefficients, steps=steps)
            worst_index = int(np.argmin(exact.minimum_eigenvalue))
            unit_deficit = max(0.0, -float(exact.minimum_eigenvalue[worst_index]))
            fixed_minimum = float(np.min(exact.fixed_scale_minimum))
            fixed_grid_deficit = -fixed_minimum if fixed_minimum < -1e-12 else 0.0
            clean_mask = exact.fixed_scale_minimum >= -1e-12
            clean_fraction = float(np.mean(clean_mask))
            clean_deficit = (
                max(0.0, -float(np.min(exact.minimum_eigenvalue[clean_mask])))
                if np.any(clean_mask)
                else None
            )

            eps1_calibration = epsilon_normalized_local_spectrum(
                epsilon_angles,
                epsilon_radii,
                coefficients,
                steps=steps,
                eps=1.0,
            )
            eps1_calibration_deficit = max(0.0, -float(np.min(eps1_calibration.minimum_eigenvalue)))
            eps1_validation = epsilon_normalized_local_spectrum(
                validation_angles,
                validation_radii,
                coefficients,
                steps=steps,
                eps=1.0,
            )
            diagonal_calibration_deficit = max(
                0.0, -float(np.min(eps1_calibration.diagonal_minimum))
            )
            diagonal_validation_deficit = max(0.0, -float(np.min(eps1_validation.diagonal_minimum)))
            eps1_validation_deficit = max(0.0, -float(np.min(eps1_validation.minimum_eigenvalue)))
            eps1_combined_deficit = max(eps1_calibration_deficit, eps1_validation_deficit)
            diagonal_combined_deficit = max(
                diagonal_calibration_deficit, diagonal_validation_deficit
            )
            rows.append(
                AuditRow(
                    baseline=coefficients.name,
                    steps=steps,
                    matrix_multiplications=polynomial_matmul_count(coefficients, steps=steps),
                    unit_exact_local_deficit=unit_deficit,
                    worst_angle_radians=float(angles[worst_index]),
                    unit_fixed_scale_deficit_grid=fixed_grid_deficit,
                    fixed_scale_minimum_at_worst_angle=float(
                        exact.fixed_scale_minimum[worst_index]
                    ),
                    clean_normalization_deficit=clean_deficit,
                    clean_normalization_grid_fraction=clean_fraction,
                    eps1_full_calibration_deficit_lower_bound=(eps1_calibration_deficit),
                    eps1_full_validation_deficit_lower_bound=eps1_validation_deficit,
                    eps1_full_combined_grid_deficit_lower_bound=eps1_combined_deficit,
                    eps1_diagonal_calibration_deficit_lower_bound=(diagonal_calibration_deficit),
                    eps1_diagonal_validation_deficit_lower_bound=(diagonal_validation_deficit),
                    eps1_diagonal_combined_grid_deficit_lower_bound=(diagonal_combined_deficit),
                    deployed_eps_local_deficit_grid_lower_bound=(
                        eps1_combined_deficit / deployed_eps
                    ),
                    deployed_eps_diagonal_grid_deficit_lower_bound=(
                        diagonal_combined_deficit / deployed_eps
                    ),
                    exact_rational_slope_mismatch_at_3_5_4_5=exact_slope_mismatch,
                    exact_global_deficit=(
                        "infinite" if exact_slope_mismatch else "not_established"
                    ),
                )
            )

    manifest = {
        "schema_version": "passive-muon-deficit-audit-v5",
        "experiment_kind": "deterministic_spectral_grid",
        "experiment": {
            "matrix_class": "2x2 diagonal spectra",
            "simulation_dtype": "float64",
            "seed": None,
            "randomness": "none; deterministic grids",
            "operator": (
                "Finite composition of odd-quintic stages "
                "q_k(X)=a_k X+b_k XX^T X+c_k(XX^T)^2X, evaluated through its "
                "singular-value response; repeated baselines reuse one stage"
            ),
            "normalizers": [
                "exact current Frobenius norm on the unit sphere",
                "current Frobenius norm plus epsilon",
                "fixed scale one",
            ],
            "polynomials": [
                _polynomial_manifest(coefficients) for coefficients, _step_values in baselines
            ],
        },
        "interpretation": (
            "Grid maxima are lower bounds, not domain certificates. Separately, each row's "
            "exact-normalizer global deficit follows from an exact rational derivative mismatch "
            "at the unit diagonal spectrum (3/5, 4/5), plus the analytic scale obstruction; "
            "that conclusion does not rely on a floating-point grid sign."
        ),
        "exact_rational_witness": {
            "unit_diagonal_spectrum": ["3/5", "4/5"],
            "criterion": "composed scalar derivatives are unequal in exact rational arithmetic",
            "consequence": (
                "the symmetric Jacobian determinant is strictly negative; scaling toward zero "
                "makes the unrestricted exact-normalizer deficit infinite"
            ),
        },
        "grid": {
            "angle_points": angle_points,
            "angle_interval": [margin, float(np.pi / 2 - margin)],
            "epsilon_angle_points": epsilon_angle_points,
            "radius_points": radius_points,
            "dimensionless_radius_over_epsilon_interval": [1e-8, 1e3],
            "eps1_radius_interval": [1e-8, 1e3],
            "deployed_radius_interval": [
                deployed_eps * 1e-8,
                deployed_eps * 1e3,
            ],
            "diagonal_validation_grid": "geometric/angular cell midpoints",
            "radius_zero_check": (
                "analytic Jacobian is positive scalar h'(0)/eps; zero is not log-sampled"
            ),
            "epsilon_reference": 1.0,
            "deployed_epsilon": deployed_eps,
            "epsilon_scaling_identity": "delta_eps = delta_1 / eps",
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        "git": _git_state(),
        "source_snapshot": _source_snapshot(
            [
                Path("experiments/matrices/run_deficit_audit.py"),
                Path("src/passive_muon/specs.py"),
                Path("src/passive_muon/spectral_analysis.py"),
                Path("src/passive_muon/metrics.py"),
            ]
        ),
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--angle-points", type=int, default=4097)
    parser.add_argument("--epsilon-angle-points", type=int, default=1025)
    parser.add_argument("--radius-points", type=int, default=513)
    parser.add_argument("--deployed-eps", type=float, default=1e-7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, manifest = run_audit(
        angle_points=args.angle_points,
        epsilon_angle_points=args.epsilon_angle_points,
        radius_points=args.radius_points,
        deployed_eps=args.deployed_eps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "deficit_audit.json"
    csv_path = args.output_dir / "deficit_audit.csv"
    payload = {**manifest, "rows": [asdict(row) for row in rows]}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
