#!/usr/bin/env python3
"""Check finite-horizon drift for two representative quadratic configurations."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from passive_muon.quadratic_experiment import (
    build_quadratic_ensemble,
    run_quadratic_sweep,
    summarize_grid_band,
)
from passive_muon.specs import (
    CLASSICAL_CUBIC,
    JORDAN_QUINTIC,
    QuinticCoefficients,
    zero_slope_gain_exact,
)

AUDIT_PATH = Path("results/summaries/deficit_audit.json")
OUTPUT_DIR = Path("results/summaries")
HORIZONS = (750, 3000)
TARGET_RATIOS = (1e-4, 1e-8)
REQUIRED_FRACTION = 0.9
REPAIR_MARGIN = 1.02
HOLD_ITERATIONS = 50
SEED = 20260831
CONDITION_NUMBERS = (1.0, 3.0, 10.0, 30.0)
SEEDS_PER_CONDITION = 8
EPSILON = 1.0
DIVERGENCE_RATIO = 1e6


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


def _polynomial_manifest(coefficients: QuinticCoefficients) -> dict[str, str]:
    return {
        "name": coefficients.name,
        "kind": "repeated_quintic",
        "a": coefficients.a,
        "b": coefficients.b,
        "c": coefficients.c,
        "source": coefficients.source,
    }


def main() -> int:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    audit_lookup = {(row["baseline"], row["steps"]): row for row in audit["rows"]}
    ensemble = build_quadratic_ensemble(
        condition_numbers=CONDITION_NUMBERS,
        seeds_per_condition=SEEDS_PER_CONDITION,
        seed=SEED,
    )
    learning_rates = np.geomspace(1e-5, 3.0, 73)
    rows: list[dict[str, Any]] = []

    for coefficients in (JORDAN_QUINTIC, CLASSICAL_CUBIC):
        steps = 5
        deficit = float(
            audit_lookup[(coefficients.name, steps)][
                "eps1_diagonal_combined_grid_deficit_lower_bound"
            ]
        )
        rho = REPAIR_MARGIN * deficit
        base_gain = float(zero_slope_gain_exact(coefficients, steps=steps))
        repaired_gain = base_gain + rho
        configurations = (
            ("unrepaired", 0.0, 1.0),
            ("raw_grid_repair_1.02x", rho, 1.0),
            (
                "gain_matched_grid_repair_1.02x",
                rho,
                base_gain / repaired_gain,
            ),
            (
                "gain_only_unrepaired_control",
                0.0,
                repaired_gain / base_gain,
            ),
        )
        for horizon in HORIZONS:
            for intervention, repair_rho, output_scale in configurations:
                outcome = run_quadratic_sweep(
                    ensemble,
                    learning_rates,
                    coefficients,
                    polynomial_steps=steps,
                    iterations=horizon,
                    normalization="current_plus_eps",
                    normalizer_scale=EPSILON,
                    repair_rho=repair_rho,
                    success_ratio=min(TARGET_RATIOS),
                    divergence_ratio=DIVERGENCE_RATIO,
                    output_scale=output_scale,
                    hold_iterations=HOLD_ITERATIONS,
                )
                effective_gain = output_scale * (base_gain + repair_rho)
                for target_ratio in TARGET_RATIOS:
                    target_fraction = np.mean(
                        outcome.bounded & (outcome.tail_maximum_ratios <= target_ratio),
                        axis=0,
                    )
                    band = summarize_grid_band(
                        target_fraction >= REQUIRED_FRACTION,
                        learning_rates,
                    )
                    rows.append(
                        {
                            "baseline": coefficients.name,
                            "steps": steps,
                            "horizon": horizon,
                            "hold_iterations": HOLD_ITERATIONS,
                            "target_ratio": target_ratio,
                            "required_problem_fraction": REQUIRED_FRACTION,
                            "intervention": intervention,
                            "repair_rho_inside_scale": repair_rho,
                            "output_scale": output_scale,
                            "effective_zero_slope_gain": effective_gain,
                            "target_component_count": band.component_count,
                            "target_lower_pass": band.lower_pass,
                            "target_upper_pass": band.upper_pass,
                            "target_lower_fail_below": band.lower_fail_below,
                            "target_upper_fail_above": band.upper_fail_above,
                            "target_log10_width": band.log10_width,
                        }
                    )

    comparisons = []
    for baseline in (JORDAN_QUINTIC.name, CLASSICAL_CUBIC.name):
        for horizon in HORIZONS:
            for target_ratio in TARGET_RATIOS:
                subset = [
                    row
                    for row in rows
                    if row["baseline"] == baseline
                    and row["horizon"] == horizon
                    and row["target_ratio"] == target_ratio
                ]
                endpoints = {row["intervention"]: row["target_upper_pass"] for row in subset}
                comparisons.append(
                    {
                        "baseline": baseline,
                        "horizon": horizon,
                        "target_ratio": target_ratio,
                        "gain_matched_repair_upper": endpoints["gain_matched_grid_repair_1.02x"],
                        "unrepaired_upper": endpoints["unrepaired"],
                        "raw_repair_upper": endpoints["raw_grid_repair_1.02x"],
                        "gain_only_control_upper": endpoints["gain_only_unrepaired_control"],
                    }
                )

    payload = {
        "schema_version": "passive-muon-quadratic-horizon-check-v2",
        "interpretation": (
            "Finite-grid robustness check only. Endpoint drift across horizons or target "
            "ratios confirms that target bands are finite-time quantities, not certified "
            "stability regions."
        ),
        "input": {
            "path": str(AUDIT_PATH),
            "sha256": hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest(),
            "schema_version": audit["schema_version"],
        },
        "experiment": {
            "problem": "2x2 diagonal matrix quadratics in a shared singular-vector basis",
            "condition_numbers": list(CONDITION_NUMBERS),
            "seeds_per_condition": SEEDS_PER_CONDITION,
            "horizons": list(HORIZONS),
            "target_ratios": list(TARGET_RATIOS),
            "hold_iterations": HOLD_ITERATIONS,
            "required_problem_fraction": REQUIRED_FRACTION,
            "problem_count": ensemble.size,
            "learning_rate_points": learning_rates.size,
            "learning_rate_interval": [float(learning_rates[0]), float(learning_rates[-1])],
            "normalization": "G / (FrobeniusNorm(G) + eps)",
            "epsilon": EPSILON,
            "operator": ("output_scale * (P_K(G / (FrobeniusNorm(G) + eps)) + rho * G)"),
            "polynomial_steps": 5,
            "polynomials": [
                _polynomial_manifest(JORDAN_QUINTIC),
                _polynomial_manifest(CLASSICAL_CUBIC),
            ],
            "repair_margin": REPAIR_MARGIN,
            "repair_source": (
                "1.02 times the maximum observed diagonal deficit over the audit's disjoint "
                "calibration and midpoint-validation grids"
            ),
            "repair_audit_dimensionless_radius_interval": audit["grid"][
                "dimensionless_radius_over_epsilon_interval"
            ],
            "divergence_ratio": DIVERGENCE_RATIO,
            "interventions": {
                "unrepaired": "rho=0, output_scale=1",
                "raw_grid_repair_1.02x": "rho=1.02*sampled_deficit, output_scale=1",
                "gain_matched_grid_repair_1.02x": (
                    "rho=1.02*sampled_deficit, output_scale=base_gain/(base_gain+rho)"
                ),
                "gain_only_unrepaired_control": (
                    "rho=0, output_scale=(base_gain+repair_rho)/base_gain"
                ),
            },
            "dtype": "float64",
            "seed": SEED,
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
        "source_snapshot": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("experiments/quadratics/run_horizon_check.py"),
                Path("src/passive_muon/quadratic_experiment.py"),
                Path("src/passive_muon/spectral_analysis.py"),
                Path("src/passive_muon/specs.py"),
            ]
        },
        "comparisons": comparisons,
        "rows": rows,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "quadratic_horizon_check.json"
    csv_path = OUTPUT_DIR / "quadratic_horizon_check.csv"
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
