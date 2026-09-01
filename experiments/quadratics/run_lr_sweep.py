#!/usr/bin/env python3
"""Run gain-matched learning-rate sweeps on controlled 2x2 quadratics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import torch
from scipy.stats import spearmanr

from passive_muon.quadratic_experiment import (
    build_quadratic_ensemble,
    run_quadratic_sweep,
    summarize_grid_band,
    summarize_true_prefix,
)
from passive_muon.specs import (
    CANS_5X4,
    CLASSICAL_CUBIC,
    JORDAN_QUINTIC,
    POLAR_EXPRESS_5,
    TAYLOR_QUINTIC,
    PolynomialCoefficients,
    StagedQuinticCoefficients,
    zero_slope_gain_exact,
)


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


def _safe_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in value.split(","))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive values")
    return values


def _zero_slope_gain(coefficients: PolynomialCoefficients, steps: int) -> float:
    return float(zero_slope_gain_exact(coefficients, steps=steps))


def _polynomial_manifest(coefficients: PolynomialCoefficients) -> dict[str, Any]:
    if isinstance(coefficients, StagedQuinticCoefficients):
        return {
            "name": coefficients.name,
            "kind": "finite_staged_quintic",
            "stages": [{"a": stage.a, "b": stage.b, "c": stage.c} for stage in coefficients.stages],
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


def _endpoint_comparison(treated: float | None, control: float | None) -> tuple[str, float | None]:
    if treated is None or control is None:
        if treated is not None:
            return "treated_only", None
        if control is not None:
            return "control_only", None
        return "neither", None
    delta = float(np.log10(treated / control))
    tolerance = 1e-12
    if delta > tolerance:
        return "right", delta
    if delta < -tolerance:
        return "left", delta
    return "unchanged", delta


def _comparison_counts(
    rows: list[dict[str, Any]],
    *,
    treated_label: str,
    control_label: str,
) -> dict[str, Any]:
    outcomes: list[str] = []
    deltas: list[float] = []
    keys = sorted({(row["baseline"], row["steps"]) for row in rows})
    for baseline, steps in keys:
        treated = next(
            row
            for row in rows
            if row["baseline"] == baseline
            and row["steps"] == steps
            and row["intervention"] == treated_label
        )
        control = next(
            row
            for row in rows
            if row["baseline"] == baseline
            and row["steps"] == steps
            and row["intervention"] == control_label
        )
        outcome, delta = _endpoint_comparison(
            treated["target_upper_pass"], control["target_upper_pass"]
        )
        outcomes.append(outcome)
        if delta is not None:
            deltas.append(delta)
    labels = ("right", "unchanged", "left", "treated_only", "control_only", "neither")
    return {
        "treated": treated_label,
        "control": control_label,
        "pair_count": len(keys),
        "upper_endpoint_counts": {label: outcomes.count(label) for label in labels},
        "median_log10_upper_endpoint_change": (float(np.median(deltas)) if deltas else None),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deficit-audit",
        type=Path,
        default=Path("results/summaries/deficit_audit.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/summaries"))
    parser.add_argument("--iterations", type=int, default=750)
    parser.add_argument("--hold-iterations", type=int, default=50)
    parser.add_argument("--lr-points", type=int, default=73)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=3.0)
    parser.add_argument("--success-ratio", type=float, default=1e-4)
    parser.add_argument("--required-success-fraction", type=float, default=0.9)
    parser.add_argument(
        "--sensitivity-success-ratios",
        type=_parse_float_tuple,
        default=(1e-3, 1e-4, 1e-5),
    )
    parser.add_argument(
        "--sensitivity-success-fractions",
        type=_parse_float_tuple,
        default=(0.75, 0.9, 1.0),
    )
    parser.add_argument("--repair-margin", type=float, default=1.02)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = json.loads(args.deficit_audit.read_text(encoding="utf-8"))
    audit_lookup = {(row["baseline"], row["steps"]): row for row in audit["rows"]}
    ensemble = build_quadratic_ensemble(
        condition_numbers=(1.0, 3.0, 10.0, 30.0),
        seeds_per_condition=8,
        seed=args.seed,
    )
    learning_rates = np.geomspace(args.lr_min, args.lr_max, args.lr_points)
    baselines = [
        (JORDAN_QUINTIC, range(1, 6)),
        (CLASSICAL_CUBIC, range(1, 6)),
        (TAYLOR_QUINTIC, range(1, 6)),
        (POLAR_EXPRESS_5, range(1, 6)),
        (CANS_5X4, range(1, 5)),
    ]
    summary_rows: list[dict[str, Any]] = []
    learning_rate_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    repair_suffix = f"{args.repair_margin:g}x"

    for coefficients, step_values in baselines:
        for polynomial_steps in step_values:
            audit_row = audit_lookup[(coefficients.name, polynomial_steps)]
            diagonal_deficit = float(audit_row["eps1_diagonal_combined_grid_deficit_lower_bound"])
            full_deficit = float(audit_row["eps1_full_combined_grid_deficit_lower_bound"])
            rho = args.repair_margin * diagonal_deficit
            base_gain = _zero_slope_gain(coefficients, polynomial_steps)
            repaired_gain = base_gain + rho
            slope_match_scale = base_gain / repaired_gain
            gain_only_scale = repaired_gain / base_gain
            configurations = [
                ("current_plus_eps", "unrepaired", 0.0, 1.0, "base_gain"),
                (
                    "current_plus_eps",
                    f"raw_grid_repair_{repair_suffix}",
                    rho,
                    1.0,
                    "raised_gain",
                ),
                (
                    "current_plus_eps",
                    f"gain_matched_grid_repair_{repair_suffix}",
                    rho,
                    slope_match_scale,
                    "base_gain",
                ),
                (
                    "current_plus_eps",
                    "gain_only_unrepaired_control",
                    0.0,
                    gain_only_scale,
                    "raised_gain",
                ),
                ("fixed_scale", "fixed_scale_control", 0.0, 1.0, "fixed_scale"),
            ]
            for normalization, intervention, repair_rho, output_scale, gain_group in configurations:
                outcome = run_quadratic_sweep(
                    ensemble,
                    learning_rates,
                    coefficients,
                    polynomial_steps=polynomial_steps,
                    iterations=args.iterations,
                    normalization=normalization,
                    normalizer_scale=1.0,
                    repair_rho=repair_rho,
                    success_ratio=args.success_ratio,
                    divergence_ratio=1e6,
                    output_scale=output_scale,
                    hold_iterations=args.hold_iterations,
                )
                success_fraction = np.mean(outcome.success, axis=0)
                bounded_fraction = np.mean(outcome.bounded, axis=0)
                unresolved_fraction = np.mean(outcome.bounded & ~outcome.success, axis=0)
                median_final = np.median(outcome.final_ratios, axis=0)
                target_band = summarize_grid_band(
                    success_fraction >= args.required_success_fraction,
                    learning_rates,
                )
                bounded_prefix = summarize_true_prefix(
                    bounded_fraction >= args.required_success_fraction,
                    learning_rates,
                )
                best_index = int(np.argmin(median_final))
                first_hits = outcome.first_target_iterations[:, best_index]
                first_hits = first_hits[first_hits >= 0]
                effective_gain = output_scale * (base_gain + repair_rho)
                residual_deficit = (
                    output_scale * max(0.0, diagonal_deficit - repair_rho)
                    if normalization == "current_plus_eps"
                    else None
                )
                config_key = (
                    f"{coefficients.name}:steps={polynomial_steps}:{normalization}:{intervention}"
                )
                summary_row = {
                    "config": config_key,
                    "baseline": coefficients.name,
                    "steps": polynomial_steps,
                    "normalization": normalization,
                    "intervention": intervention,
                    "gain_group": gain_group,
                    "repair_rho_inside_scale": repair_rho,
                    "output_scale": output_scale,
                    "base_zero_slope_gain": base_gain,
                    "effective_zero_slope_gain": effective_gain,
                    "analytic_local_lr_ceiling": 2.0 / effective_gain,
                    "measured_eps1_diagonal_deficit_lower_bound": (
                        diagonal_deficit if normalization == "current_plus_eps" else None
                    ),
                    "measured_eps1_full_deficit_lower_bound_out_of_subspace": (
                        full_deficit if normalization == "current_plus_eps" else None
                    ),
                    "sampled_diagonal_residual_after_scaled_shift": residual_deficit,
                    "target_component_count": target_band.component_count,
                    "target_lower_pass": target_band.lower_pass,
                    "target_upper_pass": target_band.upper_pass,
                    "target_lower_fail_below": target_band.lower_fail_below,
                    "target_upper_fail_above": target_band.upper_fail_above,
                    "target_left_censored": target_band.left_censored,
                    "target_right_censored": target_band.right_censored,
                    "target_log10_width": target_band.log10_width,
                    "target_lr_points": target_band.point_count,
                    "bounded_prefix_upper_pass": bounded_prefix.upper_pass,
                    "bounded_prefix_upper_fail": bounded_prefix.upper_fail,
                    "bounded_prefix_right_censored": bounded_prefix.right_censored,
                    "bounded_prefix_lr_points": bounded_prefix.point_count,
                    "best_median_lr": float(learning_rates[best_index]),
                    "best_median_normalized_lr": float(learning_rates[best_index] * effective_gain),
                    "best_median_final_ratio": _safe_float(median_final[best_index]),
                    "median_first_target_iteration_at_best_lr": (
                        float(np.median(first_hits)) if first_hits.size else None
                    ),
                    "max_target_success_fraction": float(np.max(success_fraction)),
                    "minimum_positive_gradient_norm_observed": (
                        outcome.minimum_positive_gradient_norm
                    ),
                    "maximum_gradient_norm_observed": outcome.maximum_gradient_norm,
                }
                summary_rows.append(summary_row)

                for success_ratio in args.sensitivity_success_ratios:
                    condition_success = outcome.bounded & (
                        outcome.tail_maximum_ratios <= success_ratio
                    )
                    condition_fraction = np.mean(condition_success, axis=0)
                    for required_fraction in args.sensitivity_success_fractions:
                        condition_band = summarize_grid_band(
                            condition_fraction >= required_fraction,
                            learning_rates,
                        )
                        sensitivity_rows.append(
                            {
                                "config": config_key,
                                "baseline": coefficients.name,
                                "steps": polynomial_steps,
                                "normalization": normalization,
                                "intervention": intervention,
                                "success_ratio": success_ratio,
                                "required_success_fraction": required_fraction,
                                "target_component_count": condition_band.component_count,
                                "target_lower_pass": condition_band.lower_pass,
                                "target_upper_pass": condition_band.upper_pass,
                                "target_lower_fail_below": condition_band.lower_fail_below,
                                "target_upper_fail_above": condition_band.upper_fail_above,
                                "target_left_censored": condition_band.left_censored,
                                "target_right_censored": condition_band.right_censored,
                                "target_log10_width": condition_band.log10_width,
                                "target_lr_points": condition_band.point_count,
                            }
                        )

                for index, learning_rate in enumerate(learning_rates):
                    learning_rate_rows.append(
                        {
                            "config": config_key,
                            "baseline": coefficients.name,
                            "steps": polynomial_steps,
                            "normalization": normalization,
                            "intervention": intervention,
                            "learning_rate": float(learning_rate),
                            "normalized_learning_rate": float(learning_rate * effective_gain),
                            "target_and_hold_fraction": float(success_fraction[index]),
                            "bounded_fraction": float(bounded_fraction[index]),
                            "unresolved_fraction": float(unresolved_fraction[index]),
                            "divergent_fraction": float(1.0 - bounded_fraction[index]),
                            "median_final_ratio": _safe_float(median_final[index]),
                            "median_tail_maximum_ratio": _safe_float(
                                np.median(outcome.tail_maximum_ratios[:, index])
                            ),
                        }
                    )

    current_rows = [row for row in summary_rows if row["normalization"] == "current_plus_eps"]
    raw_label = f"raw_grid_repair_{repair_suffix}"
    gain_matched_label = f"gain_matched_grid_repair_{repair_suffix}"
    primary_comparisons = {
        "gain_matched_repair_vs_unrepaired": _comparison_counts(
            current_rows,
            treated_label=gain_matched_label,
            control_label="unrepaired",
        ),
        "raw_repair_vs_gain_only_control": _comparison_counts(
            current_rows,
            treated_label=raw_label,
            control_label="gain_only_unrepaired_control",
        ),
        "raw_repair_vs_unrepaired_descriptive_only": _comparison_counts(
            current_rows,
            treated_label=raw_label,
            control_label="unrepaired",
        ),
    }

    unrepaired_rows = [row for row in current_rows if row["intervention"] == "unrepaired"]
    exploratory_x: list[float] = []
    exploratory_y: list[float] = []
    for row in unrepaired_rows:
        if row["target_upper_pass"] is not None:
            exploratory_x.append(
                row["measured_eps1_diagonal_deficit_lower_bound"] / row["effective_zero_slope_gain"]
            )
            exploratory_y.append(row["target_upper_pass"] * row["effective_zero_slope_gain"])
    normalized_correlation = spearmanr(exploratory_x, exploratory_y)
    repaired_rows = [row for row in current_rows if row["repair_rho_inside_scale"] > 0]
    audited_dimensionless_radius_min, audited_dimensionless_radius_max = (
        float(value) for value in audit["grid"]["dimensionless_radius_over_epsilon_interval"]
    )
    repaired_rows_below_audit_radius = [
        row
        for row in repaired_rows
        if row["minimum_positive_gradient_norm_observed"] < audited_dimensionless_radius_min
    ]
    repaired_rows_above_audit_radius = [
        row
        for row in repaired_rows
        if row["maximum_gradient_norm_observed"] > audited_dimensionless_radius_max
    ]

    sensitivity_headlines = []
    for success_ratio in args.sensitivity_success_ratios:
        for required_fraction in args.sensitivity_success_fractions:
            condition_rows = [
                row
                for row in sensitivity_rows
                if row["success_ratio"] == success_ratio
                and row["required_success_fraction"] == required_fraction
                and row["normalization"] == "current_plus_eps"
            ]
            sensitivity_headlines.append(
                {
                    "success_ratio": success_ratio,
                    "required_success_fraction": required_fraction,
                    "gain_matched_repair_vs_unrepaired": _comparison_counts(
                        condition_rows,
                        treated_label=gain_matched_label,
                        control_label="unrepaired",
                    ),
                    "raw_repair_vs_gain_only_control": _comparison_counts(
                        condition_rows,
                        treated_label=raw_label,
                        control_label="gain_only_unrepaired_control",
                    ),
                }
            )

    payload = {
        "schema_version": "passive-muon-quadratic-lr-sweep-v4",
        "experiment": {
            "problem": "2x2 diagonal matrix quadratics in a shared singular-vector basis",
            "condition_numbers": [1.0, 3.0, 10.0, 30.0],
            "seeds_per_condition": 8,
            "problem_count": ensemble.size,
            "iterations": args.iterations,
            "hold_iterations": args.hold_iterations,
            "learning_rate_points": args.lr_points,
            "learning_rate_interval": [args.lr_min, args.lr_max],
            "target_objective_ratio": args.success_ratio,
            "required_problem_fraction": args.required_success_fraction,
            "divergence_ratio": 1e6,
            "sensitivity_success_ratios": list(args.sensitivity_success_ratios),
            "sensitivity_success_fractions": list(args.sensitivity_success_fractions),
            "normalizer_scale_or_epsilon": 1.0,
            "normalizations": {
                "current_plus_eps": "G / (FrobeniusNorm(G) + 1)",
                "fixed_scale": "G / 1",
            },
            "simulation_dtype": "float64",
            "repair_margin": args.repair_margin,
            "seed": args.seed,
            "polynomials": [_polynomial_manifest(item) for item, _step_values in baselines],
        },
        "inputs": {
            "deficit_audit": {
                "path": str(args.deficit_audit),
                "sha256": _sha256(args.deficit_audit),
                "schema_version": audit.get("schema_version"),
            }
        },
        "interpretation": {
            "repair_status": (
                "rho is 1.02 times the maximum observed diagonal deficit over disjoint "
                "calibration and midpoint validation grids. It repairs those finite sampled "
                "points; it is not a continuous-domain upper certificate."
            ),
            "subspace_scope": (
                "The dynamics stay in a shared diagonal basis, so repair sizing uses only the "
                "diagonal Jacobian block. Full-map difference/sum-mode deficits are recorded "
                "only as out-of-subspace diagnostics."
            ),
            "target_band": (
                "Finite-grid target-and-hold band: the widest contiguous learning-rate "
                "component where the required problem fraction stays below the target for "
                "every one of the final hold iterations. It is not called a stability width."
            ),
            "causal_contrasts": (
                "Gain-matched repair versus unrepaired holds the zero-input slope fixed; raw "
                "repair versus gain-only control holds the raised zero-input slope fixed. Raw "
                "repair versus unrepaired is descriptive because +rho changes local gain. The "
                "two matched contrasts are algebraically equivalent under learning-rate "
                "rescaling, so the second is a consistency cross-check, not independent evidence."
            ),
            "local_ceiling": (
                "For lambda_max(Q)=1, linearization at the optimum gives the explicit-Euler "
                "ceiling eta < 2/effective_zero_slope_gain."
            ),
            "pvalues": (
                "Configurations are designed and dependent, so the exploratory Spearman "
                "p-value is descriptive rather than confirmatory. It is a complete-case "
                "calculation over the 16 unrepaired configurations with defined target-band "
                "endpoints; eight absent endpoints are excluded based on the outcome."
            ),
            "trajectory_scope": (
                "The sampled rho is not certified along simulated trajectories. Some observed "
                "gradient norms fall below or exceed the audit's dimensionless-radius interval; "
                "the analytic value at exactly zero does not certify the unsampled interval."
            ),
        },
        "headline": {
            "primary_gain_matched_comparisons": primary_comparisons,
            "exploratory_normalized_deficit_vs_normalized_upper_endpoint": {
                "n": len(exploratory_x),
                "excluded_absent_control_endpoint": len(unrepaired_rows) - len(exploratory_x),
                "statistic": _safe_float(normalized_correlation.statistic),
                "pvalue_descriptive_only": _safe_float(normalized_correlation.pvalue),
            },
            "sampled_repair_trajectory_domain_check": {
                "repaired_configuration_count": len(repaired_rows),
                "audit_dimensionless_radius_min": audited_dimensionless_radius_min,
                "audit_dimensionless_radius_max": audited_dimensionless_radius_max,
                "configurations_below_audit_radius": len(repaired_rows_below_audit_radius),
                "configurations_above_audit_radius": len(repaired_rows_above_audit_radius),
                "minimum_positive_gradient_norm": min(
                    row["minimum_positive_gradient_norm_observed"] for row in repaired_rows
                ),
                "maximum_observed_gradient_norm": max(
                    row["maximum_gradient_norm_observed"] for row in repaired_rows
                ),
                "certified_along_trajectories": False,
            },
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
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
                Path("experiments/quadratics/run_lr_sweep.py"),
                Path("src/passive_muon/quadratic_experiment.py"),
                Path("src/passive_muon/spectral_analysis.py"),
                Path("src/passive_muon/specs.py"),
            ]
        ),
        "sensitivity_headlines": sensitivity_headlines,
        "rows": summary_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "quadratic_lr_sweep.json"
    summary_csv = args.output_dir / "quadratic_lr_sweep.csv"
    curve_csv = args.output_dir / "quadratic_lr_curves.csv"
    sensitivity_csv = args.output_dir / "quadratic_lr_sensitivity.csv"
    json_path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with curve_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(learning_rate_rows[0]))
        writer.writeheader()
        writer.writerows(learning_rate_rows)
    with sensitivity_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    print(
        json.dumps(
            {
                "json": str(json_path),
                "summary_csv": str(summary_csv),
                "curve_csv": str(curve_csv),
                "sensitivity_csv": str(sensitivity_csv),
                "headline": payload["headline"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
