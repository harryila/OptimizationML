#!/usr/bin/env python3
"""Generate the exact P21 stored outer-loop composition certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

from passive_muon.certified_outer_loop_composition import (
    CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION as RUNTIME_SCHEMA_VERSION,
)
from passive_muon.certified_outer_loop_composition import (
    LOCKED_CANDIDATE_STEPS,
    LOCKED_OPERATOR_OUTPUT_MAX_ABS,
    LOCKED_TOTAL_STEP_MAX_ABS,
    MAXIMUM_LEARNING_RATE_FP32_BITS,
    MAXIMUM_LEARNING_RATE_FP32_EXACT,
    PRIMARY_LEARNING_RATE_FP32_BITS,
    PRIMARY_LEARNING_RATE_FP32_EXACT,
)
from passive_muon.certified_outer_loop_composition_certificate import (
    CERTIFIED_OUTER_LOOP_COMPOSITION_SCHEMA_VERSION as CERTIFICATE_SCHEMA_VERSION,
)
from passive_muon.certified_outer_loop_composition_certificate import (
    PORT_VARIABLE_ORDER,
    audit_stored_signal_port_certificate,
    centered_decay_corollary,
    exact_certificate_checks,
    zero_centered_decay_counterexample,
)
from passive_muon.certified_outer_loop_roundoff import (
    BOUNDED_DECAY_PORT_BUDGET,
    CERTIFIED_OUTER_LOOP_ROUNDOFF_SCHEMA_VERSION,
    LOCKED_EXTERNAL_GRADIENT_INTERCEPT,
    LOCKED_EXTERNAL_GRADIENT_SLOPE,
    LOCKED_MODEL_RECONSTRUCTION_INTERCEPT,
    LOCKED_MODEL_RECONSTRUCTION_SLOPE,
    LOCKED_OUTPUT_MAX_ABS,
    LOCKED_ROBUST_GRADIENT_INTERCEPT,
    LOCKED_ROBUST_GRADIENT_SLOPE,
    LOCKED_STORAGE_RADIUS,
    LOCKED_WEIGHT_DECAY_DISPLACEMENT,
    LOCKED_WEIGHT_DECAY_ROUNDED_STEP,
    LOCKED_YOUNG_PARAMETERS,
    REPORT_GRID_DENOMINATOR,
    ROBUST_GRADIENT_SOURCE_BUDGET,
    SQRT_ENCLOSURE_DENOMINATOR,
    ZERO_DECAY_PORT_BUDGET,
    ZERO_GRADIENT_SOURCE_BUDGET,
    P21RoundoffEvaluation,
    evaluate_p21_roundoff,
    exact_roundoff_checks,
)
from passive_muon.certified_outer_loop_roundoff import (
    LOCKED_TOTAL_STEP_MAX_ABS as ROUNDOFF_TOTAL_STEP_MAX_ABS,
)
from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_BITS,
    BETA_FP32_EXACT,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LOCKED_MASTER_HIGH_MAX_ABS,
    LOCKED_MASTER_LOW_MAX_ABS,
    LOCKED_MASTER_MIDDLE_MAX_ABS,
    ONE_MINUS_BETA_FP32_BITS,
    ONE_MINUS_BETA_FP32_EXACT,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_REDUCTION_BLOCK_SIZE,
    LOCKED_REPRESENTATIVE_SHAPES,
    LOCKED_SCALABLE_SHIELD_BACKEND,
)
from passive_muon.sector_projected_resolvent_certificate import LOCKED_EPSILON
from passive_muon.sector_shielded_inexact_resolvent_certificate import (
    FASTER_RATE_POINT,
    LOCKED_BETA,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    MAXIMUM_STEP_POINT,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/certified_outer_loop_composition_certificate.json"
SCHEMA_VERSION = "passive-muon-certified-outer-loop-composition-artifact-v1"

P20_ARTIFACT_PATH = "results/summaries/scalable_sector_shield_certificate.json"
P20_ARTIFACT_SHA256 = "6bca86676eb7164a617a48f34228c40355dbc682b86b4e64ddf3cd54b8d15951"
P20_SOURCE_COMMIT = "d0a9e92403a29638018b13579c2a612965dbc5b4"
P20_ARTIFACT_COMMIT = "8a0586dcf7f7d6ce933d84f4ad7a2756764f02d0"
P20_CHECKPOINT_TAG = "p20-scalable-mixed-precision-sector-shield-final-checkpoint"
P20_CHECKPOINT_TAG_OBJECT = "4cfd127117d509ae0dcd0cc26d74e9cd963db50a"

SOURCE_PATHS = (
    "scripts/certify_outer_loop_composition.py",
    "scripts/reconstruct_outer_loop_composition.py",
    "src/passive_muon/certified_outer_loop_composition.py",
    "src/passive_muon/certified_outer_loop_composition_certificate.py",
    "src/passive_muon/certified_outer_loop_roundoff.py",
    "src/passive_muon/finite_precision_outer_loop.py",
    "src/passive_muon/scalable_sector_shield.py",
    "src/passive_muon/sector_shielded_inexact_resolvent_certificate.py",
    "src/passive_muon/sector_projected_resolvent_certificate.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "src/passive_muon/p21_shadow_trace.py",
    "experiments/training/p21_shadow_trace_protocol.json",
    "experiments/training/run_p21_synthetic_shadow_trace.py",
    "theory/certified_outer_loop_composition.md",
    "theory/p21_shadow_trace_protocol.md",
    "tests/test_certified_outer_loop_composition.py",
    "tests/test_certified_outer_loop_composition_certificate.py",
    "tests/test_certified_outer_loop_roundoff.py",
    "tests/test_certified_outer_loop_composition_cli.py",
    "tests/test_certified_outer_loop_composition_reconstruction.py",
    "tests/test_p21_shadow_trace.py",
    P20_ARTIFACT_PATH,
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 60) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _matrix(matrix: tuple[tuple[Fraction, ...], ...]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    missing = [path for path in SOURCE_PATHS if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"P21 exact source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p20() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P20_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P20_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P20_ARTIFACT_COMMIT,
        "checkpoint_tag": P20_CHECKPOINT_TAG,
        "checkpoint_tag_object": P20_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p20_artifact_hash_matches": observed == P20_ARTIFACT_SHA256,
        "p20_historical_artifact_blob_matches": (
            _git_blob_sha256(P20_ARTIFACT_COMMIT, P20_ARTIFACT_PATH) == P20_ARTIFACT_SHA256
        ),
        "p20_source_commit_matches": payload["git"]["sha"] == P20_SOURCE_COMMIT,
        "p20_source_was_clean": payload["git"]["dirty"] is False,
        "p20_checkpoint_tag_matches": (
            _git_output("rev-parse", P20_CHECKPOINT_TAG) == P20_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P20_CHECKPOINT_TAG}^{{}}") == P20_ARTIFACT_COMMIT
        ),
        "p20_exact_checks_pass": payload.get("all_exact_checks_passed") is True,
        "p20_sector_matches": (
            payload["reconstruction_fields"]["sector"]["lower"] == str(LOCKED_SECTOR_LOWER)
            and payload["reconstruction_fields"]["sector"]["upper"] == str(LOCKED_SECTOR_UPPER)
            and payload["reconstruction_fields"]["sector"]["center"] == str(LOCKED_SECTOR_CENTER)
            and payload["reconstruction_fields"]["sector"]["radius"] == str(LOCKED_SECTOR_RADIUS)
        ),
    }
    return record, checks


def _envelope(envelope: Any) -> dict[str, str]:
    return {"slope": str(envelope.slope), "intercept": str(envelope.intercept)}


def _core_fields(point: Any) -> dict[str, object]:
    audit = audit_stored_signal_port_certificate(point)
    certificate = audit.certificate
    pl = certificate.pl_certificate
    function_coefficients = (
        pl.lambda_interpolation_12 - pl.lambda_interpolation_21,
        -pl.lambda_interpolation_12
        + pl.lambda_interpolation_21
        - 2 * pl.condition_ratio * pl.lambda_pl_next,
    )
    expected_coefficients = (pl.function_storage * pl.tau**2, -pl.function_storage)
    return {
        "name": point.name,
        "learning_rate": str(point.learning_rate),
        "tau": str(point.tau),
        "rate": str(point.rate_squared),
        "pl_constant": str(pl.pl_constant),
        "smoothness": str(pl.smoothness),
        "beta": str(pl.beta),
        "sector_lower": str(LOCKED_SECTOR_LOWER),
        "sector_upper": str(LOCKED_SECTOR_UPPER),
        "center_gain": str(pl.center_gain),
        "residual_lipschitz": str(pl.residual_lipschitz),
        "dimensionless_step": str(pl.dimensionless_step),
        "residual_ratio": str(pl.residual_ratio),
        "storage": _matrix(pl.storage),
        "function_storage": str(pl.function_storage),
        "interpolation_reverse_weight": str(pl.interpolation_reverse_weight),
        "lambda_interpolation_12": str(pl.lambda_interpolation_12),
        "lambda_interpolation_21": str(pl.lambda_interpolation_21),
        "lambda_pl_next": str(pl.lambda_pl_next),
        "lambda_residual_norm": str(pl.lambda_residual_norm),
        "port_gains": [str(value) for value in certificate.port_gains],
        "raw_lmi": _matrix(certificate.raw_lmi),
        "penalized_lmi": _matrix(certificate.lmi),
        "storage_leading_minors": [str(value) for value in audit.storage_leading_minors],
        "negative_lmi_leading_minors": [str(value) for value in audit.negative_lmi_leading_minors],
        "zero_port_lmi": _matrix(audit.recovered_zero_port_lmi),
        "frozen_p18_lmi": _matrix(audit.frozen_zero_port_lmi),
        "function_coefficients": [str(value) for value in function_coefficients],
        "expected_function_coefficients": [str(value) for value in expected_coefficients],
        "checks": {
            "storage_positive_definite": audit.storage_positive_definite,
            "lmi_negative_definite": audit.lmi_negative_definite,
            "zero_port_recovers_p18_exactly": audit.zero_port_recovers_p18_exactly,
            "function_values_cancel": audit.function_values_cancel,
        },
        "certified": audit.certified,
    }


def _evaluation_fields(evaluation: P21RoundoffEvaluation, *, profile: str) -> dict[str, object]:
    reduction = evaluation.reduction
    absorption = evaluation.absorption
    guard = evaluation.guard
    normalized_names = ("momentum_roundoff", "signal_roundoff", "output_equivalent_error")
    return {
        "profile": profile,
        "point": reduction.point.name,
        "shape": list(reduction.shape),
        "shield_shape": list(reduction.shield_shape),
        "transposed_for_shield": reduction.transposed_for_shield,
        "gradient_budget": {
            "name": reduction.gradient_budget.name,
            "slope": str(reduction.gradient_budget.slope),
            "intercept": str(reduction.gradient_budget.intercept),
        },
        "decay_budget": {
            "name": reduction.decay_budget.name,
            "logical_displacement": str(reduction.decay_budget.logical_displacement),
            "rounded_step": str(reduction.decay_budget.rounded_step),
        },
        "sqrt_entries_upper": reduction.sqrt_entries_upper,
        "eta_fp32": str(reduction.eta_fp32),
        "eta_representation_error": str(reduction.eta_representation_error),
        "physical_envelopes": {
            "momentum_roundoff": _envelope(reduction.momentum_roundoff),
            "signal_roundoff": _envelope(reduction.signal_roundoff),
            "stored_signal": _envelope(reduction.stored_signal),
            "shield_output": _envelope(reduction.shield_output),
            "master_roundoff": _envelope(reduction.master_roundoff),
        },
        "dead_zone_parameter_error": str(reduction.dead_zone_parameter_error),
        "normalized_ports": {
            name: _envelope(envelope)
            for name, envelope in zip(normalized_names, reduction.normalized_ports, strict=True)
        },
        "absorption": {
            "base_rate": str(absorption.base_rate),
            "absorbed_rate": str(absorption.absorbed_rate),
            "constant_forcing": str(absorption.constant_forcing),
            "storage_radius": str(absorption.storage_radius),
            "objective_gap_ultimate_bound": str(absorption.objective_gap_ultimate_bound),
            "contractive": absorption.contractive,
            "forward_invariant": absorption.forward_invariant,
        },
        "reported": {
            "rate_upper": str(evaluation.reported_rate_upper),
            "forcing_upper": str(evaluation.reported_forcing_upper),
            "objective_gap_upper": str(evaluation.reported_objective_gap_upper),
        },
        "guard": {
            "stored_signal_at_one": str(guard.stored_signal_at_one),
            "shield_output_at_one": str(guard.shield_output_at_one),
            "rounded_operator_step_at_one": str(guard.rounded_operator_step_at_one),
            "total_step_at_one": str(guard.total_step_at_one),
            "output_max_abs": str(guard.output_max_abs),
            "total_step_max_abs": str(guard.total_step_max_abs),
            "master_high_max_abs": str(guard.master_high_max_abs),
            "master_middle_max_abs": str(guard.master_middle_max_abs),
            "master_low_max_abs": str(guard.master_low_max_abs),
            "pending_after_operator_strict_upper": str(guard.pending_after_operator_strict_upper),
            "pending_after_decay_strict_upper": str(guard.pending_after_decay_strict_upper),
            "middle_candidate_strict_upper": str(guard.middle_candidate_strict_upper),
            "high_word_guard_is_conditional": guard.high_word_guard_is_conditional,
            "lower_word_guard_checks": guard.lower_word_guard_checks,
            "checks": guard.checks,
        },
        "checks": evaluation.checks,
        "certified": evaluation.certified,
    }


def _all_evaluations() -> list[dict[str, object]]:
    profiles = (
        ("zero_external_error", ZERO_GRADIENT_SOURCE_BUDGET, ZERO_DECAY_PORT_BUDGET),
        ("joint_gradient_error", ROBUST_GRADIENT_SOURCE_BUDGET, ZERO_DECAY_PORT_BUDGET),
        (
            "joint_gradient_plus_bounded_decay",
            ROBUST_GRADIENT_SOURCE_BUDGET,
            BOUNDED_DECAY_PORT_BUDGET,
        ),
    )
    return [
        _evaluation_fields(
            evaluate_p21_roundoff(
                shape,
                point,
                gradient_budget=budget,
                decay_budget=decay_budget,
            ),
            profile=name,
        )
        for name, budget, decay_budget in profiles
        for point in (FASTER_RATE_POINT, MAXIMUM_STEP_POINT)
        for shape in LOCKED_REPRESENTATIVE_SHAPES
    ]


def reconstruction_fields() -> dict[str, object]:
    """Fields rebuilt independently by the standard-library P21 replay."""

    coefficients = JORDAN_QUINTIC.fractions()
    centered = centered_decay_corollary(
        audit_stored_signal_port_certificate(FASTER_RATE_POINT).certificate,
        weight_decay=Fraction(1, 1_000_000),
        strong_convexity=Fraction(1),
    )
    control = zero_centered_decay_counterexample()
    return {
        "variable_order": list(PORT_VARIABLE_ORDER),
        "stored_graph": {
            "normalized_momentum": "z=m/L",
            "normalized_gradient": "u=grad f(W)/L",
            "momentum": "z_next=beta*z+(1-beta)*u+a_m",
            "stored_signal": "p=beta^2*z+(1-beta^2)*u+beta*a_m+a_s",
            "sector_output": "U/L=gamma*p+K_T*v; ||v||_F<=||p||_F",
            "parameter_step": "Delta W=-eta*gamma*L*(p+(K_T/gamma)*v+h)",
            "no_incremental_comparison": True,
        },
        "operator_contract": {
            "objective_domain": (
                "differentiable globally 10-smooth objectives satisfying the global PL "
                "inequality with constant 1; nonconvex objectives and nonunique minimizers "
                "are included"
            ),
            "matrix_domain": (
                "the seven P20 representative Transformer shapes and their nonduplicate "
                "transposes; transposition into P20 orientation is a Frobenius isometry"
            ),
            "candidate_normalization": "M/(||M||_F+epsilon)",
            "epsilon": str(LOCKED_EPSILON),
            "polynomial": "q(x)=a*x+b*x^3+c*x^5",
            "polynomial_coefficients": {
                "a": str(coefficients[0]),
                "b": str(coefficients[1]),
                "c": str(coefficients[2]),
            },
            "newton_schulz_stages": LOCKED_CANDIDATE_STEPS,
            "upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "upstream_muon_sha256": PINNED_MUON_PY_SHA256,
            "aspect_scaling_placement": "after BF16 candidate and before P20 shield",
            "shield_sector": {
                "lower": str(LOCKED_SECTOR_LOWER),
                "upper": str(LOCKED_SECTOR_UPPER),
                "center": str(LOCKED_SECTOR_CENTER),
                "radius": str(LOCKED_SECTOR_RADIUS),
            },
            "all_subnormal_wrapper": (
                "return zero; charge the distance to the conceptual stored-signal/2 point "
                "as an absolute parameter-update port"
            ),
        },
        "arithmetic_contract": {
            "runtime_schema": RUNTIME_SCHEMA_VERSION,
            "certificate_schema": CERTIFICATE_SCHEMA_VERSION,
            "roundoff_schema": CERTIFIED_OUTER_LOOP_ROUNDOFF_SCHEMA_VERSION,
            "backend": "torch-cpu-eager proof-reference composition",
            "shield_backend": LOCKED_SCALABLE_SHIELD_BACKEND,
            "chain": (
                "stored FP32 gradient -> FP32 EMA/Nesterov with reused bg -> five-stage "
                "BF16 candidate -> aspect scale -> P20 FP32 shield -> compensated three-word "
                "FP32 master update"
            ),
            "rounding": "IEEE-754 round-to-nearest ties-to-even at every named operation",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "fma_allowed": False,
            "reassociation_allowed": False,
            "candidate_input_dtypes": ["binary16-bfloat", "binary32"],
            "shield_output_dtype": "binary32",
            "norm_reduction_block_entries": LOCKED_REDUCTION_BLOCK_SIZE,
            "fp32_unit_roundoff": str(FP32_UNIT_ROUNDOFF),
            "fp32_half_min_subnormal": str(FP32_HALF_MIN_SUBNORMAL),
            "beta": str(LOCKED_BETA),
            "beta_fp32": str(BETA_FP32_EXACT),
            "beta_fp32_bits": f"0x{BETA_FP32_BITS:08x}",
            "one_minus_beta_fp32": str(ONE_MINUS_BETA_FP32_EXACT),
            "one_minus_beta_fp32_bits": f"0x{ONE_MINUS_BETA_FP32_BITS:08x}",
            "primary_eta_fp32": str(PRIMARY_LEARNING_RATE_FP32_EXACT),
            "primary_eta_fp32_bits": f"0x{PRIMARY_LEARNING_RATE_FP32_BITS:08x}",
            "maximum_eta_fp32": str(MAXIMUM_LEARNING_RATE_FP32_EXACT),
            "maximum_eta_fp32_bits": f"0x{MAXIMUM_LEARNING_RATE_FP32_BITS:08x}",
            "operator_output_max_abs": str(Fraction.from_float(LOCKED_OPERATOR_OUTPUT_MAX_ABS)),
            "total_step_max_abs": str(Fraction.from_float(LOCKED_TOTAL_STEP_MAX_ABS)),
            "master_high_max_abs": str(LOCKED_MASTER_HIGH_MAX_ABS),
            "master_middle_max_abs": str(LOCKED_MASTER_MIDDLE_MAX_ABS),
            "master_low_max_abs": str(LOCKED_MASTER_LOW_MAX_ABS),
        },
        "roundoff_constants": {
            "sqrt_enclosure_denominator": SQRT_ENCLOSURE_DENOMINATOR,
            "report_grid_denominator": REPORT_GRID_DENOMINATOR,
            "storage_radius": str(LOCKED_STORAGE_RADIUS),
            "young_parameters": [str(value) for value in LOCKED_YOUNG_PARAMETERS],
            "robust_gradient_slope": str(LOCKED_ROBUST_GRADIENT_SLOPE),
            "robust_gradient_intercept": str(LOCKED_ROBUST_GRADIENT_INTERCEPT),
            "external_gradient_slope": str(LOCKED_EXTERNAL_GRADIENT_SLOPE),
            "external_gradient_intercept": str(LOCKED_EXTERNAL_GRADIENT_INTERCEPT),
            "model_reconstruction_slope": str(LOCKED_MODEL_RECONSTRUCTION_SLOPE),
            "model_reconstruction_intercept": str(LOCKED_MODEL_RECONSTRUCTION_INTERCEPT),
            "gradient_source_composition": (
                "zeta=external_gradient_error+gradient_reconstruction_error; "
                "global L=10 maps ||H-W||<=a_W*sqrt(V)+b_W into "
                "10*a_W*sqrt(V)+10*b_W"
            ),
            "bounded_decay_displacement": str(LOCKED_WEIGHT_DECAY_DISPLACEMENT),
            "bounded_decay_rounded_step": str(LOCKED_WEIGHT_DECAY_ROUNDED_STEP),
            "output_guard": str(LOCKED_OUTPUT_MAX_ABS),
            "total_step_guard": str(ROUNDOFF_TOTAL_STEP_MAX_ABS),
        },
        "core_certificates": [
            _core_fields(FASTER_RATE_POINT),
            _core_fields(MAXIMUM_STEP_POINT),
        ],
        "shape_profiles": _all_evaluations(),
        "weight_decay_scope": {
            "main_pl_theorem_weight_decay": "zero",
            "bounded_port": {
                "name": BOUNDED_DECAY_PORT_BUDGET.name,
                "logical_displacement": str(LOCKED_WEIGHT_DECAY_DISPLACEMENT),
                "rounded_step": str(LOCKED_WEIGHT_DECAY_ROUNDED_STEP),
                "status": "conditional runtime premise; not inferred from wd or master range",
            },
            "centered_strong_convexity_corollary": {
                "weight_decay": str(centered.weight_decay),
                "strong_convexity": str(centered.strong_convexity),
                "normalized_output_squared_slope": str(centered.normalized_output_squared_slope),
                "certified_rate": str(centered.certified_rate),
                "contractive": centered.contractive,
            },
            "ordinary_decay_nonzero_minimizer_control": {
                "learning_rate": str(control.learning_rate),
                "weight_decay": str(control.weight_decay),
                "minimizer": str(control.minimizer),
                "next_iterate": str(control.next_iterate),
                "displacement": str(control.displacement),
                "next_objective_gap": str(control.next_objective_gap),
                "original_minimizer_is_not_an_equilibrium": (
                    control.original_minimizer_is_not_an_equilibrium
                ),
            },
        },
    }


def build_payload() -> dict[str, object]:
    prior, prior_checks = _prior_p20()
    core_checks = exact_certificate_checks()
    roundoff_checks = exact_roundoff_checks()
    checks = {**prior_checks, **core_checks, **roundoff_checks}
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": (
                "exact stored-signal port LMI plus exact rational finite-precision "
                "outer-shell composition"
            ),
            "guarantee": (
                "for each displayed pathwise gradient/model error and update-port budget, "
                "the reported affine storage inequality, invariant unit-storage domain, and "
                "objective-gap ultimate bound hold at eta=1/120 and eta=1/83"
            ),
            "sector_location": (
                "P20 is applied directly to the stored FP32 Nesterov signal; no incremental "
                "Lipschitz property of P20 is assumed"
            ),
            "weight_decay": (
                "zero in the main PL theorem; nonzero ordinary decay is a bounded update port; "
                "a zero-intercept rate corollary is limited to decay centred at a strongly "
                "convex minimizer (ordinary zero-centred decay only when that minimizer is zero)"
            ),
            "not_claimed": [
                (
                    "literal GPU, tensor-core, distributed, FMA-contracted, reassociated, "
                    "FTZ, or DAZ execution"
                ),
                "global PL for a neural-network training loss",
                "stochastic convergence without the displayed pathwise gradient-source budget",
                "unshielded upstream Muon stability",
                "an incremental Lipschitz or arbitrary-pair contraction theorem for P20",
                "model-forward parity when only the high master word is consumed",
                "an unconditional invariant for the FP32 high master word",
                "a unique minimizer or full-parameter ISS on a nonunique PL minimizer set",
            ],
        },
        "prior_p20": prior,
        "reconstruction_fields": reconstruction_fields(),
        "checks": checks,
        "all_exact_checks_passed": all(checks.values()),
        "source_snapshot": _source_snapshot(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "seed": "none; deterministic exact rational replay",
        },
        "git": _git_state(),
    }


def main() -> None:
    args = parse_args()
    payload = build_payload()
    if not payload["all_exact_checks_passed"]:
        failed = [name for name, passed in payload["checks"].items() if not passed]
        raise SystemExit(f"P21 outer-loop composition certificate failed: {failed}")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
