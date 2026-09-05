#!/usr/bin/env python3
"""Generate the exact P20 scalable mixed-precision shield certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

from passive_muon.scalable_sector_shield_certificate import (
    BF16_MIN_SUBNORMAL,
    FP32_HALF_MIN_SUBNORMAL,
    FP32_MAX_SUBNORMAL,
    FP32_MIN_NORMAL,
    FP32_MIN_SUBNORMAL,
    FP32_UNIT_ROUNDOFF,
    LOCKED_ACCEPTANCE_BUFFER,
    LOCKED_ACCEPTANCE_COEFFICIENT,
    LOCKED_CLIP_COEFFICIENT,
    LOCKED_FASTER_RATE,
    LOCKED_FASTER_RATE_STEP,
    LOCKED_MAXIMUM_STEP,
    LOCKED_MAXIMUM_STEP_RATE,
    LOCKED_REDUCTION_BLOCK_ENTRIES,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    RADIUS_GRID_DENOMINATOR,
    SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION,
    SQRT_ENCLOSURE_DENOMINATOR,
    ScalableSectorShieldAudit,
    diagnostic_sector_shield_audits,
    exact_contract_checks,
    representative_sector_shield_audits,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/scalable_sector_shield_certificate.json"
SCHEMA_VERSION = SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION

P19_ARTIFACT_PATH = "results/summaries/sector_shielded_inexact_resolvent_certificate.json"
P19_ARTIFACT_SHA256 = "8ed26627d0ce1e5389aacc954834e408e8598348b8b011c76f0b61fe768360f8"
P19_SOURCE_COMMIT = "45fd279362792a0f5cae7559d6d70fe651586580"
P19_ARTIFACT_COMMIT = "154449ed7418ed7264a7442bfb7df92f1366a4b6"
P19_CHECKPOINT_TAG = "p19-sector-shielded-inexact-resolvent-checkpoint"
P19_CHECKPOINT_TAG_OBJECT = "3842779cdb48eb798fb8b923f7ef64630636ea33"

P10_ARTIFACT_PATH = "results/summaries/outer_loop_roundoff_certificate.json"
P10_ARTIFACT_SHA256 = "a36dcf8ab10c6c45b9a31d366cc2c67def0eaddb5be1e30214222e5eb83a84ea"
P11_ARTIFACT_PATH = "results/summaries/implementation_margin_certificate.json"
P11_ARTIFACT_SHA256 = "ad74050d66f711d2bb8e3e37f80fc29f4d6162419a2f5224931e1ca7fd2e536f"

SOURCE_PATHS = (
    "scripts/certify_scalable_sector_shield.py",
    "scripts/reconstruct_scalable_sector_shield.py",
    "src/passive_muon/scalable_sector_shield_certificate.py",
    "src/passive_muon/scalable_sector_shield.py",
    "theory/scalable_mixed_precision_sector_shield.md",
    "experiments/mixed_precision/run_p20_scalable_sector_shield_study.py",
    "tests/test_scalable_sector_shield_certificate.py",
    "tests/test_scalable_sector_shield_reconstruction.py",
    "tests/test_scalable_sector_shield.py",
    "tests/test_p20_scalable_sector_shield_study.py",
    "src/passive_muon/scalable_mixed_precision_certificate.py",
    P10_ARTIFACT_PATH,
    P11_ARTIFACT_PATH,
    P19_ARTIFACT_PATH,
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
        raise RuntimeError(f"P20 exact source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p19() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P19_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = {
        "artifact_path": P19_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P19_ARTIFACT_COMMIT,
        "checkpoint_tag": P19_CHECKPOINT_TAG,
        "checkpoint_tag_object": P19_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p19_artifact_hash_matches": observed == P19_ARTIFACT_SHA256,
        "p19_historical_artifact_blob_matches": (
            _git_blob_sha256(P19_ARTIFACT_COMMIT, P19_ARTIFACT_PATH) == P19_ARTIFACT_SHA256
        ),
        "p19_source_commit_matches": payload["git"]["sha"] == P19_SOURCE_COMMIT,
        "p19_source_was_clean": payload["git"]["dirty"] is False,
        "p19_checkpoint_tag_matches": (
            _git_output("rev-parse", P19_CHECKPOINT_TAG) == P19_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P19_CHECKPOINT_TAG}^{{}}") == P19_ARTIFACT_COMMIT
        ),
        "p19_sector_matches": (
            payload["sector_shield"]["lower"]["exact"] == str(LOCKED_SECTOR_LOWER)
            and payload["sector_shield"]["upper"]["exact"] == str(LOCKED_SECTOR_UPPER)
            and payload["sector_shield"]["center"]["exact"] == str(LOCKED_SECTOR_CENTER)
            and payload["sector_shield"]["radius"]["exact"] == str(LOCKED_SECTOR_RADIUS)
        ),
    }
    return record, checks


def _prior_rounding_ledgers() -> tuple[dict[str, object], dict[str, bool]]:
    """Lock the P10/P11 artifacts whose rounding convention P20 reuses."""

    records: dict[str, object] = {}
    checks: dict[str, bool] = {}
    for name, path_text, expected_hash in (
        ("p10_outer_loop", P10_ARTIFACT_PATH, P10_ARTIFACT_SHA256),
        ("p11_implementation_margin", P11_ARTIFACT_PATH, P11_ARTIFACT_SHA256),
    ):
        path = ROOT / path_text
        observed = _sha256(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records[name] = {
            "artifact_path": path_text,
            "artifact_sha256": observed,
            "source_commit": payload.get("git", {}).get("sha", "unavailable"),
        }
        checks[f"{name}_artifact_hash_matches"] = observed == expected_hash
        checks[f"{name}_exact_checks_pass"] = (
            payload.get("audit", {}).get("all_exact_checks_passed") is True
        )
    records["reuse_scope"] = (
        "P20 reuses the P10/P11 FP32 relative-plus-absolute-crumb ledger convention; "
        "it does not yet compose their outer-loop disturbance ports with P19/P20"
    )
    return records, checks


def _shape_exact_fields(audit: ScalableSectorShieldAudit) -> dict[str, object]:
    norm = audit.norm
    return {
        "shape": [audit.shape.rows, audit.shape.columns],
        "entries": audit.shape.entries,
        "root_entries_upper": norm.root_entries_upper,
        "balanced_norm": {
            "pairwise_path_length": norm.pairwise_path_length,
            "reduction_block_entries": norm.reduction_block_entries,
            "reduction_block_count": norm.reduction_block_count,
            "pairwise_gamma": str(norm.pairwise_gamma),
            "pairwise_crumb": str(norm.pairwise_crumb),
            "division_relative_error": str(norm.division_relative_error),
            "squared_norm_lower": str(norm.squared_norm_lower),
            "squared_norm_upper": str(norm.squared_norm_upper),
            "sqrt_lower": str(norm.sqrt_lower),
            "sqrt_upper": str(norm.sqrt_upper),
            "lower_factor": str(norm.lower_factor),
            "upper_factor": str(norm.upper_factor),
        },
        "margin": {
            "normal_anchor": str(audit.normal_anchor),
            "acceptance_coefficient": str(audit.acceptance_coefficient),
            "displacement_signal_slope": str(audit.displacement_signal_slope),
            "displacement_crumb": str(audit.displacement_crumb),
            "displacement_crumb_relative": str(audit.displacement_crumb_relative),
            "accepted_candidate_radius": str(audit.accepted_candidate_radius),
            "clipped_candidate_radius": str(audit.clipped_candidate_radius),
            "clip_only_inward_margin": str(audit.clip_only_inward_margin),
            "rounded_half_radius": str(audit.rounded_half_radius),
            "certified_inward_radius_raw": str(audit.certified_inward_radius_raw),
            "certified_inward_radius": str(audit.certified_inward_radius),
            "inward_margin": str(audit.inward_margin),
        },
        "subnormal_guard": {
            "all_subnormal_input_radius": str(audit.all_subnormal_input_radius),
            "zero_instead_of_half_output_error": str(audit.zero_instead_of_half_output_error),
        },
        "checks": audit.checks,
        "certified": audit.certified,
    }


def reconstruction_fields() -> dict[str, object]:
    """Return fields rebuilt independently by the standard-library replay."""

    audits = representative_sector_shield_audits()
    return {
        "sector": {
            "lower": str(LOCKED_SECTOR_LOWER),
            "upper": str(LOCKED_SECTOR_UPPER),
            "center": str(LOCKED_SECTOR_CENTER),
            "radius": str(LOCKED_SECTOR_RADIUS),
            "acceptance_buffer": str(LOCKED_ACCEPTANCE_BUFFER),
            "acceptance_coefficient": str(LOCKED_ACCEPTANCE_COEFFICIENT),
            "clip_coefficient": str(LOCKED_CLIP_COEFFICIENT),
        },
        "arithmetic_constants": {
            "fp32_unit_roundoff": str(FP32_UNIT_ROUNDOFF),
            "fp32_half_min_subnormal": str(FP32_HALF_MIN_SUBNORMAL),
            "fp32_min_subnormal": str(FP32_MIN_SUBNORMAL),
            "fp32_min_normal": str(FP32_MIN_NORMAL),
            "fp32_max_subnormal": str(FP32_MAX_SUBNORMAL),
            "bf16_min_subnormal": str(BF16_MIN_SUBNORMAL),
            "sqrt_enclosure_denominator": SQRT_ENCLOSURE_DENOMINATOR,
            "radius_grid_denominator": RADIUS_GRID_DENOMINATOR,
            "reduction_block_entries": LOCKED_REDUCTION_BLOCK_ENTRIES,
        },
        "formulae": {
            "norm_vector": "v=x/maxabs(x); z=fl32(v)",
            "norm_reduction": (
                "qhat=balanced_pairwise_fl32(sum_i fl32(z_i*z_i)); "
                "n32=fl32(sqrt(qhat)); Nhat=float64(maxabs(x))*float64(n32)"
            ),
            "candidate_displacement": ("Dhat=fl32(C-fl32((1143/2048)*S))"),
            "acceptance": ("Nhat(Dhat)<=nextafter(fl64((891/2048)*Nhat(S)),-infinity)"),
            "clip_scalar": (
                "ratio64=down64(Nhat(S)/Nhat(Dhat)); "
                "alpha64=down64((890/2048)*ratio64); alpha32=down32(alpha64)"
            ),
            "clip_vectors": ("q=fl32(alpha32*Dhat); U=fl32(fl32((1143/2048)*S)+q)"),
            "accepted_radius": (
                "((891/2048)*U/ell+(1143/2048)*u*(1+u)+h*tau*(2+u)/sigma_min)/(1-u)"
            ),
            "clipped_radius": (
                "(890/2048)*(U/ell)*(1+u)^2+(1143/2048)*u*(2+u)+h*tau*(3+2u)/sigma_min"
            ),
            "fallback_radius": "abs(1143/2048-1/2)+h*tau/sigma_min",
            "delta": ("893/2048-ceil_2^-60(max(accepted_radius,clipped_radius,fallback_radius))"),
        },
        "shapes": [_shape_exact_fields(audit) for audit in audits],
        "diagnostic_shapes": [
            _shape_exact_fields(audit) for audit in diagnostic_sector_shield_audits()
        ],
        "operating_points": {
            "trajectory_premises": (
                "every shield call succeeds; stored S is the abstract operator-port signal; "
                "the remaining EMA/Nesterov interconnection uses exact real arithmetic"
            ),
            "maximum_step": {
                "learning_rate": str(LOCKED_MAXIMUM_STEP),
                "rate": str(LOCKED_MAXIMUM_STEP_RATE),
            },
            "faster_rate": {
                "learning_rate": str(LOCKED_FASTER_RATE_STEP),
                "rate": str(LOCKED_FASTER_RATE),
            },
        },
    }


def build_payload() -> dict[str, object]:
    prior, prior_checks = _prior_p19()
    ledgers, ledger_checks = _prior_rounding_ledgers()
    audits = representative_sector_shield_audits()
    contract_checks = exact_contract_checks()
    checks = {**contract_checks, **prior_checks, **ledger_checks}
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": (
                "shape-parameterized static binary32 inward-sector containment certificate"
            ),
            "candidate_scope": (
                "arbitrary stored BF16/FP32 candidates; finite values widen exactly and use "
                "the screen/clip graph, while nonfinite candidates take the guarded fallback"
            ),
            "signal_scope": (
                "finite stored BF16/FP32 signals; normal-anchor theorem or exact-halving fallback"
            ),
            "returned_dtype": "binary32",
            "not_claimed": [
                "literal unspecified tensor-core, BLAS, compiler-reassociated, or FTZ execution",
                "a BF16 final-output theorem",
                "P19 metric-projection nonexpansiveness",
                "global identity on exact P18 outputs",
                "an objective or Lyapunov neighborhood for rejected all-subnormal FP32 signals",
                "literal unshielded upstream Muon stability",
            ],
        },
        "arithmetic_contract": {
            "input_order": (
                "validate dtype/shape, widen BF16 exactly to FP32, reject nonfinite signal, "
                "then evaluate candidate guard"
            ),
            "rounding": "IEEE-754 round-to-nearest ties-to-even",
            "underflow": "gradual underflow required; FTZ/DAZ probe failure is fail-closed",
            "vector_dtype": "binary32",
            "accumulator_dtype": "binary32 fixed adjacent balanced tree",
            "block_schedule": (
                "aligned blocks of 2^20 entries, followed by the same zero-padded adjacent tree; "
                "adding padding zeros is exact"
            ),
            "norm_scalar": ("binary32 max scale and unit norm multiplied exactly in binary64"),
            "threshold": ("one binary64 product followed by nextafter toward -infinity"),
            "candidate_inside_action": "return stored widened FP32 candidate bit-for-bit",
            "candidate_outside_action": (
                "apply the locked downward-scalar radial clip; use fl32(S/2) only on "
                "exceptional arithmetic or an unusable clip"
            ),
            "clip_scalar_order": (
                "downward binary64 Nhat(S)/Nhat(Dhat), then downward binary64 multiply by "
                "890/2048, then downward binary32 conversion"
            ),
            "clip_vector_order": ("q=fl32(alpha32*Dhat), then U=fl32(fl32((1143/2048)*S)+q)"),
            "zero_signal": "return +0 binary32",
            "nonfinite_candidate": "ignore candidate and attempt certified S/2 fallback",
            "nonfinite_signal": "fail closed without an output",
            "subnormal_signal": (
                "below normal anchor, S/2 is returned only if every stored entry halves exactly; "
                "otherwise fail closed"
            ),
            "final_cast": "none; output remains binary32",
            "forbidden": ["FMA contraction", "reassociation", "FTZ", "DAZ", "stochastic rounding"],
        },
        "theorem": {
            "inward_disk": (
                "every successful shape-locked return U obeys "
                "||U-(1143/2048)S||_F <= (893/2048-Delta_mn)||S||_F"
            ),
            "original_p19_disk": ("therefore ||U-(1143/2048)S||_F <= (893/2048)||S||_F"),
            "runtime_exact_postcheck": "not used",
            "conditional_rate_inheritance": (
                "requires every shield call along the trajectory to succeed, stored S to equal "
                "the abstract operator-port signal, and the remaining outer loop to use exact "
                "real arithmetic"
            ),
            "subnormal_dead_zone": (
                "an all-subnormal FP32 signal has norm at most h*(2^-126-2^-149); "
                "the recorded half-distance is an input/output bound only"
            ),
        },
        "prior_p19": prior,
        "prior_rounding_ledgers": ledgers,
        "shape_summary": [
            {
                "shape": [audit.shape.rows, audit.shape.columns],
                "inward_margin": _fraction(audit.inward_margin),
                "certified_inward_radius": _fraction(audit.certified_inward_radius),
                "certified": audit.certified,
            }
            for audit in audits
        ],
        "reconstruction_fields": reconstruction_fields(),
        "checks": checks,
        "all_exact_checks_passed": all(checks.values()),
        "source_snapshot": _source_snapshot(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "git": _git_state(),
    }


def main() -> None:
    args = parse_args()
    payload = build_payload()
    if not payload["all_exact_checks_passed"]:
        raise SystemExit("P20 scalable sector-shield certificate failed")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
