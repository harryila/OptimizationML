#!/usr/bin/env python3
"""Generate the exact P22 ``768 x 2304`` shield-shape extension certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from passive_muon.p22_scalable_sector_shield_extension import (
    P22_CANONICAL_CERTIFIED_RADIUS,
    P22_CANONICAL_CERTIFIED_RADIUS_HEX,
    P22_CANONICAL_INWARD_MARGIN,
    P22_CANONICAL_INWARD_MARGIN_HEX,
    P22_CANONICAL_PARAMETER_SHAPE,
    P22_CANONICAL_SHIELD_SHAPE,
    P22_SCALABLE_SHIELD_EXTENSION_SCHEMA_VERSION,
)
from passive_muon.p22_scalable_sector_shield_extension_certificate import (
    P22_SHAPE_EXTENSION_CERTIFICATE_SCHEMA_VERSION,
    audit_p22_shape_extension,
)
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
    LOCKED_REDUCTION_BLOCK_ENTRIES,
    LOCKED_SECTOR_CENTER,
    LOCKED_SECTOR_LOWER,
    LOCKED_SECTOR_RADIUS,
    LOCKED_SECTOR_UPPER,
    RADIUS_GRID_DENOMINATOR,
    SQRT_ENCLOSURE_DENOMINATOR,
    ScalableSectorShieldAudit,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/p22_scalable_shield_shape_extension_certificate.json"
SCHEMA_VERSION = P22_SHAPE_EXTENSION_CERTIFICATE_SCHEMA_VERSION

P20_ARTIFACT_PATH = "results/summaries/scalable_sector_shield_certificate.json"
P20_ARTIFACT_SHA256 = "6bca86676eb7164a617a48f34228c40355dbc682b86b4e64ddf3cd54b8d15951"
P20_ARTIFACT_COMMIT = "8a0586dcf7f7d6ce933d84f4ad7a2756764f02d0"
P20_CHECKPOINT_TAG = "p20-scalable-mixed-precision-sector-shield-final-checkpoint"
P20_CHECKPOINT_TAG_OBJECT = "4cfd127117d509ae0dcd0cc26d74e9cd963db50a"

P21_ARTIFACT_PATH = "results/summaries/certified_outer_loop_composition_certificate.json"
P21_ARTIFACT_SHA256 = "07b911edc82332d0068ad192624c53bd9d74759a9cfe79ddc90f92fb96e75713"
P21_ARTIFACT_COMMIT = "7aa154374dc852f551926593bb5e2a05227cdc4f"
P21_CHECKPOINT_TAG = "p21-certified-outer-loop-composition-checkpoint"
P21_CHECKPOINT_TAG_OBJECT = "893f8111205e1092b2aa37fb903c9130b5bb7bf9"

SOURCE_PATHS = (
    "scripts/certify_p22_scalable_shield_shape_extension.py",
    "scripts/reconstruct_p22_scalable_shield_shape_extension.py",
    "src/passive_muon/p22_scalable_sector_shield_extension.py",
    "src/passive_muon/p22_scalable_sector_shield_extension_certificate.py",
    "tests/test_p22_scalable_shield_shape_extension.py",
    "tests/test_p22_scalable_shield_shape_extension_reconstruction.py",
    "src/passive_muon/scalable_sector_shield.py",
    "src/passive_muon/scalable_sector_shield_certificate.py",
    P20_ARTIFACT_PATH,
    P21_ARTIFACT_PATH,
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


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
        raise RuntimeError(f"could not read frozen artifact {commit}:{path}")
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
        raise RuntimeError(f"P22 shape-extension source snapshot is incomplete: {missing}")
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _frozen_artifact(
    *, path_text: str, expected_hash: str, artifact_commit: str, tag: str, tag_object: str
) -> tuple[dict[str, object], dict[str, bool]]:
    path = ROOT / path_text
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    label = tag.split("-", 1)[0]
    record = {
        "artifact_path": path_text,
        "artifact_sha256": observed,
        "artifact_commit": artifact_commit,
        "checkpoint_tag": tag,
        "checkpoint_tag_object": tag_object,
        "source_commit": payload.get("git", {}).get("sha", "unavailable"),
        "schema_version": payload.get("schema_version"),
    }
    checks = {
        f"{label}_artifact_bytes_unchanged": observed == expected_hash,
        f"{label}_checkpoint_blob_unchanged": (
            _git_blob_sha256(artifact_commit, path_text) == expected_hash
        ),
        f"{label}_checkpoint_tag_unchanged": (
            _git_output("rev-parse", tag) == tag_object
            and _git_output("rev-parse", f"{tag}^{{}}") == artifact_commit
        ),
        f"{label}_artifact_exact_checks_pass": (
            payload.get("all_exact_checks_passed") is True
            or payload.get("audit", {}).get("all_exact_checks_passed") is True
        ),
    }
    return record, checks


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
    audit = audit_p22_shape_extension()
    return {
        "extension": {
            "runtime_schema": P22_SCALABLE_SHIELD_EXTENSION_SCHEMA_VERSION,
            "shield_shape": list(P22_CANONICAL_SHIELD_SHAPE),
            "parameter_shape": list(P22_CANONICAL_PARAMETER_SHAPE),
            "parameter_to_shield_map": "transpose",
            "p20_table_mutated": False,
            "inward_margin": str(P22_CANONICAL_INWARD_MARGIN),
            "certified_inward_radius": str(P22_CANONICAL_CERTIFIED_RADIUS),
            "inward_margin_hex": P22_CANONICAL_INWARD_MARGIN_HEX,
            "certified_inward_radius_hex": P22_CANONICAL_CERTIFIED_RADIUS_HEX,
        },
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
        "operation_graph": {
            "normalization": "a=maxabs(x); z=fl32(x/a)",
            "norm": (
                "fl32 squares; fixed adjacent balanced fl32 sum; fl32 sqrt; "
                "exact float64 scale product"
            ),
            "displacement": "Dhat=fl32(C-fl32((1143/2048)*S))",
            "acceptance": "Nhat(Dhat)<=down64((891/2048)*Nhat(S))",
            "clip": (
                "alpha32=down32(down64((890/2048)*down64(Nhat(S)/Nhat(Dhat)))); "
                "U=fl32(fl32((1143/2048)*S)+fl32(alpha32*Dhat))"
            ),
            "fallback": "fl32((1/2)*S) under the P20 normal/exact-halving guard",
            "input_dtypes": ["binary32", "bfloat16 widened exactly to binary32"],
            "output_dtype": "binary32",
            "rounding": "IEEE-754 roundTiesToEven with gradual underflow",
            "fma_reassociation_ftz_daz": False,
        },
        "shape_audit": _shape_exact_fields(audit.shape_audit),
        "extension_checks": audit.checks,
    }


def build_payload() -> dict[str, object]:
    p20, p20_checks = _frozen_artifact(
        path_text=P20_ARTIFACT_PATH,
        expected_hash=P20_ARTIFACT_SHA256,
        artifact_commit=P20_ARTIFACT_COMMIT,
        tag=P20_CHECKPOINT_TAG,
        tag_object=P20_CHECKPOINT_TAG_OBJECT,
    )
    p21, p21_checks = _frozen_artifact(
        path_text=P21_ARTIFACT_PATH,
        expected_hash=P21_ARTIFACT_SHA256,
        artifact_commit=P21_ARTIFACT_COMMIT,
        tag=P21_CHECKPOINT_TAG,
        tag_object=P21_CHECKPOINT_TAG_OBJECT,
    )
    fields = reconstruction_fields()
    checks = {
        **p20_checks,
        **p21_checks,
        "supplied_exact_margin_verified": (
            fields["shape_audit"]["margin"]["inward_margin"]
            == "942123070212169/1152921504606846976"
        ),
        "canonical_shape_certified": fields["shape_audit"]["certified"] is True,
        "extension_checks_pass": all(fields["extension_checks"].values()),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": {
            "classification": "exact P22 extension of the P20 static shield margin",
            "matrix_domain": "stored finite BF16/FP32 768x2304 shield input",
            "transpose_scope": (
                "2304x768 fused-QKV parameter tensors are transposed isometrically to "
                "768x2304, shielded, and transposed back"
            ),
            "preservation": (
                "the frozen P20 and P21 artifacts, tables, and checkpoint trees are inputs; "
                "this artifact does not replace or rewrite either checkpoint"
            ),
            "not_claimed": [
                "a change to the frozen P20 seven-shape theorem",
                "GPU, tensor-core, FTZ, FMA, or reassociated execution",
                "global neural-network convergence or passivity inferred from a sampled trace",
                "unshielded upstream Muon stability",
            ],
        },
        "frozen_provenance": {"p20": p20, "p21": p21},
        "reconstruction_fields": fields,
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
        raise SystemExit("P22 scalable-shield shape-extension certificate failed")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
