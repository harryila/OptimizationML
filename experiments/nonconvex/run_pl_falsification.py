#!/usr/bin/env python3
"""Run deterministic nonconvex smooth-PL falsification trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from passive_muon.pl_experiment import PLProbeConfig, run_pl_probe
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P5_CHECKPOINT = "a549fb4c206335ef9ec264524e0f581216b250d4"
P5_RESULT_PATH = "results/summaries/nonquadratic_convergence_certificate.json"
SOURCE_PATHS = (
    "experiments/nonconvex/run_pl_falsification.py",
    "src/passive_muon/pl_experiment.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/nonquadratic_experiment.py",
    "src/passive_muon/nonquadratic_stability.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_pl_experiment.py",
    "theory/pl_convergence_certificate.md",
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
    parser.add_argument("--seed", type=int, default=PLProbeConfig.seed)
    parser.add_argument(
        "--iterations",
        type=int,
        help="override the number of repaired EMA/Nesterov updates per trajectory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one rank-deficient nonconvex 2x2 case for a smoke replay",
    )
    return parser.parse_args()


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(path) for path in SOURCE_PATHS}


def build_payload(config: PLProbeConfig | None = None) -> dict[str, Any]:
    """Run the diagnostic grid and attach certificate/source provenance."""

    selected = PLProbeConfig() if config is None else config
    payload = run_pl_probe(selected).as_dict()
    payload["certificate_reference"] = {
        "source": "src/passive_muon/pl_convergence.py",
        "source_sha256": _sha256("src/passive_muon/pl_convergence.py"),
        "theory_note": "theory/pl_convergence_certificate.md",
        "theory_note_sha256": _sha256("theory/pl_convergence_certificate.md"),
        "scope": (
            "global function-value convergence and momentum decay for every fixed "
            "differentiable globally 10-smooth objective satisfying PL with constant 1"
        ),
    }
    payload["p5_reference"] = {
        "checkpoint": P5_CHECKPOINT,
        "manifest_path": P5_RESULT_PATH,
        "manifest_sha256": _sha256(P5_RESULT_PATH),
        "scope_boundary": (
            "P5 additionally assumes strong convexity; P6 permits nonconvex and "
            "nonunique-minimizer PL objectives"
        ),
    }
    provenance = payload["experiment_provenance"]
    if not isinstance(provenance, dict):
        raise TypeError("experiment provenance must be a dictionary")
    provenance["source_snapshot"] = _source_snapshot()
    payload["upstream_provenance"] = {
        "repository": "https://github.com/KellerJordan/Muon",
        "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
        "muon_py_sha256": PINNED_MUON_PY_SHA256,
    }
    return payload


def main() -> None:
    args = parse_args()
    config = PLProbeConfig(seed=args.seed)
    if args.quick:
        config = replace(
            config,
            shapes=((2, 2),),
            rank_modes=("codimension_one",),
            transition_scales=(1.0,),
            radius_multipliers=(2.0**0.5,),
            momentum_modes=("random",),
            iterations=20,
            diagnostic_stride=1,
        )
    if args.iterations is not None:
        config = replace(config, iterations=args.iterations)
    payload = build_payload(config)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
