#!/usr/bin/env python3
"""Run the full-step nonquadratic convergence-storage falsification probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from passive_muon.nonquadratic_convergence_experiment import (
    NonquadraticConvergenceProbeConfig,
    run_nonquadratic_convergence_probe,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
P4_BASE_COMMIT = "c9636358de2d3d17bf5e62b0f03c7aff12da97bd"
P4_RESULT_PATH = "results/summaries/structure_aware_stability_certificate.json"
INCREMENTAL_P5_RESULT_PATH = "results/summaries/nonquadratic_stability_certificate.json"
SOURCE_PATHS = (
    "experiments/quadratics/run_nonquadratic_convergence_falsification.py",
    "src/passive_muon/nonquadratic_convergence_experiment.py",
    "src/passive_muon/nonquadratic_convergence.py",
    "src/passive_muon/nonquadratic_experiment.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/operator.py",
    "src/passive_muon/normalizers.py",
    "src/passive_muon/orthogonalizers.py",
    "src/passive_muon/polynomials.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_nonquadratic_convergence_experiment.py",
    "theory/nonquadratic_convergence_certificate.md",
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
    parser.add_argument("--seed", type=int, default=NonquadraticConvergenceProbeConfig.seed)
    parser.add_argument(
        "--iterations",
        type=int,
        help="override the number of EMA/Nesterov updates per trajectory",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run one changing-orientation 2x2 trajectory for a smoke replay",
    )
    return parser.parse_args()


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((REPOSITORY_ROOT / relative_path).read_bytes()).hexdigest()


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(path) for path in SOURCE_PATHS}


def build_payload(
    config: NonquadraticConvergenceProbeConfig | None = None,
) -> dict[str, Any]:
    """Run the deterministic grid and attach source/upstream provenance."""

    selected = NonquadraticConvergenceProbeConfig() if config is None else config
    payload = run_nonquadratic_convergence_probe(selected).as_dict()
    payload["certificate_references"] = {
        "full_step_convergence": {
            "source": "src/passive_muon/nonquadratic_convergence.py",
            "source_sha256": _sha256("src/passive_muon/nonquadratic_convergence.py"),
            "scope": "trajectory-to-minimizer convergence at eta=1/32000",
        },
        "p4": {
            "base_commit": P4_BASE_COMMIT,
            "manifest_path": P4_RESULT_PATH,
            "manifest_sha256": _sha256(P4_RESULT_PATH),
        },
        "incremental_p5": {
            "scope": "arbitrary-pair incremental contraction at eta=1/640000",
            "manifest_path": INCREMENTAL_P5_RESULT_PATH,
            "manifest_sha256": _sha256(INCREMENTAL_P5_RESULT_PATH),
        },
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
    config = NonquadraticConvergenceProbeConfig(seed=args.seed)
    if args.quick:
        config = replace(
            config,
            shapes=((2, 2),),
            transition_scales=(1.0,),
            radius_multipliers=(4.0,),
            momentum_modes=("random",),
            iterations=20,
            diagnostic_stride=2,
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
