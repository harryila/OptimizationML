#!/usr/bin/env python3
"""Emit the canonical exact/high-precision Jordan nonmonotonicity witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

from passive_muon.witness import canonical_jordan_witness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", type=int, default=100, help="decimal digits (>=50)")
    parser.add_argument(
        "--steps",
        type=int,
        choices=(5,),
        default=5,
        help="Jordan quintic iterations (the certified canonical fixture is fixed at 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is always printed to stdout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = canonical_jordan_witness(precision=args.precision, steps=args.steps)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    payload["software"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    payload["hardware"] = {
        "machine": platform.machine(),
        "processor": platform.processor() or None,
    }
    payload["git"] = {
        "sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
    }
    payload["experiment_provenance"] = {
        "seed": None,
        "randomness": "none; exact deterministic construction",
        "source_snapshot": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path("scripts/find_counterexample.py"),
                Path("src/passive_muon/witness.py"),
                Path("src/passive_muon/specs.py"),
            ]
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
