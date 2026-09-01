#!/usr/bin/env python3
"""Record a backend-specific canonical Keller--Jordan BF16 pairwise witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from passive_muon.bf16_witness import (
    KELLER_JORDAN_EPSILON,
    KELLER_JORDAN_STEPS,
    canonical_bf16_witness,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/record_bf16_witness.py",
    "src/passive_muon/bf16_witness.py",
    "src/passive_muon/deployed.py",
    "src/passive_muon/specs.py",
    "third_party/UPSTREAM_COMMITS.md",
    "third_party/LICENSES.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cpu",
        help="PyTorch device on which to execute the witness (default: cpu)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=KELLER_JORDAN_EPSILON,
        help="normalization epsilon (pinned upstream default: 1e-7)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        choices=(KELLER_JORDAN_STEPS,),
        default=KELLER_JORDAN_STEPS,
        help="Jordan quintic iterations (canonical deployed fixture: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSON output path; JSON is always printed to stdout",
    )
    return parser.parse_args()


def _run_git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _device_hardware(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested_device": str(device),
        "device_type": device.type,
    }
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "resolved_device": f"cuda:{index}",
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    elif device.type == "mps":
        metadata.update(
            {
                "resolved_device": "mps",
                "mps_built": torch.backends.mps.is_built(),
                "mps_available": torch.backends.mps.is_available(),
            }
        )
    elif device.type == "cpu":
        capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
        metadata.update(
            {
                "resolved_device": "cpu",
                "torch_cpu_capability": capability() if capability is not None else None,
            }
        )
    return metadata


def _source_snapshot() -> dict[str, str]:
    return {
        path: hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _add_run_provenance(payload: dict[str, Any], *, device: torch.device) -> None:
    revision = _run_git("rev-parse", "HEAD")
    status = _run_git("status", "--porcelain")
    payload["software"] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch": torch.__version__,
        "pytorch_git_version": torch.version.git_version,
        "pytorch_build_configuration": torch.__config__.show(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }
    payload["hardware"] = {
        "platform": platform.platform(),
        "operating_system": platform.system(),
        "operating_system_release": platform.release(),
        "machine_architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "byte_order": sys.byteorder,
        "execution_device": _device_hardware(device),
        "cuda_built_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    payload["git"] = {
        "sha": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()),
    }
    payload["experiment_provenance"] = {
        "seed": None,
        "randomness": "none; fixed literals and deterministic control flow",
        "source_snapshot": _source_snapshot(),
        "scope_reminder": (
            "Re-run this executable on each target backend. Do not transplant its returned "
            "values or accumulation observations into a universal BF16 claim."
        ),
    }


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    payload = canonical_bf16_witness(device=device, eps=args.eps, steps=args.steps)
    _add_run_provenance(payload, device=device)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
