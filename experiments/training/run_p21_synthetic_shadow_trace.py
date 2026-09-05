#!/usr/bin/env python3
"""Run the deterministic CPU-only P21 shadow-observer diagnostic.

This program does not load or train a model and does not observe real
gradients.  It exists solely to test the frozen acquisition schedule, metric
semantics, P20 composition, aggregation, and predeclared gate implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from passive_muon.deployed import keller_jordan_map
from passive_muon.p21_shadow_trace import (
    CAPTURE_STEPS,
    load_frozen_protocol,
    observe_aspect_scaled_candidate,
    phase_for_step,
    summarize_shadow_trace,
)
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT: Final = ROOT / "results" / "raw" / "p21_synthetic_shadow_trace.json"
SEED: Final = 20_260_907
TOKENS_PER_STEP: Final = 4_096
SCALES_BY_PHASE_OFFSET: Final = (1.1, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
SOURCE_PATHS: Final = (
    Path("experiments/training/p21_shadow_trace_protocol.json"),
    Path("experiments/training/run_p21_synthetic_shadow_trace.py"),
    Path("src/passive_muon/deployed.py"),
    Path("src/passive_muon/p21_shadow_trace.py"),
    Path("src/passive_muon/scalable_sector_shield.py"),
    Path("src/passive_muon/upstream_momentum.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD") or "unavailable",
        "branch": run("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _provenance(seed: int) -> dict[str, object]:
    return {
        "git": _git_state(),
        "seed": seed,
        "hardware": {
            "device": "cpu",
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "torch_config": torch.__config__.show(),
        },
        "source_snapshot": _source_snapshot(),
        "upstream_candidate": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "audited_muon_py_sha256": PINNED_MUON_PY_SHA256,
            "normalization": "X/(||X||_F+1e-7)",
            "coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "iteration_count": 5,
        },
        "absent_real_run_provenance": {
            "trainer": None,
            "dataset": None,
            "tokenizer": None,
            "checkpoint": None,
            "accelerator": None,
            "reason": "this is the explicitly synthetic CPU infrastructure diagnostic",
        },
    }


def _phase_offset(step: int) -> int:
    phase = phase_for_step(step)
    phase_steps = [candidate for candidate in CAPTURE_STEPS if phase_for_step(candidate) == phase]
    return phase_steps.index(step)


def _synthetic_signal(*, scale: float, layer: int, role_index: int) -> torch.Tensor:
    signal = torch.tensor(
        [[0.6, 0.0], [0.0, 0.8]],
        dtype=torch.float32,
        device="cpu",
    )
    if layer % 2:
        signal = signal.flip(0).contiguous()
    if role_index:
        signal = signal.flip(1).contiguous()
    if (layer + role_index) % 3 == 2:
        signal = -signal
    return (signal * scale).contiguous()


def run_synthetic_shadow_trace(*, seed: int = SEED) -> dict[str, object]:
    """Run all 144 deterministic synthetic observations and aggregate them."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    protocol = load_frozen_protocol()
    observations: list[dict[str, object]] = []
    roles = ("attention_output", "mlp_projection")
    intended_parameter_numel = {
        f"synthetic.layers.{layer}.{role}.weight": 4 for role in roles for layer in range(3)
    }
    for step in CAPTURE_STEPS:
        scale = SCALES_BY_PHASE_OFFSET[_phase_offset(step)]
        for role_index, role in enumerate(roles):
            for layer in range(3):
                signal = _synthetic_signal(scale=scale, layer=layer, role_index=role_index)
                raw_candidate = keller_jordan_map(signal, steps=5, eps=1.0e-7).contiguous()
                observations.append(
                    observe_aspect_scaled_candidate(
                        signal,
                        raw_candidate,
                        optimizer_step=step,
                        tokens_seen=(step + 1) * TOKENS_PER_STEP,
                        seed=seed,
                        parameter_name=f"synthetic.layers.{layer}.{role}.weight",
                        layer=layer,
                        role=role,
                        learning_rate=0.02,
                        beta=0.95,
                        weight_decay=0.0,
                    )
                )

    summary = summarize_shadow_trace(
        observations,
        evidence_kind="synthetic_cpu_diagnostic",
        provenance=_provenance(seed),
        intended_parameter_numel=intended_parameter_numel,
    )
    summary["synthetic_fixture"] = {
        "seed": seed,
        "matrix_shape": [2, 2],
        "roles": list(roles),
        "layers_per_role": 3,
        "scales_by_phase_offset": list(SCALES_BY_PHASE_OFFSET),
        "tokens_per_step": TOKENS_PER_STEP,
        "protocol_schema_version": protocol["schema_version"],
        "observation_count": len(observations),
    }
    summary["real_trace_status"] = {
        "blocked": True,
        "reason": (
            "no pinned trainer/data/checkpoint or CUDA BF16 device exists in this workspace; "
            "the analytic P21 theorem gate must pass first"
        ),
        "synthetic_result_must_not_be_cited_as_real_gradient_evidence": True,
    }
    return summary


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    payload = run_synthetic_shadow_trace(seed=arguments.seed)
    rendered = canonical_json(payload)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
