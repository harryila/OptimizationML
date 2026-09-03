from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_p13_generator_writes_complete_two_precision_payload(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p13.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_radial_passivation_tradeoff.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = json.loads(completed.stdout)
    written = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == written
    assert written["schema_version"] == ("passive-muon-radial-passivation-tradeoff-certificate-v1")
    assert written["audit"]["all_exact_and_interval_checks_passed"]
    assert [
        replay["precision_bits"] for replay in written["arb_magnitude_certificate"]["passes"]
    ] == [160, 224]


def test_p13_generator_requires_two_distinct_precisions() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_radial_passivation_tradeoff.py"),
            "--precision-bits",
            "160",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "at least two distinct Arb precisions" in completed.stderr
