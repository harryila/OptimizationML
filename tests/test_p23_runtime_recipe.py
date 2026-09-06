from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "experiments/training"
DOCKERFILE = RUNTIME_ROOT / "p23_runtime.Dockerfile"
REQUIREMENTS = RUNTIME_ROOT / "p23_runtime_requirements.txt"
SOURCE_PATH = RUNTIME_ROOT / "p23_repository_source_path.txt"

BASE_IMAGE = "python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254"
TORCH_VERSION = "2.7.0+cu128"
PYTHON_VERSION = "3.12.14"

EXPECTED_REQUIREMENTS = [
    "certifi==2026.7.22",
    "charset-normalizer==3.5.1",
    "filelock==3.32.3",
    "fsspec==2026.7.0",
    "idna==3.19",
    "Jinja2==3.1.6",
    "MarkupSafe==3.0.3",
    "mpmath==1.3.0",
    "networkx==3.6.1",
    "numpy==2.2.6",
    "nvidia-cublas-cu12==12.8.3.14",
    "nvidia-cuda-cupti-cu12==12.8.57",
    "nvidia-cuda-nvrtc-cu12==12.8.61",
    "nvidia-cuda-runtime-cu12==12.8.57",
    "nvidia-cudnn-cu12==9.7.1.26",
    "nvidia-cufft-cu12==11.3.3.41",
    "nvidia-cufile-cu12==1.13.0.11",
    "nvidia-curand-cu12==10.3.9.55",
    "nvidia-cusolver-cu12==11.7.2.55",
    "nvidia-cusparse-cu12==12.5.7.53",
    "nvidia-cusparselt-cu12==0.6.3",
    "nvidia-nccl-cu12==2.26.2",
    "nvidia-nvjitlink-cu12==12.8.61",
    "nvidia-nvtx-cu12==12.8.55",
    "regex==2026.9.3",
    "requests==2.34.2",
    "setuptools==78.1.0",
    "sympy==1.14.0",
    "tiktoken==0.14.0",
    "triton==3.3.0",
    "typing_extensions==4.16.0",
    "urllib3==2.7.0",
]


def test_p23_runtime_dockerfile_locks_the_declared_recipe() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert dockerfile.startswith(f"ARG BASE_IMAGE={BASE_IMAGE}\nFROM ${{BASE_IMAGE}}\n")
    assert re.findall(r"^ARG ([A-Z_]+)=(.+)$", dockerfile, flags=re.MULTILINE) == [
        ("BASE_IMAGE", BASE_IMAGE),
        ("TORCH_VERSION", TORCH_VERSION),
    ]
    assert re.findall(r"^USER (.+)$", dockerfile, flags=re.MULTILINE) == ["root"]
    assert dockerfile.count("--no-deps") == 2
    assert dockerfile.count("--only-binary=:all:") == 2
    assert "--index-url https://download.pytorch.org/whl/cu128" in dockerfile
    assert '"torch==${TORCH_VERSION}"' in dockerfile
    assert "--requirement /opt/p23-build/p23_runtime_requirements.txt" in dockerfile
    assert "/opt/p23-venv/bin/python -m pip check" in dockerfile
    assert f'test "$(/opt/p23-venv/bin/python --version)" = "Python {PYTHON_VERSION}"' in dockerfile
    assert (
        "apt-get install --yes --no-install-recommends ca-certificates git libgomp1" in dockerfile
    )
    assert re.findall(r"git config --system --add safe\.directory ([^ \\\n]+)", dockerfile) == [
        "/workspace/OptimizationML",
        "/workspace/inputs/nanoGPT",
        "/workspace/inputs/muon",
    ]
    assert dockerfile.count("\nCOPY ") == 2
    assert "\nADD " not in dockerfile
    assert "p23_repository_source.pth" in dockerfile

    required_destinations = {
        "/workspace/OptimizationML",
        "/workspace/inputs/nanoGPT",
        "/workspace/inputs/muon",
        "/workspace/evidence/p23",
        "/private/tmp/optimizationml-p22-data",
        "/Users/harry/Desktop/temp/OptimizationML/experiments/training",
        "/mounted-host-evidence",
        "/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py",
        "/mounted-host-evidence/image-inspect.json",
        "/mounted-host-evidence/running-container-inspect.json",
        "/mounted-host-evidence/running-mountinfo.txt",
        "/mounted-host-evidence/nvidia-smi.csv",
    }
    for destination in required_destinations:
        assert destination in dockerfile

    assert "latest" not in dockerfile.lower()


def test_p23_runtime_requirements_are_an_exact_binary_wheel_closure() -> None:
    requirements = REQUIREMENTS.read_text(encoding="utf-8")
    lines = requirements.splitlines()

    assert requirements == "\n".join(EXPECTED_REQUIREMENTS) + "\n"
    assert lines == EXPECTED_REQUIREMENTS
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s;]+", line) for line in lines)
    normalized_names = [line.split("==", 1)[0].lower().replace("_", "-") for line in lines]
    assert len(normalized_names) == len(set(normalized_names))
    assert "torch" not in normalized_names


def test_p23_repository_path_hook_is_exact_and_path_only() -> None:
    assert SOURCE_PATH.read_bytes() == b"/workspace/OptimizationML/src\n"

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert (
        "COPY p23_repository_source_path.txt \\\n"
        "    /opt/p23-venv/lib/python3.12/site-packages/p23_repository_source.pth" in dockerfile
    )
