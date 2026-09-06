ARG BASE_IMAGE=python@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254
FROM ${BASE_IMAGE}
USER root

ARG TORCH_VERSION=2.7.0+cu128

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git libgomp1 \
    && git config --system --add safe.directory /workspace/OptimizationML \
    && git config --system --add safe.directory /workspace/inputs/nanoGPT \
    && git config --system --add safe.directory /workspace/inputs/muon \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/p23-venv \
    && /opt/p23-venv/bin/python -m pip install --no-cache-dir \
        --no-deps \
        --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cu128 \
        "torch==${TORCH_VERSION}"

COPY p23_runtime_requirements.txt /opt/p23-build/p23_runtime_requirements.txt
RUN /opt/p23-venv/bin/python -m pip install --no-cache-dir \
        --no-deps \
        --only-binary=:all: \
        --requirement /opt/p23-build/p23_runtime_requirements.txt \
    && /opt/p23-venv/bin/python -m pip check \
    && test "$(/opt/p23-venv/bin/python --version)" = "Python 3.12.14"

COPY p23_repository_source_path.txt \
    /opt/p23-venv/lib/python3.12/site-packages/p23_repository_source.pth

RUN mkdir -p \
        /workspace/OptimizationML \
        /workspace/inputs/nanoGPT \
        /workspace/inputs/muon \
        /workspace/evidence/p23 \
        /private/tmp/optimizationml-p22-data \
        /Users/harry/Desktop/temp/OptimizationML/experiments/training \
        /mounted-host-evidence \
    && touch \
        /Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py \
        /mounted-host-evidence/image-inspect.json \
        /mounted-host-evidence/running-container-inspect.json \
        /mounted-host-evidence/running-mountinfo.txt \
        /mounted-host-evidence/nvidia-smi.csv
