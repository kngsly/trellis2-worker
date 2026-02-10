FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ARG BUILD_JOBS="2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    # Helps reduce fragmentation on long-running jobs.
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    # Needed when building CUDA extensions in Docker builds (no GPU visible at build time).
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    # CuMesh -> cubvh requires extended lambda support under CUDA (nvcc).
    # Export both common env vars so torch/cmake-based extension builds pick it up.
    TORCH_NVCC_FLAGS=--extended-lambda \
    NVCC_FLAGS=--extended-lambda \
    # Avoid OOM during extension builds (ninja/cmake/setuptools often respect these).
    MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS=-j${BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      git git-lfs curl wget ca-certificates \
      build-essential ninja-build pkg-config \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
      libjpeg-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Torch first (CUDA 12.4)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Clone TRELLIS.2 source (used via PYTHONPATH; the repo does not ship as a pip package).
WORKDIR /opt/trellis2
RUN git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git src
ENV PYTHONPATH=/opt/trellis2/src

WORKDIR /app
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install -r /app/requirements-docker.txt

# Extra dep used by TRELLIS.2 demos. Keep pinned to their setup.sh commit.
RUN pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# CUDA extensions (heavy). These steps may need iteration depending on your target GPU/driver.
# nvdiffrast
RUN mkdir -p /tmp/extensions \
    && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation

# CuMesh
RUN mkdir -p /tmp/extensions \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    && pip install /tmp/extensions/CuMesh --no-build-isolation

# FlexGEMM
RUN mkdir -p /tmp/extensions \
    && git clone --recursive https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM \
    && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# O-Voxel (bundled as a submodule in TRELLIS.2 repo)
RUN mkdir -p /tmp/extensions \
    && cp -r /opt/trellis2/src/o-voxel /tmp/extensions/o-voxel \
    && pip install /tmp/extensions/o-voxel --no-build-isolation

COPY server.py /app/server.py
COPY worker.py /app/worker.py

RUN mkdir -p /outputs
ENV OUTPUT_DIR=/outputs

EXPOSE 8000
CMD ["python", "server.py"]
