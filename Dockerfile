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

# Keep build tooling current; some sdists (flash-attn) fail metadata generation on older pip/setuptools.
RUN python -m pip install --upgrade pip setuptools wheel

# Torch first (CUDA 12.4)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# TRELLIS.2 sparse attention defaults to flash-attn; install it so /generate doesn't crash.
# See: https://github.com/Dao-AILab/flash-attention#installation-and-features
#
# Prefer wheels when available (fast, reliable). If a wheel isn't available for the current
# torch/cuda/python combo, fall back to building from source.
RUN pip install --only-binary=:all: flash-attn==2.7.4.post1 \
    || pip install flash-attn==2.7.4.post1 --no-build-isolation

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
    && pip install /tmp/extensions/nvdiffrast --no-build-isolation \
    # The upstream nvdiffrast build can compile the extension but fail to install the Python package
    # directory (then `import nvdiffrast` breaks at runtime). Force-copy the package into
    # site-packages and verify import.
    && python -c "import shutil,sysconfig; from pathlib import Path; src=Path('/tmp/extensions/nvdiffrast/nvdiffrast'); assert src.is_dir(), f'missing {src}'; dst=Path(sysconfig.get_paths()['purelib'])/'nvdiffrast'; shutil.copytree(src, dst, dirs_exist_ok=True); p=dst/'__init__.py'; p.write_text('''from importlib.metadata import version\\n\\ntry:\\n    __version__ = version(__package__ or \\\"nvdiffrast\\\")\\nexcept Exception:\\n    __version__ = \\\"0.0.0\\\"\\n''', encoding='utf-8'); import nvdiffrast.torch as _dr; print('nvdiffrast import OK')"

# CuMesh
RUN mkdir -p /tmp/extensions \
    && git clone --recursive https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh \
    # CuMesh's setup.py enables --extended-lambda on Windows by default, but cubvh
    # uses host/device lambdas on Linux too. Patch it in so nvcc can compile cubvh.
    && python - <<'PY'
from pathlib import Path

p = Path("/tmp/extensions/CuMesh/setup.py")
s = p.read_text(encoding="utf-8")

marker = "# trellis2-worker build patch: enable extended lambdas"
needle = "nvcc_flags = []"

if marker not in s:
    if needle not in s:
        raise SystemExit("ERROR: CuMesh setup.py changed; cannot find nvcc_flags")
    insert = (
        'nvcc_flags = []\n'
        f"{marker}\n"
        "# cubvh uses __host__/__device__ lambdas; nvcc needs --extended-lambda.\n"
        'nvcc_flags += ["--extended-lambda", "--expt-relaxed-constexpr"]\n'
    )
    s = s.replace(needle, insert, 1)
    p.write_text(s, encoding="utf-8")

print("CuMesh setup.py patched for extended lambdas")
PY
RUN grep -q -- "--extended-lambda" /tmp/extensions/CuMesh/setup.py \
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
