FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ARG BUILD_JOBS="2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
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
    NINJAFLAGS=-j${BUILD_JOBS} \
    # Prefer CUDA forward-compat runtime libs when host driver/runtime mismatch occurs.
    LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3.10-dev python3-pip \
      git git-lfs curl wget ca-certificates \
      build-essential ninja-build pkg-config \
      libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
      libjpeg-dev \
      cuda-compat-13-0 \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# Keep build tooling current; some sdists (flash-attn) fail metadata generation on older pip/setuptools.
RUN python -m pip install --upgrade pip setuptools wheel

# Torch first (CUDA 13.0)
# Use --extra-index-url instead of --index-url to ensure cuda-bindings==13.0.3 can be found
# See: https://github.com/pytorch/pytorch/issues/172926
RUN pip install torch==2.10.0 torchvision==0.25.0 --extra-index-url https://download.pytorch.org/whl/cu130

# TRELLIS.2 sparse attention defaults to flash-attn; install it so /generate doesn't crash.
# See: https://github.com/Dao-AILab/flash-attention#installation-and-features
#
# PyPI doesn't have cu130 wheels - use community-built wheels from flashattn.dev
# Flash-Attention 3 has cu130 wheels for PyTorch 2.10.0
RUN pip install flash-attn --extra-index-url https://flashattn.dev/whl/cu130/torch2.10/ \
    || echo "WARNING: flash-attn installation failed, continuing anyway (will use fallback attention)"

# Clone TRELLIS.2 source (used via PYTHONPATH; the repo does not ship as a pip package).
WORKDIR /opt/trellis2
RUN git clone --depth 1 --recursive https://github.com/microsoft/TRELLIS.2.git src
ENV PYTHONPATH=/opt/trellis2/src

# Patch upstream TRELLIS.2 for worker reliability:
# - BiRefNet dtype/device mismatch (float vs half) during rembg preprocessing
# - Robust alpha bbox cropping (avoid empty bbox when alpha is soft/empty)
# - Guard against empty sparse coords (raise a clearer error instead of crashing deeper in sparse ops)
# - Postprocess cleanup after remesh (reduce floaters / disconnected components in exported meshes)
RUN python - <<'PY'
from __future__ import annotations

import re
from pathlib import Path

root = Path("/opt/trellis2/src")

def patch_file(path: Path, fn):
    if not path.exists():
        print(f"WARN: patch target missing, skipping: {path}")
        return False
    s = path.read_text(encoding="utf-8")
    out = fn(s)
    if out != s:
        path.write_text(out, encoding="utf-8")
        return True
    return False

# 1) BiRefNet dtype fix
biref = root / "trellis2" / "pipelines" / "rembg" / "BiRefNet.py"
marker = "# trellis2-worker build patch: match input dtype/device to model params"
def patch_biref(s: str) -> str:
    if marker in s:
        return s
    # Replace hardcoded device move with a device+dtype-aware version.
    # Upstream has changed this line across commits, so keep the regex broad.
    pat = r'^(\\s*)input_images\\s*=\\s*self\\.transform_image\\(image\\)\\.unsqueeze\\(0\\)(?:\\.to\\([^\\n]+\\))?\\s*$'
    m = re.search(pat, s, flags=re.M)
    if not m:
        print("WARN: BiRefNet input_images assignment not found; skipping dtype/device patch")
        return s
    indent = m.group(1)
    repl = (
        f"{indent}input_images = self.transform_image(image).unsqueeze(0)\\n"
        f"{indent}{marker}\\n"
        f"{indent}param = next(self.model.parameters())\\n"
        f"{indent}input_images = input_images.to(device=param.device, dtype=param.dtype)\\n"
    )
    return re.sub(pat, repl.rstrip("\\n"), s, flags=re.M)

patched = patch_file(biref, patch_biref)
print(f"Patched BiRefNet dtype/device: {patched}")

# 2) Robust bbox cropping in preprocess_image
p = root / "trellis2" / "pipelines" / "trellis2_image_to_3d.py"
pre_marker = "# trellis2-worker build patch: robust alpha bbox"
def patch_preprocess_bbox(s: str) -> str:
    if pre_marker in s:
        return s
    needle = (
        "        bbox = np.argwhere(alpha > 0.8 * 255)\\n"
        "        bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])\\n"
    )
    if needle not in s:
        # Best-effort: don't fail the build if upstream already fixed it.
        print("WARN: preprocess_image bbox needle not found; skipping bbox robustness patch")
        return s
    repl = (
        f"        {pre_marker}\\n"
        "        bbox_coords = np.argwhere(alpha > 0.8 * 255)\\n"
        "        if bbox_coords.size == 0:\\n"
        "            bbox_coords = np.argwhere(alpha > 0)\\n"
        "        if bbox_coords.size == 0:\\n"
        "            bbox = (0, 0, alpha.shape[1] - 1, alpha.shape[0] - 1)\\n"
        "        else:\\n"
        "            bbox = (\\n"
        "                np.min(bbox_coords[:, 1]),\\n"
        "                np.min(bbox_coords[:, 0]),\\n"
        "                np.max(bbox_coords[:, 1]),\\n"
        "                np.max(bbox_coords[:, 0]),\\n"
        "            )\\n"
    )
    return s.replace(needle, repl)

patched = patch_file(p, patch_preprocess_bbox)
print(f"Patched preprocess_image bbox robustness: {patched}")

# 3) Empty sparse coords guard in run()
guard_marker = "# trellis2-worker build patch: explicit empty sparse coords guard"
def patch_empty_coords_guard(s: str) -> str:
    if guard_marker in s:
        return s
    block = (
        "        coords = self.sample_sparse_structure(\\n"
        "            cond_512, ss_res,\\n"
        "            num_samples, sparse_structure_sampler_params\\n"
        "        )\\n"
    )
    if block not in s:
        print("WARN: sample_sparse_structure block not found; skipping empty-coords guard patch")
        return s
    repl = (
        block
        + f"        {guard_marker}\\n"
        + "        if coords.numel() == 0 or coords.shape[0] == 0:\\n"
        + "            raise RuntimeError('empty sparse coords')\\n"
    )
    return s.replace(block, repl)

patched = patch_file(p, patch_empty_coords_guard)
print(f"Patched empty sparse coords guard: {patched}")

# 4) O-Voxel postprocess: cap remesh resolution for low-poly to reduce CuMesh simplify VRAM (avoids OOM on 24GB)
ov = root / "o-voxel" / "o_voxel" / "postprocess.py"
ov_res_marker = "# trellis2-worker build patch: cap remesh resolution for low decimation"
def patch_ovoxel_remesh_res(s: str) -> str:
    if ov_res_marker in s:
        return s
    needle = "        resolution = grid_size.max().item()\n"
    if needle not in s:
        print("WARN: o_voxel postprocess resolution needle not found; skipping remesh resolution cap")
        return s
    repl = (
        needle
        + f"        {ov_res_marker}\n"
        + "        if decimation_target <= 200000:\n"
        + "            resolution = min(resolution, 512)\n"
    )
    return s.replace(needle, repl, 1)

patched = patch_file(ov, patch_ovoxel_remesh_res)
print(f"Patched o_voxel remesh resolution cap: {patched}")

# 5) O-Voxel postprocess: cleanup after remesh (the standard branch already does this)
ov_cleanup_marker = "# trellis2-worker build patch: cleanup after remesh"
def patch_ovoxel_cleanup(s: str) -> str:
    if ov_cleanup_marker in s:
        return s
    needle = "        mesh.simplify(decimation_target, verbose=verbose)\n"
    idx = s.find(needle)
    if idx == -1:
        print("WARN: o_voxel postprocess simplify needle not found; skipping remesh cleanup patch")
        return s
    insert = (
        needle
        + f"        {ov_cleanup_marker}\n"
        + "        mesh.remove_duplicate_faces()\n"
        + "        mesh.repair_non_manifold_edges()\n"
        + "        mesh.remove_small_connected_components(1e-5)\n"
        + "        mesh.fill_holes(max_hole_perimeter=3e-2)\n"
        + "        mesh.unify_face_orientations()\n"
    )
    return s.replace(needle, insert, 1)

patched = patch_file(ov, patch_ovoxel_cleanup)
print(f"Patched o_voxel remesh cleanup: {patched}")
PY

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
        print("WARN: CuMesh setup.py changed; nvcc_flags marker not found, skipping CuMesh patch")
        print("CuMesh setup.py left unmodified")
    insert = (
        'nvcc_flags = []\n'
        f"{marker}\n"
        "# cubvh uses __host__/__device__ lambdas; nvcc needs --extended-lambda.\n"
        'nvcc_flags += ["--extended-lambda", "--expt-relaxed-constexpr"]\n'
    )
    if needle in s:
        s = s.replace(needle, insert, 1)
        p.write_text(s, encoding="utf-8")
        print("CuMesh setup.py patched for extended lambdas")
    else:
        print("CuMesh setup.py patch skipped")
else:
    print("CuMesh setup.py already patched")
PY
RUN pip install /tmp/extensions/CuMesh --no-build-isolation

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
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

RUN mkdir -p /outputs
ENV OUTPUT_DIR=/outputs

EXPOSE 8000
CMD ["/app/start.sh"]
