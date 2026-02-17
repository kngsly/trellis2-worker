#!/bin/bash
# start.sh - Conditionally disable CUDA forward-compat library before starting Python.
#
# The Docker image ships cuda-compat-13-0 which places a libcuda.so.1 in
# /usr/local/cuda/compat.  When LD_LIBRARY_PATH puts that directory first,
# the dynamic linker loads the compat lib (built for driver ~570) instead of the
# host's native libcuda.so.1 (e.g. driver 580).  If the host kernel-mode driver
# is newer than the compat lib's user-mode driver, the version mismatch produces
# CUDA Error 803 ("unsupported display driver / cuda driver combination").
#
# Fix: detect the host driver version; if it already supports CUDA 13.0 natively
# (driver major >= 570), remove /usr/local/cuda/compat from LD_LIBRARY_PATH so
# the host's own libcuda.so.1 is used.

set -euo pipefail

COMPAT_DIR="/usr/local/cuda/compat"
# CUDA 13.0 minimum driver major version
MIN_DRIVER_MAJOR=570

if echo "$LD_LIBRARY_PATH" | grep -q "$COMPAT_DIR"; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

    if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -ge "$MIN_DRIVER_MAJOR" ] 2>/dev/null; then
        # Host driver supports CUDA 13.0 natively - remove compat path to prevent Error 803
        export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^${COMPAT_DIR}$" | paste -sd ':' -)
        echo "[start.sh] Host driver ${DRIVER_VERSION} (major=${DRIVER_MAJOR}) >= ${MIN_DRIVER_MAJOR}; removed ${COMPAT_DIR} from LD_LIBRARY_PATH"
    else
        echo "[start.sh] Host driver ${DRIVER_VERSION:-unknown} (major=${DRIVER_MAJOR:-?}); keeping ${COMPAT_DIR} in LD_LIBRARY_PATH for forward compat"
    fi
fi

# Force container's cuBLAS to load before any host-mounted version.
# The NVIDIA container runtime mounts host compute libs at /usr/local/nvidia/lib64/
# which can include a cuBLAS version that conflicts with what PyTorch was compiled against.
# LD_PRELOAD ensures the container's CUDA 13.0 cuBLAS is loaded first.
CUDA_LIB="/usr/local/cuda/lib64"
_preload=""
for lib in libcublas.so libcublasLt.so; do
    if [ -f "${CUDA_LIB}/${lib}" ]; then
        _preload="${_preload:+${_preload}:}${CUDA_LIB}/${lib}"
    fi
done
if [ -n "$_preload" ]; then
    export LD_PRELOAD="${_preload}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "[start.sh] LD_PRELOAD set for container cuBLAS: ${_preload}"
fi

echo "[start.sh] Final LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "[start.sh] Final LD_PRELOAD=${LD_PRELOAD:-<unset>}"

# Log cuBLAS libraries from container vs host for diagnostics
echo "[start.sh] cuBLAS in container (${CUDA_LIB}):"
ls -la ${CUDA_LIB}/libcublas* 2>/dev/null || echo "  (none)"
echo "[start.sh] cuBLAS in host-mounted (/usr/local/nvidia/lib64):"
ls -la /usr/local/nvidia/lib64/libcublas* 2>/dev/null || echo "  (none)"

# Auto-detect attention backend: prefer flash_attn if installed, fall back to sdpa (PyTorch native).
if [ -z "${ATTN_BACKEND:-}" ]; then
    if python -c "import flash_attn" 2>/dev/null; then
        export ATTN_BACKEND="flash_attn"
        echo "[start.sh] ATTN_BACKEND=flash_attn (flash_attn installed)"
    else
        export ATTN_BACKEND="sdpa"
        export SPARSE_ATTN_BACKEND="sdpa"
        echo "[start.sh] ATTN_BACKEND=sdpa (flash_attn not available, using PyTorch native)"
    fi
else
    echo "[start.sh] ATTN_BACKEND=${ATTN_BACKEND} (explicit override)"
fi

exec python server.py
