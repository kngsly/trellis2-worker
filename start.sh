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

exec python server.py
