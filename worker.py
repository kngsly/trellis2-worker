#!/usr/bin/env python3
"""
TRELLIS.2 worker implementation.

This module is imported by server.py.
"""

from __future__ import annotations

import gc
import io
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
import ctypes
from pathlib import Path
from typing import List, Optional

_log = logging.getLogger(__name__)

# Minimum wait before exiting for "CUDA unavailable" or preload timeout (robustness:
# avoid early exit on slow driver bring-up or slow first-time model download).
_MIN_STARTUP_WAIT_SEC = 360  # 6 minutes


def _is_cuda_oom(error: Exception) -> bool:
    msg = str(error).lower()
    return "out of memory" in msg or "cuda out of memory" in msg


def _cuda_empty_cache() -> None:
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.synchronize()
    except Exception:
        pass

import numpy as np
from PIL import Image


_PIPELINE = None
_READY = {
    "status": "not_started",  # not_started | initializing_cuda | loading_weights | downloading_model | ready | error
    "detail": "",
    "started_at": 0.0,
    "ready_at": 0.0,
    "cuda_env": None,  # populated by _wait_for_cuda with driver/runtime/GPU info
}
_READY_LOCK = threading.Lock()
_CUDA_RUNTIME_PREPARED = False
_CUDA_ENV_INFO: Optional[dict] = None  # cached CUDA env info from preflight


def _get_device():
    return os.environ.get("TRELLIS2_DEVICE", "cuda")


def _find_nvidia_smi() -> Optional[Path]:
    """Locate nvidia-smi binary."""
    for p in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi"):
        pp = Path(p)
        if pp.exists():
            return pp
    return None


# Known minimum driver versions for each CUDA toolkit major.minor.
# Source: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
_CUDA_MIN_DRIVER: dict[str, int] = {
    "12.8": 570,
    "12.7": 565,
    "12.6": 560,
    "12.5": 555,
    "12.4": 550,
    "12.3": 545,
    "12.2": 535,
    "12.1": 530,
    "12.0": 525,
    "11.8": 520,
    "11.7": 515,
    "11.6": 510,
    "11.5": 495,
    "11.4": 470,
    "11.3": 465,
    "11.2": 460,
    "11.1": 455,
    "11.0": 450,
}


def _parse_driver_major(version_str: str) -> Optional[int]:
    """Extract the major version number from a driver version string like '550.54.14'."""
    try:
        return int(version_str.strip().split(".")[0])
    except (ValueError, IndexError):
        return None


def _cuda_env_info() -> dict:
    """
    Gather comprehensive CUDA environment information for diagnostics and /ready.

    Returns a dict with keys:
      driver_version, cuda_runtime_version, torch_version, torch_cuda_available,
      torch_cuda_arch_list, gpu_count, gpu_names, nvidia_smi_ok, nvidia_smi_output,
      libcuda_ok, dev_nvidia, compat_ok, compat_error, error_803
    """
    info: dict = {
        "driver_version": None,
        "cuda_runtime_version": None,
        "torch_version": None,
        "torch_cuda_available": None,
        "torch_cuda_arch_list": None,
        "gpu_count": 0,
        "gpu_names": [],
        "nvidia_smi_ok": False,
        "nvidia_smi_output": None,
        "libcuda_ok": False,
        "dev_nvidia": [],
        "compat_ok": None,
        "compat_error": None,
        "error_803": False,
    }

    # /dev/nvidia* devices
    try:
        info["dev_nvidia"] = sorted(str(p) for p in Path("/dev").glob("nvidia*"))
    except Exception:
        pass

    # libcuda.so availability
    try:
        ctypes.CDLL("libcuda.so.1")
        info["libcuda_ok"] = True
    except Exception:
        pass

    # nvidia-smi query (fast, ~1-2s)
    smi = _find_nvidia_smi()
    if smi:
        try:
            p = subprocess.run(
                [str(smi), "--query-gpu=driver_version,name,memory.total,gpu_uuid",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            if p.returncode == 0 and out:
                info["nvidia_smi_ok"] = True
                info["nvidia_smi_output"] = out[:1000]
                # Parse first GPU line: "driver_version, name, memory, uuid"
                for line in out.splitlines():
                    parts = [x.strip() for x in line.split(",")]
                    if len(parts) >= 2:
                        if info["driver_version"] is None:
                            info["driver_version"] = parts[0]
                        info["gpu_names"].append(parts[1])
                        info["gpu_count"] = len(info["gpu_names"])
            else:
                info["nvidia_smi_output"] = (err or f"rc={p.returncode}")[:500]
                # Check for error 803 specifically
                if "803" in (err or "") or "unsupported" in (err or "").lower():
                    info["error_803"] = True
                    info["compat_ok"] = False
                    info["compat_error"] = (
                        f"nvidia-smi error 803: system has unsupported display driver / "
                        f"cuda driver combination. stderr={err[:300]}"
                    )
        except subprocess.TimeoutExpired:
            info["nvidia_smi_output"] = "timeout (5s)"
        except Exception as e:
            info["nvidia_smi_output"] = f"error: {e!r}"

    # Torch CUDA info
    try:
        import torch
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["torch_cuda_arch_list"] = os.environ.get("TORCH_CUDA_ARCH_LIST")
        cuda_ver = getattr(torch.version, "cuda", None)
        info["cuda_runtime_version"] = cuda_ver
        try:
            info["torch_cuda_available"] = torch.cuda.is_available()
            info["gpu_count"] = max(info["gpu_count"], torch.cuda.device_count())
        except Exception:
            info["torch_cuda_available"] = False
    except ImportError:
        pass

    # Driver/runtime compatibility check
    if info["driver_version"] and info["cuda_runtime_version"] and not info["error_803"]:
        driver_major = _parse_driver_major(info["driver_version"])
        cuda_rt = info["cuda_runtime_version"]
        # Extract major.minor from runtime version like "12.4"
        cuda_mm = ".".join(cuda_rt.split(".")[:2]) if cuda_rt else None
        if driver_major is not None and cuda_mm and cuda_mm in _CUDA_MIN_DRIVER:
            min_driver = _CUDA_MIN_DRIVER[cuda_mm]
            if driver_major < min_driver:
                info["compat_ok"] = False
                info["compat_error"] = (
                    f"Driver {info['driver_version']} (major={driver_major}) is too old for "
                    f"CUDA runtime {cuda_rt}. Minimum driver major version: {min_driver}."
                )
            else:
                info["compat_ok"] = True
        else:
            # Can't determine compatibility; assume OK and let torch.cuda.is_available() decide
            info["compat_ok"] = True

    return info


def _log_cuda_env(info: dict) -> None:
    """Log CUDA environment info in a readable format."""
    lines = [
        "--- CUDA environment info ---",
        f"  driver_version:       {info.get('driver_version') or 'unknown'}",
        f"  cuda_runtime_version: {info.get('cuda_runtime_version') or 'unknown'}",
        f"  torch_version:        {info.get('torch_version') or 'unknown'}",
        f"  torch_cuda_available: {info.get('torch_cuda_available')}",
        f"  gpu_count:            {info.get('gpu_count', 0)}",
        f"  gpu_names:            {info.get('gpu_names', [])}",
        f"  nvidia_smi_ok:        {info.get('nvidia_smi_ok', False)}",
        f"  libcuda_ok:           {info.get('libcuda_ok', False)}",
        f"  dev_nvidia:           {info.get('dev_nvidia', [])}",
        f"  compat_ok:            {info.get('compat_ok')}",
    ]
    if info.get("compat_error"):
        lines.append(f"  compat_error:         {info['compat_error']}")
    if info.get("error_803"):
        lines.append("  error_803:            True (driver/runtime mismatch)")
    lines.append("--- end CUDA environment info ---")
    for line in lines:
        _log.info(line)
    # Also print to stdout for external systems (rent.py)
    for line in lines:
        print(line, flush=True)


def _cuda_failure_diagnostics() -> None:
    """
    Emit a concise FATAL diagnostic block when CUDA is unavailable.
    Production-only: runs on failure so rent.py can recycle the instance.
    """
    lines = [
        "--- CUDA preflight FATAL diagnostics ---",
    ]
    try:
        import torch
        lines.append(f"torch.version={getattr(torch, '__version__', 'unknown')}")
        lines.append(f"torch.cuda.is_available()={torch.cuda.is_available()}")
        lines.append(f"torch.cuda.device_count()={torch.cuda.device_count()}")
    except Exception as e:
        lines.append(f"torch import/query failed: {e!r}")

    try:
        nvidia_devs = sorted(Path("/dev").glob("nvidia*"))
        lines.append(f"/dev/nvidia* present: {[str(p) for p in nvidia_devs]}")
    except Exception as e:
        lines.append(f"/dev/nvidia* list failed: {e!r}")

    try:
        ctypes.CDLL("libcuda.so.1")
        lines.append("libcuda.so.1: dlopen success")
    except Exception as e:
        lines.append(f"libcuda.so.1: dlopen failed ({e!r})")

    smi = _find_nvidia_smi()
    if smi:
        try:
            p = subprocess.run(
                [str(smi), "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3,
                check=False,
            )
            out = (p.stdout or "").strip()
            err = (p.stderr or "").strip()
            if out:
                lines.append(f"nvidia-smi -L: {out[:500]}")
            if err:
                lines.append(f"nvidia-smi stderr: {err[:300]}")
            if not out and not err:
                lines.append(f"nvidia-smi -L rc={p.returncode}")
        except Exception as e:
            lines.append(f"nvidia-smi -L failed: {e!r}")
    else:
        lines.append("nvidia-smi: not found")

    lines.append("--- end FATAL diagnostics ---")
    for line in lines:
        print(line, flush=True)


def _try_torch_cuda_with_timeout(timeout_sec: float) -> tuple[bool, Optional[str]]:
    """
    Try torch.cuda.is_available() in a thread with a timeout.
    torch.cuda.is_available() can hang indefinitely on certain driver states,
    so we need a thread-based timeout wrapper.

    Also captures warnings (like Error 803) that appear during CUDA initialization.

    Returns: (success: bool, error_msg: Optional[str])
      - (True, None) if CUDA is available
      - (False, "timeout") if the call didn't complete within timeout_sec
      - (False, error_msg) if torch.cuda.is_available() raised an exception or warning
    """
    result = {"available": False, "error": None, "done": False, "warnings": []}

    def _check_cuda():
        import warnings
        try:
            # Capture warnings (Error 803 appears as a UserWarning)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                import torch
                available = torch.cuda.is_available()
                result["available"] = available
                # Store warning messages
                if w:
                    result["warnings"] = [str(warn.message) for warn in w]
                result["done"] = True
        except Exception as e:
            result["error"] = str(e)
            result["done"] = True

    thread = threading.Thread(target=_check_cuda, daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)

    if not result["done"]:
        # Thread didn't finish - torch.cuda.is_available() is hung
        return (False, "timeout")

    # Check for Error 803 in warnings first (most common case)
    if result.get("warnings"):
        for warn_msg in result["warnings"]:
            warn_lower = warn_msg.lower()
            if "803" in warn_lower or "unsupported display driver" in warn_lower or "cuda driver combination" in warn_lower:
                # Return the warning as an error so it triggers fast-fail
                return (False, f"Error 803 warning: {warn_msg[:300]}")

    if result["error"] is not None:
        return (False, result["error"])

    return (result["available"], None)


def _wait_for_cuda(grace_sec: int) -> bool:
    """
    Wait up to grace_sec for CUDA to become available (TRELLIS2_DEVICE=cuda only).
    Returns True if CUDA is available, False otherwise. Never exits the process;
    caller (e.g. preload worker) should set error state and continue serving /ready.

    Fast-fail: if driver/runtime incompatibility is detected (e.g. error 803),
    fails within ~10s instead of polling for the full grace period.

    CRITICAL: torch.cuda.is_available() can hang indefinitely on certain driver states.
    We use a thread-based timeout wrapper to prevent indefinite hangs.
    """
    global _CUDA_ENV_INFO
    if _get_device().lower() != "cuda":
        _log.info("CUDA preflight skipped (TRELLIS2_DEVICE!=cuda)")
        return True

    # Phase 1: fast environment check (~5s). Detect driver/runtime mismatch before
    # entering the slow polling loop. This catches error 803 and similar issues.
    _log.info("CUDA preflight phase 1: checking driver/runtime compatibility")
    with _READY_LOCK:
        _READY["detail"] = "checking driver/runtime compatibility"
    env_info = _cuda_env_info()
    _CUDA_ENV_INFO = env_info
    _log_cuda_env(env_info)

    # Fast-fail on known-incompatible configurations
    if env_info.get("compat_ok") is False:
        compat_err = env_info.get("compat_error", "driver/runtime incompatible")
        _log.error("CUDA preflight FAST FAIL: %s", compat_err)
        print(f"CUDA preflight FAST FAIL: {compat_err}", flush=True)
        with _READY_LOCK:
            _READY["detail"] = f"CUDA driver incompatible: {compat_err}"
            _READY["cuda_env"] = env_info
        _cuda_failure_diagnostics()
        return False

    # Fast-fail if nvidia-smi is present but no GPUs found and no /dev/nvidia* devices
    if not env_info.get("nvidia_smi_ok") and not env_info.get("dev_nvidia"):
        smi = _find_nvidia_smi()
        if smi is not None:
            _log.error("CUDA preflight FAST FAIL: nvidia-smi present but failed; no /dev/nvidia* devices")
            print("CUDA preflight FAST FAIL: nvidia-smi present but failed; no GPU devices visible", flush=True)
            with _READY_LOCK:
                _READY["detail"] = "no GPU devices visible (nvidia-smi failed, no /dev/nvidia*)"
                _READY["cuda_env"] = env_info
            _cuda_failure_diagnostics()
            return False

    # Phase 2: poll torch.cuda.is_available() with thread-based timeout to prevent hangs
    # Use an aggressive timeout per attempt (20s) since torch can hang indefinitely
    torch_check_timeout = max(20, min(30, int(os.environ.get("CUDA_TORCH_CHECK_TIMEOUT_SEC", "20") or "20")))
    fast_fail_sec = max(10, min(45, int(os.environ.get("CUDA_FAST_FAIL_SEC", "30") or "30")))
    _log.info(
        "CUDA preflight phase 2: waiting up to %s s for torch CUDA init "
        "(per-attempt timeout %s s, fast-fail after %s s if no progress, full grace %s s)",
        grace_sec, torch_check_timeout, fast_fail_sec, grace_sec,
    )
    with _READY_LOCK:
        _READY["detail"] = "waiting for torch CUDA initialization"

    deadline = time.time() + grace_sec
    fast_fail_deadline = time.time() + fast_fail_sec
    last_log = 0.0
    log_interval = 10.0  # Log every 10s for better visibility
    first_torch_error: Optional[str] = None
    hung_count = 0  # Track how many times torch.cuda.is_available() hung
    unavailable_count = 0  # Track how many times CUDA was unavailable (False, not error/hung)

    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            # Try torch.cuda.is_available() with a timeout wrapper
            _log.info("CUDA preflight: checking torch.cuda.is_available() (attempt #%d, timeout %ds)", attempt, torch_check_timeout)
            success, error_msg = _try_torch_cuda_with_timeout(torch_check_timeout)

            if error_msg == "timeout":
                hung_count += 1
                _log.warning(
                    "CUDA preflight: torch.cuda.is_available() hung for %ds (hung_count=%d)",
                    torch_check_timeout, hung_count
                )
                print(
                    f"[worker] CUDA preflight: torch.cuda.is_available() HUNG for {torch_check_timeout}s "
                    f"(attempt #{attempt}, hung_count={hung_count}). Likely driver issue.",
                    flush=True
                )
                # If torch has hung multiple times, fast-fail
                if hung_count >= 2:
                    _log.error("CUDA preflight FAST FAIL: torch.cuda.is_available() hung %d times", hung_count)
                    print(
                        f"CUDA preflight FAST FAIL: torch.cuda.is_available() hung {hung_count} times. "
                        f"Host driver likely incompatible or soft-locked.",
                        flush=True
                    )
                    env_info["compat_ok"] = False
                    env_info["compat_error"] = f"torch.cuda.is_available() hung {hung_count} times ({torch_check_timeout}s timeout)"
                    _CUDA_ENV_INFO = env_info
                    with _READY_LOCK:
                        _READY["detail"] = f"CUDA init hung (torch blocked {hung_count}x)"
                        _READY["cuda_env"] = env_info
                    _cuda_failure_diagnostics()
                    return False
                # Continue polling - maybe next attempt will work
                continue

            if error_msg is not None:
                # torch.cuda.is_available() raised an exception
                if first_torch_error is None:
                    first_torch_error = error_msg
                    _log.warning("CUDA preflight: torch error on attempt #%d: %s", attempt, first_torch_error)
                    print(f"[worker] CUDA preflight: torch error: {first_torch_error[:200]}", flush=True)
                # Fast-fail on Error 803 (even if it's in a warning, not an exception)
                # Error 803 means the driver/runtime combo is incompatible at the ABI/API level
                err_lower = error_msg.lower()
                if "803" in err_lower or "unsupported display driver" in err_lower:
                    _log.error("CUDA preflight FAST FAIL: Error 803 detected (driver/runtime incompatible)")
                    print(
                        f"CUDA preflight FAST FAIL: Error 803 detected in torch error message. "
                        f"Driver/runtime incompatible at ABI/API level.",
                        flush=True
                    )
                    env_info["compat_ok"] = False
                    env_info["compat_error"] = f"Error 803: {error_msg[:300]}"
                    env_info["error_803"] = True
                    _CUDA_ENV_INFO = env_info
                    with _READY_LOCK:
                        _READY["detail"] = "CUDA Error 803 (driver/runtime incompatible)"
                        _READY["cuda_env"] = env_info
                    _cuda_failure_diagnostics()
                    return False

            # If torch.cuda.is_available() returns False (not error, not hung, just False),
            # this usually means Error 803 or similar driver issue. After 3 consecutive
            # False returns within fast_fail window, give up.
            if not success and error_msg is None:
                unavailable_count += 1
                if unavailable_count >= 3 and time.time() < fast_fail_deadline + 10:
                    # Within extended fast-fail window, if CUDA consistently unavailable, bail
                    _log.error(
                        "CUDA preflight FAST FAIL: torch.cuda.is_available() returned False %d times "
                        "(driver issue likely, e.g. Error 803)",
                        unavailable_count
                    )
                    print(
                        f"CUDA preflight FAST FAIL: torch.cuda.is_available() returned False {unavailable_count} times. "
                        f"Driver/runtime likely incompatible (Error 803 or similar).",
                        flush=True
                    )
                    env_info["compat_ok"] = False
                    env_info["compat_error"] = "torch.cuda.is_available() consistently False (likely Error 803)"
                    _CUDA_ENV_INFO = env_info
                    with _READY_LOCK:
                        _READY["detail"] = "CUDA consistently unavailable (driver issue)"
                        _READY["cuda_env"] = env_info
                    _cuda_failure_diagnostics()
                    return False

            if success:
                # CUDA is available! Verify we can get device name
                try:
                    import torch
                    if torch.cuda.device_count() > 0:
                        name = torch.cuda.get_device_name(0)
                        _log.info("CUDA preflight ok: %s (attempt #%d)", name, attempt)
                        print(f"[worker] CUDA preflight OK: {name} (took {attempt} attempt(s))", flush=True)
                        # Update env info with successful torch init
                        env_info["torch_cuda_available"] = True
                        try:
                            env_info["gpu_count"] = torch.cuda.device_count()
                            env_info["gpu_names"] = [
                                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
                            ]
                        except Exception:
                            pass
                        _CUDA_ENV_INFO = env_info
                        with _READY_LOCK:
                            _READY["cuda_env"] = env_info
                        return True
                except Exception as e:
                    _log.warning("CUDA available but device_count/get_device_name failed: %s", e)
        except Exception as e:
            _log.warning("CUDA preflight: unexpected error in attempt #%d: %s", attempt, e)

        now = time.time()

        # Fast-fail check: if we're past the fast-fail deadline and torch has reported an error
        # that looks like a driver mismatch, bail out immediately
        if now > fast_fail_deadline and first_torch_error:
            err_lower = first_torch_error.lower()
            fast_fail_markers = (
                "803", "unsupported display driver", "cuda driver version is insufficient",
                "no cuda gpus are available", "cuda error",
                "forward compatibility was attempted on non supported hw",
            )
            if any(m in err_lower for m in fast_fail_markers):
                _log.error(
                    "CUDA preflight FAST FAIL after %ds: torch reported driver issue: %s",
                    fast_fail_sec, first_torch_error,
                )
                print(
                    f"CUDA preflight FAST FAIL: torch CUDA init error (likely driver mismatch): "
                    f"{first_torch_error[:300]}",
                    flush=True,
                )
                env_info["compat_ok"] = False
                env_info["compat_error"] = first_torch_error[:500]
                _CUDA_ENV_INFO = env_info
                with _READY_LOCK:
                    _READY["detail"] = f"CUDA init failed: {first_torch_error[:200]}"
                    _READY["cuda_env"] = env_info
                _cuda_failure_diagnostics()
                return False

        if now - last_log >= log_interval:
            remaining = max(0, int(deadline - now))
            _log.info("CUDA preflight still waiting (attempt #%d, ~%d s remaining, hung_count=%d)", attempt, remaining, hung_count)
            with _READY_LOCK:
                _READY["detail"] = f"waiting for CUDA device (~{remaining}s remaining, attempt #{attempt})"
            last_log = now

        if now + 1.0 > deadline:
            break

        # Brief sleep between attempts (but not too long - we want to retry quickly)
        time.sleep(2.0)

    _log.error(
        "CUDA not available after %s s grace (attempts=%d, hung_count=%d); setting error state (no exit)",
        grace_sec, attempt, hung_count,
    )
    print(
        f"CUDA not available after preflight grace (attempts={attempt}, hung_count={hung_count}); "
        f"see diagnostics below (process continues).",
        flush=True
    )
    _CUDA_ENV_INFO = env_info
    with _READY_LOCK:
        _READY["cuda_env"] = env_info
    _cuda_failure_diagnostics()
    return False


def _prepend_env_path(name: str, value: str) -> None:
    value = str(value or "").strip()
    if not value:
        return
    current = str(os.environ.get(name, "")).strip()
    parts = [p for p in current.split(":") if p]
    if value in parts:
        return
    os.environ[name] = f"{value}:{current}" if current else value


def _ensure_cuda_linker_paths() -> None:
    """
    Best-effort runtime repair for hosts where libcuda is mounted only as libcuda.so.1.
    Triton/flex_gemm compiles helper modules with `-lcuda` and needs libcuda.so visible.
    """
    global _CUDA_RUNTIME_PREPARED
    if _CUDA_RUNTIME_PREPARED:
        return
    _CUDA_RUNTIME_PREPARED = True

    if _get_device().lower() != "cuda":
        return

    # If libcuda.so already resolves, nothing to do.
    try:
        ctypes.CDLL("libcuda.so")
        return
    except Exception:
        pass

    candidate_patterns = (
        "/usr/lib/x86_64-linux-gnu/libcuda.so*",
        "/lib/x86_64-linux-gnu/libcuda.so*",
        "/usr/lib64/libcuda.so*",
        "/usr/local/nvidia/lib64/libcuda.so*",
        "/usr/local/cuda/compat/libcuda.so*",
        "/usr/lib/wsl/lib/libcuda.so*",
    )
    candidates = []
    for pat in candidate_patterns:
        try:
            candidates.extend(Path("/").glob(pat.lstrip("/")))
        except Exception:
            continue

    # Prefer an actual linker name first, then .so.1.
    preferred = None
    for p in candidates:
        if p.name == "libcuda.so" and p.exists():
            preferred = p
            break
    if preferred is None:
        for p in candidates:
            if p.name.startswith("libcuda.so.") and p.exists():
                preferred = p
                break

    if preferred is None:
        print("[worker] cuda-preflight: libcuda not found in known paths", flush=True)
        return

    link_path = preferred
    if preferred.name != "libcuda.so":
        # Try to create sibling linker name if writable; otherwise create in /tmp/libcuda.
        sibling = preferred.with_name("libcuda.so")
        if sibling.exists():
            link_path = sibling
        else:
            created = None
            try:
                sibling.symlink_to(preferred.name)
                created = sibling
            except Exception:
                try:
                    tmp_dir = Path("/tmp/libcuda")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    tmp_link = tmp_dir / "libcuda.so"
                    if tmp_link.exists() or tmp_link.is_symlink():
                        tmp_link.unlink()
                    tmp_link.symlink_to(preferred)
                    created = tmp_link
                except Exception:
                    created = None
            if created is not None:
                link_path = created

    _prepend_env_path("LIBRARY_PATH", str(link_path.parent))
    _prepend_env_path("LD_LIBRARY_PATH", str(link_path.parent))
    os.environ.setdefault("TRITON_LIBCUDA_PATH", str(link_path.parent))

    try:
        ctypes.CDLL("libcuda.so")
        print(f"[worker] cuda-preflight: using {link_path}", flush=True)
    except Exception as e:
        print(f"[worker] cuda-preflight: libcuda still unresolved ({e})", flush=True)


def _is_probably_driver_runtime_error(exc: BaseException) -> bool:
    s = str(exc or "").lower()
    if not s:
        return False
    markers = (
        "0 active drivers",
        "cuda driver",
        "cuda initialization error",
        "cuda unknown error",
        "forward compatibility was attempted on non supported hw",
        "libcuda",
        "803",  # error 803: unsupported display driver / cuda driver combination
        "unsupported display driver",
        "cuda driver version is insufficient",
        "no cuda gpus are available",
        "driver/library version mismatch",
    )
    return any(m in s for m in markers)


def _lazy_import_pipeline():
    # TRELLIS.2 is typically installed by setting PYTHONPATH to the repo root.
    from trellis2.pipelines import Trellis2ImageTo3DPipeline  # type: ignore

    return Trellis2ImageTo3DPipeline


def _bool_env(name: str, default: bool = False) -> bool:
    v = str(os.environ.get(name, "")).strip().lower()
    if not v:
        return default
    if v in ("1", "true", "t", "yes", "y", "on"):
        return True
    if v in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _configure_hf_timeouts_for_deadline(deadline: Optional[float]) -> None:
    """
    When PRELOAD_TIMEOUT/deadline is active, cap Hugging Face hub env timeouts
    to the remaining budget so from_pretrained cannot block past the deadline.
    Uses setdefault so user-set HF_HUB_* vars are not overridden.
    """
    if deadline is None:
        return
    remaining = int(max(1, deadline - time.time()))
    download_timeout = min(30, remaining)
    etag_timeout = min(30, remaining)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(etag_timeout))


def _should_avoid_gated_deps() -> bool:
    # Using a different image conditioner than the one specified by the checkpoint (pipeline.json)
    # can severely degrade output quality. Default to upstream behavior unless explicitly overridden.
    return _bool_env("TRELLIS2_AVOID_GATED_DEPS", False)


def _should_avoid_gated_rembg_deps() -> bool:
    # RMBG-2.0 is gated on HF; default to a public alternative.
    return _bool_env("TRELLIS2_AVOID_GATED_REMBG_DEPS", True)


def _is_gated_repo_error(exc: BaseException) -> bool:
    msg = (str(exc) or "").lower()
    # Common HF hub/transformers error strings for gated repos.
    if "gated repo" in msg:
        return True
    if "cannot access gated repo" in msg:
        return True
    if "access to model" in msg and "restricted" in msg:
        return True
    if "403" in msg and "huggingface.co" in msg:
        return True
    return False


def _build_pipeline_with_image_cond_override(model_id: str):
    """
    Build a Trellis2ImageTo3DPipeline but override the image conditioning model
    to avoid gated dependencies (e.g. DINOv3 on HF).
    """
    from trellis2.pipelines.base import Pipeline  # type: ignore
    from trellis2.pipelines import samplers, rembg  # type: ignore
    from trellis2.modules import image_feature_extractor  # type: ignore

    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    # Load pipeline.json + model weights without instantiating the image conditioner yet.
    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, model_id, "pipeline.json")  # type: ignore[attr-defined]
    args = getattr(pipe, "_pretrained_args", None) or {}

    # Prefer a non-gated vision backbone. TRELLIS.2 ships DinoV2FeatureExtractor.
    dinov2_name = os.environ.get("TRELLIS2_DINOV2_MODEL_NAME", "dinov2_vitl14_reg").strip()
    args["image_cond_model"] = {"name": "DinoV2FeatureExtractor", "args": {"model_name": dinov2_name}}

    if _should_avoid_gated_rembg_deps():
        # Default BiRefNet model is public; RMBG-2.0 is gated.
        rembg_id = os.environ.get("TRELLIS2_REMBG_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
        args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": rembg_id}}

    # Recreate the rest of the pipeline fields, matching upstream behavior.
    pipe.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["sparse_structure_sampler"]["args"]
    )
    pipe.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["shape_slat_sampler"]["args"]
    )
    pipe.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["tex_slat_sampler"]["args"]
    )
    pipe.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.shape_slat_normalization = args["shape_slat_normalization"]  # type: ignore[attr-defined]
    pipe.tex_slat_normalization = args["tex_slat_normalization"]  # type: ignore[attr-defined]

    pipe.image_cond_model = getattr(image_feature_extractor, args["image_cond_model"]["name"])(  # type: ignore[attr-defined]
        **args["image_cond_model"]["args"]
    )
    pipe.rembg_model = getattr(rembg, args["rembg_model"]["name"])(**args["rembg_model"]["args"])  # type: ignore[attr-defined]

    pipe.low_vram = args.get("low_vram", True)  # type: ignore[attr-defined]
    pipe.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")  # type: ignore[attr-defined]
    pipe.pbr_attr_layout = {  # type: ignore[attr-defined]
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipe._device = "cpu"  # type: ignore[attr-defined]

    return pipe


def _build_pipeline_with_rembg_override(model_id: str):
    """
    Build a Trellis2ImageTo3DPipeline but override only the rembg (background removal) model
    to avoid gated dependencies (e.g. briaai/RMBG-2.0 on HF) while keeping the default image
    conditioning model (DINOv3) intact.
    """
    from trellis2.pipelines.base import Pipeline  # type: ignore
    from trellis2.pipelines import samplers, rembg  # type: ignore
    from trellis2.modules import image_feature_extractor  # type: ignore

    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()
    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, model_id, "pipeline.json")  # type: ignore[attr-defined]
    args = getattr(pipe, "_pretrained_args", None) or {}

    rembg_id = os.environ.get("TRELLIS2_REMBG_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
    args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": rembg_id}}

    pipe.sparse_structure_sampler = getattr(samplers, args["sparse_structure_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["sparse_structure_sampler"]["args"]
    )
    pipe.sparse_structure_sampler_params = args["sparse_structure_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.shape_slat_sampler = getattr(samplers, args["shape_slat_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["shape_slat_sampler"]["args"]
    )
    pipe.shape_slat_sampler_params = args["shape_slat_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.tex_slat_sampler = getattr(samplers, args["tex_slat_sampler"]["name"])(  # type: ignore[attr-defined]
        **args["tex_slat_sampler"]["args"]
    )
    pipe.tex_slat_sampler_params = args["tex_slat_sampler"]["params"]  # type: ignore[attr-defined]

    pipe.shape_slat_normalization = args["shape_slat_normalization"]  # type: ignore[attr-defined]
    pipe.tex_slat_normalization = args["tex_slat_normalization"]  # type: ignore[attr-defined]

    pipe.image_cond_model = getattr(image_feature_extractor, args["image_cond_model"]["name"])(  # type: ignore[attr-defined]
        **args["image_cond_model"]["args"]
    )
    pipe.rembg_model = getattr(rembg, args["rembg_model"]["name"])(**args["rembg_model"]["args"])  # type: ignore[attr-defined]

    pipe.low_vram = args.get("low_vram", True)  # type: ignore[attr-defined]
    pipe.default_pipeline_type = args.get("default_pipeline_type", "1024_cascade")  # type: ignore[attr-defined]
    pipe.pbr_attr_layout = {  # type: ignore[attr-defined]
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    pipe._device = "cpu"  # type: ignore[attr-defined]

    return pipe


def _get_pipeline(deadline: Optional[float] = None):
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    # CUDA is checked in _preload_worker via _wait_for_cuda before we load the pipeline.
    _ensure_cuda_linker_paths()

    if _bool_env("TRELLIS2_DISABLE_TRITON", False):
        os.environ["TRELLIS2_DISABLE_TRITON"] = "1"
        # Some stacks respect this to avoid loading Triton/flex_gemm paths.
        os.environ.setdefault("TRITON_DISABLE", "1")

    with _READY_LOCK:
        if _READY["status"] not in ("ready", "error"):
            _READY["status"] = "loading_weights"
            if _READY["started_at"] == 0.0:
                _READY["started_at"] = time.time()
            _READY["detail"] = "CUDA ready, loading pipeline"

    model_id = os.environ.get("TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B")
    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    retries = max(1, int(os.environ.get("TRELLIS2_DRIVER_RETRY_ATTEMPTS", "4") or "4"))
    retry_sleep_sec = max(1, int(os.environ.get("TRELLIS2_DRIVER_RETRY_SLEEP_SEC", "8") or "8"))
    last_err = None

    for attempt in range(1, retries + 1):
        if deadline is not None and time.time() > deadline:
            _msg = (
                "Model preload timeout. Likely CUDA initialization failure or stalled download."
                f" HF_HUB_DOWNLOAD_TIMEOUT={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}"
                f" HF_HUB_ETAG_TIMEOUT={os.environ.get('HF_HUB_ETAG_TIMEOUT')}"
            )
            raise RuntimeError(_msg)
        try:
            _configure_hf_timeouts_for_deadline(deadline)
            with _READY_LOCK:
                _READY["status"] = "downloading_model"
                _READY["detail"] = "downloading/loading model weights"
            if _should_avoid_gated_deps():
                pipe = _build_pipeline_with_image_cond_override(model_id)
            elif _should_avoid_gated_rembg_deps():
                pipe = _build_pipeline_with_rembg_override(model_id)
            else:
                pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
            with _READY_LOCK:
                _READY["status"] = "loading_weights"
                _READY["detail"] = "moving to device"
            break
        except Exception as e:
            last_err = e
            # If the upstream config points at a gated model, fall back to a non-gated alternative.
            if _is_gated_repo_error(e):
                with _READY_LOCK:
                    _READY["status"] = "loading_weights"
                    _READY["detail"] = "moving to device"
                if _should_avoid_gated_deps():
                    pipe = _build_pipeline_with_image_cond_override(model_id)
                elif _should_avoid_gated_rembg_deps():
                    pipe = _build_pipeline_with_rembg_override(model_id)
                else:
                    raise
                break
            if attempt < retries and _is_probably_driver_runtime_error(e):
                print(
                    f"[worker] preload: driver/runtime not ready (attempt {attempt}/{retries}): {e}",
                    flush=True,
                )
                _ensure_cuda_linker_paths()
                time.sleep(retry_sleep_sec)
                continue
            raise
    else:
        raise RuntimeError(f"Pipeline init retries exhausted: {last_err}")
    dev = _get_device()
    if dev == "cuda":
        pipe.cuda()
    else:
        pipe.to(dev)
    _PIPELINE = pipe

    with _READY_LOCK:
        _READY["status"] = "ready"
        _READY["detail"] = f"pipeline loaded ({model_id})"
        _READY["ready_at"] = time.time()

    return _PIPELINE


def get_ready_state() -> dict:
    with _READY_LOCK:
        return dict(_READY)


def get_cuda_env_info() -> Optional[dict]:
    """Return cached CUDA environment info gathered during preflight, or None if not yet run."""
    return _CUDA_ENV_INFO


def _preload_heartbeat(interval_sec: float, started_at: float) -> None:
    """Log progress every interval_sec while preload is still loading/downloading."""
    while True:
        time.sleep(interval_sec)
        with _READY_LOCK:
            status = _READY.get("status", "")
            if status in ("ready", "error"):
                return
            elapsed = time.time() - started_at
            detail = (_READY.get("detail") or "").strip() or "loading"
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        print(f"[worker] preload: still {status} ({mins}m {secs}s elapsed) {detail[:80]}", flush=True)


def _preload_worker():
    global _PIPELINE
    # CUDA check runs in background after server is up; no process exit, just error state if unavailable.
    # With the fast-fail mechanism in _wait_for_cuda, driver incompatibility is detected in ~10-15s.
    # The grace_sec is the *maximum* wait; actual failure may be much faster.
    raw = str(os.environ.get("CUDA_PREFLIGHT_GRACE_SEC", str(_MIN_STARTUP_WAIT_SEC)) or str(_MIN_STARTUP_WAIT_SEC)).strip()
    try:
        requested = int(raw)
        grace_sec = max(_MIN_STARTUP_WAIT_SEC, min(600, requested))
    except ValueError:
        grace_sec = _MIN_STARTUP_WAIT_SEC
    if not _wait_for_cuda(grace_sec):
        env_info = _CUDA_ENV_INFO or {}
        err_detail = env_info.get("compat_error", "")
        with _READY_LOCK:
            _READY["status"] = "error"
            if err_detail:
                _READY["detail"] = f"CUDA driver incompatible: {err_detail[:300]}"
            else:
                _READY["detail"] = "CUDA not available after preflight grace (see logs). Process continues; /ready will report not ready."
        _log.error("preload: aborting (CUDA unavailable); server remains up for /health and /ready")
        return

    preload_retries = max(1, _int_env("TRELLIS2_PRELOAD_RETRIES", 3))
    retry_base_sec = max(1, _int_env("TRELLIS2_PRELOAD_RETRY_BASE_SEC", 15))
    heartbeat_interval_sec = max(30, _int_env("TRELLIS2_PRELOAD_HEARTBEAT_SEC", 90))
    # At least 6 min for model loading/ready so we don't exit early on slow download or first load.
    preload_timeout = max(_MIN_STARTUP_WAIT_SEC, _int_env("PRELOAD_TIMEOUT", _MIN_STARTUP_WAIT_SEC))
    start_time = time.time()
    deadline = start_time + preload_timeout

    for attempt in range(1, preload_retries + 1):
        if time.time() > deadline:
            _msg = (
                "Model preload timeout. Likely CUDA initialization failure or stalled download."
                f" HF_HUB_DOWNLOAD_TIMEOUT={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}"
                f" HF_HUB_ETAG_TIMEOUT={os.environ.get('HF_HUB_ETAG_TIMEOUT')}"
            )
            raise RuntimeError(_msg)
        try:
            if attempt > 1:
                _PIPELINE = None
                with _READY_LOCK:
                    _READY["status"] = "loading_weights"
                    _READY["detail"] = f"retry {attempt}/{preload_retries}"
                print(f"[worker] preload: retry {attempt}/{preload_retries}", flush=True)
            else:
                _log.info("preload: starting model preload")
                print("[worker] preload: starting model preload", flush=True)

            # Heartbeat so long downloads/loads don't look stuck
            started_at = time.time()
            heartbeat = threading.Thread(
                target=_preload_heartbeat,
                args=(heartbeat_interval_sec, started_at),
                daemon=True,
            )
            heartbeat.start()

            try:
                _get_pipeline(deadline=deadline)
            finally:
                # Heartbeat thread exits when status becomes ready/error
                pass

            st = get_ready_state()
            dt = 0.0
            if st.get("started_at") and st.get("ready_at"):
                dt = float(st["ready_at"]) - float(st["started_at"])
            _log.info("preload: ready (load_time_sec=%.1f)", dt)
            print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)
            return
        except Exception as e:
            if attempt < preload_retries:
                sleep_sec = min(3600, retry_base_sec * (2 ** (attempt - 1)))
                with _READY_LOCK:
                    _READY["status"] = "loading_weights"
                    _READY["detail"] = f"preload failed attempt {attempt}/{preload_retries}, retry in {sleep_sec}s"
                print(
                    f"[worker] preload: attempt {attempt}/{preload_retries} failed: {e!r}; retrying in {sleep_sec}s",
                    flush=True,
                )
                if _is_cuda_oom(e):
                    _cuda_empty_cache()
                time.sleep(sleep_sec)
                continue
            tb = traceback.format_exc()
            with _READY_LOCK:
                _READY["status"] = "error"
                _READY["detail"] = tb[-4000:] if tb else "unknown error"
            _log.error("preload: ERROR (retries exhausted): %s", e, exc_info=True)
            print("[worker] preload: ERROR (retries exhausted)\n" + (tb or "unknown error"), flush=True)


def start_preload_in_background():
    """
    Start a background thread to load the model so /ready can become true without waiting for the first /generate.
    Controlled by TRELLIS2_PRELOAD=1/0 (default 1).
    """
    v = (os.environ.get("TRELLIS2_PRELOAD", "1") or "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return
    with _READY_LOCK:
        if _READY["status"] != "not_started":
            return
        _READY["status"] = "initializing_cuda"
        _READY["started_at"] = time.time()
        _READY["detail"] = "preload scheduled"
    _log.info("preload: scheduled in background")
    t = threading.Thread(target=_preload_worker, daemon=True)
    t.start()


def _preprocess_images(pipe, images: List[Image.Image]) -> List[Image.Image]:
    out = []
    for im in images:
        out.append(pipe.preprocess_image(im))
    return out


def _float_env(name: str, default: float) -> float:
    try:
        v = str(os.environ.get(name, "")).strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _has_transparency(im: Image.Image) -> bool:
    try:
        if im.mode == "RGBA":
            mn, mx = im.getchannel("A").getextrema()
            return mn < 255 or mx < 255
        if im.mode == "LA":
            mn, mx = im.getchannel("A").getextrema()
            return mn < 255 or mx < 255
        if im.mode == "P" and "transparency" in getattr(im, "info", {}):
            return True
    except Exception:
        pass
    return False


def _crop_to_alpha(im: Image.Image, pad: int = 8) -> Image.Image:
    """Crop around non-zero alpha to avoid huge transparent borders."""
    try:
        im = im.convert("RGBA")
        a = im.getchannel("A")
        bbox = a.getbbox()
        if not bbox:
            return im
        x0, y0, x1, y1 = bbox
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(im.size[0], x1 + pad)
        y1 = min(im.size[1], y1 + pad)
        return im.crop((x0, y0, x1, y1))
    except Exception:
        return im


def _prepare_input_image(im: Image.Image) -> Image.Image:
    """
    Prepare an image for pipeline.run().

    Transparent "asset" PNGs often have huge empty borders; cropping to alpha bbox improves
    conditioning and can prevent empty sparse sampling.
    """
    # Optionally force RGB by compositing alpha onto a white background. This can help when
    # downstream models don't behave well with RGBA inputs.
    if _bool_env("TRELLIS2_FORCE_RGB", False):
        try:
            im_rgba = im.convert("RGBA")
            bg = Image.new("RGBA", im_rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im_rgba).convert("RGB")
        except Exception:
            im = im.convert("RGB")

    if _bool_env("TRELLIS2_CROP_ALPHA", True) and _has_transparency(im):
        im = _crop_to_alpha(im)

    # Preserve alpha if present, otherwise normalize to RGB.
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in getattr(im, "info", {})):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")

    # Many "asset" PNGs are tiny after alpha-cropping; upscale to give the vision backbone
    # enough pixels to work with. This only upscales, never downscales.
    if _bool_env("TRELLIS2_UPSCALE_SMALL", True):
        try:
            target = _int_env("TRELLIS2_UPSCALE_TARGET", 512)
            mx = max(im.size)
            if mx > 0 and mx < target:
                scale = float(target) / float(mx)
                im = im.resize(
                    (max(1, int(im.width * scale)), max(1, int(im.height * scale))),
                    Image.Resampling.LANCZOS,
                )
        except Exception:
            pass

    return im


def _safe_alpha_bbox(alpha: np.ndarray, primary_thresh: float = 0.8) -> Optional[tuple[int, int, int, int]]:
    """
    Return bbox (x0,y0,x1,y1) inclusive for alpha mask.
    Tries a high threshold first (matching upstream), then falls back to any non-zero alpha.
    """
    if alpha.ndim != 2:
        return None
    h, w = alpha.shape
    if h <= 0 or w <= 0:
        return None

    def _bbox_for_thresh(t: float) -> Optional[tuple[int, int, int, int]]:
        coords = np.argwhere(alpha > (t * 255.0))
        if coords.size == 0:
            return None
        y0 = int(coords[:, 0].min())
        y1 = int(coords[:, 0].max())
        x0 = int(coords[:, 1].min())
        x1 = int(coords[:, 1].max())
        return (x0, y0, x1, y1)

    bb = _bbox_for_thresh(float(primary_thresh))
    if bb is None:
        bb = _bbox_for_thresh(0.0)
    return bb


def _crop_square_around_bbox(
    w: int,
    h: int,
    bbox_xyxy: tuple[int, int, int, int],
    pad_px: int = 0,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(1, int(max(x1 - x0, y1 - y0)))
    side = side + int(max(0, pad_px)) * 2

    half = side // 2
    left = int(round(cx - half))
    top = int(round(cy - half))
    right = left + side
    bottom = top + side

    left = max(0, min(w - 1, left))
    top = max(0, min(h - 1, top))
    right = max(left + 1, min(w, right))
    bottom = max(top + 1, min(h, bottom))
    return (left, top, right, bottom)  # PIL crop is (left, top, right, bottom) with right/bottom exclusive


def _is_rembg_dtype_mismatch(exc: BaseException) -> bool:
    s = (str(exc) or "").lower()
    if "input type" in s and "bias type" in s and ("c10::half" in s or "half" in s):
        return True
    if "input type" in s and "weight type" in s and "half" in s:
        return True
    return False


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if the exception looks like a CUDA/GPU out-of-memory error."""
    s = (str(exc) or "").lower()
    if "out of memory" in s:
        return True
    if "cuda error" in s and "2" in s:  # error code 2 = OOM
        return True
    return False


def _cuda_empty_cache() -> None:
    """Clear CUDA cache when using GPU so the next request can run without leftover VRAM."""
    if _get_device().lower() != "cuda":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # finish pending work (helps after prior OOM)
            torch.cuda.empty_cache()
            gc.collect()
    except Exception:
        pass


def _safe_preprocess_image(pipe, input_im: Image.Image) -> Image.Image:
    """
    Preprocess to match the official demo behavior, but with guardrails:
    - Avoid empty bbox when alpha is soft/low.
    - Optional padding around the alpha bbox.

    Returns an RGB PIL image (alpha premultiplied like upstream).
    """
    # Downscale large inputs (matches upstream).
    max_size = max(input_im.size) if input_im.size else 0
    if max_size > 0:
        scale = min(1.0, 1024.0 / float(max_size))
        if scale < 1.0:
            input_im = input_im.resize(
                (max(1, int(input_im.width * scale)), max(1, int(input_im.height * scale))),
                Image.Resampling.LANCZOS,
            )

    has_alpha = False
    try:
        if input_im.mode == "RGBA":
            a = np.array(input_im)[:, :, 3]
            if not np.all(a == 255):
                has_alpha = True
    except Exception:
        has_alpha = False

    # If no alpha, run RMBG to get an alpha matte.
    if not has_alpha:
        im_rgb = input_im.convert("RGB")
        low_vram = bool(getattr(pipe, "low_vram", False))
        rembg_model = getattr(pipe, "rembg_model", None)
        if rembg_model is None:
            out_rgba = im_rgb.convert("RGBA")
        else:
            if low_vram:
                try:
                    rembg_model.to(pipe.device)  # type: ignore[attr-defined]
                except Exception:
                    pass

            try:
                out_rgba = rembg_model(im_rgb)  # type: ignore[attr-defined]
            except Exception as e:
                # HuggingFace remote updates can change BiRefNet runtime behavior and trigger
                # float/half dtype mismatches at inference time. Retry once with model.float().
                if _is_rembg_dtype_mismatch(e):
                    print(
                        "[worker] rembg: dtype mismatch during preprocess; retrying with rembg model forced to float32",
                        flush=True,
                    )
                    try:
                        rembg_model.float()  # type: ignore[attr-defined]
                        if low_vram:
                            rembg_model.to(pipe.device)  # type: ignore[attr-defined]
                        out_rgba = rembg_model(im_rgb)  # type: ignore[attr-defined]
                    except Exception as e2:
                        if _bool_env("TRELLIS2_PREPROCESS_DISABLE_REMBG_ON_ERROR", True):
                            print(
                                f"[worker] rembg: retry failed ({e2}); continuing without rembg preprocessing",
                                flush=True,
                            )
                            out_rgba = im_rgb.convert("RGBA")
                        else:
                            raise
                elif _bool_env("TRELLIS2_PREPROCESS_DISABLE_REMBG_ON_ERROR", True):
                    print(
                        f"[worker] rembg: preprocess failed ({e}); continuing without rembg preprocessing",
                        flush=True,
                    )
                    out_rgba = im_rgb.convert("RGBA")
                else:
                    raise
            finally:
                if low_vram:
                    try:
                        rembg_model.cpu()  # type: ignore[attr-defined]
                    except Exception:
                        pass
    else:
        out_rgba = input_im.convert("RGBA")

    out_np = np.array(out_rgba)
    if out_np.ndim != 3 or out_np.shape[2] < 4:
        # If RMBG returns unexpected format, just ensure RGB.
        return out_rgba.convert("RGB")

    alpha = out_np[:, :, 3]
    thresh = _float_env("TRELLIS2_PREPROCESS_ALPHA_BBOX_THRESHOLD", 0.8)
    pad_px = _int_env("TRELLIS2_PREPROCESS_PAD_PX", 8)
    bb = _safe_alpha_bbox(alpha, primary_thresh=thresh)
    if bb is None:
        # Nothing to crop; return premultiplied RGB of the full image.
        rgb = out_np[:, :, :3].astype(np.float32) / 255.0
        a = alpha.astype(np.float32)[:, :, None] / 255.0
        rgb = rgb * a
        return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))

    w, h = int(out_rgba.size[0]), int(out_rgba.size[1])
    crop_box = _crop_square_around_bbox(w=w, h=h, bbox_xyxy=bb, pad_px=pad_px)
    cropped = out_rgba.crop(crop_box)

    cropped_np = np.array(cropped).astype(np.float32) / 255.0
    rgb = cropped_np[:, :, :3]
    a = cropped_np[:, :, 3:4]
    rgb = rgb * a
    return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))


def _fuse_cond(cond_tensor):
    """
    Fuse per-image conditioning into a single prompt conditioning.
    The extractor returns (B, N, D). For B>1 we mean-pool over B.
    """
    import torch

    if not isinstance(cond_tensor, torch.Tensor):
        raise TypeError(f"cond must be torch.Tensor, got {type(cond_tensor)}")
    if cond_tensor.ndim >= 1 and cond_tensor.shape[0] > 1:
        return cond_tensor.mean(dim=0, keepdim=True)
    return cond_tensor


def _get_fused_cond(pipe, images: List[Image.Image], resolution: int):
    """
    pipe.get_cond expects list[Image.Image]. We use it, then fuse batch dimension.
    """
    import torch

    d = pipe.get_cond(images, resolution)
    cond = _fuse_cond(d["cond"])
    neg = torch.zeros_like(cond)
    return {"cond": cond, "neg_cond": neg}


def _resolve_pipeline_type(requested: Optional[str]) -> str:
    v = (requested or "").strip()
    if v:
        return v
    return os.environ.get("TRELLIS2_PIPELINE_TYPE", "1024_cascade").strip() or "1024_cascade"


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return int(default)


def _choose_export_params(low_poly: bool):
    if low_poly:
        return (
            _int_env("TRELLIS2_DECIMATION_TARGET_LOW_POLY", 75000),
            _int_env("TRELLIS2_TEXTURE_SIZE_LOW_POLY", 2048),
        )
    return (
        _int_env("TRELLIS2_DECIMATION_TARGET", 1000000),
        _int_env("TRELLIS2_TEXTURE_SIZE", 4096),
    )


def generate_glb_from_image_bytes_list(
    images_bytes: List[bytes],
    out_dir: Path,
    low_poly: bool = False,
    seed: Optional[int] = None,
    pipeline_type: Optional[str] = None,
    preprocess_image: Optional[bool] = None,
    post_scale_z: Optional[float] = None,
    backup_inputs: bool = True,
    export_meta: Optional[dict] = None,
) -> Path:
    if not images_bytes:
        raise ValueError("empty upload")

    _cuda_empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pipe = _get_pipeline()
    try:
        _cuda_empty_cache()  # Free VRAM from previous request so this one can run
        imgs_raw: List[Image.Image] = []
        for raw in images_bytes:
            if not raw:
                continue
            im = Image.open(io.BytesIO(raw))
            # Keep a copy of the raw image (we'll decide preprocessing later).
            imgs_raw.append(im)
        if not imgs_raw:
            raise ValueError("empty image bytes")

        if backup_inputs:
            in_dir = out_dir / "inputs"
            in_dir.mkdir(parents=True, exist_ok=True)
            for idx, (raw, im) in enumerate(zip(images_bytes, imgs_raw)):
                try:
                    # Prefer PNG to keep alpha if present.
                    p = in_dir / f"{uuid.uuid4().hex}_{idx+1:02d}.png"
                    _prepare_input_image(im).save(str(p), format="PNG")
                except Exception:
                    pass

        # IMPORTANT:
        # Use the upstream pipeline's `run()` method so we inherit the correct preprocess,
        # sampler params, and cascade behavior. Re-implementing sampling with empty/default
        # params can produce garbage outputs even when it "succeeds".
        #
        # For now we use the first image only; multi-image fusion can be added later.
        import inspect
        import o_voxel  # type: ignore

        ptype = _resolve_pipeline_type(pipeline_type)
        img0_raw = imgs_raw[0]

        import torch

        # Default seed: deterministic for easier debugging; caller can override.
        if seed is None:
            seed = _int_env("TRELLIS2_DEFAULT_SEED", 42)

        # Decide preprocessing policy once, and then run `pipe.run(..., preprocess_image=False)`
        # to match the official demo (preprocess stage is explicit, sampling stage is pure).
        do_preprocess = _bool_env("TRELLIS2_PREPROCESS_IMAGE", True) if preprocess_image is None else bool(preprocess_image)
        primary_img = _safe_preprocess_image(pipe, img0_raw) if do_preprocess else _prepare_input_image(img0_raw).convert("RGB")
        fallback_img = _prepare_input_image(img0_raw).convert("RGB") if do_preprocess else None

        # Pass only kwargs supported by this pipeline version.
        kwargs: dict = {}
        try:
            sig = inspect.signature(pipe.run)  # type: ignore[attr-defined]
            if "seed" in sig.parameters:
                kwargs["seed"] = int(seed)
            if ptype and "pipeline_type" in sig.parameters:
                kwargs["pipeline_type"] = ptype
            if "num_samples" in sig.parameters:
                kwargs["num_samples"] = 1
            if "preprocess_image" in sig.parameters:
                kwargs["preprocess_image"] = False
            # HF demo defaults (app.py): pass explicit sampler params rather than relying on pipeline.json.
            if _bool_env("TRELLIS2_USE_DEMO_SAMPLER_DEFAULTS", True):
                if "sparse_structure_sampler_params" in sig.parameters:
                    kwargs["sparse_structure_sampler_params"] = {
                        "steps": _int_env("TRELLIS2_SS_STEPS", 12),
                        "guidance_strength": _float_env("TRELLIS2_SS_GUIDANCE_STRENGTH", 7.5),
                        "guidance_rescale": _float_env("TRELLIS2_SS_GUIDANCE_RESCALE", 0.7),
                        "rescale_t": _float_env("TRELLIS2_SS_RESCALE_T", 5.0),
                    }
                if "shape_slat_sampler_params" in sig.parameters:
                    kwargs["shape_slat_sampler_params"] = {
                        "steps": _int_env("TRELLIS2_SHAPE_STEPS", 12),
                        "guidance_strength": _float_env("TRELLIS2_SHAPE_GUIDANCE_STRENGTH", 7.5),
                        "guidance_rescale": _float_env("TRELLIS2_SHAPE_GUIDANCE_RESCALE", 0.5),
                        "rescale_t": _float_env("TRELLIS2_SHAPE_RESCALE_T", 3.0),
                    }
                if "tex_slat_sampler_params" in sig.parameters:
                    kwargs["tex_slat_sampler_params"] = {
                        "steps": _int_env("TRELLIS2_TEX_STEPS", 12),
                        "guidance_strength": _float_env("TRELLIS2_TEX_GUIDANCE_STRENGTH", 1.0),
                        "guidance_rescale": _float_env("TRELLIS2_TEX_GUIDANCE_RESCALE", 0.0),
                        "rescale_t": _float_env("TRELLIS2_TEX_RESCALE_T", 3.0),
                    }
        except Exception:
            # If signature inspection fails, still attempt a minimal call.
            kwargs["seed"] = int(seed)
            if ptype:
                kwargs["pipeline_type"] = ptype

        # Retry on empty sparse sampling by shifting the seed.
        def _is_empty_sparse_error(exc: BaseException) -> bool:
            msg = (str(exc) or "").lower()
            return ("input.numel() == 0" in msg) or ("max(): expected reduction dim" in msg) or ("empty sparse coords" in msg)

        retries = _int_env("TRELLIS2_EMPTY_SPARSE_RETRIES", 4)
        last_err: Optional[BaseException] = None

        # Some inputs get wiped out by preprocessing (background removal + crop), which can
        # lead to empty sparse coords. If that happens, we auto-fallback by switching to a
        # less destructive input variant once.
        input_fallback_attempted = False

        # Free VRAM after preprocessing (rembg etc.) so pipe.run() has maximum headroom.
        _cuda_empty_cache()

        attempt = 0
        max_attempts = max(1, retries)
        img0 = primary_img
        oom_run_retried = False
        while True:
            try:
                torch.manual_seed(int(seed) + attempt)
                if "seed" in kwargs:
                    kwargs["seed"] = int(seed) + attempt
                out = pipe.run(img0, **kwargs)  # type: ignore[attr-defined]
                if not out:
                    raise RuntimeError("pipeline.run() returned no outputs")
                mesh = out[0]
                break
            except Exception as e:
                last_err = e
                # One-time retry on CUDA OOM (e.g. reused instance with fragmented VRAM).
                if _is_cuda_oom(e) and not oom_run_retried:
                    oom_run_retried = True
                    _cuda_empty_cache()
                    print("[worker] pipe.run OOM, clearing cache and retrying once", flush=True)
                    continue
                if not _is_empty_sparse_error(e):
                    raise

                attempt += 1
                if attempt < max_attempts:
                    continue

                # If we exhausted seed-shift retries, switch input once and try again.
                if (not input_fallback_attempted) and (fallback_img is not None):
                    input_fallback_attempted = True
                    attempt = 0
                    img0 = fallback_img
                    continue

                raise RuntimeError("pipeline.run() failed after retries") from last_err

        # Export to GLB via o-voxel postprocess util.
        # Free VRAM from pipeline run before mesh.simplify (CuMesh uses substantial VRAM).
        _cuda_empty_cache()
        # On CUDA OOM during mesh.simplify, retry with lower decimation to stay within VRAM.
        decimation_target, texture_size = _choose_export_params(low_poly)
        decimation_initial = int(decimation_target)
        uid = uuid.uuid4().hex
        dec_min = _int_env("TRELLIS2_DECIMATION_MIN_OOM_FALLBACK", 15000)
        glb = None
        last_glb_err = None
        oom_retries = 0
        to_glb_attempts = 0

        if export_meta is not None:
            export_meta.update({
                "pipeline_type": ptype,
                "low_poly": low_poly,
                "texture_size": int(texture_size),
                "decimation_initial": decimation_initial,
                "input_image_count": len(images_bytes),
            })

        for attempt in range(_int_env("TRELLIS2_TO_GLB_OOM_RETRIES", 4)):
            to_glb_attempts = attempt + 1
            try:
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=int(decimation_target),
                    texture_size=int(texture_size),
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    verbose=True,
                )
                break
            except RuntimeError as e:
                last_glb_err = e
                if _is_cuda_oom(e) and attempt < 3 and decimation_target > dec_min:
                    oom_retries += 1
                    _cuda_empty_cache()
                    decimation_target = max(dec_min, decimation_target // 2)
                    print(
                        f"[worker] to_glb OOM, retrying with decimation_target={decimation_target}",
                        flush=True,
                    )
                    continue
                if export_meta is not None:
                    export_meta.update({
                        "oom_exhausted": True,
                        "oom_retries": oom_retries,
                        "to_glb_attempts": to_glb_attempts,
                        "decimation_final": int(decimation_target),
                        "error_snippet": (str(e) or "")[:500],
                    })
                raise
        if glb is None:
            if export_meta is not None:
                export_meta.update({
                    "oom_exhausted": True,
                    "oom_retries": oom_retries,
                    "to_glb_attempts": to_glb_attempts,
                    "decimation_final": int(decimation_target),
                })
            raise RuntimeError("to_glb failed") from last_glb_err

        if export_meta is not None:
            export_meta.update({
                "oom_retries": oom_retries,
                "to_glb_attempts": to_glb_attempts,
                "decimation_final": int(decimation_target),
            })

        # Optional post-export axis scaling. This is intentionally Z-only so callers can
        # compensate for known vertical squash without touching X/Y proportions.
        scale_z = _float_env("TRELLIS2_POST_SCALE_Z", 1.0) if post_scale_z is None else float(post_scale_z)
        if not np.isfinite(scale_z) or scale_z <= 0:
            scale_z = 1.0
        if abs(scale_z - 1.0) > 1e-6:
            glb.apply_scale([1.0, 1.0, float(scale_z)])
            try:
                glb.fix_normals()
            except Exception:
                pass

        out_path = out_dir / f"{uid}.glb"
        # Blender compatibility: many installs do not decode embedded WebP textures.
        # Default to PNG; enable WebP explicitly if you want smaller files.
        export_webp = _bool_env("TRELLIS2_EXPORT_WEBP", False)
        glb.export(str(out_path), extension_webp=export_webp)

        dt = time.time() - t0
        print(
            f"[worker] generated {out_path} in {dt:.1f}s "
            f"(inputs={len(images_bytes)} low_poly={low_poly} pipeline_type={ptype} post_scale_z={scale_z:.4f})",
            flush=True,
        )
        return out_path
    finally:
        _cuda_empty_cache()
