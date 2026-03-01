#!/usr/bin/env python3
"""
TRELLIS.2 worker implementation.

This module is imported by server.py.
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
import uuid
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

_log = logging.getLogger(__name__)

_PIPELINE = None
_READY = {
    "status": "not_started",  # not_started | downloading | loading | ready | error
    "detail": "",
    "started_at": 0.0,
    "ready_at": 0.0,
}
_READY_LOCK = threading.Lock()
_CUDA_RUNTIME_PREPARED = False

_GENERATING_LOCK = threading.Lock()
_GENERATING_COUNT = 0


def is_generating() -> bool:
    """True if at least one /generate call is currently in progress."""
    with _GENERATING_LOCK:
        return _GENERATING_COUNT > 0

# Idle shutdown: if no generation after ready for N sec, or no new generation for N sec after last job, exit (backup so server does not idle forever).
# Disabled by default; enable via TRELLIS2_IDLE_SHUTDOWN=1 or --idle-shutdown (see server.py).
_IDLE_SHUTDOWN_TIMER: Optional[threading.Timer] = None
_IDLE_SHUTDOWN_LOCK = threading.Lock()
# Overrides set by server.py from CLI (or can be set programmatically). None = use env.
_IDLE_SHUTDOWN_ENABLED_OVERRIDE: Optional[bool] = None
_IDLE_AFTER_READY_SEC_OVERRIDE: Optional[int] = None
_IDLE_AFTER_GENERATION_SEC_OVERRIDE: Optional[int] = None


def configure_idle_shutdown(
    enabled: Optional[bool] = None,
    after_ready_sec: Optional[int] = None,
    after_generation_sec: Optional[int] = None,
) -> None:
    """
    Override idle shutdown config (e.g. from CLI). None leaves existing override or env unchanged.
    Call before starting the server when using python server.py --idle-shutdown ...
    """
    global _IDLE_SHUTDOWN_ENABLED_OVERRIDE, _IDLE_AFTER_READY_SEC_OVERRIDE, _IDLE_AFTER_GENERATION_SEC_OVERRIDE
    if enabled is not None:
        _IDLE_SHUTDOWN_ENABLED_OVERRIDE = enabled
    if after_ready_sec is not None:
        _IDLE_AFTER_READY_SEC_OVERRIDE = after_ready_sec
    if after_generation_sec is not None:
        _IDLE_AFTER_GENERATION_SEC_OVERRIDE = after_generation_sec


def _idle_shutdown_enabled() -> bool:
    if _IDLE_SHUTDOWN_ENABLED_OVERRIDE is not None:
        return _IDLE_SHUTDOWN_ENABLED_OVERRIDE
    v = str(os.environ.get("TRELLIS2_IDLE_SHUTDOWN", "")).strip().lower()
    return v in ("1", "true", "t", "yes", "y", "on")


def _idle_after_ready_sec() -> int:
    if _IDLE_AFTER_READY_SEC_OVERRIDE is not None:
        return _IDLE_AFTER_READY_SEC_OVERRIDE
    try:
        return int(str(os.environ.get("TRELLIS2_IDLE_SHUTDOWN_AFTER_READY_SEC", "300")).strip() or "300")
    except Exception:
        return 300


def _idle_after_generation_sec() -> int:
    if _IDLE_AFTER_GENERATION_SEC_OVERRIDE is not None:
        return _IDLE_AFTER_GENERATION_SEC_OVERRIDE
    try:
        return int(str(os.environ.get("TRELLIS2_IDLE_SHUTDOWN_AFTER_GENERATION_SEC", "120")).strip() or "120")
    except Exception:
        return 120


def _get_device():
    return os.environ.get("TRELLIS2_DEVICE", "cuda")


# ---------------------------------------------------------------------------
# CUDA OOM helpers
# ---------------------------------------------------------------------------

def _is_cuda_oom(exc: BaseException) -> bool:
    """True if the exception looks like a CUDA/GPU out-of-memory error."""
    s = (str(exc) or "").lower()
    return "out of memory" in s or "cuda out of memory" in s


def _cuda_empty_cache() -> None:
    """Clear CUDA cache so the next request can run without leftover VRAM."""
    if _get_device().lower() != "cuda":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.ipc_collect()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Signal handling / cleanup
# ---------------------------------------------------------------------------

def _cleanup_on_exit(signum, frame):
    _log.info("received signal %s, cleaning up", signum)
    _cuda_empty_cache()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _cleanup_on_exit)
signal.signal(signal.SIGINT, _cleanup_on_exit)


# ---------------------------------------------------------------------------
# CUDA linker path repair (needed on some Vast.ai hosts)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def _lazy_import_pipeline():
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


def _int_env(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, "")).strip() or default)
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        v = str(os.environ.get(name, "")).strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _should_avoid_gated_deps() -> bool:
    return _bool_env("TRELLIS2_AVOID_GATED_DEPS", False)


def _should_avoid_gated_rembg_deps() -> bool:
    return _bool_env("TRELLIS2_AVOID_GATED_REMBG_DEPS", True)


def _is_gated_repo_error(exc: BaseException) -> bool:
    msg = (str(exc) or "").lower()
    if "gated repo" in msg:
        return True
    if "cannot access gated repo" in msg:
        return True
    if "access to model" in msg and "restricted" in msg:
        return True
    if "403" in msg and "huggingface.co" in msg:
        return True
    return False


def _runtime_downloads_allowed() -> bool:
    """
    If true (default), missing Hugging Face assets are downloaded at runtime.
    Set TRELLIS2_DISABLE_RUNTIME_DOWNLOADS=1 to require all assets to be pre-baked.
    """
    return not _bool_env("TRELLIS2_DISABLE_RUNTIME_DOWNLOADS", False)


def _split_repo_and_subpath(ref: str) -> tuple[Optional[str], Optional[str]]:
    s = str(ref or "").strip()
    if not s or s.startswith(("/", "./", "../")):
        return None, None
    parts = [p for p in s.split("/") if p]
    if len(parts) < 3:
        return None, None
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def _resolve_model_snapshot(model_ref: str, purpose: str) -> str:
    """
    Resolve a HF model reference to a local snapshot directory.
    """
    ref = str(model_ref or "").strip()
    if not ref:
        raise RuntimeError(f"{purpose}: empty model reference")

    if os.path.isdir(ref):
        return ref

    repo_id = ref
    parsed_repo_id, _ = _split_repo_and_subpath(ref)
    if parsed_repo_id:
        repo_id = parsed_repo_id

    local_only = not _runtime_downloads_allowed()

    try:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(repo_id, repo_type="model", local_files_only=local_only)
        _log.info(
            "resolved model snapshot: ref=%s repo=%s local_only=%s path=%s",
            ref,
            repo_id,
            local_only,
            snapshot_dir,
        )
        return snapshot_dir
    except Exception as e:
        if local_only:
            raise RuntimeError(
                f"{purpose}: local snapshot for '{repo_id}' not found and runtime downloads "
                "are disabled (TRELLIS2_DISABLE_RUNTIME_DOWNLOADS=1). Rebuild the image with "
                f"pre-downloaded assets or unset TRELLIS2_DISABLE_RUNTIME_DOWNLOADS. Original error: {e}"
            ) from e
        raise


def _link_cross_repo_dependencies(model_snapshot_dir: str) -> None:
    """
    Ensure cross-repo model references in pipeline.json resolve locally.
    """
    p = Path(model_snapshot_dir) / "pipeline.json"
    if not p.exists():
        return

    cfg = json.loads(p.read_text(encoding="utf-8"))
    models = (cfg.get("args") or {}).get("models") or {}
    if not isinstance(models, dict):
        return

    snapshot_root = Path(model_snapshot_dir).resolve()
    for ref in models.values():
        if not isinstance(ref, str):
            continue
        repo_id, _ = _split_repo_and_subpath(ref)
        if not repo_id:
            continue

        dep_snapshot = _resolve_model_snapshot(repo_id, f"pipeline dependency '{repo_id}'")
        dep_root = Path(dep_snapshot).resolve()
        if dep_root == snapshot_root:
            continue

        alias_path = Path(model_snapshot_dir) / repo_id
        if alias_path.exists() or alias_path.is_symlink():
            continue
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        alias_path.symlink_to(dep_snapshot, target_is_directory=True)


def _build_pipeline_with_image_cond_override(model_source: str):
    """
    Build a Trellis2ImageTo3DPipeline but override the image conditioning model
    to avoid gated dependencies (e.g. DINOv3 on HF).
    """
    from trellis2.pipelines.base import Pipeline  # type: ignore
    from trellis2.pipelines import samplers, rembg  # type: ignore
    from trellis2.modules import image_feature_extractor  # type: ignore

    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, model_source, "pipeline.json")  # type: ignore[attr-defined]
    args = getattr(pipe, "_pretrained_args", None) or {}

    dinov2_name = os.environ.get("TRELLIS2_DINOV2_MODEL_NAME", "dinov2_vitl14_reg").strip()
    args["image_cond_model"] = {"name": "DinoV2FeatureExtractor", "args": {"model_name": dinov2_name}}

    if _should_avoid_gated_rembg_deps():
        rembg_id = os.environ.get("TRELLIS2_REMBG_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
        rembg_source = _resolve_model_snapshot(rembg_id, f"rembg model '{rembg_id}'")
        args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": rembg_source}}

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


def _build_pipeline_with_rembg_override(model_source: str):
    """
    Build a Trellis2ImageTo3DPipeline but override only the rembg (background removal) model
    to avoid gated dependencies (e.g. briaai/RMBG-2.0 on HF) while keeping the default image
    conditioning model (DINOv3) intact.
    """
    from trellis2.pipelines.base import Pipeline  # type: ignore
    from trellis2.pipelines import samplers, rembg  # type: ignore
    from trellis2.modules import image_feature_extractor  # type: ignore

    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()
    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, model_source, "pipeline.json")  # type: ignore[attr-defined]
    args = getattr(pipe, "_pretrained_args", None) or {}

    rembg_id = os.environ.get("TRELLIS2_REMBG_MODEL_ID", "ZhengPeng7/BiRefNet").strip()
    rembg_source = _resolve_model_snapshot(rembg_id, f"rembg model '{rembg_id}'")
    args["rembg_model"] = {"name": "BiRefNet", "args": {"model_name": rembg_source}}

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


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _ensure_cuda_linker_paths()

    with _READY_LOCK:
        if _READY["status"] in ("not_started", "downloading"):
            _READY["status"] = "loading"
            if not _READY.get("started_at"):
                _READY["started_at"] = time.time()
            _READY["detail"] = "initializing pipeline"

    model_id = os.environ.get("TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B")
    model_source = _resolve_model_snapshot(model_id, f"pipeline model '{model_id}'")
    _link_cross_repo_dependencies(model_source)
    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    try:
        if _should_avoid_gated_deps():
            pipe = _build_pipeline_with_image_cond_override(model_source)
        elif _should_avoid_gated_rembg_deps():
            pipe = _build_pipeline_with_rembg_override(model_source)
        else:
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_source)
    except Exception as e:
        if _is_gated_repo_error(e):
            if _should_avoid_gated_deps():
                pipe = _build_pipeline_with_image_cond_override(model_source)
            elif _should_avoid_gated_rembg_deps():
                pipe = _build_pipeline_with_rembg_override(model_source)
            else:
                raise
        else:
            raise
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


# ---------------------------------------------------------------------------
# Readiness / preload
# ---------------------------------------------------------------------------

def get_ready_state() -> dict:
    with _READY_LOCK:
        return dict(_READY)


# ---------------------------------------------------------------------------
# Idle shutdown (backup: exit if no work after ready, or after last generation)
# ---------------------------------------------------------------------------

def _on_idle_shutdown_fire():
    _log.info("idle shutdown: no activity within window, exiting process")
    print("[worker] idle shutdown: exiting", flush=True)
    sys.exit(0)


def _schedule_idle_shutdown(seconds: float) -> None:
    """Schedule process exit in `seconds` if not cancelled. Cancels any existing timer."""
    global _IDLE_SHUTDOWN_TIMER
    with _IDLE_SHUTDOWN_LOCK:
        if _IDLE_SHUTDOWN_TIMER is not None:
            _IDLE_SHUTDOWN_TIMER.cancel()
            _IDLE_SHUTDOWN_TIMER = None
        if seconds <= 0:
            return
        _IDLE_SHUTDOWN_TIMER = threading.Timer(seconds, _on_idle_shutdown_fire)
        _IDLE_SHUTDOWN_TIMER.daemon = True
        _IDLE_SHUTDOWN_TIMER.start()
        _log.info("idle shutdown: scheduled exit in %.0fs", seconds)


def cancel_idle_shutdown() -> None:
    """Cancel any scheduled idle shutdown (e.g. when a generate request starts)."""
    global _IDLE_SHUTDOWN_TIMER
    with _IDLE_SHUTDOWN_LOCK:
        if _IDLE_SHUTDOWN_TIMER is not None:
            _IDLE_SHUTDOWN_TIMER.cancel()
            _IDLE_SHUTDOWN_TIMER = None
            _log.info("idle shutdown: cancelled")


def schedule_idle_shutdown_after_generation() -> None:
    """Call after a generate completes: reset timer so we exit if no new job in N sec."""
    if not _idle_shutdown_enabled():
        return
    _schedule_idle_shutdown(float(_idle_after_generation_sec()))


def _on_ready_start_idle_timer() -> None:
    """Call once when server becomes ready and queue is effectively empty: exit if no generation for N sec."""
    if not _idle_shutdown_enabled():
        return
    sec = _idle_after_ready_sec()
    if sec <= 0:
        return
    _schedule_idle_shutdown(float(sec))
    _log.info("idle shutdown: ready with no job; will exit in %s s if no generation", sec)


def _warmup_gpu_kernels(pipe) -> None:
    """
    Run minimal dummy passes to pre-compile Triton/FlexGEMM kernels before the first real job.
    Enabled by TRELLIS2_WARMUP=1 (default: off). Adds ~1–3 min to startup but eliminates
    JIT overhead on the first generation.
    """
    import inspect
    import torch

    print("[worker] warmup: pre-compiling GPU kernels (TRELLIS2_WARMUP=1)...", flush=True)
    t0 = time.time()
    try:
        dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8) + 128, mode="RGB")

        # 1. Warm up the image conditioning model (DINOv2 feature extractor).
        try:
            with torch.no_grad():
                pipe.get_cond([dummy], 512)
            print("[worker] warmup: image conditioning model OK", flush=True)
        except Exception as e:
            print(f"[worker] warmup: get_cond skipped ({e!r})", flush=True)

        _cuda_empty_cache()

        # 2. Run a single-step generation to trigger sparse attention / FlexGEMM compilation.
        try:
            sig = inspect.signature(pipe.run)
            kw: dict = {"num_samples": 1, "preprocess_image": False, "seed": 0}
            if "pipeline_type" in sig.parameters:
                kw["pipeline_type"] = "512"
            if "resolution" in sig.parameters:
                kw["resolution"] = 512
            for param, cfg in (
                ("sparse_structure_sampler_params", {"steps": 1, "guidance_strength": 7.5, "guidance_rescale": 0.7, "rescale_t": 5.0}),
                ("shape_slat_sampler_params", {"steps": 1, "guidance_strength": 7.5, "guidance_rescale": 0.5, "rescale_t": 3.0}),
                ("tex_slat_sampler_params", {"steps": 1, "guidance_strength": 1.0, "guidance_rescale": 0.0, "rescale_t": 3.0}),
            ):
                if param in sig.parameters:
                    kw[param] = cfg
            with torch.no_grad():
                pipe.run(dummy, **kw)
            print("[worker] warmup: generation pass OK", flush=True)
        except Exception as e:
            print(f"[worker] warmup: generation pass skipped ({e!r})", flush=True)

        _cuda_empty_cache()
        dt = time.time() - t0
        print(f"[worker] warmup: done in {dt:.1f}s", flush=True)
    except Exception as e:
        print(f"[worker] warmup: unexpected error ({e!r}), continuing", flush=True)
        _cuda_empty_cache()


def _preload_worker():
    try:
        _log.info("preload: starting model preload")
        print("[worker] preload: starting model preload", flush=True)
        pipe = _get_pipeline()
        st = get_ready_state()
        dt = 0.0
        if st.get("started_at") and st.get("ready_at"):
            dt = float(st["ready_at"]) - float(st["started_at"])
        _log.info("preload: ready (load_time_sec=%.1f)", dt)
        print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)
        if _bool_env("TRELLIS2_WARMUP", False):
            _warmup_gpu_kernels(pipe)
        _on_ready_start_idle_timer()
    except Exception:
        tb = traceback.format_exc()
        with _READY_LOCK:
            _READY["status"] = "error"
            _READY["detail"] = tb[-4000:] if tb else "unknown error"
        _log.error("preload: ERROR\n%s", tb or "unknown error")
        print("[worker] preload: ERROR\n" + (tb or "unknown error"), flush=True)


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
        _READY["status"] = "downloading"
        _READY["started_at"] = time.time()
        _READY["detail"] = "preload scheduled"
    t = threading.Thread(target=_preload_worker, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess_images(pipe, images: List[Image.Image]) -> List[Image.Image]:
    out = []
    for im in images:
        out.append(pipe.preprocess_image(im))
    return out


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
    if _bool_env("TRELLIS2_FORCE_RGB", False):
        try:
            im_rgba = im.convert("RGBA")
            bg = Image.new("RGBA", im_rgba.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im_rgba).convert("RGB")
        except Exception:
            im = im.convert("RGB")

    if _bool_env("TRELLIS2_CROP_ALPHA", True) and _has_transparency(im):
        im = _crop_to_alpha(im)

    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in getattr(im, "info", {})):
        im = im.convert("RGBA")
    else:
        im = im.convert("RGB")

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
    Uses robust thresholding + largest connected component to avoid tiny noisy halos
    around the frame from dominating the crop.
    """
    if alpha.ndim != 2:
        return None
    h, w = alpha.shape
    if h <= 0 or w <= 0:
        return None

    positive = alpha[alpha > 0]
    if positive.size == 0:
        return None

    min_area_frac = max(0.0, _float_env("TRELLIS2_PREPROCESS_MIN_MASK_AREA_FRAC", 0.00005))
    min_area_px = max(16, int(round(float(h * w) * min_area_frac)))

    def _largest_component_bbox(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
        if mask.ndim != 2:
            return None
        mask = (mask > 0).astype(np.uint8)
        if int(mask.sum()) < min_area_px:
            return None

        try:
            import cv2  # type: ignore

            n_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n_labels <= 1:
                return None
            # Skip label 0 (background), pick the largest foreground component.
            areas = stats[1:, cv2.CC_STAT_AREA]
            if areas.size == 0:
                return None
            best = int(np.argmax(areas)) + 1
            area = int(stats[best, cv2.CC_STAT_AREA])
            if area < min_area_px:
                return None
            x0 = int(stats[best, cv2.CC_STAT_LEFT])
            y0 = int(stats[best, cv2.CC_STAT_TOP])
            ww = int(stats[best, cv2.CC_STAT_WIDTH])
            hh = int(stats[best, cv2.CC_STAT_HEIGHT])
            x1 = x0 + max(0, ww - 1)
            y1 = y0 + max(0, hh - 1)
            return (x0, y0, x1, y1)
        except Exception:
            # Fallback when cv2 isn't available: bbox of all non-zero pixels.
            coords = np.argwhere(mask > 0)
            if coords.size == 0:
                return None
            y0 = int(coords[:, 0].min())
            y1 = int(coords[:, 0].max())
            x0 = int(coords[:, 1].min())
            x1 = int(coords[:, 1].max())
            return (x0, y0, x1, y1)

    thr_primary = float(np.clip(primary_thresh, 0.0, 1.0))
    maxv = float(positive.max()) / 255.0
    thresholds: List[float] = [thr_primary, 0.6, 0.4, 0.25, 0.12]
    thresholds.extend([max(0.005, 0.6 * maxv), max(0.005, 0.3 * maxv)])
    try:
        q95 = float(np.quantile(positive, 0.95)) / 255.0
        q85 = float(np.quantile(positive, 0.85)) / 255.0
        q70 = float(np.quantile(positive, 0.70)) / 255.0
        thresholds.extend([q95, q85, q70])
    except Exception:
        pass

    tried = set()
    for t in sorted((float(np.clip(v, 0.0, 1.0)) for v in thresholds), reverse=True):
        # Deduplicate thresholds after clipping/rounding.
        k = int(round(t * 1000))
        if k in tried:
            continue
        tried.add(k)
        bb = _largest_component_bbox(alpha > (t * 255.0))
        if bb is not None:
            return bb

    if _bool_env("TRELLIS2_PREPROCESS_ALLOW_ALPHA_NONZERO_FALLBACK", False):
        return _largest_component_bbox(alpha > 0)
    return None


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
    return (left, top, right, bottom)


def _is_rembg_dtype_mismatch(exc: BaseException) -> bool:
    s = (str(exc) or "").lower()
    if "input type" in s and "bias type" in s and ("c10::half" in s or "half" in s):
        return True
    if "input type" in s and "weight type" in s and "half" in s:
        return True
    return False


def _run_rembg_rgba(pipe, im_rgb: Image.Image) -> Image.Image:
    """
    Execute rembg model with low-vram guardrails and dtype-mismatch fallback.
    Returns RGBA output.
    """
    low_vram = bool(getattr(pipe, "low_vram", False))
    rembg_model = getattr(pipe, "rembg_model", None)
    if rembg_model is None:
        print("[worker] rembg: WARNING rembg_model is None, skipping background removal", flush=True)
        return im_rgb.convert("RGBA")

    print(
        f"[worker] rembg: running BiRefNet on {im_rgb.size[0]}x{im_rgb.size[1]} image (low_vram={low_vram}, device={getattr(pipe, 'device', '?')})",
        flush=True,
    )

    if low_vram:
        try:
            target_dev = getattr(pipe, "device", "cuda")
            inner = getattr(rembg_model, "model", None)
            if inner is not None and hasattr(inner, "to"):
                inner.to(target_dev)
            else:
                rembg_model.to(target_dev)  # type: ignore[attr-defined]
        except Exception as move_err:
            print(f"[worker] rembg: WARNING failed to move model to device: {move_err}", flush=True)

    rembg_failed = False
    try:
        out_rgba = rembg_model(im_rgb)  # type: ignore[attr-defined]
    except Exception as e:
        if _is_rembg_dtype_mismatch(e):
            print(
                "[worker] rembg: dtype mismatch during preprocess; retrying with rembg model forced to float32",
                flush=True,
            )
            try:
                # The BiRefNet wrapper may not expose .float() directly — try the inner model.
                inner = getattr(rembg_model, "model", None)
                if inner is not None and hasattr(inner, "float"):
                    inner.float()
                    print("[worker] rembg: forced inner model to float32", flush=True)
                elif hasattr(rembg_model, "float"):
                    rembg_model.float()  # type: ignore[attr-defined]
                    print("[worker] rembg: forced wrapper to float32", flush=True)
                else:
                    print("[worker] rembg: WARNING no .float() available on model or wrapper", flush=True)
                if low_vram:
                    target_dev = getattr(pipe, "device", "cuda")
                    if inner is not None and hasattr(inner, "to"):
                        inner.to(target_dev)
                    elif hasattr(rembg_model, "to"):
                        rembg_model.to(target_dev)  # type: ignore[attr-defined]
                out_rgba = rembg_model(im_rgb)  # type: ignore[attr-defined]
            except Exception as e2:
                if _bool_env("TRELLIS2_PREPROCESS_DISABLE_REMBG_ON_ERROR", True):
                    print(
                        f"[worker] rembg: retry failed ({e2}); continuing without rembg preprocessing",
                        flush=True,
                    )
                    out_rgba = im_rgb.convert("RGBA")
                    rembg_failed = True
                else:
                    raise
        elif _bool_env("TRELLIS2_PREPROCESS_DISABLE_REMBG_ON_ERROR", True):
            print(
                f"[worker] rembg: preprocess failed ({e}); continuing without rembg preprocessing",
                flush=True,
            )
            out_rgba = im_rgb.convert("RGBA")
            rembg_failed = True
        else:
            raise
    finally:
        if low_vram:
            try:
                inner = getattr(rembg_model, "model", None)
                if inner is not None and hasattr(inner, "to"):
                    inner.cpu()
                else:
                    rembg_model.cpu()  # type: ignore[attr-defined]
            except Exception:
                pass

    # Verify output has meaningful transparency (background actually removed).
    try:
        out_arr = np.array(out_rgba)
        if out_arr.ndim == 3 and out_arr.shape[2] >= 4:
            alpha_ch = out_arr[:, :, 3]
            opaque_frac = float((alpha_ch == 255).sum()) / float(max(1, alpha_ch.size))
            transparent_frac = float((alpha_ch == 0).sum()) / float(max(1, alpha_ch.size))
            print(
                f"[worker] rembg: result mode={out_rgba.mode} size={out_rgba.size} "
                f"alpha_stats: opaque={opaque_frac:.3f} transparent={transparent_frac:.3f} "
                f"min={int(alpha_ch.min())} max={int(alpha_ch.max())} failed={rembg_failed}",
                flush=True,
            )
            if not rembg_failed and opaque_frac > 0.99:
                print(
                    "[worker] rembg: WARNING output is >99% opaque — background removal may have failed silently!",
                    flush=True,
                )
        else:
            print(f"[worker] rembg: result mode={out_rgba.mode} size={out_rgba.size} (no alpha channel)", flush=True)
    except Exception:
        pass

    return out_rgba


# ---------------------------------------------------------------------------
# Alpha quality assessment & adaptive step-up
# ---------------------------------------------------------------------------

@dataclass
class AlphaQualityReport:
    level: int
    score: float              # 0.0 (terrible) to 1.0 (perfect)
    passed: bool
    bg_white_residue_pct: float
    fg_white_residue_pct: float   # opaque white pixels likely trapped background
    shadow_residue_pct: float
    foreground_coverage_pct: float
    semi_transparent_pct: float
    edge_halo_pct: float
    opaque_pct: float
    transparent_pct: float
    detail: str


def _assess_alpha_quality(
    rgba: Image.Image,
    original_rgb: np.ndarray,
    level: int,
) -> AlphaQualityReport:
    """
    Pure analysis: evaluate how clean the background removal is.
    Returns an AlphaQualityReport with per-defect metrics and overall score.
    """
    bg_alpha_thresh = _int_env("TRELLIS2_QA_BG_ALPHA_THRESH", 25)
    white_luma_min = _int_env("TRELLIS2_QA_WHITE_LUMA_MIN", 200)
    white_chroma_max = _int_env("TRELLIS2_QA_WHITE_CHROMA_MAX", 40)
    shadow_luma_max = _int_env("TRELLIS2_QA_SHADOW_LUMA_MAX", 120)
    edge_erode_px = _int_env("TRELLIS2_QA_EDGE_ERODE_PX", 3)
    pass_threshold = _float_env("TRELLIS2_QA_PASS_THRESHOLD", 0.70)

    arr = np.array(rgba)
    if arr.ndim != 3 or arr.shape[2] < 4:
        return AlphaQualityReport(
            level=level, score=1.0, passed=True,
            bg_white_residue_pct=0, fg_white_residue_pct=0,
            shadow_residue_pct=0,
            foreground_coverage_pct=100, semi_transparent_pct=0,
            edge_halo_pct=0, opaque_pct=100, transparent_pct=0,
            detail="no alpha channel",
        )

    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    total_px = max(1, alpha.size)

    # Luminance & chroma from RGB
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    # Basic alpha stats
    opaque_mask = alpha > 200
    transparent_mask = alpha < bg_alpha_thresh
    semi_mask = (alpha >= 1) & (alpha <= 254)
    opaque_pct = 100.0 * float(opaque_mask.sum()) / total_px
    transparent_pct = 100.0 * float(transparent_mask.sum()) / total_px
    semi_transparent_pct = 100.0 * float(semi_mask.sum()) / total_px

    # 1) Background white residue: bg pixels with white-ish RGB
    bg_mask = alpha < bg_alpha_thresh
    bg_count = max(1, int(bg_mask.sum()))
    white_in_bg = bg_mask & (luma > white_luma_min) & (chroma < white_chroma_max)
    bg_white_residue_pct = 100.0 * float(white_in_bg.sum()) / total_px

    # 1b) Foreground white residue: opaque pixels that look like trapped background
    #     (bright, achromatic — likely white bg that flood-fill couldn't reach)
    fg_white = opaque_mask & (luma > white_luma_min) & (chroma < white_chroma_max)
    fg_white_residue_pct = 100.0 * float(fg_white.sum()) / total_px

    # 2) Foreground coverage
    foreground_coverage_pct = opaque_pct

    # 3) Shadow residue: erode foreground, find edge band, check for dark semi-transparent
    import cv2  # type: ignore
    fg_binary = (alpha > bg_alpha_thresh).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_erode_px * 2 + 1, edge_erode_px * 2 + 1))
    eroded = cv2.erode(fg_binary, kernel, iterations=1)
    edge_band = (fg_binary > 0) & (eroded == 0)
    edge_count = max(1, int(edge_band.sum()))
    shadow_in_edge = edge_band & (luma < shadow_luma_max) & (alpha > 10) & (alpha < 200)
    shadow_residue_pct = 100.0 * float(shadow_in_edge.sum()) / total_px

    # 4) Edge halo: semi-transparent pixels with high luminance and low chroma (light fringes)
    halo_mask = semi_mask & (luma > 180) & (chroma < white_chroma_max)
    edge_halo_pct = 100.0 * float(halo_mask.sum()) / total_px

    # 5) Overall score: start at 1.0, penalize each defect
    score = 1.0
    score -= bg_white_residue_pct * 0.03   # white bg residue is a strong signal
    score -= shadow_residue_pct * 0.02
    score -= edge_halo_pct * 0.025
    # Penalize opaque white pixels trapped in foreground (likely un-removed background).
    # Use a threshold so tiny amounts (subject highlights, etc.) don't trigger.
    fg_white_thresh = _float_env("TRELLIS2_QA_FG_WHITE_THRESH_PCT", 10.0)
    if fg_white_residue_pct > fg_white_thresh:
        score -= (fg_white_residue_pct - fg_white_thresh) * 0.02
    # Penalize if too much or too little foreground
    if foreground_coverage_pct > 90.0:
        score -= (foreground_coverage_pct - 90.0) * 0.05
    if foreground_coverage_pct < 5.0:
        score -= (5.0 - foreground_coverage_pct) * 0.05
    # Penalize excessive semi-transparency (but some is natural)
    if semi_transparent_pct > 10.0:
        score -= (semi_transparent_pct - 10.0) * 0.01

    score = max(0.0, min(1.0, score))
    passed = score >= pass_threshold

    details = []
    if bg_white_residue_pct > 1.0:
        details.append(f"white_bg={bg_white_residue_pct:.1f}%")
    if fg_white_residue_pct > 5.0:
        details.append(f"fg_white={fg_white_residue_pct:.1f}%")
    if shadow_residue_pct > 0.5:
        details.append(f"shadow={shadow_residue_pct:.1f}%")
    if edge_halo_pct > 1.0:
        details.append(f"halo={edge_halo_pct:.1f}%")
    if foreground_coverage_pct > 90.0:
        details.append(f"fg_too_high={foreground_coverage_pct:.1f}%")

    return AlphaQualityReport(
        level=level,
        score=round(score, 3),
        passed=passed,
        bg_white_residue_pct=round(bg_white_residue_pct, 2),
        fg_white_residue_pct=round(fg_white_residue_pct, 2),
        shadow_residue_pct=round(shadow_residue_pct, 2),
        foreground_coverage_pct=round(foreground_coverage_pct, 2),
        semi_transparent_pct=round(semi_transparent_pct, 2),
        edge_halo_pct=round(edge_halo_pct, 2),
        opaque_pct=round(opaque_pct, 2),
        transparent_pct=round(transparent_pct, 2),
        detail="; ".join(details) if details else "clean",
    )


def _postprocess_alpha(rgba: Image.Image, level: int) -> Image.Image:
    """
    Apply progressively stronger alpha cleanup.
    Level 2: hard-threshold + erosion.
    Level 3: level 2 + shadow/halo removal + morphological close.
    Returns a new RGBA image.
    """
    if level < 2:
        return rgba

    import cv2  # type: ignore

    arr = np.array(rgba).copy()
    if arr.ndim != 3 or arr.shape[2] < 4:
        return rgba

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3].astype(np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    chroma = np.max(rgb, axis=2) - np.min(rgb, axis=2)

    white_chroma_max = _int_env("TRELLIS2_QA_WHITE_CHROMA_MAX", 40)
    shadow_luma_max = _int_env("TRELLIS2_QA_SHADOW_LUMA_MAX", 120)
    edge_erode_px = _int_env("TRELLIS2_QA_EDGE_ERODE_PX", 3)

    # Level 2: hard threshold + erosion
    alpha = np.where(alpha < 128, np.uint8(0), np.uint8(255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.erode(alpha, kernel, iterations=1)

    if level >= 3:
        # Build edge band from the hard-thresholded alpha
        fg_binary = (alpha > 0).astype(np.uint8) * 255
        edge_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (edge_erode_px * 2 + 1, edge_erode_px * 2 + 1)
        )
        eroded_fg = cv2.erode(fg_binary, edge_kernel, iterations=1)
        edge_band = (fg_binary > 0) & (eroded_fg == 0)

        # Shadow removal: zero alpha on dark semi-transparent edge pixels
        shadow_mask = edge_band & (luma < shadow_luma_max) & (chroma < white_chroma_max)
        alpha[shadow_mask] = 0

        # Halo removal: zero alpha on bright, achromatic near-edge pixels
        halo_mask = edge_band & (luma > 180) & (chroma < white_chroma_max)
        alpha[halo_mask] = 0

        # Morphological close to clean small holes
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    arr[:, :, 3] = alpha
    return Image.fromarray(arr)


def _check_foreground_preserved(
    original_rgba: Image.Image, processed_rgba: Image.Image
) -> bool:
    """
    Safety guard: reject post-processing if too many foreground pixels were removed.
    Returns True if the result is acceptable (foreground preserved).
    """
    max_loss_pct = _float_env("TRELLIS2_QA_MAX_FG_LOSS_PCT", 15.0)

    orig_arr = np.array(original_rgba)
    proc_arr = np.array(processed_rgba)
    orig_alpha = orig_arr[:, :, 3] if orig_arr.ndim == 3 and orig_arr.shape[2] >= 4 else None
    proc_alpha = proc_arr[:, :, 3] if proc_arr.ndim == 3 and proc_arr.shape[2] >= 4 else None

    if orig_alpha is None or proc_alpha is None:
        return True

    orig_fg = int((orig_alpha > 128).sum())
    proc_fg = int((proc_alpha > 128).sum())

    if orig_fg == 0:
        return True

    loss_pct = 100.0 * (orig_fg - proc_fg) / orig_fg
    if loss_pct > max_loss_pct:
        print(
            f"[worker] preprocess: foreground preservation check FAILED: "
            f"lost {loss_pct:.1f}% of foreground pixels (max={max_loss_pct}%)",
            flush=True,
        )
        return False
    return True


def _safe_preprocess_image(
    pipe,
    input_im: Image.Image,
    return_rgba: bool = False,
    out_dir: Optional[Path] = None,
    uid: Optional[str] = None,
):
    """
    Preprocess to match the official demo behavior, with adaptive quality
    checking and step-up system for background removal.

    Levels:
      0 — Use input alpha as-is (if client sent alpha)
      1 — Run BiRefNet (standard background removal)
      2 — Reuse BiRefNet result + moderate post-processing
      3 — Reuse BiRefNet result + aggressive post-processing

    Returns an RGB PIL image (alpha premultiplied like upstream).
    If return_rgba=True, returns (rgb_premultiplied, rgba_transparent, qa_report) tuple.
    """
    max_size = max(input_im.size) if input_im.size else 0
    if max_size > 0:
        scale = min(1.0, 1024.0 / float(max_size))
        if scale < 1.0:
            input_im = input_im.resize(
                (max(1, int(input_im.width * scale)), max(1, int(input_im.height * scale))),
                Image.Resampling.LANCZOS,
            )

    qa_enabled = _bool_env("TRELLIS2_QA_ENABLE", True)
    max_level = _int_env("TRELLIS2_QA_MAX_LEVEL", 3)
    save_debug = _bool_env("TRELLIS2_QA_SAVE_DEBUG", False)

    has_alpha = False
    try:
        if input_im.mode == "RGBA":
            a = np.array(input_im)[:, :, 3]
            if not np.all(a == 255):
                has_alpha = True
    except Exception:
        has_alpha = False

    original_rgb = np.array(input_im.convert("RGB"))

    # -- Adaptive quality loop --
    birefnet_rgba: Optional[Image.Image] = None  # cached BiRefNet result
    best_rgba: Optional[Image.Image] = None
    best_report: Optional[AlphaQualityReport] = None
    level_scores: List[str] = []

    start_level = 0 if has_alpha else 1
    end_level = max_level if qa_enabled else (1 if not has_alpha else 0)

    for level in range(start_level, end_level + 1):
        t_level = time.time()

        if level == 0:
            # Use input alpha as-is
            candidate_rgba = input_im.convert("RGBA")
            # Guardrail: broken/near-empty alpha mattes can cause full-frame box crops.
            if _bool_env("TRELLIS2_PREPROCESS_REMBG_ON_BAD_ALPHA", True):
                alpha_probe = np.array(candidate_rgba)[:, :, 3]
                probe_thresh = _float_env("TRELLIS2_PREPROCESS_ALPHA_BBOX_THRESHOLD", 0.8)
                bb_probe = _safe_alpha_bbox(alpha_probe, primary_thresh=probe_thresh)
                bad_alpha = bb_probe is None
                if not bad_alpha:
                    w_probe, h_probe = candidate_rgba.size
                    x0, y0, x1, y1 = bb_probe
                    bb_area = max(1, (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1))
                    cover = float(bb_area) / float(max(1, w_probe * h_probe))
                    strong_cover = float((alpha_probe > int(0.8 * 255.0)).mean())
                    max_cover = _float_env("TRELLIS2_PREPROCESS_BAD_ALPHA_MAX_BBOX_COVER", 0.98)
                    min_strong = _float_env("TRELLIS2_PREPROCESS_BAD_ALPHA_MIN_STRONG_COVER", 0.70)
                    if cover >= max_cover and strong_cover < min_strong:
                        bad_alpha = True
                if bad_alpha:
                    print(
                        "[worker] preprocess: level 0 alpha matte looks unreliable; skipping to level 1",
                        flush=True,
                    )
                    continue

        elif level == 1:
            # Run BiRefNet (standard)
            im_rgb = input_im.convert("RGB")
            birefnet_rgba = _run_rembg_rgba(pipe, im_rgb)
            candidate_rgba = birefnet_rgba

        elif level >= 2:
            # Post-process the BiRefNet result (levels 2 and 3)
            if birefnet_rgba is None:
                # BiRefNet hasn't run yet (e.g. started at level 0 and skipped level 1)
                im_rgb = input_im.convert("RGB")
                birefnet_rgba = _run_rembg_rgba(pipe, im_rgb)
            candidate_rgba = _postprocess_alpha(birefnet_rgba, level)
            # Safety: check foreground preservation
            if not _check_foreground_preserved(birefnet_rgba, candidate_rgba):
                print(
                    f"[worker] preprocess: level {level} rejected (foreground clipped), keeping previous best",
                    flush=True,
                )
                continue

        dt = time.time() - t_level

        # Save debug export if enabled
        if save_debug and out_dir is not None and uid is not None:
            try:
                debug_name = f"{uid}_qa_level{level}.png"
                candidate_rgba.save(str(out_dir / debug_name), format="PNG")
            except Exception:
                pass

        if not qa_enabled:
            # QA disabled — use this result directly
            best_rgba = candidate_rgba
            best_report = AlphaQualityReport(
                level=level, score=1.0, passed=True,
                bg_white_residue_pct=0, fg_white_residue_pct=0,
                shadow_residue_pct=0,
                foreground_coverage_pct=0, semi_transparent_pct=0,
                edge_halo_pct=0, opaque_pct=0, transparent_pct=0,
                detail="qa_disabled",
            )
            break

        # Assess quality
        report = _assess_alpha_quality(candidate_rgba, original_rgb, level)
        level_scores.append(f"L{level}:{report.score:.2f}")

        print(
            f"[worker] preprocess: level {level} QA: score={report.score:.3f} passed={report.passed} "
            f"bg_white_residue={report.bg_white_residue_pct:.1f}% "
            f"fg_white_residue={report.fg_white_residue_pct:.1f}% "
            f"shadow_residue={report.shadow_residue_pct:.1f}% "
            f"foreground_coverage={report.foreground_coverage_pct:.1f}% "
            f"semi_transparent={report.semi_transparent_pct:.1f}% "
            f"edge_halo={report.edge_halo_pct:.1f}% dt={dt:.2f}s",
            flush=True,
        )

        # Track best result
        if best_report is None or report.score > best_report.score:
            best_rgba = candidate_rgba
            best_report = report

        if report.passed:
            if level >= 1:
                print(
                    f"[worker] preprocess: level {level} PASSED (score={report.score:.3f})",
                    flush=True,
                )
                break
            else:
                # Level 0 (input alpha): always try BiRefNet too so we can compare.
                # Input alpha from client-side flood-fill may have trapped white regions
                # that BiRefNet (neural segmentation) handles better.
                print(
                    f"[worker] preprocess: level {level} scored {report.score:.3f}, "
                    f"continuing to BiRefNet for comparison",
                    flush=True,
                )
        else:
            if level < end_level:
                print(
                    f"[worker] preprocess: level {level} FAILED (score={report.score:.3f}), stepping up to level {level + 1}",
                    flush=True,
                )

    # QA summary
    if qa_enabled and level_scores:
        final_level = best_report.level if best_report else -1
        final_score = best_report.score if best_report else 0.0
        print(
            f"[worker] preprocess: QA summary: {' | '.join(level_scores)} → using level {final_level} (score={final_score:.3f})",
            flush=True,
        )

    # Fallback: if nothing was set (shouldn't happen), use input as-is
    if best_rgba is None:
        best_rgba = input_im.convert("RGBA")
        best_report = AlphaQualityReport(
            level=-1, score=0.0, passed=False,
            bg_white_residue_pct=0, fg_white_residue_pct=0,
            shadow_residue_pct=0,
            foreground_coverage_pct=0, semi_transparent_pct=0,
            edge_halo_pct=0, opaque_pct=0, transparent_pct=0,
            detail="fallback",
        )

    out_rgba = best_rgba

    # -- Existing bbox detection, cropping, alpha premultiply --
    out_np = np.array(out_rgba)
    if out_np.ndim != 3 or out_np.shape[2] < 4:
        rgb_result = out_rgba.convert("RGB")
        if return_rgba:
            return rgb_result, out_rgba.convert("RGBA"), best_report
        return rgb_result

    alpha = out_np[:, :, 3]
    thresh = _float_env("TRELLIS2_PREPROCESS_ALPHA_BBOX_THRESHOLD", 0.8)
    pad_px = _int_env("TRELLIS2_PREPROCESS_PAD_PX", 8)
    bb = _safe_alpha_bbox(alpha, primary_thresh=thresh)
    if bb is None:
        rgb = out_np[:, :, :3].astype(np.float32) / 255.0
        a = alpha.astype(np.float32)[:, :, None] / 255.0
        rgb = rgb * a
        rgb_result = Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))
        if return_rgba:
            return rgb_result, out_rgba, best_report
        return rgb_result

    w, h = int(out_rgba.size[0]), int(out_rgba.size[1])
    crop_box = _crop_square_around_bbox(w=w, h=h, bbox_xyxy=bb, pad_px=pad_px)
    cropped = out_rgba.crop(crop_box)

    cropped_np = np.array(cropped).astype(np.float32) / 255.0
    rgb = cropped_np[:, :, :3]
    a = cropped_np[:, :, 3:4]
    rgb = rgb * a
    rgb_result = Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))
    if return_rgba:
        return rgb_result, cropped, best_report
    return rgb_result


# ---------------------------------------------------------------------------
# Conditioning fusion (multi-image support)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

ALLOWED_MESH_PROFILES = {"game_ready", "hd"}
ALLOWED_GEOMETRY_RESOLUTIONS = {512, 1024, 1536}
ALLOWED_TEXTURE_MODES = {"fast_512", "native_1024", "cascade_512_1024"}
ALLOWED_TEXTURE_SIZES = {1024, 2048, 4096}


def _normalize_mesh_profile(raw: dict) -> str:
    mesh_profile = str(raw.get("mesh_profile") or "").strip().lower()
    if mesh_profile in ALLOWED_MESH_PROFILES:
        return mesh_profile
    if bool(raw.get("enable_hd")):
        return "hd"
    if str(raw.get("low_poly") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return "game_ready"
    if str(raw.get("quality") or "").strip().lower() == "game_ready":
        return "game_ready"
    return "hd"


def _normalize_geometry_resolution(raw: dict, mesh_profile: str) -> int:
    value = raw.get("geometry_resolution", raw.get("resolution"))
    if value is None:
        p = str(raw.get("pipeline_type") or "").strip().lower()
        if p == "512":
            return 512
        if p == "1024":
            return 1024
        if p in {"1024_cascade", "cascade_512_1024"}:
            return 1536
        return 512 if mesh_profile == "game_ready" else 1024
    try:
        n = int(str(value).strip())
    except Exception:
        n = 1024
    if n not in ALLOWED_GEOMETRY_RESOLUTIONS:
        n = min(ALLOWED_GEOMETRY_RESOLUTIONS, key=lambda x: abs(x - n))
    return int(n)


def _normalize_texture_mode(raw: dict) -> str:
    mode = str(raw.get("texture_generation_mode") or raw.get("texture_mode") or "").strip().lower()
    if mode in ALLOWED_TEXTURE_MODES:
        return mode
    p = str(raw.get("pipeline_type") or "").strip().lower()
    if p == "512":
        return "fast_512"
    if p == "1024":
        return "native_1024"
    if p in {"1024_cascade", "cascade_512_1024"}:
        return "cascade_512_1024"
    return "native_1024"


def _normalize_texture_size(raw: dict, mesh_profile: str) -> int:
    value = raw.get("texture_output_size", raw.get("texture_size"))
    if value is None:
        return 2048 if mesh_profile == "game_ready" else 4096
    try:
        n = int(str(value).strip())
    except Exception:
        n = 2048
    if n not in ALLOWED_TEXTURE_SIZES:
        n = min(ALLOWED_TEXTURE_SIZES, key=lambda x: abs(x - n))
    return int(n)


def _texture_mode_to_pipeline_type(texture_mode: str) -> str:
    if texture_mode == "fast_512":
        return "512"
    if texture_mode == "cascade_512_1024":
        return "1024_cascade"
    return "1024"


def normalize_generation_request(raw: dict, *, strict_4k_geometry: bool = False) -> tuple[dict, list[str]]:
    """
    Normalize request fields while keeping legacy compatibility.
    Separates geometry, texture generation strategy, and bake texture resolution.
    """
    adjustments: list[str] = []

    mesh_profile = _normalize_mesh_profile(raw)
    geometry_resolution = _normalize_geometry_resolution(raw, mesh_profile)
    texture_generation_mode = _normalize_texture_mode(raw)
    texture_output_size = _normalize_texture_size(raw, mesh_profile)
    steps_raw = raw.get("steps")
    try:
        steps = int(str(steps_raw).strip()) if steps_raw is not None else 12
    except Exception:
        steps = 12
    steps = max(1, int(steps))

    default_decimation = 150000 if mesh_profile == "game_ready" else 2000000
    decimation_raw = raw.get("decimation_target")
    try:
        decimation_target = int(str(decimation_raw).strip()) if decimation_raw is not None else default_decimation
    except Exception:
        decimation_target = default_decimation
    decimation_target = max(1, int(decimation_target))

    if texture_output_size == 4096 and geometry_resolution < 1024:
        if strict_4k_geometry:
            raise ValueError("texture_output_size=4096 requires geometry_resolution >= 1024")
        texture_output_size = 2048
        adjustments.append("downgraded texture_output_size 4096->2048 because geometry_resolution < 1024")

    if mesh_profile == "game_ready" and decimation_target > 300000:
        decimation_target = 300000
        adjustments.append("capped decimation_target to 300000 for mesh_profile=game_ready")

    if texture_generation_mode == "fast_512" and texture_output_size > 2048:
        texture_output_size = 2048
        adjustments.append("downgraded texture_output_size to 2048 for texture_generation_mode=fast_512")

    normalized = {
        "mesh_profile": mesh_profile,
        "geometry_resolution": int(geometry_resolution),
        "decimation_target": int(decimation_target),
        "texture_generation_mode": texture_generation_mode,
        "texture_output_size": int(texture_output_size),
        "steps": int(steps),
        "pipeline_type": _texture_mode_to_pipeline_type(texture_generation_mode),
        "enable_hd": bool(mesh_profile == "hd"),
    }
    return normalized, adjustments


def _resolve_pipeline_type(requested: Optional[str]) -> str:
    v = (requested or "").strip()
    if v:
        return v
    return os.environ.get("TRELLIS2_PIPELINE_TYPE", "1024_cascade").strip() or "1024_cascade"


def _choose_decimation_target(mesh_profile: str, decimation_target: Optional[int] = None) -> int:
    if decimation_target and decimation_target > 0:
        return int(decimation_target)
    if mesh_profile == "game_ready":
        return _int_env("TRELLIS2_DECIMATION_TARGET_LOW_POLY", 75000)
    return _int_env("TRELLIS2_DECIMATION_TARGET", 1000000)


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

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
    decimation_target: Optional[int] = None,
    mesh_profile: Optional[str] = None,
    geometry_resolution: Optional[int] = None,
    texture_generation_mode: Optional[str] = None,
    texture_output_size: Optional[int] = None,
    steps: Optional[int] = None,
) -> Path:
    if not images_bytes:
        raise ValueError("empty upload")

    global _GENERATING_COUNT
    with _GENERATING_LOCK:
        _GENERATING_COUNT += 1

    _cuda_empty_cache()
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    normalized_request, payload_adjustments = normalize_generation_request(
        {
            "mesh_profile": mesh_profile,
            "resolution": geometry_resolution,
            "decimation_target": decimation_target,
            "texture_mode": texture_generation_mode,
            "texture_size": texture_output_size,
            "steps": steps,
            "low_poly": low_poly,
            "pipeline_type": pipeline_type,
        },
        strict_4k_geometry=False,
    )
    for note in payload_adjustments:
        print(f"[worker] request auto-adjustment: {note}", flush=True)
    print(
        "[worker] request settings: "
        f"geometry_resolution={normalized_request['geometry_resolution']} "
        f"texture_generation_mode={normalized_request['texture_generation_mode']} "
        f"texture_bake_resolution={normalized_request['texture_output_size']} "
        f"mesh_profile={normalized_request['mesh_profile']} "
        f"hd_enabled={1 if normalized_request['enable_hd'] else 0}",
        flush=True,
    )

    pipe = _get_pipeline()
    try:
        _cuda_empty_cache()
        imgs_raw: List[Image.Image] = []
        for raw in images_bytes:
            if not raw:
                continue
            im = Image.open(io.BytesIO(raw))
            imgs_raw.append(im)
        if not imgs_raw:
            raise ValueError("empty image bytes")

        if backup_inputs:
            in_dir = out_dir / "inputs"
            in_dir.mkdir(parents=True, exist_ok=True)
            for idx, (raw, im) in enumerate(zip(images_bytes, imgs_raw)):
                try:
                    p = in_dir / f"{uuid.uuid4().hex}_{idx+1:02d}.png"
                    _prepare_input_image(im).save(str(p), format="PNG")
                except Exception:
                    pass

        import inspect
        import o_voxel  # type: ignore

        # Texture generation strategy controls internal pipeline family.
        # Geometry resolution and UV texture bake size are handled separately.
        requested_steps = int(normalized_request["steps"])
        ptype = _resolve_pipeline_type(str(normalized_request["pipeline_type"] or pipeline_type or ""))
        img0_raw = imgs_raw[0]

        import torch

        if seed is None:
            seed = _int_env("TRELLIS2_DEFAULT_SEED", 42)

        do_preprocess = _bool_env("TRELLIS2_PREPROCESS_IMAGE", True) if preprocess_image is None else bool(preprocess_image)
        print(
            f"[worker] preprocess decision: do_preprocess={do_preprocess} "
            f"preprocess_image_arg={preprocess_image} "
            f"num_images={len(imgs_raw)} "
            f"img0_mode={img0_raw.mode} img0_size={img0_raw.size}",
            flush=True,
        )

        # Preprocess all input images and collect RGBA exports for background-removed versions.
        rembg_export_names: List[str] = []
        uid = uuid.uuid4().hex
        primary_qa_report: Optional[AlphaQualityReport] = None
        if do_preprocess:
            result = _safe_preprocess_image(pipe, img0_raw, return_rgba=True, out_dir=out_dir, uid=uid)
            primary_img, primary_rgba, primary_qa_report = result
            # Add QA metrics to export_meta
            if export_meta is not None and primary_qa_report is not None:
                export_meta.update({
                    "bg_removal_final_level": primary_qa_report.level,
                    "bg_removal_final_score": primary_qa_report.score,
                    "bg_removal_passed": primary_qa_report.passed,
                    "bg_removal_bg_white_residue_pct": primary_qa_report.bg_white_residue_pct,
                    "bg_removal_fg_white_residue_pct": primary_qa_report.fg_white_residue_pct,
                    "bg_removal_shadow_residue_pct": primary_qa_report.shadow_residue_pct,
                    "bg_removal_foreground_coverage_pct": primary_qa_report.foreground_coverage_pct,
                    "bg_removal_edge_halo_pct": primary_qa_report.edge_halo_pct,
                    "bg_removal_detail": primary_qa_report.detail,
                })
            # Save RGBA for all input images.
            for rembg_idx, rembg_im_raw in enumerate(imgs_raw):
                try:
                    if rembg_idx == 0:
                        rgba_out = primary_rgba
                    else:
                        sec_result = _safe_preprocess_image(pipe, rembg_im_raw, return_rgba=True, out_dir=out_dir, uid=uid)
                        _, rgba_out, _ = sec_result
                    fname = f"{uid}_rembg_{rembg_idx:02d}.png"
                    rgba_out.save(str(out_dir / fname), format="PNG")
                    rembg_export_names.append(fname)
                    print(f"[worker] saved rembg export: {fname} ({rgba_out.size[0]}x{rgba_out.size[1]})", flush=True)
                except Exception as e:
                    print(f"[worker] WARNING: failed to save rembg export for image {rembg_idx}: {e}", flush=True)
        else:
            primary_img = _prepare_input_image(img0_raw).convert("RGB")
        fallback_img = _prepare_input_image(img0_raw).convert("RGB") if do_preprocess else None

        # Pass only kwargs supported by this pipeline version.
        kwargs: dict = {}
        try:
            sig = inspect.signature(pipe.run)  # type: ignore[attr-defined]
            if "seed" in sig.parameters:
                kwargs["seed"] = int(seed)
            if ptype and "pipeline_type" in sig.parameters:
                kwargs["pipeline_type"] = ptype
            if "resolution" in sig.parameters:
                kwargs["resolution"] = int(normalized_request["geometry_resolution"])
            if "num_samples" in sig.parameters:
                kwargs["num_samples"] = 1
            if "preprocess_image" in sig.parameters:
                kwargs["preprocess_image"] = False
            if _bool_env("TRELLIS2_USE_DEMO_SAMPLER_DEFAULTS", True):
                if "sparse_structure_sampler_params" in sig.parameters:
                    kwargs["sparse_structure_sampler_params"] = {
                        "steps": requested_steps,
                        "guidance_strength": _float_env("TRELLIS2_SS_GUIDANCE_STRENGTH", 7.5),
                        "guidance_rescale": _float_env("TRELLIS2_SS_GUIDANCE_RESCALE", 0.7),
                        "rescale_t": _float_env("TRELLIS2_SS_RESCALE_T", 5.0),
                    }
                if "shape_slat_sampler_params" in sig.parameters:
                    kwargs["shape_slat_sampler_params"] = {
                        "steps": requested_steps,
                        "guidance_strength": _float_env("TRELLIS2_SHAPE_GUIDANCE_STRENGTH", 7.5),
                        "guidance_rescale": _float_env("TRELLIS2_SHAPE_GUIDANCE_RESCALE", 0.5),
                        "rescale_t": _float_env("TRELLIS2_SHAPE_RESCALE_T", 3.0),
                    }
                if "tex_slat_sampler_params" in sig.parameters:
                    kwargs["tex_slat_sampler_params"] = {
                        "steps": requested_steps,
                        "guidance_strength": _float_env("TRELLIS2_TEX_GUIDANCE_STRENGTH", 1.0),
                        "guidance_rescale": _float_env("TRELLIS2_TEX_GUIDANCE_RESCALE", 0.0),
                        "rescale_t": _float_env("TRELLIS2_TEX_RESCALE_T", 3.0),
                    }
        except Exception:
            kwargs["seed"] = int(seed)
            if ptype:
                kwargs["pipeline_type"] = ptype

        def _is_empty_sparse_error(exc: BaseException) -> bool:
            msg = (str(exc) or "").lower()
            return ("input.numel() == 0" in msg) or ("max(): expected reduction dim" in msg) or ("empty sparse coords" in msg)

        def _is_retryable_generation_error(exc: BaseException) -> bool:
            msg = (str(exc) or "").lower()
            if _is_empty_sparse_error(exc):
                return True
            if "pipeline produced empty mesh" in msg:
                return True
            if "pipeline produced planar mesh" in msg:
                return True
            return False

        def _mesh_axis_ratio(mesh_obj) -> Optional[tuple[float, float, float]]:
            try:
                verts = getattr(mesh_obj, "vertices", None)
                if verts is None:
                    return None
                if hasattr(verts, "detach"):
                    v = verts.detach().float().cpu().numpy()
                else:
                    v = np.asarray(verts)
                if v.ndim != 2 or v.shape[0] < 3 or v.shape[1] < 3:
                    return None
                xyz = v[:, :3].astype(np.float32, copy=False)
                ext = np.max(xyz, axis=0) - np.min(xyz, axis=0)
                min_extent = float(np.min(ext))
                max_extent = float(np.max(ext))
                if max_extent <= 1e-9:
                    return (min_extent, max_extent, 0.0)
                return (min_extent, max_extent, min_extent / max_extent)
            except Exception:
                return None

        retries = _int_env("TRELLIS2_GENERATION_RETRIES", _int_env("TRELLIS2_EMPTY_SPARSE_RETRIES", 4))
        last_err: Optional[BaseException] = None
        input_fallback_attempted = False

        # Free VRAM after preprocessing (rembg etc.) so pipe.run() has maximum headroom.
        _cuda_empty_cache()

        attempt = 0
        max_attempts = max(1, retries)
        img0 = primary_img
        oom_quality_level = 0
        original_kwargs = dict(kwargs)
        mesh_axis_stats: Optional[tuple[float, float, float]] = None

        while True:
            try:
                torch.manual_seed(int(seed) + attempt)
                if "seed" in kwargs:
                    kwargs["seed"] = int(seed) + attempt
                out = pipe.run(img0, **kwargs)  # type: ignore[attr-defined]
                if not out:
                    raise RuntimeError("pipeline.run() returned no outputs")
                mesh = out[0]

                # Validate mesh output
                if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
                    raise RuntimeError("pipeline output missing vertices or faces attributes")
                vert_count = len(mesh.vertices) if hasattr(mesh.vertices, "__len__") else 0
                face_count = len(mesh.faces) if hasattr(mesh.faces, "__len__") else 0
                if vert_count == 0 or face_count == 0:
                    raise RuntimeError(f"pipeline produced empty mesh: {vert_count} verts, {face_count} faces")
                if face_count > vert_count * 10:
                    print(
                        f"[worker] WARNING: unusual mesh topology: {vert_count} verts, {face_count} faces "
                        f"(ratio {face_count/vert_count:.1f})",
                        flush=True,
                    )
                min_thin_axis_ratio = max(0.0, _float_env("TRELLIS2_MIN_THIN_AXIS_RATIO", 0.001))
                if min_thin_axis_ratio > 0.0:
                    mesh_axis_stats = _mesh_axis_ratio(mesh)
                    if mesh_axis_stats is not None:
                        min_extent, max_extent, thin_axis_ratio = mesh_axis_stats
                        if thin_axis_ratio < min_thin_axis_ratio:
                            raise RuntimeError(
                                "pipeline produced planar mesh: "
                                f"thin_axis_ratio={thin_axis_ratio:.6f}, "
                                f"min_extent={min_extent:.6f}, max_extent={max_extent:.6f}"
                            )

                if oom_quality_level > 0:
                    degradation_desc = []
                    if "sparse_structure_sampler_params" in kwargs:
                        degradation_desc.append(f"ss_steps={kwargs['sparse_structure_sampler_params'].get('steps', '?')}")
                    if "shape_slat_sampler_params" in kwargs:
                        degradation_desc.append(f"shape_steps={kwargs['shape_slat_sampler_params'].get('steps', '?')}")
                    if "tex_slat_sampler_params" in kwargs:
                        degradation_desc.append(f"tex_steps={kwargs['tex_slat_sampler_params'].get('steps', '?')}")
                    print(
                        f"[worker] pipe.run succeeded with quality degradation level {oom_quality_level} "
                        f"(pipeline_type={ptype}, {', '.join(degradation_desc)})",
                        flush=True,
                    )
                break
            except Exception as e:
                last_err = e

                # Progressive OOM handling: preserve texture quality as long as possible.
                # For 1024+ pipelines, step reductions rarely help — peak VRAM is
                # determined by resolution, not step count.  Skip directly to a
                # smaller pipeline to avoid burning minutes on doomed retries.
                if _is_cuda_oom(e):
                    _cuda_empty_cache()

                    # Level 0: Simple cache clear
                    if oom_quality_level == 0:
                        oom_quality_level = 1
                        print("[worker] pipe.run OOM (level 0), clearing cache and retrying", flush=True)
                        continue

                    # Levels 1-6: Smart pipeline fallback
                    # For non-512 pipelines, resolution is the bottleneck — step
                    # reductions don't meaningfully reduce peak VRAM.  Jump to a
                    # smaller pipeline with full original steps for best quality.
                    elif oom_quality_level < 7 and ptype != "512":
                        if "cascade" in (ptype or "").lower() and oom_quality_level < 4:
                            # Cascade pipeline: try dropping cascade first (one attempt)
                            oom_quality_level = 4
                            fallback_ptype = ptype.replace("_cascade", "").replace("cascade_", "")
                            print(f"[worker] pipe.run OOM (level 1), dropping cascade: {ptype} -> {fallback_ptype}", flush=True)
                            if "pipeline_type" in kwargs:
                                kwargs["pipeline_type"] = fallback_ptype
                            ptype = fallback_ptype
                        else:
                            # Non-cascade 1024 (or cascade already dropped): switch to 512
                            oom_quality_level = 7
                            print(f"[worker] pipe.run OOM (level {oom_quality_level - 1}), {ptype} won't fit in VRAM, switching to 512 pipeline", flush=True)
                            if "pipeline_type" in kwargs:
                                kwargs["pipeline_type"] = "512"
                            ptype = "512"
                        # Restore original steps for best quality at the new pipeline
                        for _k in ("sparse_structure_sampler_params", "shape_slat_sampler_params", "tex_slat_sampler_params"):
                            if _k in kwargs and _k in original_kwargs and isinstance(kwargs[_k], dict) and isinstance(original_kwargs.get(_k), dict):
                                kwargs[_k]["steps"] = original_kwargs[_k].get("steps", 12)
                        continue

                    # Levels 1-5 for 512 pipeline: gradual step reductions
                    elif oom_quality_level < 7 and ptype == "512":
                        if oom_quality_level == 1:
                            oom_quality_level = 2
                            if "sparse_structure_sampler_params" in kwargs and isinstance(kwargs["sparse_structure_sampler_params"], dict):
                                orig_steps = kwargs["sparse_structure_sampler_params"].get("steps", 12)
                                new_steps = max(4, orig_steps // 2)
                                kwargs["sparse_structure_sampler_params"]["steps"] = new_steps
                                print(f"[worker] pipe.run OOM (level 1), reducing sparse_structure steps {orig_steps}->{new_steps}", flush=True)
                            continue
                        elif oom_quality_level == 2:
                            oom_quality_level = 3
                            if "shape_slat_sampler_params" in kwargs and isinstance(kwargs["shape_slat_sampler_params"], dict):
                                orig_steps = kwargs["shape_slat_sampler_params"].get("steps", 12)
                                new_steps = max(4, orig_steps // 2)
                                kwargs["shape_slat_sampler_params"]["steps"] = new_steps
                                print(f"[worker] pipe.run OOM (level 2), reducing shape_slat steps {orig_steps}->{new_steps}", flush=True)
                            continue
                        elif oom_quality_level == 3:
                            oom_quality_level = 5
                            for _k in ("sparse_structure_sampler_params", "shape_slat_sampler_params"):
                                if _k in kwargs and isinstance(kwargs[_k], dict):
                                    kwargs[_k]["steps"] = 4
                            print("[worker] pipe.run OOM (level 3), all geometry steps->4", flush=True)
                            continue
                        else:
                            # 512 with min steps still OOMs — fall through to image downscaling
                            oom_quality_level = 7
                            continue

                    # Level 7: Downscale input image to 768
                    elif oom_quality_level == 7:
                        oom_quality_level = 8
                        w, h = img0.size
                        max_dim = max(w, h)
                        if max_dim > 768:
                            sc = 768.0 / max_dim
                            new_w = int(w * sc)
                            new_h = int(h * sc)
                            img0 = img0.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            print(f"[worker] pipe.run OOM (level 7), downscaled image {w}x{h}->{new_w}x{new_h}", flush=True)
                        continue

                    # Level 8: Downscale to 512
                    elif oom_quality_level == 8:
                        oom_quality_level = 9
                        w, h = img0.size
                        max_dim = max(w, h)
                        if max_dim > 512:
                            sc = 512.0 / max_dim
                            new_w = int(w * sc)
                            new_h = int(h * sc)
                            img0 = img0.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            print(f"[worker] pipe.run OOM (level 8), downscaled image {w}x{h}->{new_w}x{new_h}", flush=True)
                        continue

                    # Level 9: Reduce texture steps by 50%
                    elif oom_quality_level == 9:
                        oom_quality_level = 10
                        if "tex_slat_sampler_params" in kwargs and isinstance(kwargs["tex_slat_sampler_params"], dict):
                            orig_steps = kwargs["tex_slat_sampler_params"].get("steps", 12)
                            new_steps = max(4, orig_steps // 2)
                            kwargs["tex_slat_sampler_params"]["steps"] = new_steps
                            print(f"[worker] pipe.run OOM (level 9), reducing tex_slat steps {orig_steps}->{new_steps}", flush=True)
                        continue

                    # Level 10: Downscale to 384
                    elif oom_quality_level == 10:
                        oom_quality_level = 11
                        w, h = img0.size
                        max_dim = max(w, h)
                        if max_dim > 384:
                            sc = 384.0 / max_dim
                            new_w = int(w * sc)
                            new_h = int(h * sc)
                            img0 = img0.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            print(f"[worker] pipe.run OOM (level 10), downscaled image {w}x{h}->{new_w}x{new_h}", flush=True)
                        continue

                    # Level 11: Reduce texture to minimum
                    elif oom_quality_level == 11:
                        oom_quality_level = 12
                        if "tex_slat_sampler_params" in kwargs and isinstance(kwargs["tex_slat_sampler_params"], dict):
                            kwargs["tex_slat_sampler_params"]["steps"] = 4
                            print("[worker] pipe.run OOM (level 11), tex_slat steps->4", flush=True)
                        continue

                    # Level 12: Absolute minimum - tiny image with minimal steps
                    elif oom_quality_level == 12:
                        oom_quality_level = 13
                        w, h = img0.size
                        max_dim = max(w, h)
                        if max_dim > 256:
                            sc = 256.0 / max_dim
                            new_w = int(w * sc)
                            new_h = int(h * sc)
                            img0 = img0.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            print(f"[worker] pipe.run OOM (level 12), downscaled to absolute minimum {w}x{h}->{new_w}x{new_h}", flush=True)
                        if "sparse_structure_sampler_params" in kwargs:
                            kwargs["sparse_structure_sampler_params"]["steps"] = 4
                        if "shape_slat_sampler_params" in kwargs:
                            kwargs["shape_slat_sampler_params"]["steps"] = 4
                        if "tex_slat_sampler_params" in kwargs:
                            kwargs["tex_slat_sampler_params"]["steps"] = 4
                        continue

                    # Level 13+: All fallbacks exhausted
                    else:
                        print("[worker] pipe.run OOM after all 13 quality degradation attempts", flush=True)
                        raise RuntimeError("Worker ran out of memory, model was too complex.") from e

                # Handle non-OOM retryable errors with seed shift/fallback image.
                if not _is_retryable_generation_error(e):
                    raise

                attempt += 1
                if attempt < max_attempts:
                    continue

                if (not input_fallback_attempted) and (fallback_img is not None):
                    input_fallback_attempted = True
                    attempt = 0
                    img0 = fallback_img
                    continue

                raise RuntimeError("pipeline.run() failed after retries") from last_err

        # Export to GLB via o-voxel postprocess util.
        _cuda_empty_cache()
        # Keep geometry complexity and UV bake resolution independent.
        decimation_target = _choose_decimation_target(
            str(normalized_request["mesh_profile"]),
            decimation_target=int(normalized_request["decimation_target"]),
        )
        texture_size = int(normalized_request["texture_output_size"])
        decimation_initial = int(decimation_target)
        dec_min = _int_env("TRELLIS2_DECIMATION_MIN_OOM_FALLBACK", 15000)
        glb = None
        last_glb_err = None
        oom_retries = 0
        to_glb_attempts = 0

        if export_meta is not None:
            export_meta.update({
                "pipeline_type": ptype,
                "low_poly": bool(str(normalized_request["mesh_profile"]) == "game_ready"),
                "mesh_profile": str(normalized_request["mesh_profile"]),
                "geometry_resolution": int(normalized_request["geometry_resolution"]),
                "texture_generation_mode": str(normalized_request["texture_generation_mode"]),
                "texture_size": int(texture_size),
                "decimation_initial": decimation_initial,
                "input_image_count": len(images_bytes),
                "oom_quality_degradation_level": oom_quality_level,
                "enable_hd": bool(normalized_request["enable_hd"]),
                "preprocessed_images": rembg_export_names if rembg_export_names else None,
            })
            if "sparse_structure_sampler_params" in kwargs and isinstance(kwargs.get("sparse_structure_sampler_params"), dict):
                export_meta["sparse_structure_steps"] = kwargs["sparse_structure_sampler_params"].get("steps")
            if "shape_slat_sampler_params" in kwargs and isinstance(kwargs.get("shape_slat_sampler_params"), dict):
                export_meta["shape_slat_steps"] = kwargs["shape_slat_sampler_params"].get("steps")
            if "tex_slat_sampler_params" in kwargs and isinstance(kwargs.get("tex_slat_sampler_params"), dict):
                export_meta["tex_slat_steps"] = kwargs["tex_slat_sampler_params"].get("steps")
            if mesh_axis_stats is not None:
                export_meta["mesh_min_extent"] = float(mesh_axis_stats[0])
                export_meta["mesh_max_extent"] = float(mesh_axis_stats[1])
                export_meta["mesh_thin_axis_ratio"] = float(mesh_axis_stats[2])

        for glb_attempt in range(_int_env("TRELLIS2_TO_GLB_OOM_RETRIES", 4)):
            to_glb_attempts = glb_attempt + 1
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
                if _is_cuda_oom(e) and glb_attempt < 3 and decimation_target > dec_min:
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
                if _is_cuda_oom(e):
                    raise RuntimeError(
                        "GPU OOM during UV texture bake "
                        f"(texture_size={int(texture_size)}, decimation_target={int(decimation_target)})."
                    ) from e
                raise
        if glb is None:
            if export_meta is not None:
                export_meta.update({
                    "oom_exhausted": True,
                    "oom_retries": oom_retries,
                    "to_glb_attempts": to_glb_attempts,
                    "decimation_final": int(decimation_target),
                })
            if last_glb_err is not None and _is_cuda_oom(last_glb_err):
                raise RuntimeError(
                    "GPU OOM during UV texture bake after retries "
                    f"(texture_size={int(texture_size)}, decimation_target={int(decimation_target)})."
                ) from last_glb_err
            raise RuntimeError("to_glb failed") from last_glb_err

        # Extract mesh statistics
        try:
            glb_vert_count = len(glb.vertices) if hasattr(glb, "vertices") and hasattr(glb.vertices, "__len__") else 0
            glb_face_count = len(glb.faces) if hasattr(glb, "faces") and hasattr(glb.faces, "__len__") else 0
            print(
                f"[worker] GLB export complete: {glb_vert_count} vertices, {glb_face_count} faces "
                f"(decimation_target={decimation_target}, texture_size={texture_size})",
                flush=True,
            )
            print(
                "[worker] final settings: "
                f"geometry_resolution={normalized_request['geometry_resolution']} "
                f"texture_generation_mode={normalized_request['texture_generation_mode']} "
                f"texture_bake_resolution={texture_size} "
                f"final_polygon_count={glb_face_count} "
                f"hd_enabled={1 if normalized_request['enable_hd'] else 0}",
                flush=True,
            )
            if export_meta is not None:
                export_meta.update({
                    "oom_retries": oom_retries,
                    "to_glb_attempts": to_glb_attempts,
                    "decimation_final": int(decimation_target),
                    "glb_vertex_count": glb_vert_count,
                    "glb_face_count": glb_face_count,
                })
        except Exception as e:
            print(f"[worker] WARNING: failed to extract GLB mesh statistics: {e}", flush=True)
            if export_meta is not None:
                export_meta.update({
                    "oom_retries": oom_retries,
                    "to_glb_attempts": to_glb_attempts,
                    "decimation_final": int(decimation_target),
                })

        # Optional post-export Z-axis scaling
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
        with _GENERATING_LOCK:
            _GENERATING_COUNT = max(0, _GENERATING_COUNT - 1)
