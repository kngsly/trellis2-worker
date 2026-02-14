#!/usr/bin/env python3
"""
TRELLIS.2 worker implementation.

This module is imported by server.py.
"""

from __future__ import annotations

import gc
import io
import os
import subprocess
import threading
import time
import traceback
import uuid
import ctypes
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


_PIPELINE = None
_READY = {
    "status": "not_started",  # not_started | downloading | loading | ready | error
    "detail": "",
    "started_at": 0.0,
    "ready_at": 0.0,
}
_READY_LOCK = threading.Lock()
_CUDA_RUNTIME_PREPARED = False


def _get_device():
    return os.environ.get("TRELLIS2_DEVICE", "cuda")


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
    )
    return any(m in s for m in markers)


def _cuda_runtime_probe() -> tuple[bool, str]:
    driver_info = ""
    # 1) Device nodes present.
    if not (Path("/dev/nvidiactl").exists() and Path("/dev/nvidia-uvm").exists()):
        return (False, "missing /dev/nvidiactl or /dev/nvidia-uvm")

    # 2) Driver linker name resolves.
    try:
        ctypes.CDLL("libcuda.so")
    except Exception as e:
        return (False, f"libcuda unresolved ({e})")

    # 3) nvidia-smi reports at least one GPU (best effort).
    try:
        p = subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
        out = (p.stdout or "").strip()
        if p.returncode != 0 or "GPU " not in out:
            err = (p.stderr or "").strip()
            return (False, f"nvidia-smi not ready rc={p.returncode} out={out[:140]!r} err={err[:140]!r}")
        # Add driver hint to later probe failures (useful for CUDA/toolkit compatibility triage).
        p_drv = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            check=False,
        )
        if p_drv.returncode == 0:
            driver_info = ((p_drv.stdout or "").strip().splitlines() or [""])[0].strip()
    except Exception as e:
        return (False, f"nvidia-smi probe failed ({e})")

    # 4) Torch sees CUDA (defer heavy imports until needed).
    try:
        import torch

        if not torch.cuda.is_available():
            if driver_info:
                return (False, f"torch.cuda.is_available()=False (driver={driver_info})")
            return (False, "torch.cuda.is_available()=False")
        if int(torch.cuda.device_count()) <= 0:
            if driver_info:
                return (False, f"torch.cuda.device_count()=0 (driver={driver_info})")
            return (False, "torch.cuda.device_count()=0")
    except Exception as e:
        return (False, f"torch cuda probe failed ({e})")

    return (True, "cuda runtime ready")


def _wait_for_cuda_runtime_ready() -> None:
    if _get_device().lower() != "cuda":
        return
    max_wait_sec = int(os.environ.get("TRELLIS2_CUDA_READY_TIMEOUT_SEC", "180") or "180")
    interval_sec = max(1, int(os.environ.get("TRELLIS2_CUDA_READY_POLL_SEC", "3") or "3"))
    strict = _bool_env("TRELLIS2_CUDA_READY_STRICT", False)
    deadline = time.time() + max(1, max_wait_sec)
    last_detail = ""

    while True:
        ok, detail = _cuda_runtime_probe()
        if ok:
            if last_detail:
                print(f"[worker] cuda-preflight: recovered ({detail})", flush=True)
            return
        last_detail = detail
        if time.time() >= deadline:
            msg = f"CUDA runtime did not become ready within {max_wait_sec}s: {detail}"
            if strict:
                raise RuntimeError(msg)
            # Non-strict mode: continue into pipeline init retries to avoid false negatives on slow hosts.
            print(f"[worker] cuda-preflight: warning: {msg}; continuing with deferred preload retries", flush=True)
            return
        print(f"[worker] cuda-preflight: waiting ({detail})", flush=True)
        time.sleep(interval_sec)


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


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _ensure_cuda_linker_paths()
    _wait_for_cuda_runtime_ready()

    with _READY_LOCK:
        if _READY["status"] == "not_started":
            _READY["status"] = "loading"
            _READY["started_at"] = time.time()
            _READY["detail"] = "initializing pipeline"

    model_id = os.environ.get("TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B")
    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    retries = max(1, int(os.environ.get("TRELLIS2_DRIVER_RETRY_ATTEMPTS", "4") or "4"))
    retry_sleep_sec = max(1, int(os.environ.get("TRELLIS2_DRIVER_RETRY_SLEEP_SEC", "8") or "8"))
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            if _should_avoid_gated_deps():
                pipe = _build_pipeline_with_image_cond_override(model_id)
            elif _should_avoid_gated_rembg_deps():
                pipe = _build_pipeline_with_rembg_override(model_id)
            else:
                pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
            break
        except Exception as e:
            last_err = e
            # If the upstream config points at a gated model, fall back to a non-gated alternative.
            if _is_gated_repo_error(e):
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


def _preload_worker():
    try:
        print("[worker] preload: starting model preload", flush=True)
        _get_pipeline()
        st = get_ready_state()
        dt = 0.0
        if st.get("started_at") and st.get("ready_at"):
            dt = float(st["ready_at"]) - float(st["started_at"])
        print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)
    except Exception:
        tb = traceback.format_exc()
        with _READY_LOCK:
            _READY["status"] = "error"
            _READY["detail"] = tb[-4000:] if tb else "unknown error"
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
            _int_env("TRELLIS2_DECIMATION_TARGET_LOW_POLY", 250000),
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
) -> Path:
    if not images_bytes:
        raise ValueError("empty upload")

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

        attempt = 0
        max_attempts = max(1, retries)
        img0 = primary_img
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
        # On CUDA OOM during mesh.simplify, retry with lower decimation to stay within VRAM.
        decimation_target, texture_size = _choose_export_params(low_poly)
        uid = uuid.uuid4().hex
        dec_min = _int_env("TRELLIS2_DECIMATION_MIN_OOM_FALLBACK", 50000)
        glb = None
        last_glb_err = None
        for attempt in range(_int_env("TRELLIS2_TO_GLB_OOM_RETRIES", 3)):
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
                if _is_cuda_oom(e) and attempt < 2 and decimation_target > dec_min:
                    _cuda_empty_cache()
                    decimation_target = max(dec_min, decimation_target // 2)
                    print(
                        f"[worker] to_glb OOM, retrying with decimation_target={decimation_target}",
                        flush=True,
                    )
                    continue
                raise
        if glb is None:
            raise RuntimeError("to_glb failed") from last_glb_err

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
