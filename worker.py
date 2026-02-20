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
import signal
import threading
import time
import traceback
import uuid
import ctypes
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


def _build_pipeline_with_image_cond_override(model_id: str):
    """
    Build a Trellis2ImageTo3DPipeline but override the image conditioning model
    to avoid gated dependencies (e.g. DINOv3 on HF).
    """
    from trellis2.pipelines.base import Pipeline  # type: ignore
    from trellis2.pipelines import samplers, rembg  # type: ignore
    from trellis2.modules import image_feature_extractor  # type: ignore

    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, model_id, "pipeline.json")  # type: ignore[attr-defined]
    args = getattr(pipe, "_pretrained_args", None) or {}

    dinov2_name = os.environ.get("TRELLIS2_DINOV2_MODEL_NAME", "dinov2_vitl14_reg").strip()
    args["image_cond_model"] = {"name": "DinoV2FeatureExtractor", "args": {"model_name": dinov2_name}}

    if _should_avoid_gated_rembg_deps():
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

    with _READY_LOCK:
        if _READY["status"] == "not_started":
            _READY["status"] = "loading"
            _READY["started_at"] = time.time()
            _READY["detail"] = "initializing pipeline"

    model_id = os.environ.get("TRELLIS2_MODEL_ID", "microsoft/TRELLIS.2-4B")
    Trellis2ImageTo3DPipeline = _lazy_import_pipeline()

    try:
        if _should_avoid_gated_deps():
            pipe = _build_pipeline_with_image_cond_override(model_id)
        elif _should_avoid_gated_rembg_deps():
            pipe = _build_pipeline_with_rembg_override(model_id)
        else:
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    except Exception as e:
        if _is_gated_repo_error(e):
            if _should_avoid_gated_deps():
                pipe = _build_pipeline_with_image_cond_override(model_id)
            elif _should_avoid_gated_rembg_deps():
                pipe = _build_pipeline_with_rembg_override(model_id)
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


def _preload_worker():
    try:
        _log.info("preload: starting model preload")
        print("[worker] preload: starting model preload", flush=True)
        _get_pipeline()
        st = get_ready_state()
        dt = 0.0
        if st.get("started_at") and st.get("ready_at"):
            dt = float(st["ready_at"]) - float(st["started_at"])
        _log.info("preload: ready (load_time_sec=%.1f)", dt)
        print(f"[worker] preload: ready (load_time_sec={dt:.1f})", flush=True)
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
        return im_rgb.convert("RGBA")

    if low_vram:
        try:
            rembg_model.to(pipe.device)  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        out_rgba = rembg_model(im_rgb)  # type: ignore[attr-defined]
    except Exception as e:
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

    return out_rgba


def _safe_preprocess_image(pipe, input_im: Image.Image) -> Image.Image:
    """
    Preprocess to match the official demo behavior, but with guardrails:
    - Avoid empty bbox when alpha is soft/low.
    - Optional padding around the alpha bbox.

    Returns an RGB PIL image (alpha premultiplied like upstream).
    """
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

    if not has_alpha:
        im_rgb = input_im.convert("RGB")
        out_rgba = _run_rembg_rgba(pipe, im_rgb)
    else:
        out_rgba = input_im.convert("RGBA")
        # Guardrail: broken/near-empty alpha mattes can cause full-frame box crops.
        # If alpha looks unusable, optionally re-run rembg on RGB.
        if _bool_env("TRELLIS2_PREPROCESS_REMBG_ON_BAD_ALPHA", True):
            alpha_probe = np.array(out_rgba)[:, :, 3]
            probe_thresh = _float_env("TRELLIS2_PREPROCESS_ALPHA_BBOX_THRESHOLD", 0.8)
            bb_probe = _safe_alpha_bbox(alpha_probe, primary_thresh=probe_thresh)
            bad_alpha = bb_probe is None
            if not bad_alpha:
                w_probe, h_probe = out_rgba.size
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
                    "[worker] preprocess: alpha matte looks unreliable; retrying with rembg segmentation",
                    flush=True,
                )
                out_rgba = _run_rembg_rgba(pipe, input_im.convert("RGB"))

    out_np = np.array(out_rgba)
    if out_np.ndim != 3 or out_np.shape[2] < 4:
        return out_rgba.convert("RGB")

    alpha = out_np[:, :, 3]
    thresh = _float_env("TRELLIS2_PREPROCESS_ALPHA_BBOX_THRESHOLD", 0.8)
    pad_px = _int_env("TRELLIS2_PREPROCESS_PAD_PX", 8)
    bb = _safe_alpha_bbox(alpha, primary_thresh=thresh)
    if bb is None:
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

                # Progressive OOM handling: preserve texture quality as long as possible
                if _is_cuda_oom(e):
                    _cuda_empty_cache()

                    # Level 0: Simple cache clear
                    if oom_quality_level == 0:
                        oom_quality_level = 1
                        print("[worker] pipe.run OOM (level 0), clearing cache and retrying", flush=True)
                        continue

                    # Level 1: Reduce sparse_structure steps by 50%
                    elif oom_quality_level == 1:
                        oom_quality_level = 2
                        if "sparse_structure_sampler_params" in kwargs and isinstance(kwargs["sparse_structure_sampler_params"], dict):
                            orig_steps = kwargs["sparse_structure_sampler_params"].get("steps", 12)
                            new_steps = max(4, orig_steps // 2)
                            kwargs["sparse_structure_sampler_params"]["steps"] = new_steps
                            print(f"[worker] pipe.run OOM (level 1), reducing sparse_structure steps {orig_steps}->{new_steps}", flush=True)
                        continue

                    # Level 2: Reduce shape_slat steps by 50%
                    elif oom_quality_level == 2:
                        oom_quality_level = 3
                        if "shape_slat_sampler_params" in kwargs and isinstance(kwargs["shape_slat_sampler_params"], dict):
                            orig_steps = kwargs["shape_slat_sampler_params"].get("steps", 12)
                            new_steps = max(4, orig_steps // 2)
                            kwargs["shape_slat_sampler_params"]["steps"] = new_steps
                            print(f"[worker] pipe.run OOM (level 2), reducing shape_slat steps {orig_steps}->{new_steps}", flush=True)
                        continue

                    # Level 3: Switch from cascade to non-cascade pipeline
                    elif oom_quality_level == 3 and ptype and "cascade" in ptype.lower():
                        oom_quality_level = 4
                        fallback_ptype = ptype.replace("_cascade", "").replace("cascade_", "")
                        print(f"[worker] pipe.run OOM (level 3), switching from {ptype} to {fallback_ptype}", flush=True)
                        if "pipeline_type" in kwargs:
                            kwargs["pipeline_type"] = fallback_ptype
                        ptype = fallback_ptype
                        # Restore original steps for the new pipeline
                        if "sparse_structure_sampler_params" in kwargs and "sparse_structure_sampler_params" in original_kwargs:
                            kwargs["sparse_structure_sampler_params"]["steps"] = original_kwargs["sparse_structure_sampler_params"].get("steps", 12)
                        if "shape_slat_sampler_params" in kwargs and "shape_slat_sampler_params" in original_kwargs:
                            kwargs["shape_slat_sampler_params"]["steps"] = original_kwargs["shape_slat_sampler_params"].get("steps", 12)
                        if "tex_slat_sampler_params" in kwargs and "tex_slat_sampler_params" in original_kwargs:
                            kwargs["tex_slat_sampler_params"]["steps"] = original_kwargs["tex_slat_sampler_params"].get("steps", 12)
                        continue

                    # Level 4: Reduce sparse_structure to minimum
                    elif oom_quality_level in (3, 4):
                        oom_quality_level = 5
                        if "sparse_structure_sampler_params" in kwargs and isinstance(kwargs["sparse_structure_sampler_params"], dict):
                            kwargs["sparse_structure_sampler_params"]["steps"] = 4
                            print("[worker] pipe.run OOM (level 4), sparse_structure steps->4", flush=True)
                        continue

                    # Level 5: Reduce shape_slat to minimum
                    elif oom_quality_level == 5:
                        oom_quality_level = 6
                        if "shape_slat_sampler_params" in kwargs and isinstance(kwargs["shape_slat_sampler_params"], dict):
                            kwargs["shape_slat_sampler_params"]["steps"] = 4
                            print("[worker] pipe.run OOM (level 5), shape_slat steps->4", flush=True)
                        continue

                    # Level 6: Switch to 512 pipeline
                    elif oom_quality_level in (3, 4, 5, 6) and ptype != "512":
                        oom_quality_level = 7
                        print(f"[worker] pipe.run OOM (level 6), switching from {ptype} to 512 pipeline", flush=True)
                        if "pipeline_type" in kwargs:
                            kwargs["pipeline_type"] = "512"
                        ptype = "512"
                        if "tex_slat_sampler_params" in kwargs and "tex_slat_sampler_params" in original_kwargs:
                            kwargs["tex_slat_sampler_params"]["steps"] = original_kwargs["tex_slat_sampler_params"].get("steps", 12)
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
        uid = uuid.uuid4().hex
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
