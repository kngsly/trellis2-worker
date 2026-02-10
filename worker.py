#!/usr/bin/env python3
"""
TRELLIS.2 worker implementation.

This module is imported by server.py.
"""

from __future__ import annotations

import io
import os
import threading
import time
import traceback
import uuid
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


def _get_device():
    return os.environ.get("TRELLIS2_DEVICE", "cuda")


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
    # Default to avoiding gated repos so the worker can boot on fresh machines.
    return _bool_env("TRELLIS2_AVOID_GATED_DEPS", True)


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
        # If the upstream config points at a gated model, fall back to a non-gated alternative.
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
    backup_inputs: bool = True,
) -> Path:
    if not images_bytes:
        raise ValueError("empty upload")

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pipe = _get_pipeline()
    imgs = []
    for raw in images_bytes:
        if not raw:
            continue
        im = Image.open(io.BytesIO(raw))
        im = _prepare_input_image(im)
        imgs.append(im)
    if not imgs:
        raise ValueError("empty image bytes")

    if backup_inputs:
        in_dir = out_dir / "inputs"
        in_dir.mkdir(parents=True, exist_ok=True)
        for idx, (raw, im) in enumerate(zip(images_bytes, imgs)):
            try:
                # Prefer PNG to keep alpha if present.
                p = in_dir / f"{uuid.uuid4().hex}_{idx+1:02d}.png"
                im.save(str(p), format="PNG")
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
    img0 = imgs[0]

    import torch

    # Default seed: deterministic for easier debugging; caller can override.
    if seed is None:
        seed = _int_env("TRELLIS2_DEFAULT_SEED", 42)

    # Pass only kwargs supported by this pipeline version.
    kwargs = {}
    try:
        sig = inspect.signature(pipe.run)  # type: ignore[attr-defined]
        if "seed" in sig.parameters:
            kwargs["seed"] = int(seed)
        if ptype and "pipeline_type" in sig.parameters:
            kwargs["pipeline_type"] = ptype
        if "num_samples" in sig.parameters:
            kwargs["num_samples"] = 1
        if "preprocess_image" in sig.parameters:
            if preprocess_image is None:
                kwargs["preprocess_image"] = _bool_env("TRELLIS2_PREPROCESS_IMAGE", True)
            else:
                kwargs["preprocess_image"] = bool(preprocess_image)
    except Exception:
        # If signature inspection fails, still attempt a minimal call.
        kwargs["seed"] = int(seed)
        if ptype:
            kwargs["pipeline_type"] = ptype

    # Retry on empty sparse sampling by shifting the seed.
    def _is_empty_sparse_error(exc: BaseException) -> bool:
        msg = (str(exc) or "").lower()
        return ("input.numel() == 0" in msg) or ("max(): expected reduction dim" in msg)

    retries = _int_env("TRELLIS2_EMPTY_SPARSE_RETRIES", 4)
    last_err: Optional[BaseException] = None

    # Some inputs get wiped out by TRELLIS preprocessing (background removal + crop), which can
    # lead to empty sparse coords. If that happens, we auto-fallback by toggling preprocess_image.
    preprocess_toggle_attempted = False
    base_preprocess = kwargs.get("preprocess_image", None)

    attempt = 0
    max_attempts = max(1, retries)
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

            # If we exhausted seed-shift retries, flip preprocess_image once and try again.
            if (not preprocess_toggle_attempted) and ("preprocess_image" in kwargs):
                preprocess_toggle_attempted = True
                attempt = 0
                kwargs["preprocess_image"] = (not bool(base_preprocess))
                continue

            raise RuntimeError("pipeline.run() failed after retries") from last_err

    # Export to GLB via o-voxel postprocess util.
    decimation_target, texture_size = _choose_export_params(low_poly)
    uid = uuid.uuid4().hex
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

    out_path = out_dir / f"{uid}.glb"
    # Use webp textures to keep output compact.
    glb.export(str(out_path), extension_webp=True)

    dt = time.time() - t0
    print(
        f"[worker] generated {out_path} in {dt:.1f}s "
        f"(inputs={len(images_bytes)} low_poly={low_poly} pipeline_type={ptype})",
        flush=True,
    )
    return out_path
