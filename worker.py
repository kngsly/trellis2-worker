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
    args["image_cond_model"] = {
        "name": "DinoV2FeatureExtractor",
        "args": {"model_name": dinov2_name},
    }

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
        else:
            pipe = Trellis2ImageTo3DPipeline.from_pretrained(model_id)
    except Exception as e:
        # If the upstream config points at a gated model (e.g. facebook/dinov3-*),
        # fall back to a non-gated image conditioner.
        if (not _should_avoid_gated_deps()) and _is_gated_repo_error(e):
            pipe = _build_pipeline_with_image_cond_override(model_id)
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
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
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

    # Preprocess (background removal + crop)
    pre = _preprocess_images(pipe, imgs)

    import torch
    import o_voxel  # type: ignore

    if seed is None:
        # A deterministic default is useful for debugging; caller can override.
        seed = 42
    torch.manual_seed(int(seed))

    ptype = _resolve_pipeline_type(pipeline_type)
    cond_512 = _get_fused_cond(pipe, pre, 512)
    cond_1024 = _get_fused_cond(pipe, pre, 1024) if ptype != "512" else None

    # Replicate Trellis2ImageTo3DPipeline.run(), but use fused cond from multiple images.
    if ptype == "512":
        ss_res = 32
        coords = pipe.sample_sparse_structure(cond_512, ss_res, num_samples=1, sampler_params={})
        shape_slat = pipe.sample_shape_slat(cond_512, pipe.models["shape_slat_flow_model_512"], coords, sampler_params={})
        tex_slat = pipe.sample_tex_slat(cond_512, pipe.models["tex_slat_flow_model_512"], shape_slat, sampler_params={})
        res = 512
    elif ptype == "1024":
        if cond_1024 is None:
            raise ValueError("cond_1024 missing")
        ss_res = 64
        coords = pipe.sample_sparse_structure(cond_512, ss_res, num_samples=1, sampler_params={})
        shape_slat = pipe.sample_shape_slat(cond_1024, pipe.models["shape_slat_flow_model_1024"], coords, sampler_params={})
        tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, sampler_params={})
        res = 1024
    elif ptype == "1024_cascade":
        if cond_1024 is None:
            raise ValueError("cond_1024 missing")
        coords = pipe.sample_sparse_structure(cond_512, 32, num_samples=1, sampler_params={})
        shape_slat, res = pipe.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipe.models["shape_slat_flow_model_512"],
            pipe.models["shape_slat_flow_model_1024"],
            512,
            1024,
            coords,
            sampler_params={},
            max_num_tokens=_int_env("TRELLIS2_MAX_NUM_TOKENS", 49152),
        )
        tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, sampler_params={})
    elif ptype == "1536_cascade":
        if cond_1024 is None:
            raise ValueError("cond_1024 missing")
        coords = pipe.sample_sparse_structure(cond_512, 32, num_samples=1, sampler_params={})
        shape_slat, res = pipe.sample_shape_slat_cascade(
            cond_512,
            cond_1024,
            pipe.models["shape_slat_flow_model_512"],
            pipe.models["shape_slat_flow_model_1024"],
            512,
            1536,
            coords,
            sampler_params={},
            max_num_tokens=_int_env("TRELLIS2_MAX_NUM_TOKENS", 49152),
        )
        tex_slat = pipe.sample_tex_slat(cond_1024, pipe.models["tex_slat_flow_model_1024"], shape_slat, sampler_params={})
    else:
        raise ValueError(f"invalid pipeline_type: {ptype}")

    torch.cuda.empty_cache()
    mesh = pipe.decode_latent(shape_slat, tex_slat, res)[0]

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
