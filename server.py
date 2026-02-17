#!/usr/bin/env python3
"""
FastAPI server for TRELLIS.2 image-to-3D generation.

Endpoints:
  - GET  /health -> {"status":"OK"}
  - GET  /ready  -> {"ready": true|false, ...}
  - POST /generate (multipart 'image' and/or repeated 'images') -> {"success": true, "glb_path": "/outputs/<file>.glb"} or {"success": false, "error": "..."}
  - GET  /download/{filename} -> returns file bytes

Response shape mirrors other workers so a client can reuse the same protocol.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import sys
import traceback
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import List, Optional

# Configure logging first so we see messages even if later imports or startup fail.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stderr,
    force=True,
)

_log = logging.getLogger(__name__)
_log.info("worker process starting")

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

from worker import (
    generate_glb_from_image_bytes_list,
    get_ready_state,
    start_preload_in_background,
)

_log.info("worker module loaded")

APP_PORT = int(os.environ.get("PORT", "8000"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _log_exit():
    _log.info("process exiting")
    try:
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:
        pass


atexit.register(_log_exit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _log.info("lifespan startup begin")
    # Start server first so container is reachable; CUDA check + model load run in background.
    start_preload_in_background()
    _log.info("HTTP server listening on 0.0.0.0:%s; preload (CUDA + model) running in background", APP_PORT)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return JSONResponse({"status": "OK"}, status_code=200)


@app.get("/ready")
def ready():
    st = get_ready_state()
    ok = st.get("status") == "ready"
    resp = {"ready": bool(ok), **st}
    return JSONResponse(resp, status_code=200)


def _parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _parse_float(
    v: Optional[str],
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[float]:
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        x = float(s)
    except Exception:
        return default
    if min_value is not None and x < min_value:
        x = float(min_value)
    if max_value is not None and x > max_value:
        x = float(max_value)
    return x


@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    low_poly: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    pipeline_type: Optional[str] = Form(None),
    preprocess_image: Optional[str] = Form(None),
    post_scale_z: Optional[str] = Form(None),
    backup_inputs: Optional[str] = Form(None),
):
    export_meta: dict = {}
    try:
        want_low_poly = _parse_bool(low_poly, default=False)
        want_backup = _parse_bool(backup_inputs, default=True)
        want_preprocess = _parse_bool(preprocess_image, default=True) if preprocess_image is not None else None
        want_post_scale_z = _parse_float(post_scale_z, default=None, min_value=0.01, max_value=8.0)
        safe_seed = None
        if seed is not None and str(seed).strip():
            try:
                safe_seed = int(str(seed).strip())
            except Exception:
                safe_seed = None

        uploads: List[UploadFile] = []
        if images:
            uploads.extend([u for u in images if u is not None])
        if image is not None and not uploads:
            uploads.append(image)

        if not uploads:
            return JSONResponse({"success": False, "error": "missing image(s)"}, status_code=200)

        raw_list = []
        for u in uploads:
            raw = await u.read()
            if raw:
                raw_list.append(raw)

        if not raw_list:
            return JSONResponse({"success": False, "error": "empty upload(s)"}, status_code=200)

        # Run the blocking CUDA generation in a thread so the event loop stays
        # responsive for /health and /ready probes during long jobs.
        out_path = await asyncio.to_thread(
            partial(
                generate_glb_from_image_bytes_list,
                raw_list,
                out_dir=OUTPUT_DIR,
                low_poly=want_low_poly,
                seed=safe_seed,
                pipeline_type=str(pipeline_type).strip() if pipeline_type else None,
                preprocess_image=want_preprocess,
                post_scale_z=want_post_scale_z,
                backup_inputs=want_backup,
                export_meta=export_meta,
            )
        )
        resp = {"success": True, "glb_path": str(out_path)}
        if export_meta:
            resp["worker_export"] = export_meta
        return JSONResponse(resp, status_code=200)
    except Exception:
        tb = traceback.format_exc()
        if len(tb) > 20000:
            tb = tb[:20000] + "\n...[truncated]...\n"
        resp = {"success": False, "error": tb}
        if export_meta:
            resp["worker_export"] = export_meta
        return JSONResponse(resp, status_code=200)


@app.get("/download/{filename}")
def download(filename: str):
    name = os.path.basename(filename or "")
    p = OUTPUT_DIR / name
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary", filename=name)


if __name__ == "__main__":
    import uvicorn

    _log.info("starting uvicorn on 0.0.0.0:%s", APP_PORT)
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
