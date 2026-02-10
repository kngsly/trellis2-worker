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

import os
import traceback
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse

from worker import (
    generate_glb_from_image_bytes_list,
    get_ready_state,
    start_preload_in_background,
)


APP_PORT = int(os.environ.get("PORT", "8000"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()


@app.get("/health")
def health():
    return JSONResponse({"status": "OK"}, status_code=200)


@app.get("/ready")
def ready():
    st = get_ready_state()
    ok = st.get("status") == "ready"
    return JSONResponse({"ready": bool(ok), **st}, status_code=200)


@app.on_event("startup")
def _startup():
    start_preload_in_background()


def _parse_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


@app.post("/generate")
async def generate(
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    low_poly: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    pipeline_type: Optional[str] = Form(None),
    preprocess_image: Optional[str] = Form(None),
    backup_inputs: Optional[str] = Form(None),
):
    try:
        want_low_poly = _parse_bool(low_poly, default=False)
        want_backup = _parse_bool(backup_inputs, default=True)
        want_preprocess = _parse_bool(preprocess_image, default=True) if preprocess_image is not None else None
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

        out_path = generate_glb_from_image_bytes_list(
            raw_list,
            out_dir=OUTPUT_DIR,
            low_poly=want_low_poly,
            seed=safe_seed,
            pipeline_type=str(pipeline_type).strip() if pipeline_type else None,
            preprocess_image=want_preprocess,
            backup_inputs=want_backup,
        )
        return JSONResponse({"success": True, "glb_path": str(out_path)}, status_code=200)
    except Exception:
        tb = traceback.format_exc()
        if len(tb) > 20000:
            tb = tb[:20000] + "\n...[truncated]...\n"
        return JSONResponse({"success": False, "error": tb}, status_code=200)


@app.get("/download/{filename}")
def download(filename: str):
    name = os.path.basename(filename or "")
    p = OUTPUT_DIR / name
    if not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(p), media_type="model/gltf-binary", filename=name)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=APP_PORT, log_level="info")
