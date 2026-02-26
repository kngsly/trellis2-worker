#!/usr/bin/env python3
"""
FastAPI server for TRELLIS.2 image-to-3D generation.

Endpoints:
  - GET  /health -> {"status":"OK"}
  - GET  /ready  -> {"ready": true|false, ...}
  - POST /generate (multipart 'image' and/or repeated 'images') -> {"success": true, "glb_path": "/outputs/<file>.glb"} or {"success": false, "error": "..."}
  - GET  /download/{filename} -> returns file bytes

Response shape mirrors other workers so a client can reuse the same protocol.

Usage:
  Default (no idle shutdown):
    python3 server.py
    uvicorn server:app

  With idle shutdown and custom timeouts:
    python3 server.py --idle-shutdown --idle-after-ready-sec 600 --idle-after-generation-sec 180
    uvicorn server:app  # then set env: TRELLIS2_IDLE_SHUTDOWN=1 TRELLIS2_IDLE_SHUTDOWN_AFTER_READY_SEC=600 TRELLIS2_IDLE_SHUTDOWN_AFTER_GENERATION_SEC=180
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sys
import time
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


def _parse_server_args():
    import argparse
    p = argparse.ArgumentParser(description="TRELLIS.2 worker server")
    p.add_argument(
        "--idle-shutdown",
        action="store_true",
        help="Enable idle shutdown (exit after idle/generation timeouts). Default: disabled.",
    )
    p.add_argument(
        "--idle-after-ready-sec",
        type=int,
        default=300,
        metavar="SEC",
        help="Seconds idle after ready before exit (default: 300). Only used if --idle-shutdown.",
    )
    p.add_argument(
        "--idle-after-generation-sec",
        type=int,
        default=120,
        metavar="SEC",
        help="Seconds idle after last generation before exit (default: 120). Only used if --idle-shutdown.",
    )
    return p.parse_known_args()


def _apply_server_args():
    args, _ = _parse_server_args()
    os.environ["TRELLIS2_IDLE_SHUTDOWN"] = "1" if args.idle_shutdown else "0"
    os.environ["TRELLIS2_IDLE_SHUTDOWN_AFTER_READY_SEC"] = str(args.idle_after_ready_sec)
    os.environ["TRELLIS2_IDLE_SHUTDOWN_AFTER_GENERATION_SEC"] = str(args.idle_after_generation_sec)
    return args


_apply_server_args()

from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from worker import (
    cancel_idle_shutdown,
    generate_glb_from_image_bytes_list,
    get_ready_state,
    is_generating,
    normalize_generation_request,
    schedule_idle_shutdown_after_generation,
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
    resp = {"ready": bool(ok), "generating": is_generating(), **st}
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


def _parse_int(v: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if v is None:
        return default
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(s)
    except Exception:
        return default


@app.post("/generate")
async def generate(
    request: Request,
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    low_poly: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    pipeline_type: Optional[str] = Form(None),
    preprocess_image: Optional[str] = Form(None),
    post_scale_z: Optional[str] = Form(None),
    backup_inputs: Optional[str] = Form(None),
    decimation_target: Optional[str] = Form(None),
    mesh_profile: Optional[str] = Form(None),
    geometry_resolution: Optional[str] = Form(None),
    texture_generation_mode: Optional[str] = Form(None),
    texture_output_size: Optional[str] = Form(None),
    steps: Optional[str] = Form(None),
    generate_timeout_sec: Optional[str] = Form(None),
    # Legacy aliases retained for compatibility.
    resolution: Optional[str] = Form(None),
    texture_mode: Optional[str] = Form(None),
    texture_size: Optional[str] = Form(None),
    hd: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
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
        safe_decimation_target = _parse_int(decimation_target, default=None)
        raw_request = {
            "mesh_profile": mesh_profile,
            "geometry_resolution": _parse_int(geometry_resolution, default=None),
            "decimation_target": safe_decimation_target,
            "texture_generation_mode": texture_generation_mode,
            "texture_output_size": _parse_int(texture_output_size, default=None),
            "steps": _parse_int(steps, default=None),
            # Legacy compatibility.
            "low_poly": want_low_poly,
            "pipeline_type": str(pipeline_type).strip() if pipeline_type else None,
            "resolution": _parse_int(resolution, default=None),
            "texture_mode": texture_mode,
            "texture_size": _parse_int(texture_size, default=None),
            "enable_hd": _parse_bool(hd, default=False),
            "quality": quality,
        }
        normalized, adjustments = normalize_generation_request(raw_request, strict_4k_geometry=False)
        for note in adjustments:
            _log.info("request auto-adjustment: %s", note)
        _log.info(
            "request normalized geometry_resolution=%s texture_generation_mode=%s texture_bake_resolution=%s hd_enabled=%s",
            normalized["geometry_resolution"],
            normalized["texture_generation_mode"],
            normalized["texture_output_size"],
            1 if normalized["enable_hd"] else 0,
        )

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

        cancel_idle_shutdown()

        # Server-side generation timeout.
        # The client can pass generate_timeout_sec; fall back to generous defaults.
        timeout_sec = _parse_int(generate_timeout_sec, default=None)
        if timeout_sec is None or timeout_sec <= 0:
            is_standard = normalized["mesh_profile"] == "game_ready"
            timeout_sec = 10 * 60 if is_standard else 20 * 60
        timeout_sec = max(30, min(timeout_sec, 30 * 60))  # clamp 30s–30min
        _log.info("generate timeout_sec=%s", timeout_sec)

        # Run the blocking CUDA generation in a thread so the event loop stays
        # responsive for /health and /ready probes during long jobs.
        # Stream NDJSON keepalive pings every 15s to prevent idle TCP timeouts.
        KEEPALIVE_INTERVAL = 15

        gen_kwargs = dict(
            out_dir=OUTPUT_DIR,
            low_poly=bool(normalized["mesh_profile"] == "game_ready"),
            seed=safe_seed,
            pipeline_type=normalized["pipeline_type"],
            preprocess_image=want_preprocess,
            post_scale_z=want_post_scale_z,
            backup_inputs=want_backup,
            export_meta=export_meta,
            decimation_target=normalized["decimation_target"],
            mesh_profile=normalized["mesh_profile"],
            geometry_resolution=normalized["geometry_resolution"],
            texture_generation_mode=normalized["texture_generation_mode"],
            texture_output_size=normalized["texture_output_size"],
            steps=normalized["steps"],
        )

        async def _stream():
            start = time.monotonic()
            gen_task = asyncio.ensure_future(
                asyncio.to_thread(
                    partial(generate_glb_from_image_bytes_list, raw_list, **gen_kwargs)
                )
            )
            try:
                while not gen_task.done():
                    # Check for client disconnect
                    if await request.is_disconnected():
                        elapsed = time.monotonic() - start
                        _log.warning(
                            "client disconnected during generation at %.1fs", elapsed
                        )
                        gen_task.cancel()
                        return

                    done, _ = await asyncio.wait({gen_task}, timeout=KEEPALIVE_INTERVAL)
                    if done:
                        break

                    elapsed = time.monotonic() - start
                    if elapsed > timeout_sec:
                        gen_task.cancel()
                        _log.warning("generation timed out after %ss", timeout_sec)
                        resp = {
                            "type": "result",
                            "success": False,
                            "error": f"Generation timed out after {timeout_sec}s",
                        }
                        if export_meta:
                            resp["worker_export"] = export_meta
                        yield json.dumps(resp) + "\n"
                        return
                    yield json.dumps(
                        {"type": "keepalive", "elapsed_sec": round(elapsed, 1)}
                    ) + "\n"

                elapsed = time.monotonic() - start
                client_connected = not await request.is_disconnected()
                _log.info(
                    "generation wall_time=%.1fs client_connected=%s",
                    elapsed,
                    client_connected,
                )

                out_path = gen_task.result()
                resp = {
                    "type": "result",
                    "success": True,
                    "glb_path": str(out_path),
                }
                if export_meta:
                    resp["worker_export"] = export_meta
                yield json.dumps(resp) + "\n"
            except asyncio.CancelledError:
                elapsed = time.monotonic() - start
                _log.warning("generation cancelled at %.1fs", elapsed)
                resp = {
                    "type": "result",
                    "success": False,
                    "error": "Generation cancelled (client disconnected)",
                }
                if export_meta:
                    resp["worker_export"] = export_meta
                yield json.dumps(resp) + "\n"
            except Exception:
                tb = traceback.format_exc()
                if len(tb) > 20000:
                    tb = tb[:20000] + "\n...[truncated]...\n"
                resp = {"type": "result", "success": False, "error": tb}
                if export_meta:
                    resp["worker_export"] = export_meta
                yield json.dumps(resp) + "\n"
            finally:
                schedule_idle_shutdown_after_generation()

        return StreamingResponse(
            _stream(),
            status_code=200,
            media_type="application/x-ndjson",
        )
    except Exception:
        schedule_idle_shutdown_after_generation()
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
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(
        description="TRELLIS.2 image-to-3D worker server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--idle-shutdown",
        action="store_true",
        help="Enable idle shutdown: exit after N s with no generation (after ready) or after last job.",
    )
    parser.add_argument(
        "--idle-after-ready-sec",
        type=int,
        default=None,
        metavar="SEC",
        help="Seconds with no generation after server ready before exit. Implies --idle-shutdown if set.",
    )
    parser.add_argument(
        "--idle-after-generation-sec",
        type=int,
        default=None,
        metavar="SEC",
        help="Seconds with no new generation after last job before exit. Implies --idle-shutdown if set.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host for uvicorn.")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default: PORT env or 8000).")
    args = parser.parse_args()

    if args.idle_shutdown or args.idle_after_ready_sec is not None or args.idle_after_generation_sec is not None:
        from worker import configure_idle_shutdown

        configure_idle_shutdown(
            enabled=args.idle_shutdown or (args.idle_after_ready_sec is not None) or (args.idle_after_generation_sec is not None),
            after_ready_sec=args.idle_after_ready_sec,
            after_generation_sec=args.idle_after_generation_sec,
        )
        _log.info(
            "idle shutdown: enabled (after_ready_sec=%s, after_generation_sec=%s)",
            args.idle_after_ready_sec,
            args.idle_after_generation_sec,
        )

    port = args.port if args.port is not None else APP_PORT
    _log.info("starting uvicorn on %s:%s", args.host, port)
    uvicorn.run(app, host=args.host, port=port, log_level="info")
