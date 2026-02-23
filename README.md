# trellis2-worker

Dedicated Docker image + HTTP worker for TRELLIS.2 image-to-3D, intended to run on GPU VMs (e.g. Vast.ai).

This repo exists so you can deploy/build the worker on GPU machines without copying a larger private app repo (or any secrets) onto those machines.

## What Runs

- `server.py`: FastAPI app
- `worker.py`: loads TRELLIS.2 pipeline and serves `/generate`
- `Dockerfile`: builds a CUDA 12.4 image with TRELLIS.2 + required CUDA extensions

## Build (on the GPU VM)

```bash
git clone <YOUR_WORKER_REPO_URL> /root/trellis2-worker
cd /root/trellis2-worker

docker build --progress=plain -t <dockerhub_user>/<image_name>:<tag> .
```

## Run (on the GPU VM)

```bash
export HF_TOKEN="...optional..."

docker rm -f trellis2-worker || true
docker run -d --name trellis2-worker --gpus all \
  -e PORT=8000 \
  -e TRELLIS2_MODEL_ID="microsoft/TRELLIS.2-4B" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TRELLIS2_DISABLE_RUNTIME_DOWNLOADS=0 \
  -e TRELLIS2_AVOID_GATED_DEPS=1 \
  -e TRELLIS2_DINOV2_MODEL_NAME="dinov2_vitl14_reg" \
  -p 8000:8000 \
  <dockerhub_user>/<image_name>:<tag>

docker logs -f trellis2-worker
```

Notes:
- `TRELLIS2_AVOID_GATED_DEPS=1` (default) avoids gated Hugging Face repos (e.g. DINOv3) by switching the image-conditioning backbone to DINOv2.
- If you later get approved for gated repos and want the upstream default behavior, run with `-e TRELLIS2_AVOID_GATED_DEPS=0`.
- Runtime downloads are **enabled by default**: if a model asset is missing from the image cache it is downloaded at startup automatically.
- Set `TRELLIS2_DISABLE_RUNTIME_DOWNLOADS=1` to opt out and force local-cache-only loading. Startup fails fast if any asset is missing — useful to catch a broken build early.

### Model Asset Caching (Persistent Volume — Recommended)

The fastest startup is achieved by mounting a persistent volume for the HuggingFace cache. Models download once on first run and are reused on every subsequent start, regardless of which host the container lands on.

On Vast.ai, attach a network volume mounted at `/root/.cache/huggingface` (or any path), then run:

```bash
docker run -d --name trellis2-worker --gpus all \
  -v /path/to/persistent/volume:/root/.cache/huggingface \
  -e PORT=8000 \
  -e TRELLIS2_MODEL_ID="microsoft/TRELLIS.2-4B" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TRELLIS2_AVOID_GATED_DEPS=1 \
  -p 8000:8000 \
  <dockerhub_user>/<image_name>:<tag>
```

First run downloads models (~15 GB, ~5–8 min on a fast connection). Every subsequent start skips the download entirely and goes straight to model loading (~2–4 min).

To force local-only loading and fail fast if any asset is missing (useful after the first successful run):

```bash
docker run ... -e TRELLIS2_DISABLE_RUNTIME_DOWNLOADS=1 ...
```

## Idle shutdown

Idle shutdown is **disabled by default**. When enabled, the process exits after a period of no activity so the server does not sit idle indefinitely (e.g. on spot/preemptible GPU VMs).

- **Enable via CLI** (when starting the process directly, e.g. `python3 server.py`):
  - `--idle-shutdown` — turn idle shutdown on
  - `--idle-after-ready-sec SEC` — seconds with no job after the model becomes ready before exit (default: 300)
  - `--idle-after-generation-sec SEC` — seconds with no new job after the last generation before exit (default: 120)
- **Enable via env** (e.g. in Docker): set `TRELLIS2_IDLE_SHUTDOWN=1`, and optionally:
  - `TRELLIS2_IDLE_SHUTDOWN_AFTER_READY_SEC` (default: 300)
  - `TRELLIS2_IDLE_SHUTDOWN_AFTER_GENERATION_SEC` (default: 120)

Example with env (Docker):

```bash
docker run -d --name trellis2-worker --gpus all \
  -e TRELLIS2_IDLE_SHUTDOWN=1 \
  -e TRELLIS2_IDLE_SHUTDOWN_AFTER_READY_SEC=600 \
  -e TRELLIS2_IDLE_SHUTDOWN_AFTER_GENERATION_SEC=180 \
  ...other options...
```

When idle shutdown is enabled, the worker exits (cleanly) after the configured idle window; an orchestrator or process manager can restart it when new work is available.

## Health / Ready

```bash
curl -sS http://localhost:8000/health
curl -sS http://localhost:8000/ready
```

## Generate (example)

```bash
curl -sS -X POST \
  -F "image=@/path/to/input.png" \
  -F "low_poly=false" \
  -F "seed=42" \
  -F "preprocess_image=true" \
  http://localhost:8000/generate
```

The response includes a `glb_path` (inside the container, under `/outputs`). Download via:

```bash
resp="$(curl -sS -X POST -F "image=@/path/to/input.png" http://localhost:8000/generate)"
name="$(printf '%s' "$resp" | python3 -c 'import json,sys,os; d=json.load(sys.stdin); print(os.path.basename(d["glb_path"]))')"
curl -fL -o out.glb "http://localhost:8000/download/$name"
```

Notes:
- If a particular image fails with an "empty sparse coords" error, try `-F "preprocess_image=false"` (skips background removal/cropping inside TRELLIS).
- For transparent PNG assets, the worker also supports alpha-crop/upscale before calling TRELLIS. Control with:
  - `TRELLIS2_CROP_ALPHA=1` (default)
  - `TRELLIS2_UPSCALE_SMALL=1` (default) and `TRELLIS2_UPSCALE_TARGET=512`
  - `TRELLIS2_FORCE_RGB=1` (optional, composites alpha onto white)

### Troubleshooting: Box Artifacts / Flat 2D Results

If outputs show a square box around the object or collapse into a textured 2D slab:

- Keep `preprocess_image=true` (default) so the worker can sanitize masks and retry bad mattes.
- Prefer clean transparent PNGs where background alpha is truly zero.
- Check `worker_export` in the `/generate` response:
  - `oom_quality_degradation_level`: high values can reduce geometric quality.
  - `mesh_thin_axis_ratio`: near-zero values indicate a planar/flat collapse.
- If you have access to gated upstream models, use the original conditioning stack:
  - `TRELLIS2_AVOID_GATED_DEPS=0`
  - `TRELLIS2_AVOID_GATED_REMBG_DEPS=0`

Advanced mask controls:

- `TRELLIS2_PREPROCESS_REMBG_ON_BAD_ALPHA=1` (default): if alpha appears unreliable, retry segmentation with rembg.
- `TRELLIS2_PREPROCESS_MIN_MASK_AREA_FRAC=0.00005`: ignore tiny alpha noise while finding bbox.
- `TRELLIS2_PREPROCESS_ALLOW_ALPHA_NONZERO_FALLBACK=0` (default): avoids full-frame bbox from faint nonzero alpha haze.
- `TRELLIS2_MIN_THIN_AXIS_RATIO=0.001`: reject near-planar meshes and retry with a new seed.

## Export Image From VM (download locally, then push from your machine)

On the VM:

```bash
docker save <dockerhub_user>/<image_name>:<tag> | zstd -T0 -19 -o /root/trellis2-worker_<tag>.tar.zst
ls -lh /root/trellis2-worker_<tag>.tar.zst
```

On your local machine:

```bash
scp -P <SSH_PORT> root@<VM_IP>:/root/trellis2-worker_<tag>.tar.zst .
```

Then locally:

```bash
zstd -d ./trellis2-worker_<tag>.tar.zst -c | docker load

docker tag <dockerhub_user>/<image_name>:<tag> <dockerhub_user>/<image_name>:latest
docker push <dockerhub_user>/<image_name>:<tag>
docker push <dockerhub_user>/<image_name>:latest
```

## Access From Your Local Machine

If your worker is running on a remote VM but only reachable on the VM's localhost, use SSH port forwarding:

```bash
ssh -p <SSH_PORT> -L 8000:127.0.0.1:8000 root@<VM_IP>
```

Then in another local terminal, use `http://127.0.0.1:8000` as the base URL.
