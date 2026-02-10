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
  -e TRELLIS2_AVOID_GATED_DEPS=1 \
  -e TRELLIS2_DINOV2_MODEL_NAME="dinov2_vitl14_reg" \
  -p 8000:8000 \
  <dockerhub_user>/<image_name>:<tag>

docker logs -f trellis2-worker
```

Notes:
- `TRELLIS2_AVOID_GATED_DEPS=1` (default) avoids gated Hugging Face repos (e.g. DINOv3) by switching the image-conditioning backbone to DINOv2.
- If you later get approved for gated repos and want the upstream default behavior, run with `-e TRELLIS2_AVOID_GATED_DEPS=0`.

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
