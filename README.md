# trellis2-worker

Dedicated Docker image + HTTP worker for TRELLIS.2 image-to-3D, intended to run on GPU VMs (e.g. Vast.ai).

This repo exists so we never have to upload the full `the private application repo` repo (and any secrets) to random GPU boxes.

## What Runs

- `server.py`: FastAPI app
- `worker.py`: loads TRELLIS.2 pipeline and serves `/generate`
- `Dockerfile`: builds a CUDA 12.4 image with TRELLIS.2 + required CUDA extensions

## Build (on the GPU VM)

```bash
git clone https://github.com/kngsly/trellis2-worker.git /root/trellis2-worker
cd /root/trellis2-worker

docker build --progress=plain -t kngsly/trellis2-worker:2026-02-10 .
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
  kngsly/trellis2-worker:2026-02-10

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
  http://localhost:8000/generate
```

The response includes a `glb_path` (inside the container, under `/outputs`). Download via:

```bash
curl -fL -o out.glb "http://localhost:8000/download/$(basename /outputs/<file>.glb)"
```

## Export Image From VM (download locally, then push from your machine)

On the VM:

```bash
docker save kngsly/trellis2-worker:2026-02-10 | zstd -T0 -19 -o /root/trellis2-worker_2026-02-10.tar.zst
ls -lh /root/trellis2-worker_2026-02-10.tar.zst
```

On your local machine (download into `/media/user/Zoomer/`):

```bash
scp -P <SSH_PORT> root@<VM_IP>:/root/trellis2-worker_2026-02-10.tar.zst /media/user/Zoomer/
```

Then locally:

```bash
zstd -d /media/user/Zoomer/trellis2-worker_2026-02-10.tar.zst -c | docker load

docker tag kngsly/trellis2-worker:2026-02-10 kngsly/trellis2-worker:latest
docker push kngsly/trellis2-worker:2026-02-10
docker push kngsly/trellis2-worker:latest
```

