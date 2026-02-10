# TRELLIS.2 Worker Image

This folder contains a FastAPI worker container intended to run on GPU VMs and provide a stable HTTP API for a client:

- `GET /health` -> `{"status":"OK"}`
- `GET /ready`  -> `{"ready": true|false, ...}`
- `POST /generate` (multipart: `image` and/or repeated `images`) -> JSON `{success, glb_path}` or `{success:false, error}`
- `GET /download/{filename}` -> returns file bytes

The API shape mirrors other workers so a client can keep using the same protocol.

## Multi-image support

`POST /generate` accepts:
- `image` (single file, required for backward compatibility)
- `images` (0..N files). If provided, the worker will fuse conditioning across images (simple mean over extracted features).

## Environment variables (worker container)

- `TRELLIS2_MODEL_ID` (default: `microsoft/TRELLIS.2-4B`)
- `TRELLIS2_PIPELINE_TYPE` (default: `1024_cascade`) one of `512`, `1024`, `1024_cascade`, `1536_cascade`
- `TRELLIS2_TEXTURE_SIZE` (default: `4096`)
- `TRELLIS2_TEXTURE_SIZE_LOW_POLY` (default: `2048`)
- `TRELLIS2_DECIMATION_TARGET` (default: `1000000`)
- `TRELLIS2_DECIMATION_TARGET_LOW_POLY` (default: `250000`)
- `OUTPUT_DIR` (default: `/outputs`)
- `PORT` (default: `8000`)

## Notes

TRELLIS.2 has heavyweight CUDA/C++ dependencies (`o-voxel`, `CuMesh`, `FlexGEMM`, `nvdiffrast`, etc.).
The `Dockerfile` in this folder is a pragmatic starting point, but you may need to iterate based on your target GPU/driver.
