# NEXT_STEPS — docker_mlx_cpp

## Last Updated: 2026-04-01

## Vision
The NVIDIA Container Toolkit for Mac. Any Docker container gets full Metal GPU access — inference, training, image gen, audio, embeddings, raw GPU compute. First-mover on Apple Silicon.

## Status: v2 Complete — All Engines Fixed + Deepened

## Accomplished This Session
- [x] **AUDIT**: Found 4/5 engines broken (wrong imports, wrong APIs, wrong return types)
- [x] **A1 inference.py**: Fixed stream_generate (GenerationResponse objects, not strings), added VLM support via mlx-vlm
- [x] **A2 embeddings.py**: Fixed load/generate API (was using nonexistent `encode`)
- [x] **A3 audio.py**: Fixed STT (generate_transcription) + TTS (load_model + stream generate)
- [x] **A4 image_gen.py**: Replaced broken stable_diffusion import with mflux (pip-installable FLUX)
- [x] **A5 training.py**: Replaced fake mlx_lm.lora() with real mlx_lm.tuner.train() Python API
- [x] **A6 gateway**: Fixed SSE streaming proxy (httpx streaming, not buffered)
- [x] **A7 pyproject.toml**: All MLX packages now CORE deps (not optional)
- [x] **A8 presets.yaml**: Fixed audio model IDs to verified working models
- [x] **B1 VLM**: Vision-language model support with image input detection
- [x] **B2 File upload**: POST /files/upload for containers to push data to host
- [x] **B4 GPU compute**: New compute engine — direct MLX tensor ops (matmul, softmax, sort, benchmark)
- [x] **B4 GPU monitoring**: mx.metal.get_active_memory/peak_memory/cache_memory
- [x] **Engine availability**: GET /engines shows which MLX packages are installed

## TODO — Validate on M5
- [ ] pip install -e "." on M5 Mac
- [ ] mlx-cpp serve → verify all engines load
- [ ] Test each endpoint (11-point checklist in plan)
- [ ] docker compose up → test gateway routing
- [ ] Container → gateway → daemon → Metal GPU → response

## TODO — Sprint C: Harden
- [ ] Pydantic schemas (daemon/schemas.py)
- [ ] Request queue (asyncio.Semaphore for GPU serialization)
- [ ] Integration tests
- [ ] Async wrapping for GPU calls

## Blockers
- None

## MVP Readiness: 75%
