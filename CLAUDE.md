# docker_mlx_cpp — The NVIDIA Container Toolkit for Mac

## What This Is
Host-side MLX daemon + Docker gateway that gives **any container** Metal GPU access on Apple Silicon. Supports inference (50+ LLM architectures), training (LoRA/QLoRA/DPO), image generation (SD/FLUX), audio (Whisper/TTS), and embeddings.

## Architecture
```
Container → mlx-gateway:8080 (Docker) → MLX Daemon:12435 (host) → Metal GPU
                                       → DMR (fallback for GGUF)
```

## Key Files
```
daemon/mlx_daemon.py         # Host-side daemon (FastAPI, port 12435)
daemon/engines/inference.py  # LLM/VLM inference via mlx-lm
daemon/engines/training.py   # LoRA/QLoRA fine-tuning
daemon/engines/image_gen.py  # Stable Diffusion / FLUX
daemon/engines/audio.py      # Whisper STT + TTS
daemon/engines/embeddings.py # Text embeddings
daemon/model_manager.py      # Model pull/cache/convert from HuggingFace
gateway/server.py            # Container gateway (FastAPI reverse proxy)
cli/mlx_cpp.py               # CLI tool (mlx-cpp command)
models/presets.yaml           # Curated model presets
docker-compose.yml            # Full stack definition
pyproject.toml                # Python packaging + dependencies
```

## Dev Commands
```bash
# Install
pip install -e ".[all,dev]"

# Run daemon
mlx-cpp serve                          # or: python -m daemon.mlx_daemon
mlx-cpp health                         # Check daemon + GPU

# Run gateway
docker compose up -d mlx-gateway
docker compose logs -f

# Quick inference
mlx-cpp run chat-small "hello"
mlx-cpp run mlx-community/Llama-3.2-3B-Instruct-4bit "hello"

# Models
mlx-cpp models list
mlx-cpp models pull mlx-community/SmolLM2-360M-Instruct-4bit

# Training
mlx-cpp train lora --model <id> --data <path>

# Test
pytest tests/
ruff check .
```

## Conventions
- Use `rg` (ripgrep) instead of `grep`
- MLX models from `mlx-community/` namespace on HuggingFace
- OpenAI-compatible API where possible, custom REST for training
- Presets in `models/presets.yaml` map short names → full model IDs
- Daemon on port 12435 (next to Docker Model Runner's 12434)
- Gateway on port 8080 inside Docker

## Dependencies
- MLX >= 0.22, mlx-lm >= 0.21 (core)
- FastAPI + uvicorn + httpx (server)
- click + rich (CLI)
- huggingface-hub (model management)

# currentDate
Today's date is 2026-04-01.
