# Changelog

## [0.1.0] - 2026-04-01

### Added
- MLX Daemon: host-side FastAPI service exposing Metal GPU to containers (port 12435)
- 107 GPU compute operations across 15 categories (arithmetic, linear algebra, reductions, transforms, activations, convolutions, pooling, attention, normalization, random, FFT, sorting, comparison, Metal memory, benchmarks)
- LLM inference engine: 50+ model architectures via mlx-lm with streaming support
- VLM inference: vision-language models via mlx-vlm with image input support
- Training engine: LoRA/QLoRA fine-tuning via mlx-lm tuner Python API
- Audio engine: Whisper STT + Kokoro TTS via mlx-audio
- Image generation: FLUX via mflux
- Embedding generation via mlx-embeddings
- Model manager: pull from HuggingFace, cache, presets, auto-download
- Gateway: Docker container reverse proxy with dual upstream (MLX Daemon + DMR fallback)
- CLI tool (mlx-cpp): serve, run, models, health, gpu, benchmark, train
- 14 curated model presets (chat, code, vision, image-gen, audio, embeddings)
- Container SDK (docker_mlx): Python package for GPU access from containers
- Base Docker image (robotflowlabs/mlx-base) for building GPU containers
- One-line installer: `curl -fsSL https://raw.githubusercontent.com/RobotFlow-Labs/docker_mlx_cpp/main/install.sh | bash`
- GPU test suite: validates all operations from inside Docker container
- Developer guide: docs/SHIPPING_GPU_CONTAINERS.md
- File upload endpoint for training data from containers
- Real-time GPU memory monitoring via MLX API

### Tested
- 107/108 GPU operations passing on Apple M5 (24GB)
- ~95 TFLOPS matmul throughput on M5
- Flash Attention: 1.6ms (batch=2, heads=4, seq=128)
