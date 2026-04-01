# NEXT_STEPS — docker_mlx_cpp

## Last Updated: 2026-04-01

## Vision
The NVIDIA Container Toolkit for Mac. Any Docker container gets Metal GPU access — inference, training, image gen, audio, embeddings. First-mover on Apple Silicon GPU containers.

## Status: Sprint 1 Complete
Full daemon + gateway + CLI + all engines built. Ready for testing.

## Accomplished This Session
- [x] Initialized git repo → github.com/RobotFlow-Labs/docker_mlx_cpp
- [x] Project structure: daemon/, gateway/, cli/, sdk/, models/, tests/
- [x] pyproject.toml with all dependencies and CLI entry point
- [x] MLX Daemon (daemon/mlx_daemon.py) — FastAPI on port 12435
  - Inference engine (mlx-lm, 50+ architectures, streaming)
  - Training engine (LoRA, QLoRA, DPO, async job queue)
  - Image generation engine (Stable Diffusion, FLUX)
  - Audio engine (Whisper STT, TTS)
  - Embeddings engine (mlx-embeddings, Jina v5)
  - Model manager (HuggingFace pull, cache, presets)
- [x] Gateway (gateway/server.py) — dual upstream routing
  - MLX Daemon (primary) + Docker Model Runner (fallback)
  - Rate limiting, metrics, CORS, health aggregation
- [x] CLI tool (cli/mlx_cpp.py) — serve, run, models, health, gpu, benchmark, train
- [x] Model presets (models/presets.yaml) — 14 curated presets
- [x] Docker compose with gateway, examples, training profiles
- [x] launchd plist for macOS auto-start
- [x] Updated setup script with full verification
- [x] README with architecture diagram and full API docs

## TODO — Sprint 2: Test & Validate
- [ ] pip install -e ".[all]" and verify imports
- [ ] mlx-cpp serve → test daemon starts
- [ ] docker compose up -d → test gateway connects
- [ ] End-to-end: container → gateway → daemon → Metal GPU → response
- [ ] Benchmark: compare with llama.cpp baseline
- [ ] Test training job submission and tracking
- [ ] Test model pull and caching

## TODO — Sprint 3: Polish
- [ ] Python SDK (sdk/python/docker_mlx/)
- [ ] Homebrew formula
- [ ] Integration test suite
- [ ] Documentation site
- [ ] Demo GIF recording

## TODO — Sprint 4: Launch
- [ ] PyPI publish
- [ ] Hacker News "Show HN"
- [ ] Blog post
- [ ] ProductHunt

## Blockers
- None

## MVP Readiness: 50%
