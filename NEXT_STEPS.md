# NEXT_STEPS — docker_mlx_cpp

## Last Updated: 2026-04-01

## Vision
The go-to tool for all Mac M-series users who need GPU acceleration from Docker containers. The "NVIDIA Container Toolkit" equivalent for Apple Silicon — not just for LLMs, but for any GPU workload: simulators, ML training, rendering, anything that currently falls back to CPU in Docker.

## Status
MVP foundation built. Gateway + compose stack + examples ready for testing.

## Accomplished This Session
- Initialized git repo, added remote (github.com/RobotFlow-Labs/docker_mlx_cpp)
- Created CLAUDE.md with project instructions
- Set up .claude/ config (settings.json with push-to-main enabled, rules)
- Built LLM Gateway (FastAPI reverse proxy to Docker Model Runner)
  - OpenAI API passthrough (/v1/*)
  - Anthropic API passthrough (/anthropic/v1/*)
  - Health checks, rate limiting, request logging
- Created docker-compose.yml with gateway + example services
- Created Python client example (OpenAI SDK → gateway → Metal GPU)
- Created curl test scripts and benchmark script
- Created setup.sh for first-time setup
- Created README.md with architecture diagram and comparison table

## TODO — Phase 1: Validate MVP
- [ ] Run setup.sh to verify DMR is working
- [ ] `docker compose up -d` and test gateway health
- [ ] Run curl-test.sh end-to-end
- [ ] Run python-client example
- [ ] Run benchmark.sh baseline numbers

## TODO — Phase 2: Beyond LLMs (GPU for everything)
- [ ] Add Metal compute proxy for non-LLM GPU workloads (image generation, simulation)
- [ ] Create `mlx-compute` container for general Metal shader dispatch
- [ ] Explore virtio-gpu bridge for direct Metal access in containers
- [ ] Add support for MLX training workloads (not just inference)
- [ ] Create Homebrew formula for easy install

## TODO — Phase 3: Go-to-Market
- [ ] Create landing page (robotflowlabs.com/docker-mlx)
- [ ] Write blog post: "NVIDIA containers, but for Mac"
- [ ] Create demo video showing container → Metal GPU inference
- [ ] Submit to Hacker News, Reddit r/MachineLearning, r/docker
- [ ] Docker Hub marketplace listing

## Blockers
- None currently

## MVP Readiness: 25%
