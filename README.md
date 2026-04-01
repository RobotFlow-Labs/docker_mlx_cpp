# docker_mlx_cpp

**The NVIDIA Container Toolkit — for Mac.**

Give **any Docker container** access to your Apple Silicon GPU. No CUDA. No passthrough hacks. Just Metal.

```
┌─────────────────────────────────────────────────┐
│  Your Container (Linux)                         │
│  ┌───────────────────────────────────────────┐  │
│  │  app → http://llm-gateway:8080/v1/...    │  │
│  └──────────────────┬────────────────────────┘  │
│                     │                            │
│  ┌──────────────────▼────────────────────────┐  │
│  │  mlx-gateway (proxy, logs, rate limits)   │  │
│  └──────────────────┬────────────────────────┘  │
└─────────────────────┼───────────────────────────┘
                      │ model-runner.docker.internal
┌─────────────────────▼───────────────────────────┐
│  macOS Host                                      │
│  ┌───────────────────────────────────────────┐  │
│  │  Docker Model Runner                      │  │
│  │  ├── llama.cpp (GGUF) ──► Metal GPU      │  │
│  │  └── vllm-metal (MLX) ──► Metal GPU      │  │
│  └───────────────────────────────────────────┘  │
│  Apple Silicon M1/M2/M3/M4 — Metal API          │
└──────────────────────────────────────────────────┘
```

## Why

On Linux, `nvidia-docker` gives containers GPU access with `--gpus all`. On Mac, there's nothing equivalent — Metal GPU can't be passed into Docker's Linux VM.

**docker_mlx_cpp** solves this by running GPU inference on the host via Docker Model Runner and exposing it to all containers through a standard API gateway. Your containers speak OpenAI API, your Mac does the Metal compute.

## Quick Start

```bash
# 1. Prerequisites: Docker Desktop 4.62+ with Model Runner enabled
#    Settings → AI → Enable Docker Model Runner

# 2. Setup — pull a model and verify
./scripts/setup.sh

# 3. Start the gateway
docker compose up -d

# 4. Test from any container
docker compose run --rm curl-test \
  curl -s http://llm-gateway:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ai/smollm2:360M-Q4_K_M","messages":[{"role":"user","content":"hello from docker"}]}'

# 5. Run the Python example
docker compose --profile examples up example-app
```

## What You Get

| Feature | NVIDIA Container Toolkit | docker_mlx_cpp |
|---------|------------------------|----------------|
| GPU from containers | `--gpus all` | `http://llm-gateway:8080` |
| Hardware | CUDA GPUs | Apple Silicon Metal |
| API | Custom per framework | OpenAI-compatible (standard) |
| Model format | Framework-specific | GGUF (llama.cpp) / MLX (vllm-metal) |
| Setup | nvidia-docker runtime | Docker Desktop Model Runner |
| Container changes | None (direct GPU) | Set `OPENAI_BASE_URL` env var |

## API Endpoints

The gateway proxies all Docker Model Runner APIs:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Gateway + DMR health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat inference (OpenAI format) |
| `POST /v1/embeddings` | Embeddings |
| `POST /anthropic/v1/messages` | Anthropic-compatible API |

## Use in Your docker-compose.yml

```yaml
services:
  your-app:
    image: your-app
    environment:
      - OPENAI_BASE_URL=http://llm-gateway:8080/v1
      - OPENAI_API_KEY=not-needed
      - OPENAI_MODEL=ai/smollm2:360M-Q4_K_M
    networks:
      - mlx-network

networks:
  mlx-network:
    external: true
```

## Supported Models

Any model Docker Model Runner supports:

```bash
docker model pull ai/smollm2:360M-Q4_K_M       # Small, fast
docker model pull ai/mistral:7B-Q4_K_M          # General purpose
docker model pull ai/llama3.2:3B-Q4_K_M         # Meta Llama
docker model pull ai/gemma3:4B-Q4_K_M           # Google Gemma
```

MLX models route automatically to vllm-metal when installed:
```bash
docker model install-runner --backend vllm
```

## Benchmarking

```bash
./scripts/benchmark.sh                    # Default: 5 runs
RUNS=20 MODEL=ai/mistral:7B-Q4_K_M ./scripts/benchmark.sh
```

## Project Structure

```
docker_mlx_cpp/
├── CLAUDE.md                 # Project instructions for Claude Code
├── docker-compose.yml        # Main stack definition
├── gateway/
│   ├── Dockerfile            # Gateway container
│   ├── server.py             # FastAPI reverse proxy to DMR
│   └── requirements.txt      # Python deps
├── examples/
│   ├── python-client/        # Python OpenAI SDK example
│   │   ├── Dockerfile
│   │   └── app.py
│   └── curl-test.sh          # Quick curl smoke test
├── scripts/
│   ├── setup.sh              # First-time setup + model pull
│   └── benchmark.sh          # Inference latency benchmark
├── models/                   # Model configs and presets
└── tests/                    # Integration tests
```

## License

MIT

---

**Built by [RobotFlow Labs](https://robotflowlabs.com) — Making Mac GPUs work like NVIDIA for Docker.**
