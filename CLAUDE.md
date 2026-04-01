# Docker MLX — Mac GPU Model Runner Platform

## What This Is
A platform that lets **any Docker container** use Apple Silicon GPU inference via Docker Model Runner (DMR). Metal GPU runs on the host; containers call it over HTTP via OpenAI-compatible API.

## Architecture
- **Host layer:** Docker Desktop + DMR — inference engines run as host processes (Metal)
- **Container layer:** Any container calls `http://model-runner.docker.internal/engines/v1/*`
- **Backends:** llama.cpp (GGUF, default) and vllm-metal (MLX Safetensors)
- **API formats:** OpenAI (`/engines/v1/`), Anthropic (`/anthropic/v1/`), Ollama-compatible

## Key Files
```
docker_mlx.md          # Full architecture doc and mega plan
CLAUDE.md              # This file — project instructions
NEXT_STEPS.md          # Session tracking (read first every session)
docker-compose.yml     # (to be created) Compose stack with model definitions
```

## Prerequisites
- Docker Desktop 4.62+ with Model Runner enabled (Settings > AI tab)
- Apple Silicon Mac
- Optional: vllm-metal backend (`docker model install-runner --backend vllm`)
- Optional: host-side TCP (`localhost:12434`)

## Dev Commands
```bash
# Model management
docker model pull <model>                    # Pull a model
docker model run <model> "prompt"            # Quick inference test
docker model ls                              # List local models
docker model version                         # Verify DMR is working
docker model package --gguf <file> --push    # Package model as OCI artifact

# Container testing
docker compose up -d                         # Start the stack
docker compose logs -f                       # Watch logs
docker compose down                          # Stop everything

# Verify GPU inference from container
docker run --rm curlimages/curl:8.5.0 \
  -s http://model-runner.docker.internal/engines/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"ai/smollm2:360M-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'
```

## Conventions
- Use `rg` (ripgrep) instead of `grep`
- Models referenced by OCI artifact names: `org/model:params-quant`
- Environment injection via Compose `models:` block (preferred over hardcoded URLs)
- API key not required by DMR — use `"not-needed"` placeholder for SDK compatibility

## API Endpoints
| From | Base URL |
|------|----------|
| Container | `http://model-runner.docker.internal/engines/v1` |
| Host | `http://localhost:12434/engines/v1` |

## Important Constraints
- No Metal GPU passthrough into containers — inference MUST run host-side
- `model-runner.docker.internal` not available from ECI containers
- Context size increases memory: ~100-500MB per 1000 extra tokens
- llama.cpp default context: 4096 tokens

# currentDate
Today's date is 2026-04-01.
