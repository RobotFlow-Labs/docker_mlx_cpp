# Contributing to docker_mlx_cpp

Thanks for your interest in making Metal GPU accessible from Docker containers.

## Quick Start for Contributors

```bash
# Clone and install
git clone https://github.com/RobotFlow-Labs/docker_mlx_cpp.git
cd docker_mlx_cpp
pip install -e ".[dev]"

# Run linter
ruff check .

# Run tests (requires Apple Silicon Mac with MLX)
pytest tests/

# Start the daemon for manual testing
mlx-cpp serve
```

## What to Work On

- **Issues labeled `good first issue`** — great starting points
- **Issues labeled `help wanted`** — we need community help
- **New GPU operations** — add ops to `daemon/engines/compute.py`
- **New examples** — add language examples to `examples/`
- **Documentation** — improve docs, fix typos, add tutorials

## Architecture

```
daemon/              # Host-side MLX daemon (runs on macOS, Metal GPU)
  mlx_daemon.py      # FastAPI app, port 12435
  engines/           # One engine per capability
    compute.py       # 107 raw GPU operations
    inference.py     # LLM + VLM inference
    training.py      # LoRA/QLoRA fine-tuning
    audio.py         # Whisper STT + Kokoro TTS
    image_gen.py     # FLUX image generation
    embeddings.py    # Text embeddings
  model_manager.py   # HuggingFace model pull/cache

gateway/             # Docker container reverse proxy
  server.py          # FastAPI, routes to daemon

cli/                 # CLI tool (mlx-cpp command)
  mlx_cpp.py         # Click-based CLI

images/              # Docker base images
  mlx-base/          # FROM robotflowlabs/mlx-base
    sdk/             # Container SDK (docker_mlx Python package)
```

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Run `ruff check .` and fix any issues
4. Run `pytest tests/` and ensure tests pass
5. Update docs if you changed behavior
6. Submit a PR with a clear description

## Code Style

- Python 3.11+
- Ruff for linting (config in `pyproject.toml`)
- Line length: 120 characters
- Type hints where practical
- Docstrings on public functions

## Adding a New GPU Operation

1. Add the operation to `daemon/engines/compute.py` in the `eval_operation()` dispatcher
2. Add it to the `list_operations()` category listing
3. Add a test case to `tests/test_compute.py`
4. Run the test: `pytest tests/test_compute.py -k "test_op_name"`

## Adding a New Example

1. Create `examples/your-example/` with a `Dockerfile` and app code
2. Add a service to `docker-compose.yml` under a profile
3. Test it: `docker compose --profile your-example up`
4. Document it in the README

## Reporting Issues

Use GitHub Issues. Include:
- macOS version and chip (M1/M2/M3/M4/M5)
- MLX version (`python -c "import mlx; print(mlx.__version__)"`)
- Docker version (`docker --version`)
- Steps to reproduce
- Full error output
