#!/usr/bin/env bash
# docker_mlx_cpp — Setup script
# Installs dependencies, verifies MLX + Docker, and starts the daemon
set -euo pipefail

echo "============================================"
echo "  docker_mlx_cpp — Mac GPU Platform Setup"
echo "  The NVIDIA Container Toolkit for Mac"
echo "============================================"
echo

# ── Check platform ──────────────────────────────────────────────────────────
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
  echo "WARNING: docker_mlx_cpp is designed for Apple Silicon (arm64)."
  echo "  Detected: $ARCH"
  echo "  MLX requires Apple Silicon for Metal GPU acceleration."
  echo
fi

# ── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  echo "ERROR: Python 3 not found."
  echo "  Install: brew install python@3.12"
  exit 1
fi
PYVER=$(python3 --version)
echo "[OK] Python: $PYVER"

# ── Check Docker ────────────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker not found. Install Docker Desktop 4.62+"
  exit 1
fi
echo "[OK] Docker: $(docker --version)"

# ── Install Python dependencies ─────────────────────────────────────────────
echo
echo "Installing docker_mlx_cpp dependencies..."

if command -v uv &>/dev/null; then
  echo "  Using uv (fast)..."
  uv pip install -e ".[all,dev]" 2>/dev/null || uv pip install -e ".[dev]"
elif command -v pip &>/dev/null; then
  echo "  Using pip..."
  pip install -e ".[all,dev]" 2>/dev/null || pip install -e ".[dev]"
else
  echo "ERROR: Neither uv nor pip found."
  exit 1
fi
echo "[OK] Dependencies installed"

# ── Verify MLX ──────────────────────────────────────────────────────────────
echo
echo "Checking MLX installation..."
if python3 -c "import mlx.core as mx; print(f'  MLX device: {mx.default_device()}')" 2>/dev/null; then
  echo "[OK] MLX available with Metal GPU"
else
  echo "WARNING: MLX not available or no Metal GPU."
  echo "  Install: pip install mlx"
  echo "  Note: MLX requires Apple Silicon."
fi

# ── Check Docker Model Runner (optional) ────────────────────────────────────
echo
echo "Checking Docker Model Runner (optional)..."
if docker model version &>/dev/null; then
  echo "[OK] Docker Model Runner: available"
else
  echo "[--] Docker Model Runner: not enabled (optional)"
  echo "  Enable in Docker Desktop: Settings → AI → Enable Docker Model Runner"
fi

# ── Verify CLI ──────────────────────────────────────────────────────────────
echo
echo "Checking mlx-cpp CLI..."
if command -v mlx-cpp &>/dev/null; then
  echo "[OK] mlx-cpp CLI installed"
else
  echo "[--] mlx-cpp not in PATH (run from project root with: python -m cli.mlx_cpp)"
fi

# ── Build gateway image ─────────────────────────────────────────────────────
echo
echo "Building gateway Docker image..."
docker compose build mlx-gateway 2>/dev/null && echo "[OK] Gateway image built" || echo "[--] Gateway build skipped"

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Quick start:"
echo "    1. mlx-cpp serve              # Start MLX Daemon"
echo "    2. docker compose up -d       # Start gateway"
echo "    3. mlx-cpp health             # Verify everything"
echo ""
echo "  First model:"
echo "    mlx-cpp models pull mlx-community/SmolLM2-360M-Instruct-4bit"
echo "    mlx-cpp run chat-small 'Hello from Metal GPU!'"
echo "============================================"
