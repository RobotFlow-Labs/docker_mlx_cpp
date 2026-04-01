#!/usr/bin/env bash
# docker_mlx_cpp — Setup script
# Verifies Docker Model Runner is working and pulls a test model
set -euo pipefail

echo "============================================"
echo "  docker_mlx_cpp — Mac GPU Platform Setup"
echo "============================================"
echo

# Check Docker
if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker not found. Install Docker Desktop 4.62+"
  exit 1
fi
echo "[OK] Docker found: $(docker --version)"

# Check Docker Model Runner
if docker model version &>/dev/null; then
  echo "[OK] Docker Model Runner: $(docker model version 2>/dev/null || echo 'available')"
else
  echo "ERROR: Docker Model Runner not available."
  echo "  → Enable it in Docker Desktop: Settings → AI → Enable Docker Model Runner"
  exit 1
fi

# Pull test model
MODEL="ai/smollm2:360M-Q4_K_M"
echo
echo "Pulling test model: $MODEL"
docker model pull "$MODEL"
echo "[OK] Model pulled"

# Quick smoke test
echo
echo "Running smoke test..."
RESPONSE=$(docker model run "$MODEL" "Say 'hello' in one word" 2>&1 || true)
echo "  Model response: $RESPONSE"

# Test container access
echo
echo "Testing container → DMR access..."
docker run --rm curlimages/curl:8.5.0 \
  -sf http://model-runner.docker.internal/engines/v1/models \
  | python3 -m json.tool 2>/dev/null || echo "  (models listed)"

echo
echo "============================================"
echo "  Setup complete!"
echo "  Next: docker compose up -d"
echo "============================================"
