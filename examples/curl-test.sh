#!/usr/bin/env bash
# Quick test: call Mac GPU inference from any container via the gateway
#
# Usage:
#   # From host (if TCP enabled):
#   ./examples/curl-test.sh
#
#   # From inside a container on mlx-network:
#   curl -s http://llm-gateway:8080/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"ai/smollm2:360M-Q4_K_M","messages":[{"role":"user","content":"hello"}]}'

set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
MODEL="${MODEL:-ai/smollm2:360M-Q4_K_M}"

echo "[docker_mlx_cpp] Testing GPU inference via gateway..."
echo "  Gateway: $GATEWAY_URL"
echo "  Model:   $MODEL"
echo

# Health check
echo "==> Health check"
curl -sf "$GATEWAY_URL/health" | python3 -m json.tool
echo

# List models
echo "==> Available models"
curl -sf "$GATEWAY_URL/v1/models" | python3 -m json.tool
echo

# Chat completion
echo "==> Chat completion"
curl -sf "$GATEWAY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello from Apple Silicon Metal GPU in one sentence.\"}],
    \"max_tokens\": 64
  }" | python3 -m json.tool

echo
echo "[docker_mlx_cpp] Done — Mac GPU inference working from container!"
