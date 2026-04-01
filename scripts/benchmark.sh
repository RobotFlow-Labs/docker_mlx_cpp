#!/usr/bin/env bash
# docker_mlx_cpp — Benchmark: compare llama.cpp vs vllm-metal backends
set -euo pipefail

GATEWAY_URL="${GATEWAY_URL:-http://localhost:8080}"
MODEL="${MODEL:-ai/smollm2:360M-Q4_K_M}"
PROMPT="Write a short paragraph about artificial intelligence."
RUNS="${RUNS:-5}"

echo "============================================"
echo "  docker_mlx_cpp — Inference Benchmark"
echo "============================================"
echo "  Gateway: $GATEWAY_URL"
echo "  Model:   $MODEL"
echo "  Runs:    $RUNS"
echo

total_ms=0
for i in $(seq 1 "$RUNS"); do
  start=$(python3 -c "import time; print(int(time.time()*1000))")

  curl -sf "$GATEWAY_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"$MODEL\",
      \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
      \"max_tokens\": 128
    }" > /dev/null

  end=$(python3 -c "import time; print(int(time.time()*1000))")
  elapsed=$((end - start))
  total_ms=$((total_ms + elapsed))
  echo "  Run $i: ${elapsed}ms"
done

avg=$((total_ms / RUNS))
echo
echo "  Average: ${avg}ms over $RUNS runs"
echo "============================================"
