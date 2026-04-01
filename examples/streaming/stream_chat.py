"""
Streaming example: real-time token-by-token output from Metal GPU inside Docker.
"""

import os
import httpx

MLX_URL = os.environ.get("MLX_URL", "http://mlx-gateway:8080")

print(f"[streaming] SSE streaming from Metal GPU via {MLX_URL}")
print()

with httpx.Client(timeout=120.0) as client:
    with client.stream(
        "POST",
        f"{MLX_URL}/v1/chat/completions",
        json={
            "model": "chat-small",
            "messages": [{"role": "user", "content": "Write a haiku about Metal GPU computing."}],
            "max_tokens": 128,
            "stream": True,
        },
    ) as response:
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                import json
                chunk = json.loads(line[6:])
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)

                # Show metrics on final chunk
                if chunk.get("mlx_metrics"):
                    m = chunk["mlx_metrics"]
                    print(f"\n\n[streaming] {m.get('generation_tps', '?')} tok/s | peak memory: {m.get('peak_memory_gb', '?')} GB")

print("\n[streaming] Done!")
