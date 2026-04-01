"""
Example: Python app using Mac GPU inference from inside a Docker container.

This demonstrates the "NVIDIA alternative" pattern for macOS:
    Container → mlx-gateway → MLX Daemon → Apple Silicon Metal GPU

Environment variables (injected by docker-compose.yml):
    OPENAI_BASE_URL  — gateway URL (http://mlx-gateway:8080/v1)
    OPENAI_API_KEY   — not needed, set to "not-needed"
    OPENAI_MODEL     — model to use (MLX community or preset name)
"""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("OPENAI_BASE_URL", "http://mlx-gateway:8080/v1"),
    api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
)

model = os.environ.get("OPENAI_MODEL", "mlx-community/SmolLM2-360M-Instruct-4bit")


def main():
    print(f"[docker_mlx_cpp] Calling model: {model}")
    print(f"[docker_mlx_cpp] Via gateway: {client.base_url}")
    print()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant running on Apple Silicon Metal GPU via Docker Model Runner."},
            {"role": "user", "content": "What hardware are you running on? Explain briefly."},
        ],
        max_tokens=128,
    )

    print(f"Response: {response.choices[0].message.content}")
    print(f"Tokens: {response.usage.prompt_tokens} prompt + {response.usage.completion_tokens} completion")
    print()
    print("[docker_mlx_cpp] GPU inference from container — success!")


if __name__ == "__main__":
    main()
