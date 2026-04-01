# docker_mlx_cpp — Base Image for Metal GPU Containers
#
# This is the "nvidia/cuda" equivalent for Apple Silicon.
#
# Any container built FROM this image gets:
#   - Python SDK to call Metal GPU from inside Docker
#   - Auto-discovery of the mlx-gateway on mlx-network
#   - OpenAI-compatible client for inference
#   - Direct GPU compute via /compute/eval
#
# Usage in your Dockerfile:
#   FROM robotflowlabs/mlx-base:latest
#   COPY my_app.py .
#   CMD ["python", "my_app.py"]
#
# Then in your code:
#   from docker_mlx import gpu, llm, vision, audio, embeddings, images

FROM python:3.12-slim

LABEL maintainer="RobotFlow Labs <dev@robotflowlabs.com>"
LABEL description="Base image for Metal GPU containers — the nvidia/cuda equivalent for Mac"
LABEL org.opencontainers.image.source="https://github.com/RobotFlow-Labs/docker_mlx_cpp"

# Install SDK
RUN pip install --no-cache-dir httpx openai numpy rich

# Copy the container SDK
COPY sdk/ /usr/lib/python3/docker_mlx/
ENV PYTHONPATH="/usr/lib/python3:${PYTHONPATH}"

# Default gateway URL (override with MLX_URL env var)
ENV MLX_URL="http://mlx-gateway:8080"
ENV OPENAI_BASE_URL="http://mlx-gateway:8080/v1"
ENV OPENAI_API_KEY="not-needed"

# Metadata
ENV DOCKER_MLX_VERSION="0.1.0"
