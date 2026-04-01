"""
docker_mlx — Container SDK for Apple Silicon Metal GPU.

Use this inside any Docker container to access the host Metal GPU.
The "import torch" equivalent, but for Mac containers.

Quick start:
    from docker_mlx import gpu, llm

    # Check GPU
    info = gpu.device_info()
    print(f"Metal GPU: {info['default_device']}")
    print(f"Memory: {info['active_memory_gb']} GB used")

    # Run inference
    response = llm.chat("Hello from a Docker container!", model="chat-small")
    print(response)

    # Raw GPU compute
    result = gpu.matmul(a_shape=[1024, 1024], b_shape=[1024, 1024])
    print(f"Matmul: {result['elapsed_ms']}ms on {result['device']}")
"""

from docker_mlx.client import gpu, llm, vision, audio, embeddings, images, training, files

__version__ = "0.1.0"
__all__ = ["gpu", "llm", "vision", "audio", "embeddings", "images", "training", "files"]
