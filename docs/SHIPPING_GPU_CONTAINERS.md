# Shipping GPU Containers for Apple Silicon

**A guide for the MLX community: how to build Docker containers that actually use the Mac GPU.**

## The Problem

When you build a Docker container on Mac, it runs inside a Linux VM. The Metal GPU is on the other side of the VM boundary. Your container sees **zero GPUs** — every ML workload falls back to CPU.

```
Standard Docker on Mac:

  Your Container (Linux VM)     macOS Host
  ┌─────────────────────┐      ┌──────────────┐
  │  import mlx          │      │  Metal GPU   │
  │  → CPU ONLY ❌       │  ✗   │  (unused)    │
  │  No Metal access     │      │              │
  └─────────────────────┘      └──────────────┘
```

On Linux, NVIDIA solved this with `nvidia-docker` — the `--gpus all` flag. On Mac, **there is no equivalent**. Until now.

## The Solution: docker_mlx_cpp

docker_mlx_cpp runs a daemon on the host with direct Metal GPU access. Your container calls it over HTTP. The GPU work happens on the host, results come back to your container.

```
docker_mlx_cpp:

  Your Container (Linux VM)     macOS Host
  ┌─────────────────────┐      ┌──────────────┐
  │  from docker_mlx     │      │  MLX Daemon  │
  │    import gpu, llm   │ ───► │  ↓           │
  │  gpu.matmul(...)     │      │  Metal GPU ✅│
  │  llm.chat("hello")  │ ◄─── │  ↑           │
  │  → GPU RESULTS ✅    │      │  Results     │
  └─────────────────────┘      └──────────────┘
```

## Quick Start: Build a GPU Container in 5 Minutes

### 1. Install docker_mlx_cpp on your Mac

```bash
pip install docker-mlx-cpp
mlx-cpp serve  # starts the GPU daemon
```

### 2. Start the gateway

```bash
docker compose up -d mlx-gateway
```

### 3. Use the base image in your Dockerfile

```dockerfile
FROM robotflowlabs/mlx-base:latest

COPY my_app.py .
CMD ["python", "my_app.py"]
```

### 4. Write your GPU code

```python
# my_app.py — runs INSIDE a Docker container, uses Mac GPU
from docker_mlx import gpu, llm, embeddings

# Check GPU access
info = gpu.device_info()
print(f"Metal GPU: {info['default_device']}")
print(f"Memory: {info['active_memory_gb']} GB")

# Run inference on Metal GPU (not CPU!)
response = llm.chat("Explain quantum computing", model="chat-default")
print(response)

# Run raw GPU compute
result = gpu.matmul(a_shape=[2048, 2048], b_shape=[2048, 2048])
print(f"2048x2048 matmul: {result['elapsed_ms']}ms")

# Generate embeddings
vectors = embeddings.embed(["hello world", "machine learning"])
print(f"Embedding dim: {len(vectors[0])}")
```

### 5. Run it

```bash
docker compose up my-app
# → All computation happens on Metal GPU, not CPU
```

## What Your Container Gets

| Capability | How to Use | GPU Backend |
|-----------|-----------|-------------|
| **LLM inference** | `llm.chat("prompt")` | mlx-lm (Metal) |
| **Vision + images** | `vision.describe(image_url, "describe")` | mlx-vlm (Metal) |
| **Embeddings** | `embeddings.embed(["text"])` | mlx-embeddings (Metal) |
| **Speech-to-text** | `audio.transcribe("file.wav")` | mlx-audio Whisper (Metal) |
| **Text-to-speech** | `audio.speak("hello", output="out.wav")` | mlx-audio Kokoro (Metal) |
| **Image generation** | `images.generate("a cat in space")` | mflux FLUX (Metal) |
| **LoRA training** | `training.lora(model="...", dataset="...")` | mlx-lm tuner (Metal) |
| **Raw GPU compute** | `gpu.eval("matmul", ...)` | mlx.core (Metal) |
| **100+ GPU ops** | `gpu.eval("softmax")`, `gpu.eval("conv2d")`, etc. | mlx.core + mlx.nn |

## GPU Compute Operations (100+)

Your container can run ANY of these on the Mac Metal GPU:

### Arithmetic
```python
gpu.eval("matmul", a={"shape": [1024, 1024]}, b={"shape": [1024, 1024]})
gpu.eval("add", a={"shape": [4096, 4096]}, b={"shape": [4096, 4096]})
gpu.eval("exp", x={"shape": [1024, 1024]})
gpu.eval("sqrt", x={"shape": [1024, 1024], "fill": 4.0})
```

### Linear Algebra
```python
gpu.eval("svd", x={"shape": [512, 512]})
gpu.eval("qr", x={"shape": [256, 256]})
gpu.eval("cholesky", size=256)
gpu.eval("inv", size=256)
gpu.eval("norm", x={"shape": [1024, 1024]})
```

### Neural Network Layers
```python
gpu.eval("conv2d", in_channels=3, out_channels=64, kernel_size=3, batch_size=8, height=224, width=224)
gpu.eval("scaled_dot_product_attention", batch_size=4, num_heads=8, seq_len=2048, head_dim=64)
gpu.eval("layer_norm", dims=768, x={"shape": [32, 512, 768]})
gpu.eval("rms_norm", dims=768, x={"shape": [32, 512, 768]})
```

### Activations
```python
gpu.eval("relu", x={"shape": [1024, 1024]})
gpu.eval("gelu", x={"shape": [1024, 1024]})
gpu.eval("silu", x={"shape": [1024, 1024]})
gpu.eval("softmax", x={"shape": [1024, 1024]}, axis=-1)
```

### FFT
```python
gpu.eval("fft", x={"shape": [65536]})
gpu.eval("fft2", x={"shape": [1024, 1024]})
```

### Benchmarking
```python
result = gpu.benchmark(size=2048, ops=100)
print(f"{result['gflops']} GFLOPS / {result['tflops']} TFLOPS")

bw = gpu.memory_bandwidth(size_mb=512)
print(f"{bw['gb_per_sec']} GB/s memory bandwidth")
```

## For Framework Authors: Integrating with docker_mlx_cpp

If you maintain an MLX-based library and want containers to use your framework on Metal GPU, here's how:

### Option 1: Add an HTTP backend to your library

```python
# In your library, add a "remote" backend option
class MyModel:
    def __init__(self, backend="local"):
        if backend == "docker-mlx":
            from docker_mlx import gpu
            self._compute = lambda op, **kw: gpu.eval(op, **kw)
        else:
            import mlx.core as mx
            self._compute = lambda op, **kw: getattr(mx, op)(**kw)
```

### Option 2: Just use the OpenAI SDK

If your library serves an OpenAI-compatible API, containers already work:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://mlx-gateway:8080/v1",
    api_key="not-needed",
)

# This hits Metal GPU through docker_mlx_cpp
response = client.chat.completions.create(
    model="mlx-community/Llama-3.2-3B-Instruct-4bit",
    messages=[{"role": "user", "content": "hello"}],
)
```

### Option 3: Use the compute endpoint for custom operations

```python
import httpx

def my_custom_gpu_op(data):
    resp = httpx.post("http://mlx-gateway:8080/compute/eval", json={
        "op": "matmul",
        "args": {"a": {"data": data}, "b": {"shape": [len(data), 256]}}
    })
    return resp.json()
```

## Architecture for Container Authors

```
┌─────────────────────────────────────────────────────┐
│  YOUR CONTAINER (any base image)                     │
│  ┌───────────────────────────────────────────────┐  │
│  │  from docker_mlx import gpu, llm              │  │
│  │  OR: curl http://mlx-gateway:8080/compute/... │  │
│  │  OR: OpenAI SDK with base_url=mlx-gateway     │  │
│  └──────────────────────┬────────────────────────┘  │
└─────────────────────────┼───────────────────────────┘
                          │ HTTP (mlx-network)
┌─────────────────────────▼───────────────────────────┐
│  mlx-gateway:8080 (Docker container)                 │
│  Routes all requests to MLX Daemon on host           │
└─────────────────────────┬───────────────────────────┘
                          │ host.docker.internal:12435
┌─────────────────────────▼───────────────────────────┐
│  MLX Daemon (native macOS process)                   │
│  ├── mlx-lm (LLM, 50+ architectures)               │
│  ├── mlx-vlm (Vision-Language)                      │
│  ├── mlx-audio (Whisper STT, Kokoro TTS)            │
│  ├── mlx-embeddings (text embeddings)               │
│  ├── mflux (FLUX image generation)                  │
│  ├── mlx.core (100+ raw GPU operations)             │
│  └── mlx.nn (conv, attention, norm, pool)           │
└─────────────────────────┬───────────────────────────┘
                          │ Metal API
┌─────────────────────────▼───────────────────────────┐
│  Apple Silicon M1/M2/M3/M4/M5                       │
│  Metal GPU + Unified Memory + Neural Engine          │
└─────────────────────────────────────────────────────┘
```

## docker-compose.yml Template

Copy this into your project to ship a GPU-enabled container:

```yaml
services:
  my-app:
    build: .
    environment:
      - MLX_URL=http://mlx-gateway:8080
      - OPENAI_BASE_URL=http://mlx-gateway:8080/v1
      - OPENAI_API_KEY=not-needed
    depends_on:
      mlx-gateway:
        condition: service_healthy
    networks:
      - mlx-network

networks:
  mlx-network:
    external: true
```

**Prerequisites for users of your container:**
1. macOS on Apple Silicon
2. `pip install docker-mlx-cpp && mlx-cpp serve`
3. `docker compose up -d mlx-gateway` (from docker_mlx_cpp repo)

## FAQ

**Q: Can I use `import mlx` directly inside a container?**
No. MLX requires Metal, which requires macOS. Docker containers run Linux. Use `from docker_mlx import gpu` instead — same operations, runs on the host GPU.

**Q: What's the performance overhead?**
HTTP round-trip adds ~1-5ms per call. For inference (100ms+), this is negligible. For tight loops of small ops, batch them.

**Q: Can multiple containers share the GPU?**
Yes. The daemon serializes requests. Multiple containers can call the gateway concurrently.

**Q: Does this work with Kubernetes?**
Yes, if the MLX Daemon runs on a Mac node. Containers on that node can reach the gateway.

**Q: What models are supported?**
Anything MLX supports: 50+ LLM architectures (Llama, Mistral, Qwen, Gemma, Phi, etc.), VLMs, Whisper, Kokoro TTS, FLUX image gen, and all mlx-community models on HuggingFace.
