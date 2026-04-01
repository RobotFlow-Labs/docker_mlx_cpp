"""
docker_mlx.client — Python SDK for Metal GPU from Docker containers.

Every function calls the mlx-gateway (HTTP) which routes to the MLX Daemon
running on the host with direct Metal GPU access.

Container → mlx-gateway:8080 → MLX Daemon:12435 → Metal GPU → result
"""

import os
from typing import Any

import httpx

_BASE_URL = os.environ.get("MLX_URL", "http://mlx-gateway:8080")
_TIMEOUT = float(os.environ.get("MLX_TIMEOUT", "120"))


def _url(path: str) -> str:
    return f"{_BASE_URL}{path}"


def _get(path: str) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as c:
        return c.get(_url(path)).json()


def _post(path: str, data: dict | None = None) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as c:
        return c.post(_url(path), json=data or {}).json()


# ── GPU: Direct Metal compute ────────────────────────────────────────────────

class GPU:
    """Direct access to Apple Silicon Metal GPU from any container."""

    def device_info(self) -> dict:
        """Get Metal GPU info: device, memory, status."""
        return _get("/compute/devices")

    def list_ops(self) -> dict:
        """List all 100+ supported GPU operations."""
        return _get("/compute/ops")

    def eval(self, op: str, **kwargs) -> dict:
        """Run any MLX operation on Metal GPU.

        Examples:
            gpu.eval("matmul", a={"shape": [1024, 1024]}, b={"shape": [1024, 1024]})
            gpu.eval("softmax", x={"shape": [512, 512]})
            gpu.eval("conv2d", in_channels=3, out_channels=64, kernel_size=3)
            gpu.eval("scaled_dot_product_attention", batch_size=4, num_heads=8, seq_len=512)
        """
        return _post("/compute/eval", {"op": op, "args": kwargs})

    def matmul(self, a_shape: list = None, b_shape: list = None) -> dict:
        return self.eval("matmul", a={"shape": a_shape or [512, 512]}, b={"shape": b_shape or [512, 512]})

    def softmax(self, shape: list = None, axis: int = -1) -> dict:
        return self.eval("softmax", x={"shape": shape or [1024, 1024]}, axis=axis)

    def attention(self, batch_size: int = 4, num_heads: int = 8, seq_len: int = 512, head_dim: int = 64) -> dict:
        return self.eval("scaled_dot_product_attention", batch_size=batch_size, num_heads=num_heads, seq_len=seq_len, head_dim=head_dim)

    def benchmark(self, size: int = 1024, ops: int = 100) -> dict:
        return self.eval("matmul_throughput", size=size, ops=ops)

    def memory_bandwidth(self, size_mb: int = 256) -> dict:
        return self.eval("memory_bandwidth", size_mb=size_mb)

    def clear_cache(self) -> dict:
        return _post("/compute/clear-cache")


# ── LLM: Language model inference ────────────────────────────────────────────

class LLM:
    """LLM inference on Metal GPU — 50+ architectures."""

    def chat(self, message: str, model: str = "chat-default", max_tokens: int = 512, temperature: float = 0.7, stream: bool = False) -> str | dict:
        """Send a chat message and get a response.

        Returns the response text (or full dict if you need usage stats).
        """
        result = _post("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        })
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return result

    def complete(self, messages: list[dict], model: str = "chat-default", max_tokens: int = 512, **kwargs) -> dict:
        """Full OpenAI-compatible chat completions."""
        return _post("/v1/chat/completions", {
            "model": model, "messages": messages, "max_tokens": max_tokens, **kwargs
        })

    def models(self) -> list:
        """List available models."""
        result = _get("/v1/models")
        return result.get("data", [])


# ── Vision: VLM inference with images ────────────────────────────────────────

class Vision:
    """Vision-language models on Metal GPU."""

    def describe(self, image_url: str, prompt: str = "Describe this image in detail.", model: str = "vision", max_tokens: int = 512) -> str:
        """Send an image + text prompt to a VLM."""
        result = _post("/v1/chat/completions", {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "max_tokens": max_tokens,
        })
        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        return str(result)


# ── Audio: STT + TTS ────────────────────────────────────────────────────────

class Audio:
    """Speech-to-text and text-to-speech on Metal GPU."""

    def transcribe(self, audio_path: str, model: str = "audio-stt") -> str:
        """Transcribe audio file to text."""
        with open(audio_path, "rb") as f:
            with httpx.Client(timeout=_TIMEOUT) as c:
                resp = c.post(
                    _url("/v1/audio/transcriptions"),
                    files={"file": (audio_path, f)},
                    data={"model": model},
                )
                result = resp.json()
        return result.get("text", str(result))

    def speak(self, text: str, voice: str = "af_heart", output_path: str = "speech.wav") -> str:
        """Convert text to speech, save to file."""
        with httpx.Client(timeout=_TIMEOUT) as c:
            resp = c.post(_url("/v1/audio/speech"), json={"input": text, "voice": voice})
            with open(output_path, "wb") as f:
                f.write(resp.content)
        return output_path


# ── Embeddings ───────────────────────────────────────────────────────────────

class Embeddings:
    """Text embeddings on Metal GPU."""

    def embed(self, texts: str | list[str], model: str = "embeddings") -> list:
        """Generate embedding vectors."""
        if isinstance(texts, str):
            texts = [texts]
        result = _post("/v1/embeddings", {"model": model, "input": texts})
        if "data" in result:
            return [d["embedding"] for d in result["data"]]
        return result


# ── Images: Generation ───────────────────────────────────────────────────────

class Images:
    """Image generation on Metal GPU (FLUX)."""

    def generate(self, prompt: str, model: str = "flux-schnell", size: str = "512x512", n: int = 1) -> dict:
        """Generate images from text prompt."""
        return _post("/v1/images/generations", {
            "prompt": prompt, "model": model, "size": size, "n": n,
        })


# ── Training ─────────────────────────────────────────────────────────────────

class Training:
    """LoRA/QLoRA fine-tuning on Metal GPU."""

    def lora(self, model: str, dataset: str, epochs: int = 1, lora_rank: int = 8, **kwargs) -> dict:
        """Start a LoRA fine-tuning job."""
        return _post("/train/lora", {
            "model": model, "dataset": dataset, "epochs": epochs, "lora_rank": lora_rank, **kwargs,
        })

    def status(self, job_id: str) -> dict:
        """Check training job status."""
        return _get(f"/train/jobs/{job_id}")

    def jobs(self) -> list:
        """List all training jobs."""
        return _get("/train/jobs").get("jobs", [])


# ── Files ────────────────────────────────────────────────────────────────────

class Files:
    """Upload files from containers to host for training/processing."""

    def upload(self, file_path: str, purpose: str = "general") -> dict:
        """Upload a file to the host."""
        with open(file_path, "rb") as f:
            with httpx.Client(timeout=_TIMEOUT) as c:
                resp = c.post(
                    _url("/files/upload"),
                    files={"file": (file_path, f)},
                    data={"purpose": purpose},
                )
                return resp.json()

    def list(self) -> list:
        return _get("/files/list").get("files", [])


# ── Module-level instances ───────────────────────────────────────────────────

gpu = GPU()
llm = LLM()
vision = Vision()
audio = Audio()
embeddings = Embeddings()
images = Images()
training = Training()
files = Files()
