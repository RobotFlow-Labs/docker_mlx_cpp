"""
MLX Daemon — Host-side GPU compute service for docker_mlx_cpp.

This is the core of docker_mlx_cpp: a FastAPI service running natively on macOS
that exposes Apple Silicon Metal GPU capabilities to Docker containers.

Runs on port 12435 (next to Docker Model Runner's 12434).

Endpoints:
    /v1/chat/completions    — OpenAI-compatible chat (LLM/VLM)
    /v1/completions         — OpenAI-compatible completions
    /v1/embeddings          — Embedding generation
    /v1/models              — List available models
    /v1/audio/transcriptions — Whisper STT
    /v1/audio/speech         — TTS
    /v1/images/generations   — Image generation
    /train/lora             — LoRA fine-tuning jobs
    /train/jobs/{id}        — Training job status
    /models/pull            — Pull model from HuggingFace
    /models/delete          — Delete cached model
    /health                 — Health + GPU status
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from daemon.model_manager import ModelManager

DAEMON_PORT = int(os.environ.get("MLX_DAEMON_PORT", "12435"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("mlx-daemon")

model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MLX Daemon starting on port %d", DAEMON_PORT)
    logger.info("GPU info: %s", model_manager.get_gpu_info())
    logger.info("Cached models: %d", len(model_manager.list_models()))
    yield
    logger.info("MLX Daemon stopped")


app = FastAPI(
    title="MLX Daemon",
    description="Apple Silicon Metal GPU compute for Docker containers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    from daemon.engines.inference import list_loaded_models

    gpu = model_manager.get_gpu_info()
    return {
        "status": "healthy",
        "daemon": "mlx-daemon",
        "version": "0.1.0",
        "gpu": gpu,
        "loaded_models": list_loaded_models(),
        "cached_models": len(model_manager.list_models()),
    }


# ── Models ───────────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models():
    models = model_manager.list_models()
    return {"object": "list", "data": models}


@app.post("/models/pull")
async def pull_model(request: Request):
    body = await request.json()
    model_id = body.get("model")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model field required"})

    model_id = model_manager.resolve_model(model_id)
    path = await model_manager.pull_model(model_id, force=body.get("force", False))
    return {"status": "pulled", "model": model_id, "path": str(path)}


@app.post("/models/delete")
async def delete_model(request: Request):
    body = await request.json()
    model_id = body.get("model")
    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model field required"})

    deleted = model_manager.delete_model(model_id)
    if deleted:
        return {"status": "deleted", "model": model_id}
    return JSONResponse(status_code=404, content={"error": f"Model {model_id} not found"})


# ── Chat Completions (OpenAI-compatible) ─────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    model_id = body.get("model", "")
    messages = body.get("messages", [])
    max_tokens = body.get("max_tokens", 512)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)
    stream = body.get("stream", False)
    stop = body.get("stop")

    if not model_id:
        return JSONResponse(status_code=400, content={"error": "model field required"})
    if not messages:
        return JSONResponse(status_code=400, content={"error": "messages field required"})

    # Resolve preset → model ID, then ensure downloaded
    model_id = model_manager.resolve_model(model_id)
    model_path = model_manager.ensure_model(model_id)

    from daemon.engines.inference import generate_completion

    result = generate_completion(
        model_path=str(model_path),
        model_id=model_id,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        stream=stream,
    )

    if stream:
        return StreamingResponse(result, media_type="text/event-stream")
    return result


# ── Embeddings ───────────────────────────────────────────────────────────────

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    model_id = body.get("model", "")
    input_text = body.get("input", "")

    if not model_id or not input_text:
        return JSONResponse(status_code=400, content={"error": "model and input required"})

    model_id = model_manager.resolve_model(model_id)

    from daemon.engines.embeddings import generate_embeddings

    return generate_embeddings(model_id, input_text)


# ── Audio ────────────────────────────────────────────────────────────────────

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    from daemon.engines.audio import transcribe

    # Handle multipart form data (file upload)
    form = await request.form()
    audio_file = form.get("file")
    model = form.get("model", "mlx-community/whisper-large-v3-turbo-asr-fp16")

    if not audio_file:
        return JSONResponse(status_code=400, content={"error": "file field required"})

    content = await audio_file.read()
    return transcribe(content, model_manager.resolve_model(model))


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    from daemon.engines.audio import text_to_speech

    body = await request.json()
    text = body.get("input", "")
    model = body.get("model", "tts")
    voice = body.get("voice", "alloy")

    if not text:
        return JSONResponse(status_code=400, content={"error": "input field required"})

    audio_bytes = text_to_speech(text, voice)
    return StreamingResponse(iter([audio_bytes]), media_type="audio/mpeg")


# ── Image Generation ─────────────────────────────────────────────────────────

@app.post("/v1/images/generations")
async def image_generations(request: Request):
    from daemon.engines.image_gen import generate_image

    body = await request.json()
    prompt = body.get("prompt", "")
    model = body.get("model", "stable-diffusion")
    size = body.get("size", "512x512")
    n = body.get("n", 1)

    if not prompt:
        return JSONResponse(status_code=400, content={"error": "prompt field required"})

    return generate_image(prompt, model, size, n)


# ── Training ─────────────────────────────────────────────────────────────────

@app.post("/train/lora")
async def train_lora(request: Request):
    from daemon.engines.training import start_lora_training

    body = await request.json()
    return start_lora_training(body)


@app.get("/train/jobs/{job_id}")
async def get_training_job(job_id: str):
    from daemon.engines.training import get_job_status

    status = get_job_status(job_id)
    if status is None:
        return JSONResponse(status_code=404, content={"error": f"Job {job_id} not found"})
    return status


@app.get("/train/jobs")
async def list_training_jobs():
    from daemon.engines.training import list_jobs

    return {"jobs": list_jobs()}


# ── Direct MLX GPU Compute ───────────────────────────────────────────────────

@app.get("/compute/devices")
async def compute_devices():
    """Get MLX Metal GPU device info and memory stats."""
    from daemon.engines.compute import get_device_info
    return get_device_info()


@app.post("/compute/eval")
async def compute_eval(request: Request):
    """Execute an MLX GPU operation (matmul, softmax, sort, etc.)."""
    from daemon.engines.compute import eval_operation

    body = await request.json()
    op = body.get("op", "")
    args = body.get("args", {})

    if not op:
        return JSONResponse(status_code=400, content={"error": "op field required"})
    return eval_operation(op, args)


@app.post("/compute/benchmark")
async def compute_benchmark(request: Request):
    """Benchmark raw Metal GPU compute throughput."""
    from daemon.engines.compute import eval_operation

    body = await request.json()
    size = body.get("size", 1024)
    ops = body.get("ops", 100)
    return eval_operation("benchmark", {"size": size, "ops": ops})


@app.post("/compute/clear-cache")
async def compute_clear_cache():
    """Clear MLX Metal memory cache."""
    from daemon.engines.compute import clear_cache
    return clear_cache()


# ── File Upload (for training data, images, audio) ──────────────────────────

@app.post("/files/upload")
async def upload_file(request: Request):
    """Upload a file from a container to the host for training/processing."""
    import uuid as _uuid
    from pathlib import Path

    uploads_dir = Path.home() / ".docker-mlx" / "uploads"

    form = await request.form()
    file = form.get("file")
    purpose = form.get("purpose", "general")

    if not file:
        return JSONResponse(status_code=400, content={"error": "file field required"})

    file_id = f"file-{_uuid.uuid4().hex[:12]}"
    dest_dir = uploads_dir / purpose / file_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file.filename

    content = await file.read()
    dest.write_bytes(content)

    return {"id": file_id, "filename": file.filename, "path": str(dest), "bytes": len(content), "purpose": purpose}


@app.get("/files/list")
async def list_files():
    """List uploaded files."""
    from pathlib import Path

    uploads_dir = Path.home() / ".docker-mlx" / "uploads"
    files = []
    if uploads_dir.exists():
        for purpose_dir in uploads_dir.iterdir():
            if purpose_dir.is_dir():
                for file_dir in purpose_dir.iterdir():
                    if file_dir.is_dir():
                        for f in file_dir.iterdir():
                            files.append({
                                "id": file_dir.name,
                                "filename": f.name,
                                "purpose": purpose_dir.name,
                                "path": str(f),
                                "bytes": f.stat().st_size,
                            })
    return {"files": files}


# ── GPU Info ─────────────────────────────────────────────────────────────────

@app.get("/gpu")
async def gpu_info():
    """Detailed GPU info — memory, device, loaded models."""
    from daemon.engines.inference import list_loaded_models
    from daemon.engines.compute import get_device_info

    gpu = get_device_info()
    gpu["loaded_models"] = list_loaded_models()
    gpu["cached_models"] = len(model_manager.list_models())
    return gpu


# ── Engine Availability ─────────────────────────────────────────────────────

@app.get("/engines")
async def engine_status():
    """Check which MLX engines are available (installed)."""
    engines = {}
    for name, module in [
        ("inference", "mlx_lm"),
        ("vlm", "mlx_vlm"),
        ("audio", "mlx_audio"),
        ("embeddings", "mlx_embeddings"),
        ("image_gen", "mflux"),
        ("compute", "mlx.core"),
    ]:
        try:
            __import__(module)
            engines[name] = {"status": "available", "module": module}
        except ImportError:
            engines[name] = {"status": "not_installed", "module": module}
    return {"engines": engines}


# ── Entry point ──────────────────────────────────────────────────────────────

def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=DAEMON_PORT)


if __name__ == "__main__":
    run()
