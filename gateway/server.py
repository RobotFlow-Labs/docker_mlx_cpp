"""
MLX Gateway — Unified reverse proxy routing containers to Metal GPU compute.

Routes to two upstreams:
    1. MLX Daemon (host:12435) — full MLX stack (inference, training, image gen, audio, embeddings)
    2. Docker Model Runner (model-runner.docker.internal) — DMR fallback for GGUF/llama.cpp

Architecture:
    Container → mlx-gateway:8080 → MLX Daemon (host:12435) → Metal GPU
                                 → DMR (fallback)

API surface:
    /v1/chat/completions      → MLX Daemon (inference)
    /v1/completions           → MLX Daemon (inference)
    /v1/embeddings            → MLX Daemon (embeddings)
    /v1/models                → MLX Daemon (model list)
    /v1/audio/transcriptions  → MLX Daemon (Whisper STT)
    /v1/audio/speech          → MLX Daemon (TTS)
    /v1/images/generations    → MLX Daemon (Stable Diffusion / FLUX)
    /train/*                  → MLX Daemon (LoRA/QLoRA training)
    /models/*                 → MLX Daemon (model management)
    /anthropic/v1/*           → DMR (Anthropic-compatible)
    /engines/v1/*             → DMR (direct DMR passthrough)
    /health                   → Aggregated health
    /metrics                  → Prometheus metrics
"""

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# ── Configuration ────────────────────────────────────────────────────────────

MLX_DAEMON_URL = os.environ.get("MLX_DAEMON_URL", "http://host.docker.internal:12435")
DMR_UPSTREAM = os.environ.get("DMR_UPSTREAM", "http://model-runner.docker.internal")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "120"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlx-gateway")

# ── State ────────────────────────────────────────────────────────────────────

_request_counts: dict[str, list[float]] = {}
_metrics = {
    "requests_total": 0,
    "requests_by_engine": {"mlx": 0, "dmr": 0},
    "errors_total": 0,
    "latency_sum_ms": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.mlx_client = httpx.AsyncClient(base_url=MLX_DAEMON_URL, timeout=300.0)
    app.state.dmr_client = httpx.AsyncClient(base_url=DMR_UPSTREAM, timeout=120.0)
    logger.info("MLX Gateway started")
    logger.info("  MLX Daemon:  %s", MLX_DAEMON_URL)
    logger.info("  DMR:         %s", DMR_UPSTREAM)
    yield
    await app.state.mlx_client.aclose()
    await app.state.dmr_client.aclose()
    logger.info("MLX Gateway stopped")


app = FastAPI(title="docker_mlx_cpp Gateway", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = _request_counts.setdefault(client_ip, [])
    _request_counts[client_ip] = [t for t in window if now - t < 60]
    if len(_request_counts[client_ip]) >= RATE_LIMIT_RPM:
        return False
    _request_counts[client_ip].append(now)
    return True


async def _proxy(client: httpx.AsyncClient, method: str, url: str, body: bytes, content_type: str, engine: str) -> Response:
    """Proxy a request to an upstream, with metrics."""
    start = time.monotonic()
    _metrics["requests_total"] += 1
    _metrics["requests_by_engine"][engine] = _metrics["requests_by_engine"].get(engine, 0) + 1

    try:
        resp = await client.request(
            method=method,
            url=url,
            content=body,
            headers={"content-type": content_type},
        )
        elapsed_ms = (time.monotonic() - start) * 1000
        _metrics["latency_sum_ms"] += elapsed_ms
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
    except httpx.ConnectError:
        _metrics["errors_total"] += 1
        name = "MLX Daemon" if engine == "mlx" else "Docker Model Runner"
        return JSONResponse(status_code=502, content={"error": f"{name} unreachable"})


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Aggregated health: gateway + MLX Daemon + DMR."""
    result = {"status": "healthy", "gateway": "up"}

    # Check MLX Daemon
    try:
        resp = await request.app.state.mlx_client.get("/health")
        result["mlx_daemon"] = resp.json() if resp.status_code == 200 else {"status": "error", "code": resp.status_code}
    except httpx.ConnectError:
        result["mlx_daemon"] = {"status": "unreachable"}

    # Check DMR
    try:
        resp = await request.app.state.dmr_client.get("/engines/v1/models")
        result["dmr"] = {"status": "healthy", "code": resp.status_code}
    except httpx.ConnectError:
        result["dmr"] = {"status": "unreachable"}

    # Overall status
    if result.get("mlx_daemon", {}).get("status") == "unreachable":
        result["status"] = "degraded"

    return result


@app.get("/metrics")
async def metrics():
    """Prometheus-style metrics."""
    return _metrics


# ── MLX Daemon Routes (primary) ─────────────────────────────────────────────

@app.get("/v1/models")
async def list_models(request: Request):
    """List models from MLX Daemon."""
    return await _proxy(request.app.state.mlx_client, "GET", "/v1/models", b"", "application/json", "mlx")


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """Chat completions → MLX Daemon. Supports SSE streaming."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    body = await request.body()

    # Detect streaming request
    import json as _json
    try:
        body_json = _json.loads(body) if body else {}
    except _json.JSONDecodeError:
        body_json = {}
    is_streaming = body_json.get("stream", False)

    start = time.monotonic()

    try:
        if is_streaming:
            # SSE streaming: proxy chunks in real-time (not buffered)
            req = request.app.state.mlx_client.build_request(
                method="POST",
                url="/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )
            resp = await request.app.state.mlx_client.send(req, stream=True)

            async def sse_proxy():
                try:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                finally:
                    await resp.aclose()

            _metrics["requests_total"] += 1
            _metrics["requests_by_engine"]["mlx"] += 1
            logger.info("POST /v1/chat/completions → MLX STREAM [%s]", client_ip)
            return StreamingResponse(sse_proxy(), media_type="text/event-stream")
        else:
            # Non-streaming: standard proxy
            resp = await request.app.state.mlx_client.request(
                method="POST",
                url="/v1/chat/completions",
                content=body,
                headers={"content-type": "application/json"},
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            _metrics["requests_total"] += 1
            _metrics["requests_by_engine"]["mlx"] += 1
            _metrics["latency_sum_ms"] += elapsed_ms
            logger.info("POST /v1/chat/completions → MLX %d (%.0fms) [%s]", resp.status_code, elapsed_ms, client_ip)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")

    except httpx.ConnectError:
        logger.warning("MLX Daemon unreachable, falling back to DMR")
        return await _proxy(request.app.state.dmr_client, "POST", "/engines/v1/chat/completions", body, "application/json", "dmr")


@app.api_route("/v1/embeddings", methods=["POST"])
async def embeddings(request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, "POST", "/v1/embeddings", body, "application/json", "mlx")


@app.api_route("/v1/audio/transcriptions", methods=["POST"])
async def audio_transcriptions(request: Request):
    body = await request.body()
    ct = request.headers.get("content-type", "multipart/form-data")
    return await _proxy(request.app.state.mlx_client, "POST", "/v1/audio/transcriptions", body, ct, "mlx")


@app.api_route("/v1/audio/speech", methods=["POST"])
async def audio_speech(request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, "POST", "/v1/audio/speech", body, "application/json", "mlx")


@app.api_route("/v1/images/generations", methods=["POST"])
async def image_generations(request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, "POST", "/v1/images/generations", body, "application/json", "mlx")


# ── Training Routes → MLX Daemon ────────────────────────────────────────────

@app.api_route("/train/{path:path}", methods=["GET", "POST"])
async def proxy_training(path: str, request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, request.method, f"/train/{path}", body, "application/json", "mlx")


# ── Model Management Routes → MLX Daemon ────────────────────────────────────

@app.api_route("/models/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_models(path: str, request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, request.method, f"/models/{path}", body, "application/json", "mlx")


# ── Compute Routes → MLX Daemon (direct Metal GPU) ──────────────────────────

@app.api_route("/compute/{path:path}", methods=["GET", "POST"])
async def proxy_compute(path: str, request: Request):
    body = await request.body()
    return await _proxy(request.app.state.mlx_client, request.method, f"/compute/{path}", body, "application/json", "mlx")


# ── File Upload Routes → MLX Daemon ─────────────────────────────────────────

@app.api_route("/files/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy_files(path: str, request: Request):
    body = await request.body()
    ct = request.headers.get("content-type", "application/json")
    return await _proxy(request.app.state.mlx_client, request.method, f"/files/{path}", body, ct, "mlx")


# ── GPU + Engine Status → MLX Daemon ────────────────────────────────────────

@app.get("/gpu")
async def gpu_info(request: Request):
    return await _proxy(request.app.state.mlx_client, "GET", "/gpu", b"", "application/json", "mlx")


@app.get("/engines")
async def engine_status(request: Request):
    return await _proxy(request.app.state.mlx_client, "GET", "/engines", b"", "application/json", "mlx")


# ── DMR Direct Routes (fallback / compatibility) ────────────────────────────

@app.api_route("/engines/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_dmr(path: str, request: Request):
    """Direct passthrough to Docker Model Runner."""
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    body = await request.body()
    start = time.monotonic()
    result = await _proxy(request.app.state.dmr_client, request.method, f"/engines/v1/{path}", body, "application/json", "dmr")
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("DMR %s /engines/v1/%s → (%.0fms) [%s]", request.method, path, elapsed_ms, client_ip)
    return result


@app.api_route("/anthropic/v1/{path:path}", methods=["GET", "POST"])
async def proxy_anthropic(path: str, request: Request):
    """Anthropic-compatible → DMR."""
    body = await request.body()
    return await _proxy(request.app.state.dmr_client, request.method, f"/anthropic/v1/{path}", body, "application/json", "dmr")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
