"""
LLM Gateway — Reverse proxy from containers to Docker Model Runner (host Metal GPU).

This is the core of docker_mlx_cpp: a lightweight gateway that gives any container
access to Apple Silicon GPU inference via Docker Model Runner's OpenAI-compatible API.

Architecture:
    Container → llm-gateway:8080 → model-runner.docker.internal → Host Metal GPU

Provides:
    - Stable internal DNS (llm-gateway) so apps never hardcode Docker Desktop hostnames
    - Request/response logging with token usage tracking
    - Health checks for DMR connectivity
    - Rate limiting per caller
    - OpenAI, Anthropic, and Ollama API passthrough
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

DMR_UPSTREAM = os.environ.get("DMR_UPSTREAM", "http://model-runner.docker.internal")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "info").upper()
RATE_LIMIT_RPM = int(os.environ.get("RATE_LIMIT_RPM", "60"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlx-gateway")

# Rate limiting state
_request_counts: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = httpx.AsyncClient(base_url=DMR_UPSTREAM, timeout=120.0)
    logger.info("MLX Gateway started — upstream: %s", DMR_UPSTREAM)
    yield
    await app.state.http_client.aclose()
    logger.info("MLX Gateway stopped")


app = FastAPI(title="MLX Gateway", lifespan=lifespan)


def _check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window = _request_counts.setdefault(client_ip, [])
    # Prune old entries (60s window)
    _request_counts[client_ip] = [t for t in window if now - t < 60]
    if len(_request_counts[client_ip]) >= RATE_LIMIT_RPM:
        return False
    _request_counts[client_ip].append(now)
    return True


@app.get("/health")
async def health(request: Request):
    """Health check — verifies gateway is up and DMR is reachable."""
    try:
        resp = await request.app.state.http_client.get("/engines/v1/models")
        return {"status": "healthy", "dmr_status": resp.status_code, "upstream": DMR_UPSTREAM}
    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": "Cannot reach Docker Model Runner", "upstream": DMR_UPSTREAM},
        )


@app.get("/v1/models")
@app.get("/engines/v1/models")
async def list_models(request: Request):
    """List available models from DMR."""
    resp = await request.app.state.http_client.get("/engines/v1/models")
    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@app.api_route("/engines/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_openai(path: str, request: Request):
    """Proxy all OpenAI-compatible API calls to DMR."""
    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    body = await request.body()
    start = time.monotonic()

    try:
        resp = await request.app.state.http_client.request(
            method=request.method,
            url=f"/engines/v1/{path}",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={"error": "Docker Model Runner unreachable"})

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("%s /v1/%s → %d (%.0fms) [%s]", request.method, path, resp.status_code, elapsed_ms, client_ip)

    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


@app.api_route("/anthropic/v1/{path:path}", methods=["GET", "POST"])
async def proxy_anthropic(path: str, request: Request):
    """Proxy Anthropic-compatible API calls to DMR."""
    client_ip = request.client.host if request.client else "unknown"

    if not _check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Rate limit exceeded"})

    body = await request.body()
    start = time.monotonic()

    try:
        resp = await request.app.state.http_client.request(
            method=request.method,
            url=f"/anthropic/v1/{path}",
            content=body,
            headers={"content-type": request.headers.get("content-type", "application/json")},
        )
    except httpx.ConnectError:
        return JSONResponse(status_code=502, content={"error": "Docker Model Runner unreachable"})

    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("ANTHROPIC %s /%s → %d (%.0fms) [%s]", request.method, path, resp.status_code, elapsed_ms, client_ip)

    return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT)
