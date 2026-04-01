"""
mlx-cpp CLI — Command-line interface for docker_mlx_cpp.

The NVIDIA Container Toolkit for Mac.

Usage:
    mlx-cpp serve                           Start the MLX Daemon
    mlx-cpp run <model> "prompt"            Quick inference
    mlx-cpp models list                     List cached models
    mlx-cpp models pull <model>             Pull from HuggingFace
    mlx-cpp models remove <model>           Delete cached model
    mlx-cpp health                          Check daemon + GPU status
    mlx-cpp gpu                             Show GPU info
    mlx-cpp benchmark <model>               Run inference benchmark
    mlx-cpp train lora --model ... --data . LoRA fine-tuning
"""

import json
import sys
import time

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

DAEMON_URL = "http://localhost:12435"


def _request(method: str, path: str, data: dict | None = None, timeout: float = 120.0) -> dict:
    """Make HTTP request to MLX Daemon."""
    import httpx

    url = f"{DAEMON_URL}{path}"
    try:
        with httpx.Client(timeout=timeout) as client:
            if method == "GET":
                resp = client.get(url)
            else:
                resp = client.post(url, json=data)
            return resp.json()
    except httpx.ConnectError:
        console.print("[red]Error:[/red] MLX Daemon not running. Start it with: mlx-cpp serve")
        sys.exit(1)


# ── Main CLI Group ───────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="docker-mlx-cpp")
def cli():
    """docker_mlx_cpp — The NVIDIA Container Toolkit for Mac."""
    pass


# ── Serve ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--port", default=12435, help="Port to listen on")
@click.option("--log-level", default="info", help="Log level")
def serve(port: int, log_level: str):
    """Start the MLX Daemon (host-side GPU service)."""
    import os
    os.environ["MLX_DAEMON_PORT"] = str(port)
    os.environ["LOG_LEVEL"] = log_level

    console.print(Panel.fit(
        f"[bold green]MLX Daemon[/bold green] starting on port {port}\n"
        f"Metal GPU → OpenAI API for Docker containers",
        title="docker_mlx_cpp",
    ))

    from daemon.mlx_daemon import run
    run()


# ── Run (quick inference) ────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.argument("prompt")
@click.option("--max-tokens", default=256, help="Max tokens to generate")
@click.option("--temperature", default=0.7, help="Sampling temperature")
def run(model: str, prompt: str, max_tokens: int, temperature: float):
    """Quick inference: mlx-cpp run <model> 'prompt'"""
    console.print(f"[dim]Model: {model}[/dim]")
    console.print(f"[dim]Prompt: {prompt}[/dim]")
    console.print()

    start = time.monotonic()
    result = _request("POST", "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    })
    elapsed = time.monotonic() - start

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    content = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})

    console.print(content)
    console.print()
    console.print(
        f"[dim]{usage.get('prompt_tokens', '?')} prompt + "
        f"{usage.get('completion_tokens', '?')} completion tokens | "
        f"{elapsed:.1f}s[/dim]"
    )


# ── Models ───────────────────────────────────────────────────────────────────

@cli.group()
def models():
    """Manage MLX models."""
    pass


@models.command("list")
def models_list():
    """List cached models."""
    result = _request("GET", "/v1/models")
    data = result.get("data", [])

    if not data:
        console.print("[dim]No models cached. Pull one with: mlx-cpp models pull <model>[/dim]")
        return

    table = Table(title="Cached Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Owner")

    for m in data:
        size = m.get("size_bytes")
        size_str = f"{size // (1024*1024)} MB" if size else "?"
        table.add_row(m["id"], size_str, m.get("owned_by", "?"))

    console.print(table)


@models.command("pull")
@click.argument("model_id")
@click.option("--force", is_flag=True, help="Re-download even if cached")
def models_pull(model_id: str, force: bool):
    """Pull a model from HuggingFace."""
    console.print(f"Pulling [cyan]{model_id}[/cyan]...")
    result = _request("POST", "/models/pull", {"model": model_id, "force": force}, timeout=600.0)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[green]Pulled:[/green] {result.get('model')} → {result.get('path')}")


@models.command("remove")
@click.argument("model_id")
def models_remove(model_id: str):
    """Remove a cached model."""
    result = _request("POST", "/models/delete", {"model": model_id})
    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[green]Deleted:[/green] {model_id}")


# ── Health ───────────────────────────────────────────────────────────────────

@cli.command()
def health():
    """Check MLX Daemon and GPU status."""
    result = _request("GET", "/health")

    gpu = result.get("gpu", {})
    status = result.get("status", "unknown")
    color = "green" if status == "healthy" else "red"

    console.print(Panel.fit(
        f"Status:       [{color}]{status}[/{color}]\n"
        f"Metal GPU:    {'Yes' if gpu.get('metal_available') else 'No'}\n"
        f"MLX Backend:  {gpu.get('mlx_backend', 'N/A')}\n"
        f"Memory:       {gpu.get('memory_total_gb', '?')} GB total\n"
        f"Chip:         {gpu.get('chip', '?')}\n"
        f"Loaded:       {', '.join(result.get('loaded_models', [])) or 'none'}\n"
        f"Cached:       {result.get('cached_models', 0)} models",
        title="MLX Daemon Health",
    ))


# ── GPU Info ─────────────────────────────────────────────────────────────────

@cli.command()
def gpu():
    """Show Apple Silicon GPU information."""
    result = _request("GET", "/health")
    gpu_info = result.get("gpu", {})

    console.print(Panel.fit(
        f"Chip:        {gpu_info.get('chip', '?')}\n"
        f"Platform:    {gpu_info.get('platform', '?')}\n"
        f"Metal:       {'Available' if gpu_info.get('metal_available') else 'Not available'}\n"
        f"MLX Backend: {gpu_info.get('mlx_backend', 'N/A')}\n"
        f"Total RAM:   {gpu_info.get('memory_total_gb', '?')} GB\n"
        f"(Unified memory — shared between CPU and GPU)",
        title="Apple Silicon GPU",
    ))


# ── Benchmark ────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("model")
@click.option("--runs", default=3, help="Number of benchmark runs")
@click.option("--max-tokens", default=128, help="Tokens per run")
def benchmark(model: str, runs: int, max_tokens: int):
    """Benchmark inference speed for a model."""
    console.print(f"Benchmarking [cyan]{model}[/cyan] ({runs} runs, {max_tokens} tokens each)...")
    console.print()

    latencies = []
    token_counts = []

    for i in range(runs):
        start = time.monotonic()
        result = _request("POST", "/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "Write a short paragraph about artificial intelligence."}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
        })
        elapsed = time.monotonic() - start
        latencies.append(elapsed)

        tokens = result.get("usage", {}).get("completion_tokens", 0)
        token_counts.append(tokens)
        tps = tokens / elapsed if elapsed > 0 else 0

        console.print(f"  Run {i+1}: {elapsed:.2f}s | {tokens} tokens | {tps:.1f} tok/s")

    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    avg_tps = avg_tokens / avg_latency if avg_latency > 0 else 0

    console.print()
    console.print(Panel.fit(
        f"Model:         {model}\n"
        f"Avg Latency:   {avg_latency:.2f}s\n"
        f"Avg Tokens:    {avg_tokens:.0f}\n"
        f"Avg Tok/s:     {avg_tps:.1f}\n"
        f"Runs:          {runs}",
        title="Benchmark Results",
    ))


# ── Train ────────────────────────────────────────────────────────────────────

@cli.group()
def train():
    """Fine-tuning commands."""
    pass


@train.command("lora")
@click.option("--model", required=True, help="Model ID to fine-tune")
@click.option("--data", required=True, help="Training data path")
@click.option("--epochs", default=1, help="Number of epochs")
@click.option("--batch-size", default=4, help="Batch size")
@click.option("--lr", default=1e-5, type=float, help="Learning rate")
@click.option("--rank", default=8, help="LoRA rank")
@click.option("--qlora", is_flag=True, help="Use QLoRA (4-bit quantized)")
def train_lora(model: str, data: str, epochs: int, batch_size: int, lr: float, rank: int, qlora: bool):
    """Start LoRA fine-tuning job."""
    console.print(f"Starting {'QLoRA' if qlora else 'LoRA'} training...")
    console.print(f"  Model: {model}")
    console.print(f"  Data:  {data}")
    console.print(f"  Rank:  {rank}, Epochs: {epochs}, LR: {lr}")

    result = _request("POST", "/train/lora", {
        "model": model,
        "dataset": data,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "lora_rank": rank,
        "qlora": qlora,
    }, timeout=600.0)

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
    else:
        console.print(f"[green]Job started:[/green] {result.get('job_id')}")
        console.print(f"  Check status: mlx-cpp train status {result.get('job_id')}")


@train.command("status")
@click.argument("job_id")
def train_status(job_id: str):
    """Check training job status."""
    result = _request("GET", f"/train/jobs/{job_id}")

    if "error" in result:
        console.print(f"[red]Error:[/red] {result['error']}")
        return

    status = result.get("status", "unknown")
    color = {"queued": "yellow", "running": "blue", "completed": "green", "failed": "red"}.get(status, "white")

    console.print(Panel.fit(
        f"Job ID:  {result.get('id')}\n"
        f"Status:  [{color}]{status}[/{color}]\n"
        f"Model:   {result.get('model')}\n"
        f"Output:  {result.get('output_dir', 'N/A')}",
        title=f"Training Job {job_id}",
    ))


@train.command("jobs")
def train_jobs():
    """List all training jobs."""
    result = _request("GET", "/train/jobs")
    jobs = result.get("jobs", [])

    if not jobs:
        console.print("[dim]No training jobs.[/dim]")
        return

    table = Table(title="Training Jobs")
    table.add_column("Job ID", style="cyan")
    table.add_column("Status")
    table.add_column("Model")

    for j in jobs:
        status = j.get("status", "?")
        color = {"queued": "yellow", "running": "blue", "completed": "green", "failed": "red"}.get(status, "white")
        table.add_row(j["id"], f"[{color}]{status}[/{color}]", j.get("model", "?"))

    console.print(table)


# ── Docker Init (scaffold a GPU-enabled project) ────────────────────────────

@cli.group()
def docker():
    """Docker container scaffolding commands."""
    pass


@docker.command("init")
@click.argument("name")
def docker_init(name: str):
    """Scaffold a new GPU-enabled Docker project.

    Creates a ready-to-run project with Dockerfile, docker-compose.yml,
    and a Python app that uses Metal GPU from inside a container.
    """
    from pathlib import Path

    project_dir = Path(name)
    if project_dir.exists():
        console.print(f"[red]Error:[/red] Directory '{name}' already exists")
        return

    project_dir.mkdir(parents=True)

    # Dockerfile
    (project_dir / "Dockerfile").write_text(f"""FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir httpx openai rich

COPY . .

ENV MLX_URL=http://mlx-gateway:8080
ENV OPENAI_BASE_URL=http://mlx-gateway:8080/v1
ENV OPENAI_API_KEY=not-needed

CMD ["python", "app.py"]
""")

    # docker-compose.yml
    (project_dir / "docker-compose.yml").write_text(f"""# {name} — GPU-enabled Docker project (powered by docker_mlx_cpp)
#
# Prerequisites:
#   1. mlx-cpp serve (start Metal GPU daemon)
#   2. docker_mlx_cpp gateway running on mlx-network
#
# Usage:
#   docker compose up

services:
  {name}:
    build: .
    environment:
      - MLX_URL=http://mlx-gateway:8080
      - OPENAI_BASE_URL=http://mlx-gateway:8080/v1
      - OPENAI_API_KEY=not-needed
    networks:
      - mlx-network

networks:
  mlx-network:
    external: true
""")

    # app.py
    (project_dir / "app.py").write_text(f"""\"\"\"
{name} — GPU-accelerated app running inside Docker on Apple Silicon Metal.

This container uses the Metal GPU via docker_mlx_cpp.
\"\"\"

import os
import httpx

MLX_URL = os.environ.get("MLX_URL", "http://mlx-gateway:8080")


def main():
    print(f"[{name}] Connecting to Metal GPU via {{MLX_URL}}")
    print()

    # Check GPU
    gpu = httpx.get(f"{{MLX_URL}}/compute/devices").json()
    print(f"  Metal GPU: {{gpu.get('default_device', 'N/A')}}")
    print(f"  Memory: {{gpu.get('active_memory_gb', '?')}} GB active")
    print()

    # Run a GPU operation
    result = httpx.post(f"{{MLX_URL}}/compute/eval", json={{
        "op": "matmul",
        "args": {{"a": {{"shape": [512, 512]}}, "b": {{"shape": [512, 512]}}}}
    }}).json()
    print(f"  Matmul 512x512: {{result.get('elapsed_ms', '?')}}ms on {{result.get('device', '?')}}")
    print()

    # Run LLM inference
    resp = httpx.post(f"{{MLX_URL}}/v1/chat/completions", json={{
        "model": "chat-small",
        "messages": [{{"role": "user", "content": "Hello from Docker!"}}],
        "max_tokens": 64,
    }}).json()

    if "choices" in resp:
        print(f"  LLM: {{resp['choices'][0]['message']['content']}}")
    else:
        print(f"  LLM: {{resp}}")

    print()
    print(f"[{name}] Metal GPU from Docker container — working!")


if __name__ == "__main__":
    main()
""")

    # README.md
    (project_dir / "README.md").write_text(f"""# {name}

GPU-accelerated Docker app powered by [docker_mlx_cpp](https://github.com/RobotFlow-Labs/docker_mlx_cpp).

## Run

```bash
# 1. Ensure docker_mlx_cpp is running
mlx-cpp serve                    # Terminal 1
cd ~/.docker-mlx/repo && docker compose up -d  # Terminal 2

# 2. Run this app
docker compose up
```

## What this does

This container runs inside Docker but uses your Mac's Metal GPU for:
- LLM inference (50+ model architectures)
- Raw GPU compute (matmul, softmax, FFT, etc.)
- And everything else docker_mlx_cpp supports
""")

    console.print(f"[green]Created project:[/green] {name}/")
    console.print(f"  {name}/Dockerfile")
    console.print(f"  {name}/docker-compose.yml")
    console.print(f"  {name}/app.py")
    console.print(f"  {name}/README.md")
    console.print()
    console.print(f"[bold]Next steps:[/bold]")
    console.print(f"  cd {name}")
    console.print(f"  docker compose up")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
