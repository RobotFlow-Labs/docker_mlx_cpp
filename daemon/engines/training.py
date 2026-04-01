"""
Training Engine — LoRA/QLoRA fine-tuning via MLX.

Uses the real mlx_lm.tuner Python API:
    from mlx_lm.tuner import train, TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers

Falls back to subprocess CLI (python -m mlx_lm.lora --train) if Python API unavailable.

Jobs run asynchronously in background threads.
"""

import logging
import subprocess
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger("mlx-daemon.training")

# Job registry
_jobs: dict[str, dict] = {}
_job_lock = threading.Lock()

ADAPTERS_DIR = Path.home() / ".docker-mlx" / "adapters"
ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)


def start_lora_training(config: dict) -> dict:
    """Start a LoRA fine-tuning job."""
    job_id = f"train-{uuid.uuid4().hex[:8]}"

    model = config.get("model")
    dataset = config.get("dataset")
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 4)
    learning_rate = config.get("learning_rate", 1e-5)
    lora_rank = config.get("lora_rank", 8)
    lora_layers = config.get("lora_layers", 16)
    iters = config.get("iters", epochs * 100)
    use_qlora = config.get("qlora", False)

    if not model:
        return {"error": "model field required"}
    if not dataset:
        return {"error": "dataset field required"}

    output_dir = ADAPTERS_DIR / job_id

    job = {
        "id": job_id,
        "status": "queued",
        "model": model,
        "dataset": dataset,
        "config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "lora_rank": lora_rank,
            "lora_layers": lora_layers,
            "iters": iters,
            "qlora": use_qlora,
        },
        "output_dir": str(output_dir),
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    with _job_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_training, args=(job_id,), daemon=True)
    thread.start()

    logger.info("Training job %s queued: %s on %s", job_id, model, dataset)
    return {"job_id": job_id, "status": "queued"}


def _run_training(job_id: str):
    """Execute training — try Python API first, fall back to CLI."""
    with _job_lock:
        job = _jobs[job_id]
        job["status"] = "running"
        job["started_at"] = time.time()

    try:
        _run_training_python_api(job)
    except ImportError:
        logger.info("Python tuner API not available, falling back to CLI for job %s", job_id)
        try:
            _run_training_cli(job)
        except Exception as e:
            with _job_lock:
                job["status"] = "failed"
                job["error"] = str(e)
                job["completed_at"] = time.time()
            logger.error("Training job %s failed (CLI): %s", job_id, e)
            return
    except Exception as e:
        with _job_lock:
            job["status"] = "failed"
            job["error"] = str(e)
            job["completed_at"] = time.time()
        logger.error("Training job %s failed: %s", job_id, e)
        return

    with _job_lock:
        job["status"] = "completed"
        job["completed_at"] = time.time()
        elapsed = job["completed_at"] - job["started_at"]
    logger.info("Training job %s completed in %.1fs", job_id, elapsed)


def _run_training_python_api(job: dict):
    """Train using mlx_lm.tuner Python API."""
    from mlx_lm import load
    from mlx_lm.tuner import train, TrainingArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers
    import mlx.optimizers as optim

    config = job["config"]
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_file = str(output_dir / "adapters.safetensors")

    logger.info("Loading model %s for training...", job["model"])
    model, tokenizer = load(job["model"])

    # Freeze base model, apply LoRA
    model.freeze()
    linear_to_lora_layers(
        model,
        lora_layers=config["lora_layers"],
        lora_parameters={"rank": config["lora_rank"]},
    )

    training_args = TrainingArgs(
        batch_size=config["batch_size"],
        iters=config["iters"],
        adapter_file=adapter_file,
        steps_per_report=10,
        steps_per_eval=50,
        steps_per_save=50,
    )

    optimizer = optim.Adam(learning_rate=config["learning_rate"])

    # Load dataset — expects path to directory or HF dataset
    from mlx_lm.tuner.datasets import load_dataset
    train_set, val_set, _ = load_dataset(job["dataset"], tokenizer)

    logger.info("Starting LoRA training: %d iters, rank=%d", config["iters"], config["lora_rank"])

    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
    )

    logger.info("Adapter saved to %s", adapter_file)


def _run_training_cli(job: dict):
    """Train using mlx_lm.lora CLI subprocess."""
    config = job["config"]
    output_dir = Path(job["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", job["model"],
        "--data", job["dataset"],
        "--adapter-path", str(output_dir),
        "--iters", str(config["iters"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--train",
    ]

    logger.info("Training via CLI: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stdout:
        logger.info("Training output: ...%s", result.stdout[-500:])


def get_job_status(job_id: str) -> dict | None:
    """Get status of a training job."""
    with _job_lock:
        return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    """List all training jobs."""
    with _job_lock:
        return [
            {
                "id": j["id"],
                "status": j["status"],
                "model": j["model"],
                "created_at": j["created_at"],
                "output_dir": j["output_dir"],
            }
            for j in _jobs.values()
        ]
