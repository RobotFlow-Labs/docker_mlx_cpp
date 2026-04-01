"""
Model Manager — Pull, cache, convert, and serve MLX models.

Manages the local model registry for docker_mlx_cpp. Models are pulled from
HuggingFace (mlx-community and others), cached locally, and served on demand.

Registry location: ~/.docker-mlx/models/
"""

import json
import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

logger = logging.getLogger("mlx-daemon.models")

DEFAULT_CACHE_DIR = Path.home() / ".docker-mlx" / "models"
PRESETS_FILE = Path(__file__).parent.parent / "models" / "presets.yaml"


class ModelManager:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.cache_dir / "registry.json"
        self._registry = self._load_registry()
        self._hf_api = HfApi()

    def _load_registry(self) -> dict:
        if self.registry_file.exists():
            return json.loads(self.registry_file.read_text())
        return {"models": {}}

    def _save_registry(self):
        self.registry_file.write_text(json.dumps(self._registry, indent=2))

    def _model_dir(self, model_id: str) -> Path:
        """Get local path for a model, using / → -- as separator."""
        safe_name = model_id.replace("/", "--")
        return self.cache_dir / safe_name

    def list_models(self) -> list[dict]:
        """List all locally cached models."""
        models = []
        for model_id, info in self._registry.get("models", {}).items():
            model_dir = self._model_dir(model_id)
            models.append({
                "id": model_id,
                "object": "model",
                "owned_by": model_id.split("/")[0] if "/" in model_id else "local",
                "local_path": str(model_dir),
                "size_bytes": info.get("size_bytes"),
                "pulled_at": info.get("pulled_at"),
            })
        return models

    def get_model_path(self, model_id: str) -> Path | None:
        """Get local path for a model, or None if not cached."""
        model_dir = self._model_dir(model_id)
        if model_dir.exists() and model_id in self._registry.get("models", {}):
            return model_dir
        return None

    async def pull_model(self, model_id: str, force: bool = False) -> Path:
        """Pull a model from HuggingFace to local cache."""
        model_dir = self._model_dir(model_id)

        if model_dir.exists() and not force:
            if model_id in self._registry.get("models", {}):
                logger.info("Model %s already cached at %s", model_id, model_dir)
                return model_dir

        logger.info("Pulling model %s from HuggingFace...", model_id)

        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        # Calculate size
        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

        import datetime

        self._registry.setdefault("models", {})[model_id] = {
            "local_path": str(local_path),
            "size_bytes": size_bytes,
            "pulled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._save_registry()

        logger.info("Model %s pulled (%d MB)", model_id, size_bytes // (1024 * 1024))
        return model_dir

    def ensure_model(self, model_id: str) -> Path:
        """Synchronously ensure a model is available (for inference startup)."""
        model_dir = self._model_dir(model_id)
        if model_dir.exists() and model_id in self._registry.get("models", {}):
            return model_dir

        logger.info("Auto-downloading model %s...", model_id)
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
        )

        size_bytes = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())

        import datetime

        self._registry.setdefault("models", {})[model_id] = {
            "local_path": str(local_path),
            "size_bytes": size_bytes,
            "pulled_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        self._save_registry()
        return model_dir

    def delete_model(self, model_id: str) -> bool:
        """Remove a model from local cache."""
        model_dir = self._model_dir(model_id)
        if model_dir.exists():
            shutil.rmtree(model_dir)
        if model_id in self._registry.get("models", {}):
            del self._registry["models"][model_id]
            self._save_registry()
            logger.info("Deleted model %s", model_id)
            return True
        return False

    def resolve_preset(self, preset_name: str) -> str | None:
        """Resolve a preset name to a model ID."""
        try:
            import yaml

            if PRESETS_FILE.exists():
                presets = yaml.safe_load(PRESETS_FILE.read_text())
                preset = presets.get("presets", {}).get(preset_name)
                if preset:
                    return preset.get("model")
        except ImportError:
            pass
        return None

    def resolve_model(self, model_id_or_preset: str) -> str:
        """Resolve a model ID or preset name to a model ID."""
        resolved = self.resolve_preset(model_id_or_preset)
        return resolved if resolved else model_id_or_preset

    def get_gpu_info(self) -> dict:
        """Get Apple Silicon GPU information."""
        import platform
        import subprocess

        info = {
            "chip": platform.processor(),
            "platform": platform.machine(),
            "metal_available": False,
            "memory_total_gb": None,
            "memory_used_gb": None,
        }

        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, check=True,
            )
            total_bytes = int(result.stdout.strip())
            info["memory_total_gb"] = round(total_bytes / (1024**3), 1)
        except (subprocess.CalledProcessError, ValueError):
            pass

        try:
            import mlx.core as mx
            info["metal_available"] = True
            info["mlx_backend"] = str(mx.default_device())
        except ImportError:
            info["metal_available"] = False

        return info
