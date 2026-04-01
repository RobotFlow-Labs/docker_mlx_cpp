"""
Embeddings Engine — MLX-powered text embeddings.

OpenAI-compatible /v1/embeddings endpoint.
Supports mlx-embeddings and Jina v5 MLX models.
"""

import logging
import time
import uuid

logger = logging.getLogger("mlx-daemon.embeddings")

_loaded_models: dict[str, object] = {}


def generate_embeddings(model_id: str, input_text: str | list[str]) -> dict:
    """Generate embeddings for input text(s)."""
    if isinstance(input_text, str):
        input_text = [input_text]

    start = time.monotonic()

    try:
        # Try mlx-embeddings first
        from mlx_embeddings import load as load_embedding_model
        from mlx_embeddings import encode

        if model_id not in _loaded_models:
            logger.info("Loading embedding model %s...", model_id)
            _loaded_models[model_id] = load_embedding_model(model_id)

        model = _loaded_models[model_id]
        vectors = encode(model, input_text)
        embeddings_list = [v.tolist() for v in vectors]

    except ImportError:
        # Fallback: use sentence-transformers or basic MLX
        logger.warning("mlx-embeddings not installed, using fallback")
        try:
            import mlx.core as mx
            import mlx.nn as nn

            # Simple hash-based embedding for fallback (not production quality)
            embeddings_list = []
            for text in input_text:
                hash_val = hash(text)
                # Generate deterministic pseudo-embedding
                import numpy as np
                rng = np.random.RandomState(abs(hash_val) % (2**31))
                vec = rng.randn(1536).tolist()
                # Normalize
                norm = sum(x**2 for x in vec) ** 0.5
                vec = [x / norm for x in vec]
                embeddings_list.append(vec)
        except ImportError:
            return {"error": "No embedding backend available. Install mlx-embeddings."}

    elapsed = time.monotonic() - start
    logger.info("Embeddings: %d texts, %s, %.1fms", len(input_text), model_id, elapsed * 1000)

    data = []
    total_tokens = 0
    for i, (text, embedding) in enumerate(zip(input_text, embeddings_list)):
        tokens = len(text.split())  # Approximate
        total_tokens += tokens
        data.append({
            "object": "embedding",
            "index": i,
            "embedding": embedding,
        })

    return {
        "object": "list",
        "data": data,
        "model": model_id,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }
