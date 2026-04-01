"""
Embeddings Engine — MLX-powered text embeddings.

OpenAI-compatible /v1/embeddings endpoint.
Uses mlx-embeddings with load() + generate() API.
"""

import logging
import time

logger = logging.getLogger("mlx-daemon.embeddings")

_loaded_models: dict[str, tuple] = {}  # model_id -> (model, tokenizer)


def generate_embeddings(model_id: str, input_text: str | list[str]) -> dict:
    """Generate embeddings for input text(s)."""
    if isinstance(input_text, str):
        input_text = [input_text]

    start = time.monotonic()

    try:
        from mlx_embeddings import load, generate

        if model_id not in _loaded_models:
            logger.info("Loading embedding model %s...", model_id)
            model, tokenizer = load(model_id)
            _loaded_models[model_id] = (model, tokenizer)

        model, tokenizer = _loaded_models[model_id]
        output = generate(model, tokenizer, texts=input_text)
        embeddings_list = [v.tolist() for v in output]

    except ImportError:
        logger.error("mlx-embeddings not installed. pip install 'docker-mlx-cpp[embeddings]'")
        return {"error": "mlx-embeddings not installed. Install with: pip install 'docker-mlx-cpp[embeddings]'"}

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
