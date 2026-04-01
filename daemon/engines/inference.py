"""
Inference Engine — MLX-powered LLM and VLM inference.

LLM: mlx-lm (load, generate, stream_generate) — 50+ architectures
VLM: mlx-vlm (load, generate) — vision-language with image input

stream_generate yields GenerationResponse objects with:
    .text, .finish_reason, .generation_tokens, .generation_tps,
    .prompt_tokens, .prompt_tps, .peak_memory

Performance: 20-30% faster than llama.cpp on Apple Silicon Metal.
"""

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator

logger = logging.getLogger("mlx-daemon.inference")

# Lazy-loaded modules and model cache
_mlx_lm = None
_loaded_models: dict[str, tuple] = {}  # model_id -> (model, tokenizer)
_loaded_vlms: dict[str, tuple] = {}    # model_id -> (model, processor, config)


def _get_mlx_lm():
    global _mlx_lm
    if _mlx_lm is None:
        import mlx_lm
        _mlx_lm = mlx_lm
    return _mlx_lm


def _load_model(model_path: str, model_id: str) -> tuple:
    """Load an MLX LLM model + tokenizer, with caching."""
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    logger.info("Loading LLM %s from %s...", model_id, model_path)
    mlx_lm = _get_mlx_lm()
    model, tokenizer = mlx_lm.load(model_path)
    _loaded_models[model_id] = (model, tokenizer)
    logger.info("LLM %s loaded and ready", model_id)
    return model, tokenizer


def _load_vlm(model_path: str, model_id: str) -> tuple:
    """Load an MLX VLM model + processor + config, with caching."""
    if model_id in _loaded_vlms:
        return _loaded_vlms[model_id]

    logger.info("Loading VLM %s from %s...", model_id, model_path)
    from mlx_vlm import load as vlm_load
    from mlx_vlm.utils import load_config

    model, processor = vlm_load(model_path)
    config = load_config(model_path)
    _loaded_vlms[model_id] = (model, processor, config)
    logger.info("VLM %s loaded and ready", model_id)
    return model, processor, config


def _is_vlm_request(messages: list[dict]) -> bool:
    """Check if any message contains image content (OpenAI vision format)."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _extract_vlm_content(messages: list[dict]) -> tuple[str, list[str]]:
    """Extract text prompt and image URLs from OpenAI vision-format messages."""
    text_parts = []
    images = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {})
                        if isinstance(url, dict):
                            images.append(url.get("url", ""))
                        else:
                            images.append(str(url))
    return " ".join(text_parts), images


def unload_model(model_id: str) -> bool:
    """Unload a model from memory."""
    removed = False
    if model_id in _loaded_models:
        del _loaded_models[model_id]
        removed = True
    if model_id in _loaded_vlms:
        del _loaded_vlms[model_id]
        removed = True
    if removed:
        logger.info("Unloaded model %s", model_id)
    return removed


def list_loaded_models() -> list[str]:
    """List currently loaded model IDs."""
    return list(set(list(_loaded_models.keys()) + list(_loaded_vlms.keys())))


def generate_completion(
    model_path: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    stop: list[str] | None = None,
    stream: bool = False,
) -> dict | AsyncGenerator:
    """Generate a chat completion using MLX (LLM or VLM)."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # VLM path: messages contain images
    if _is_vlm_request(messages):
        return _generate_vlm(model_path, model_id, messages, max_tokens, temperature, completion_id, created)

    # LLM path
    model, tokenizer = _load_model(model_path, model_id)
    mlx_lm = _get_mlx_lm()

    # Apply chat template
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages
        )
        prompt += "\nassistant: "

    if stream:
        return _stream_generate(
            model, tokenizer, mlx_lm, prompt, model_id,
            completion_id, created, max_tokens, temperature, top_p,
        )

    # Non-streaming generation
    start = time.monotonic()
    response_text = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
    )
    elapsed = time.monotonic() - start

    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(response_text))

    logger.info(
        "Inference: %s | %d prompt + %d completion tokens | %.1fs",
        model_id, prompt_tokens, completion_tokens, elapsed,
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _generate_vlm(
    model_path: str,
    model_id: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    completion_id: str,
    created: int,
) -> dict:
    """Generate completion from a vision-language model."""
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, config = _load_vlm(model_path, model_id)

    prompt_text, images = _extract_vlm_content(messages)

    formatted_prompt = apply_chat_template(
        processor, config, prompt_text, num_images=len(images)
    )

    start = time.monotonic()
    output = vlm_generate(
        model, processor, formatted_prompt,
        images if images else None,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    elapsed = time.monotonic() - start

    # output is a string for vlm generate
    response_text = output if isinstance(output, str) else str(output)

    logger.info("VLM inference: %s | %d images | %.1fs", model_id, len(images), elapsed)

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,  # VLM doesn't expose this easily
            "completion_tokens": len(response_text.split()),
            "total_tokens": len(response_text.split()),
        },
    }


async def _stream_generate(
    model, tokenizer, mlx_lm, prompt: str, model_id: str,
    completion_id: str, created: int,
    max_tokens: int, temperature: float, top_p: float,
) -> AsyncGenerator[str, None]:
    """Stream tokens as SSE events (OpenAI format).

    mlx_lm.stream_generate yields GenerationResponse objects with:
        .text           — the token text
        .finish_reason  — None during gen, "stop"/"length" at end
        .generation_tokens, .generation_tps, .peak_memory
        .prompt_tokens, .prompt_tps
    """
    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
    ):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": response.text},
                    "finish_reason": response.finish_reason,
                }
            ],
        }

        # Include usage + metrics on final chunk
        if response.finish_reason is not None:
            chunk["usage"] = {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.generation_tokens,
                "total_tokens": response.prompt_tokens + response.generation_tokens,
            }
            chunk["mlx_metrics"] = {
                "generation_tps": round(response.generation_tps, 2),
                "prompt_tps": round(response.prompt_tps, 2),
                "peak_memory_gb": round(response.peak_memory, 3),
            }

        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"
