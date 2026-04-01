"""
Image Generation Engine — MLX-native FLUX via mflux.

Uses mflux (pip install mflux) for FLUX image generation on Apple Silicon Metal.
OpenAI-compatible /v1/images/generations endpoint.
"""

import base64
import io
import logging
import time

logger = logging.getLogger("mlx-daemon.image_gen")


def generate_image(
    prompt: str,
    model: str = "flux-schnell",
    size: str = "512x512",
    n: int = 1,
    steps: int = 4,
    seed: int | None = None,
) -> dict:
    """Generate images from a text prompt using FLUX via mflux on Metal GPU."""
    try:
        width, height = _parse_size(size)
    except ValueError:
        return {"error": f"Invalid size: {size}. Use format WxH (e.g., 512x512)"}

    try:
        from mflux import Flux1Schnell

        images_data = []
        for i in range(n):
            logger.info("Generating image %d/%d: '%s' (%dx%d)", i + 1, n, prompt[:50], width, height)

            flux = Flux1Schnell(quantize=8)
            image = flux.generate_image(
                prompt=prompt,
                seed=seed or (int(time.time()) + i),
                num_inference_steps=steps,
                width=width,
                height=height,
            )

            # Convert PIL Image to base64 PNG
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            images_data.append({
                "b64_json": b64,
                "revised_prompt": prompt,
            })

        elapsed_total = time.monotonic()
        logger.info("Generated %d images", n)

        return {
            "created": int(time.time()),
            "data": images_data,
        }

    except ImportError:
        return {
            "error": "Image generation requires mflux. Install with: pip install 'docker-mlx-cpp[image]'",
        }
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return {"error": str(e)}


def _parse_size(size: str) -> tuple[int, int]:
    """Parse 'WxH' string to (width, height)."""
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size}")
    return int(parts[0]), int(parts[1])
