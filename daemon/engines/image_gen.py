"""
Image Generation Engine — MLX-powered Stable Diffusion / FLUX.

Supports:
    - Stable Diffusion 1.5 / SDXL
    - FLUX (via mlx-community models)

OpenAI-compatible /v1/images/generations endpoint.
"""

import base64
import io
import logging
import time
import uuid

logger = logging.getLogger("mlx-daemon.image_gen")


def generate_image(
    prompt: str,
    model: str = "stable-diffusion",
    size: str = "512x512",
    n: int = 1,
) -> dict:
    """Generate images from a text prompt using MLX."""
    start = time.monotonic()

    try:
        width, height = _parse_size(size)
    except ValueError:
        return {"error": f"Invalid size: {size}. Use format WxH (e.g., 512x512)"}

    try:
        # Try MLX stable diffusion
        from stable_diffusion import StableDiffusion

        sd = StableDiffusion()

        images_data = []
        for i in range(n):
            logger.info("Generating image %d/%d: '%s' (%s)", i + 1, n, prompt[:50], size)

            image = sd.generate(
                prompt,
                n_steps=20,
                cfg_weight=7.5,
                width=width,
                height=height,
            )

            # Convert to base64 PNG
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            images_data.append({
                "b64_json": b64,
                "revised_prompt": prompt,
            })

        elapsed = time.monotonic() - start
        logger.info("Generated %d images in %.1fs", n, elapsed)

        return {
            "created": int(time.time()),
            "data": images_data,
        }

    except ImportError:
        logger.error("MLX stable-diffusion not available. Install mlx SD examples.")
        return {
            "error": "Image generation requires MLX stable-diffusion. "
                     "Clone mlx-examples and install stable_diffusion module.",
            "hint": "git clone https://github.com/ml-explore/mlx-examples && cd mlx-examples/stable_diffusion && pip install .",
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
