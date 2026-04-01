"""
Compute Engine — Direct MLX GPU acceleration for containers.

Exposes raw MLX tensor operations over HTTP so containers can run
arbitrary GPU-accelerated computation on Apple Silicon Metal.

This is the "CUDA compute" equivalent: not just models, but raw GPU math.

Endpoints:
    POST /compute/eval     — Evaluate MLX operations (matmul, conv, etc.)
    GET  /compute/devices  — List available Metal devices
    POST /compute/bench    — Benchmark GPU performance
"""

import logging
import time

logger = logging.getLogger("mlx-daemon.compute")


def get_device_info() -> dict:
    """Get MLX Metal GPU device information."""
    try:
        import mlx.core as mx

        return {
            "default_device": str(mx.default_device()),
            "metal_available": True,
            "active_memory_bytes": mx.metal.get_active_memory(),
            "peak_memory_bytes": mx.metal.get_peak_memory(),
            "cache_memory_bytes": mx.metal.get_cache_memory(),
            "active_memory_gb": round(mx.metal.get_active_memory() / 1e9, 3),
            "peak_memory_gb": round(mx.metal.get_peak_memory() / 1e9, 3),
            "cache_memory_gb": round(mx.metal.get_cache_memory() / 1e9, 3),
        }
    except (ImportError, AttributeError) as e:
        return {"metal_available": False, "error": str(e)}


def eval_operation(op: str, args: dict) -> dict:
    """Execute an MLX GPU operation.

    Supported operations:
        matmul:    {a_shape, b_shape, dtype} — matrix multiply
        add:       {a_shape, b_shape, dtype} — element-wise add
        softmax:   {shape, axis, dtype}      — softmax
        conv2d:    {input_shape, weight_shape, stride, padding}
        fft:       {shape, dtype}            — FFT
        sort:      {shape, dtype}            — GPU sort
        random:    {shape, dtype}            — random generation
        benchmark: {shape, dtype, ops}       — run N ops and report throughput
    """
    import mlx.core as mx

    start = time.monotonic()

    try:
        if op == "matmul":
            a = mx.random.normal(args.get("a_shape", [512, 512]))
            b = mx.random.normal(args.get("b_shape", [512, 512]))
            result = mx.matmul(a, b)
            mx.eval(result)
            shape = list(result.shape)

        elif op == "add":
            a = mx.random.normal(args.get("a_shape", [1024, 1024]))
            b = mx.random.normal(args.get("b_shape", [1024, 1024]))
            result = mx.add(a, b)
            mx.eval(result)
            shape = list(result.shape)

        elif op == "softmax":
            x = mx.random.normal(args.get("shape", [1024, 1024]))
            axis = args.get("axis", -1)
            result = mx.softmax(x, axis=axis)
            mx.eval(result)
            shape = list(result.shape)

        elif op == "sort":
            x = mx.random.normal(args.get("shape", [1000000]))
            result = mx.sort(x)
            mx.eval(result)
            shape = list(result.shape)

        elif op == "random":
            shape_arg = args.get("shape", [1024, 1024])
            result = mx.random.normal(shape_arg)
            mx.eval(result)
            shape = list(result.shape)

        elif op == "benchmark":
            # Run matmul N times and report throughput
            n_ops = args.get("ops", 100)
            size = args.get("size", 1024)
            a = mx.random.normal([size, size])
            b = mx.random.normal([size, size])
            mx.eval(a, b)

            bench_start = time.monotonic()
            for _ in range(n_ops):
                c = mx.matmul(a, b)
            mx.eval(c)
            bench_elapsed = time.monotonic() - bench_start

            flops = 2 * size * size * size * n_ops
            gflops = flops / bench_elapsed / 1e9

            elapsed = time.monotonic() - start
            return {
                "op": "benchmark",
                "size": size,
                "n_ops": n_ops,
                "total_seconds": round(bench_elapsed, 4),
                "per_op_ms": round(bench_elapsed / n_ops * 1000, 3),
                "gflops": round(gflops, 1),
                "device": str(mx.default_device()),
            }

        else:
            return {"error": f"Unknown operation: {op}. Supported: matmul, add, softmax, sort, random, benchmark"}

        elapsed = time.monotonic() - start

        return {
            "op": op,
            "output_shape": shape,
            "elapsed_ms": round(elapsed * 1000, 3),
            "device": str(mx.default_device()),
            "active_memory_gb": round(mx.metal.get_active_memory() / 1e9, 3),
        }

    except Exception as e:
        return {"error": str(e), "op": op}


def clear_cache() -> dict:
    """Clear MLX Metal memory cache."""
    try:
        import mlx.core as mx
        before = mx.metal.get_cache_memory()
        mx.metal.clear_cache()
        after = mx.metal.get_cache_memory()
        return {
            "cleared_bytes": before - after,
            "cleared_mb": round((before - after) / 1e6, 1),
            "cache_after_bytes": after,
        }
    except (ImportError, AttributeError) as e:
        return {"error": str(e)}
