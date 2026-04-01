"""
Compute Engine — Full MLX GPU surface for containers.

The CUDA equivalent for Apple Silicon. Exposes EVERY category of MLX
operations over HTTP so containers can build arbitrary GPU-accelerated
applications on Metal — not just models, but raw GPU math.

Operation categories:
    ARITHMETIC     — add, multiply, divide, power, abs, neg, exp, log, sqrt
    LINEAR ALGEBRA — matmul, inner, outer, norm, svd, qr, eig, solve, inv, cholesky
    REDUCTIONS     — sum, mean, max, min, prod, var, std, argmax, argmin, all, any
    TRANSFORMS     — reshape, transpose, flatten, expand_dims, squeeze, broadcast, stack, concat, split, pad, tile, repeat
    ACTIVATIONS    — relu, gelu, silu, sigmoid, softmax, tanh, leaky_relu, elu, softplus, mish, celu
    CONVOLUTIONS   — conv1d, conv2d, conv_transpose1d, conv_transpose2d
    POOLING        — avg_pool1d, avg_pool2d, max_pool1d, max_pool2d
    ATTENTION      — scaled_dot_product_attention (Flash Attention on Metal)
    NORMALIZATION  — layer_norm, rms_norm, group_norm, batch_norm, instance_norm
    RANDOM         — normal, uniform, bernoulli, categorical, randint, truncated_normal
    FFT            — fft, ifft, rfft, irfft, fft2, ifft2
    SORTING        — sort, argsort, topk, partition
    COMPARISON     — equal, greater, less, where, clip, maximum, minimum
    METAL MEMORY   — get_active_memory, get_peak_memory, clear_cache, set_memory_limit
    BENCHMARK      — matmul throughput, memory bandwidth, latency profiling
"""

import logging
import time

import mlx.core as mx

logger = logging.getLogger("mlx-daemon.compute")

# ── Dtype mapping ────────────────────────────────────────────────────────────

DTYPES = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "int32": mx.int32,
    "int64": mx.int64,
    "int16": mx.int16,
    "int8": mx.int8,
    "uint8": mx.uint8,
    "uint32": mx.uint32,
    "bool": mx.bool_,
}


def _dtype(name: str | None):
    return DTYPES.get(name or "float32", mx.float32)


def _make_tensor(spec: dict | list) -> mx.array:
    """Create a tensor from spec: {shape, dtype, fill} or {data} or just a shape list."""
    if isinstance(spec, list):
        return mx.random.normal(spec)
    if "data" in spec:
        return mx.array(spec["data"], dtype=_dtype(spec.get("dtype")))
    shape = spec.get("shape", [512, 512])
    dtype = _dtype(spec.get("dtype"))
    fill = spec.get("fill")
    if fill is not None:
        return mx.full(shape, fill, dtype=dtype)
    return mx.random.normal(shape).astype(dtype)


def _result(op: str, result: mx.array, start: float, extra: dict | None = None) -> dict:
    """Standard result format."""
    mx.eval(result)
    elapsed = time.monotonic() - start
    out = {
        "op": op,
        "output_shape": list(result.shape),
        "output_dtype": str(result.dtype),
        "elapsed_ms": round(elapsed * 1000, 3),
        "device": str(mx.default_device()),
        "active_memory_gb": round(mx.metal.get_active_memory() / 1e9, 3),
    }
    if extra:
        out.update(extra)
    # Return small results inline
    if result.size <= 100:
        out["data"] = result.tolist()
    return out


# ── Device Info ──────────────────────────────────────────────────────────────

def get_device_info() -> dict:
    """Full Metal GPU device information."""
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


def clear_cache() -> dict:
    """Clear MLX Metal memory cache."""
    before = mx.metal.get_cache_memory()
    mx.metal.clear_cache()
    after = mx.metal.get_cache_memory()
    return {
        "cleared_bytes": before - after,
        "cleared_mb": round((before - after) / 1e6, 1),
        "cache_after_bytes": after,
    }


# ── Supported Operations ────────────────────────────────────────────────────

def list_operations() -> dict:
    """List all supported GPU operations by category."""
    return {
        "categories": {
            "arithmetic": ["add", "subtract", "multiply", "divide", "power", "abs", "neg", "exp", "log", "sqrt", "square", "reciprocal", "sin", "cos", "floor", "ceil"],
            "linear_algebra": ["matmul", "inner", "outer", "norm", "svd", "qr", "cholesky", "inv", "trace", "diagonal", "eye", "tri"],
            "reductions": ["sum", "mean", "max", "min", "prod", "var", "std", "argmax", "argmin", "all", "any", "logsumexp"],
            "transforms": ["reshape", "transpose", "flatten", "expand_dims", "squeeze", "stack", "concatenate", "split", "pad", "tile", "repeat", "flip", "roll"],
            "activations": ["relu", "gelu", "silu", "sigmoid", "softmax", "tanh", "leaky_relu", "elu", "softplus", "mish", "celu", "log_softmax", "hard_swish"],
            "convolutions": ["conv1d", "conv2d"],
            "pooling": ["avg_pool1d", "avg_pool2d", "max_pool1d", "max_pool2d"],
            "attention": ["scaled_dot_product_attention"],
            "normalization": ["layer_norm", "rms_norm", "group_norm", "batch_norm", "instance_norm"],
            "random": ["normal", "uniform", "bernoulli", "categorical", "randint", "truncated_normal"],
            "fft": ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2"],
            "sorting": ["sort", "argsort", "topk", "partition"],
            "comparison": ["equal", "greater", "less", "not_equal", "greater_equal", "less_equal", "where", "clip", "maximum", "minimum"],
            "metal": ["device_info", "clear_cache", "set_memory_limit", "reset_peak_memory"],
            "benchmark": ["matmul_throughput", "memory_bandwidth", "all_ops"],
        },
        "total_operations": 100,
        "device": str(mx.default_device()),
    }


# ── Main dispatcher ─────────────────────────────────────────────────────────

def eval_operation(op: str, args: dict) -> dict:
    """Execute any MLX GPU operation by name."""
    start = time.monotonic()

    try:
        # ── Arithmetic ───────────────────────────────────────────────────
        if op == "add":
            a, b = _make_tensor(args.get("a", {"shape": [1024, 1024]})), _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            return _result(op, a + b, start)

        elif op == "subtract":
            a, b = _make_tensor(args.get("a", {"shape": [1024, 1024]})), _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            return _result(op, a - b, start)

        elif op == "multiply":
            a, b = _make_tensor(args.get("a", {"shape": [1024, 1024]})), _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            return _result(op, a * b, start)

        elif op == "divide":
            a = _make_tensor(args.get("a", {"shape": [1024, 1024]}))
            b = _make_tensor(args.get("b", {"shape": [1024, 1024], "fill": 2.0}))
            return _result(op, a / b, start)

        elif op == "power":
            a = _make_tensor(args.get("a", {"shape": [1024, 1024]}))
            exp = args.get("exponent", 2)
            return _result(op, mx.power(a, exp), start)

        elif op in ("abs", "neg", "exp", "log", "sqrt", "square", "reciprocal", "sin", "cos", "floor", "ceil"):
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            fn = getattr(mx, op)
            return _result(op, fn(x), start)

        # ── Linear Algebra ───────────────────────────────────────────────
        elif op == "matmul":
            a = _make_tensor(args.get("a", {"shape": [512, 512]}))
            b = _make_tensor(args.get("b", {"shape": [512, 512]}))
            return _result(op, mx.matmul(a, b), start)

        elif op == "inner":
            a = _make_tensor(args.get("a", {"shape": [1024]}))
            b = _make_tensor(args.get("b", {"shape": [1024]}))
            return _result(op, mx.inner(a, b), start)

        elif op == "outer":
            a = _make_tensor(args.get("a", {"shape": [512]}))
            b = _make_tensor(args.get("b", {"shape": [512]}))
            return _result(op, mx.outer(a, b), start)

        elif op == "norm":
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            axis = args.get("axis")
            return _result(op, mx.linalg.norm(x, axis=axis), start)

        elif op == "svd":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            U, S, Vt = mx.linalg.svd(x, stream=mx.gpu)
            mx.eval(U, S, Vt)
            elapsed = time.monotonic() - start
            return {"op": op, "U_shape": list(U.shape), "S_shape": list(S.shape), "Vt_shape": list(Vt.shape), "elapsed_ms": round(elapsed * 1000, 3)}

        elif op == "qr":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            Q, R = mx.linalg.qr(x, stream=mx.gpu)
            mx.eval(Q, R)
            elapsed = time.monotonic() - start
            return {"op": op, "Q_shape": list(Q.shape), "R_shape": list(R.shape), "elapsed_ms": round(elapsed * 1000, 3)}

        elif op == "cholesky":
            n = args.get("size", 256)
            x = mx.random.normal([n, n])
            x = mx.matmul(x, x.T) + n * mx.eye(n)  # positive definite
            return _result(op, mx.linalg.cholesky(x), start)

        elif op == "inv":
            n = args.get("size", 256)
            x = mx.random.normal([n, n])
            x = x + n * mx.eye(n)  # invertible
            return _result(op, mx.linalg.inv(x), start)

        elif op == "trace":
            x = _make_tensor(args.get("x", {"shape": [512, 512]}))
            return _result(op, mx.trace(x), start)

        elif op == "diagonal":
            x = _make_tensor(args.get("x", {"shape": [512, 512]}))
            return _result(op, mx.diagonal(x), start)

        elif op == "eye":
            n = args.get("size", 512)
            return _result(op, mx.eye(n), start)

        elif op == "tri":
            n = args.get("size", 512)
            return _result(op, mx.tri(n), start)

        # ── Reductions ───────────────────────────────────────────────────
        elif op in ("sum", "mean", "max", "min", "prod", "var", "std", "logsumexp"):
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            axis = args.get("axis")
            keepdims = args.get("keepdims", False)
            fn = getattr(mx, op)
            return _result(op, fn(x, axis=axis, keepdims=keepdims), start)

        elif op in ("argmax", "argmin"):
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            axis = args.get("axis", -1)
            fn = getattr(mx, op)
            return _result(op, fn(x, axis=axis), start)

        elif op in ("all", "any"):
            x = _make_tensor(args.get("x", {"shape": [1024]}))
            x = x > 0  # convert to bool
            fn = getattr(mx, op)
            return _result(op, fn(x), start)

        # ── Transforms ───────────────────────────────────────────────────
        elif op == "reshape":
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            new_shape = args.get("new_shape", [-1])
            return _result(op, mx.reshape(x, new_shape), start)

        elif op == "transpose":
            x = _make_tensor(args.get("x", {"shape": [512, 1024]}))
            axes = args.get("axes")
            return _result(op, mx.transpose(x, axes=axes) if axes else mx.transpose(x), start)

        elif op == "flatten":
            x = _make_tensor(args.get("x", {"shape": [32, 32, 32]}))
            return _result(op, mx.flatten(x), start)

        elif op == "expand_dims":
            x = _make_tensor(args.get("x", {"shape": [1024]}))
            axis = args.get("axis", 0)
            return _result(op, mx.expand_dims(x, axis=axis), start)

        elif op == "squeeze":
            x = _make_tensor(args.get("x", {"shape": [1, 1024, 1]}))
            return _result(op, mx.squeeze(x), start)

        elif op == "stack":
            n = args.get("n", 4)
            shape = args.get("shape", [256, 256])
            arrays = [mx.random.normal(shape) for _ in range(n)]
            return _result(op, mx.stack(arrays), start)

        elif op == "concatenate":
            n = args.get("n", 4)
            shape = args.get("shape", [256, 256])
            axis = args.get("axis", 0)
            arrays = [mx.random.normal(shape) for _ in range(n)]
            return _result(op, mx.concatenate(arrays, axis=axis), start)

        elif op == "split":
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            n = args.get("n", 4)
            parts = mx.split(x, n)
            mx.eval(*parts)
            elapsed = time.monotonic() - start
            return {"op": op, "n_parts": len(parts), "part_shape": list(parts[0].shape), "elapsed_ms": round(elapsed * 1000, 3)}

        elif op == "pad":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            pad_width = args.get("pad_width", [[1, 1], [1, 1]])
            return _result(op, mx.pad(x, pad_width), start)

        elif op == "tile":
            x = _make_tensor(args.get("x", {"shape": [64, 64]}))
            reps = args.get("reps", [4, 4])
            return _result(op, mx.tile(x, reps), start)

        elif op == "repeat":
            x = _make_tensor(args.get("x", {"shape": [256]}))
            repeats = args.get("repeats", 4)
            return _result(op, mx.repeat(x, repeats), start)

        elif op == "flip":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            axis = args.get("axis", 0)
            return _result(op, mx.flip(x, axis=axis), start)

        elif op == "roll":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            shift = args.get("shift", 10)
            axis = args.get("axis", 0)
            return _result(op, mx.roll(x, shift=shift, axis=axis), start)

        # ── Activations (nn module) ──────────────────────────────────────
        elif op in ("relu", "gelu", "silu", "sigmoid", "tanh", "softplus", "mish", "celu", "hard_swish", "log_softmax"):
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            fn_map = {
                "relu": nn.relu, "gelu": nn.gelu, "silu": nn.silu,
                "sigmoid": mx.sigmoid, "tanh": mx.tanh,
                "softplus": nn.softplus, "mish": nn.mish, "celu": nn.celu,
                "hard_swish": nn.hard_swish, "log_softmax": nn.log_softmax,
            }
            return _result(op, fn_map[op](x), start)

        elif op == "softmax":
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            axis = args.get("axis", -1)
            return _result(op, mx.softmax(x, axis=axis), start)

        elif op == "leaky_relu":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            slope = args.get("negative_slope", 0.01)
            return _result(op, nn.leaky_relu(x, negative_slope=slope), start)

        elif op == "elu":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            alpha = args.get("alpha", 1.0)
            return _result(op, nn.elu(x, alpha=alpha), start)

        # ── Convolutions ─────────────────────────────────────────────────
        elif op == "conv1d":
            import mlx.nn as nn
            in_ch = args.get("in_channels", 32)
            out_ch = args.get("out_channels", 64)
            kernel = args.get("kernel_size", 3)
            batch = args.get("batch_size", 8)
            seq_len = args.get("seq_len", 256)
            conv = nn.Conv1d(in_ch, out_ch, kernel)
            x = mx.random.normal([batch, seq_len, in_ch])
            return _result(op, conv(x), start)

        elif op == "conv2d":
            import mlx.nn as nn
            in_ch = args.get("in_channels", 3)
            out_ch = args.get("out_channels", 64)
            kernel = args.get("kernel_size", 3)
            batch = args.get("batch_size", 4)
            h = args.get("height", 64)
            w = args.get("width", 64)
            conv = nn.Conv2d(in_ch, out_ch, kernel)
            x = mx.random.normal([batch, h, w, in_ch])
            return _result(op, conv(x), start)

        # ── Pooling ──────────────────────────────────────────────────────
        elif op == "avg_pool1d":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [4, 256, 64]}))
            kernel = args.get("kernel_size", 4)
            stride = args.get("stride", 4)
            return _result(op, nn.AvgPool1d(kernel, stride)(x), start)

        elif op == "avg_pool2d":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [4, 64, 64, 32]}))
            kernel = args.get("kernel_size", 2)
            stride = args.get("stride", 2)
            return _result(op, nn.AvgPool2d(kernel, stride)(x), start)

        elif op == "max_pool1d":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [4, 256, 64]}))
            kernel = args.get("kernel_size", 4)
            stride = args.get("stride", 4)
            return _result(op, nn.MaxPool1d(kernel, stride)(x), start)

        elif op == "max_pool2d":
            import mlx.nn as nn
            x = _make_tensor(args.get("x", {"shape": [4, 64, 64, 32]}))
            kernel = args.get("kernel_size", 2)
            stride = args.get("stride", 2)
            return _result(op, nn.MaxPool2d(kernel, stride)(x), start)

        # ── Attention (Flash Attention on Metal) ─────────────────────────
        elif op == "scaled_dot_product_attention":
            batch = args.get("batch_size", 4)
            heads = args.get("num_heads", 8)
            seq_len = args.get("seq_len", 512)
            head_dim = args.get("head_dim", 64)
            q = mx.random.normal([batch, heads, seq_len, head_dim])
            k = mx.random.normal([batch, heads, seq_len, head_dim])
            v = mx.random.normal([batch, heads, seq_len, head_dim])
            scale = args.get("scale", head_dim ** -0.5)
            result = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            return _result(op, result, start, {"batch": batch, "heads": heads, "seq_len": seq_len, "head_dim": head_dim})

        # ── Normalization ────────────────────────────────────────────────
        elif op == "layer_norm":
            import mlx.nn as nn
            dims = args.get("dims", 512)
            x = _make_tensor(args.get("x", {"shape": [32, 128, dims]}))
            ln = nn.LayerNorm(dims)
            return _result(op, ln(x), start)

        elif op == "rms_norm":
            import mlx.nn as nn
            dims = args.get("dims", 512)
            x = _make_tensor(args.get("x", {"shape": [32, 128, dims]}))
            rn = nn.RMSNorm(dims)
            return _result(op, rn(x), start)

        elif op == "group_norm":
            import mlx.nn as nn
            groups = args.get("groups", 32)
            channels = args.get("channels", 256)
            x = _make_tensor(args.get("x", {"shape": [4, 64, 64, channels]}))
            gn = nn.GroupNorm(groups, channels)
            return _result(op, gn(x), start)

        elif op == "batch_norm":
            import mlx.nn as nn
            features = args.get("features", 256)
            x = _make_tensor(args.get("x", {"shape": [32, features]}))
            bn = nn.BatchNorm(features)
            return _result(op, bn(x), start)

        elif op == "instance_norm":
            import mlx.nn as nn
            dims = args.get("dims", 64)
            x = _make_tensor(args.get("x", {"shape": [4, 32, 32, dims]}))
            inn = nn.InstanceNorm(dims)
            return _result(op, inn(x), start)

        # ── Random ───────────────────────────────────────────────────────
        elif op == "normal":
            shape = args.get("shape", [1024, 1024])
            return _result(op, mx.random.normal(shape), start)

        elif op == "uniform":
            shape = args.get("shape", [1024, 1024])
            low = args.get("low", 0.0)
            high = args.get("high", 1.0)
            return _result(op, mx.random.uniform(low=low, high=high, shape=shape), start)

        elif op == "bernoulli":
            shape = args.get("shape", [1024, 1024])
            p = args.get("p", 0.5)
            return _result(op, mx.random.bernoulli(p=p, shape=shape), start)

        elif op == "categorical":
            logits = _make_tensor(args.get("logits", {"shape": [32, 1000]}))
            return _result(op, mx.random.categorical(logits), start)

        elif op == "randint":
            shape = args.get("shape", [1024])
            low = args.get("low", 0)
            high = args.get("high", 100)
            return _result(op, mx.random.randint(low=low, high=high, shape=shape), start)

        elif op == "truncated_normal":
            shape = args.get("shape", [1024, 1024])
            low = args.get("low", -2.0)
            high = args.get("high", 2.0)
            return _result(op, mx.random.truncated_normal(low=low, high=high, shape=shape), start)

        # ── FFT ──────────────────────────────────────────────────────────
        elif op == "fft":
            x = _make_tensor(args.get("x", {"shape": [4096]}))
            return _result(op, mx.fft.fft(x), start)

        elif op == "ifft":
            x = _make_tensor(args.get("x", {"shape": [4096]}))
            f = mx.fft.fft(x)
            return _result(op, mx.fft.ifft(f), start)

        elif op == "rfft":
            x = _make_tensor(args.get("x", {"shape": [4096]}))
            return _result(op, mx.fft.rfft(x), start)

        elif op == "irfft":
            x = _make_tensor(args.get("x", {"shape": [4096]}))
            f = mx.fft.rfft(x)
            return _result(op, mx.fft.irfft(f), start)

        elif op == "fft2":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            return _result(op, mx.fft.fft2(x), start)

        elif op == "ifft2":
            x = _make_tensor(args.get("x", {"shape": [256, 256]}))
            f = mx.fft.fft2(x)
            return _result(op, mx.fft.ifft2(f), start)

        # ── Sorting / Searching ──────────────────────────────────────────
        elif op == "sort":
            x = _make_tensor(args.get("x", {"shape": [1000000]}))
            axis = args.get("axis", -1)
            return _result(op, mx.sort(x, axis=axis), start)

        elif op == "argsort":
            x = _make_tensor(args.get("x", {"shape": [1000000]}))
            axis = args.get("axis", -1)
            return _result(op, mx.argsort(x, axis=axis), start)

        elif op == "topk":
            x = _make_tensor(args.get("x", {"shape": [10000]}))
            k = args.get("k", 10)
            sorted_x = mx.sort(x)
            return _result(op, sorted_x[-k:], start, {"k": k})

        elif op == "partition":
            x = _make_tensor(args.get("x", {"shape": [10000]}))
            kth = args.get("kth", 100)
            return _result(op, mx.partition(x, kth=kth), start)

        # ── Comparison ───────────────────────────────────────────────────
        elif op in ("equal", "greater", "less", "not_equal", "greater_equal", "less_equal"):
            a = _make_tensor(args.get("a", {"shape": [1024, 1024]}))
            b = _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            fn = getattr(mx, op)
            return _result(op, fn(a, b), start)

        elif op == "where":
            cond = mx.random.bernoulli(shape=args.get("shape", [1024, 1024]))
            a = _make_tensor(args.get("a", {"shape": [1024, 1024]}))
            b = _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            return _result(op, mx.where(cond, a, b), start)

        elif op == "clip":
            x = _make_tensor(args.get("x", {"shape": [1024, 1024]}))
            a_min = args.get("min", -1.0)
            a_max = args.get("max", 1.0)
            return _result(op, mx.clip(x, a_min=a_min, a_max=a_max), start)

        elif op in ("maximum", "minimum"):
            a = _make_tensor(args.get("a", {"shape": [1024, 1024]}))
            b = _make_tensor(args.get("b", {"shape": [1024, 1024]}))
            fn = getattr(mx, op)
            return _result(op, fn(a, b), start)

        # ── Metal Memory Management ──────────────────────────────────────
        elif op == "device_info":
            return get_device_info()

        elif op == "clear_cache":
            return clear_cache()

        elif op == "set_memory_limit":
            limit_gb = args.get("limit_gb", 0)
            if limit_gb > 0:
                mx.metal.set_memory_limit(int(limit_gb * 1e9))
            return {"memory_limit_gb": limit_gb, "status": "set"}

        elif op == "reset_peak_memory":
            mx.metal.reset_peak_memory()
            return {"status": "peak_memory_reset"}

        # ── Benchmarks ───────────────────────────────────────────────────
        elif op == "benchmark" or op == "matmul_throughput":
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

            return {
                "op": "matmul_throughput",
                "matrix_size": size,
                "n_ops": n_ops,
                "total_seconds": round(bench_elapsed, 4),
                "per_op_ms": round(bench_elapsed / n_ops * 1000, 3),
                "gflops": round(gflops, 1),
                "tflops": round(gflops / 1000, 3),
                "device": str(mx.default_device()),
            }

        elif op == "memory_bandwidth":
            size_mb = args.get("size_mb", 256)
            n_elements = int(size_mb * 1e6 / 4)  # float32 = 4 bytes
            a = mx.random.normal([n_elements])
            b = mx.random.normal([n_elements])
            mx.eval(a, b)

            bench_start = time.monotonic()
            for _ in range(10):
                c = a + b
            mx.eval(c)
            bench_elapsed = time.monotonic() - bench_start

            bytes_moved = n_elements * 4 * 3 * 10  # read a, read b, write c, x10
            gb_per_sec = bytes_moved / bench_elapsed / 1e9

            return {
                "op": "memory_bandwidth",
                "size_mb": size_mb,
                "gb_per_sec": round(gb_per_sec, 1),
                "device": str(mx.default_device()),
            }

        elif op == "all_ops":
            # Quick benchmark of key operations
            results = {}
            for test_op in ["matmul", "softmax", "sort", "fft", "conv2d", "layer_norm", "scaled_dot_product_attention"]:
                r = eval_operation(test_op, {})
                results[test_op] = r.get("elapsed_ms", -1)
            return {"op": "all_ops", "results_ms": results, "device": str(mx.default_device())}

        elif op == "list":
            return list_operations()

        else:
            ops = list_operations()
            all_ops = []
            for cat_ops in ops["categories"].values():
                all_ops.extend(cat_ops)
            return {"error": f"Unknown op: {op}", "available": all_ops}

    except Exception as e:
        logger.error("Compute op '%s' failed: %s", op, e)
        return {"error": str(e), "op": op}
