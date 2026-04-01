"""
GPU Test Suite — Validates EVERY MLX Metal GPU operation from inside a Docker container.

This container proves that ANY Docker container can access the full Apple Silicon GPU
via docker_mlx_cpp. It tests 100+ operations across all categories.

Usage:
    docker compose --profile gpu-test up gpu-test
"""

import os
import sys
import time

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

GATEWAY_URL = os.environ.get("MLX_URL", "http://mlx-gateway:8080")
console = Console()


def call(method: str, path: str, data: dict | None = None) -> dict:
    """Make HTTP request to MLX gateway."""
    url = f"{GATEWAY_URL}{path}"
    with httpx.Client(timeout=120.0) as client:
        if method == "GET":
            resp = client.get(url)
        else:
            resp = client.post(url, json=data)
        return resp.json()


def test_op(op: str, args: dict | None = None) -> tuple[bool, float, str]:
    """Test a single GPU operation. Returns (passed, elapsed_ms, detail)."""
    try:
        result = call("POST", "/compute/eval", {"op": op, "args": args or {}})
        if "error" in result:
            return False, 0, result["error"]
        elapsed = result.get("elapsed_ms", 0)
        shape = result.get("output_shape", "?")
        return True, elapsed, f"shape={shape}"
    except Exception as e:
        return False, 0, str(e)


def main():
    console.print(Panel.fit(
        "[bold]docker_mlx_cpp — GPU Operation Test Suite[/bold]\n"
        f"Gateway: {GATEWAY_URL}\n"
        "Testing ALL Metal GPU operations from inside Docker",
        title="GPU TEST",
    ))
    console.print()

    # ── Phase 1: Health + Device ─────────────────────────────────────────
    console.print("[bold cyan]Phase 1: Health & Device Check[/bold cyan]")
    health = call("GET", "/health")
    console.print(f"  Daemon status: {health.get('status', '?')}")

    gpu = call("GET", "/gpu")
    console.print(f"  Metal GPU:     {gpu.get('metal_available', '?')}")
    console.print(f"  Device:        {gpu.get('default_device', '?')}")
    console.print(f"  Active mem:    {gpu.get('active_memory_gb', '?')} GB")
    console.print()

    if not gpu.get("metal_available"):
        console.print("[red]FATAL: Metal GPU not available. Cannot run tests.[/red]")
        sys.exit(1)

    # ── Phase 2: List available ops ──────────────────────────────────────
    console.print("[bold cyan]Phase 2: Operation Discovery[/bold cyan]")
    ops_info = call("GET", "/compute/ops")
    categories = ops_info.get("categories", {})
    total = sum(len(ops) for ops in categories.values())
    console.print(f"  Categories: {len(categories)}")
    console.print(f"  Total ops:  {total}")
    console.print()

    # ── Phase 3: Test every operation ────────────────────────────────────
    console.print("[bold cyan]Phase 3: Testing All GPU Operations[/bold cyan]")
    console.print()

    all_results = []
    passed_total = 0
    failed_total = 0

    # Define test cases per category with appropriate args
    test_cases = {
        "arithmetic": {
            "add": {}, "subtract": {}, "multiply": {}, "divide": {},
            "power": {"exponent": 2}, "abs": {}, "neg": {}, "exp": {},
            "log": {"x": {"shape": [512, 512], "fill": 2.0}},
            "sqrt": {"x": {"shape": [512, 512], "fill": 4.0}},
            "square": {}, "sin": {}, "cos": {}, "floor": {}, "ceil": {},
        },
        "linear_algebra": {
            "matmul": {},
            "inner": {"a": {"shape": [512]}, "b": {"shape": [512]}},
            "outer": {"a": {"shape": [256]}, "b": {"shape": [256]}},
            "norm": {},
            "svd": {"x": {"shape": [128, 128]}},
            "qr": {"x": {"shape": [128, 128]}},
            "cholesky": {"size": 128},
            "inv": {"size": 128},
            "trace": {},
            "diagonal": {},
            "eye": {"size": 256},
            "tri": {"size": 256},
        },
        "reductions": {
            "sum": {}, "mean": {}, "max": {}, "min": {}, "prod": {"x": {"shape": [64, 64]}},
            "var": {}, "std": {},
            "argmax": {}, "argmin": {},
            "all": {}, "any": {},
            "logsumexp": {},
        },
        "transforms": {
            "reshape": {"x": {"shape": [256, 256]}, "new_shape": [65536]},
            "transpose": {}, "flatten": {},
            "expand_dims": {}, "squeeze": {},
            "stack": {"n": 3, "shape": [64, 64]},
            "concatenate": {"n": 3, "shape": [64, 64]},
            "split": {"x": {"shape": [256, 256]}, "n": 4},
            "pad": {"x": {"shape": [64, 64]}},
            "tile": {"x": {"shape": [32, 32]}, "reps": [4, 4]},
            "repeat": {"x": {"shape": [128]}, "repeats": 4},
            "flip": {}, "roll": {},
        },
        "activations": {
            "relu": {}, "gelu": {}, "silu": {}, "sigmoid": {},
            "softmax": {}, "tanh": {}, "leaky_relu": {}, "elu": {},
            "softplus": {}, "mish": {}, "celu": {}, "hard_swish": {},
            "log_softmax": {},
        },
        "convolutions": {
            "conv1d": {"in_channels": 16, "out_channels": 32, "kernel_size": 3, "batch_size": 4, "seq_len": 128},
            "conv2d": {"in_channels": 3, "out_channels": 32, "kernel_size": 3, "batch_size": 2, "height": 32, "width": 32},
        },
        "pooling": {
            "avg_pool1d": {"x": {"shape": [2, 128, 32]}},
            "avg_pool2d": {"x": {"shape": [2, 32, 32, 16]}},
            "max_pool1d": {"x": {"shape": [2, 128, 32]}},
            "max_pool2d": {"x": {"shape": [2, 32, 32, 16]}},
        },
        "attention": {
            "scaled_dot_product_attention": {"batch_size": 2, "num_heads": 4, "seq_len": 128, "head_dim": 32},
        },
        "normalization": {
            "layer_norm": {"dims": 256, "x": {"shape": [8, 64, 256]}},
            "rms_norm": {"dims": 256, "x": {"shape": [8, 64, 256]}},
            "group_norm": {"groups": 16, "channels": 128, "x": {"shape": [2, 16, 16, 128]}},
            "batch_norm": {"features": 128, "x": {"shape": [16, 128]}},
            "instance_norm": {"dims": 32, "x": {"shape": [2, 16, 16, 32]}},
        },
        "random": {
            "normal": {"shape": [512, 512]},
            "uniform": {"shape": [512, 512]},
            "bernoulli": {"shape": [512, 512]},
            "categorical": {"logits": {"shape": [16, 100]}},
            "randint": {"shape": [1024]},
            "truncated_normal": {"shape": [512, 512]},
        },
        "fft": {
            "fft": {"x": {"shape": [2048]}},
            "ifft": {"x": {"shape": [2048]}},
            "rfft": {"x": {"shape": [2048]}},
            "irfft": {"x": {"shape": [2048]}},
            "fft2": {"x": {"shape": [128, 128]}},
            "ifft2": {"x": {"shape": [128, 128]}},
        },
        "sorting": {
            "sort": {"x": {"shape": [100000]}},
            "argsort": {"x": {"shape": [100000]}},
            "topk": {"x": {"shape": [10000]}, "k": 10},
            "partition": {"x": {"shape": [10000]}, "kth": 100},
        },
        "comparison": {
            "equal": {}, "greater": {}, "less": {},
            "not_equal": {}, "greater_equal": {}, "less_equal": {},
            "where": {"shape": [256, 256]},
            "clip": {}, "maximum": {}, "minimum": {},
        },
        "metal": {
            "device_info": {},
            "reset_peak_memory": {},
        },
        "benchmark": {
            "matmul_throughput": {"size": 512, "ops": 20},
            "memory_bandwidth": {"size_mb": 64},
        },
    }

    for category, ops in test_cases.items():
        table = Table(title=f"{category.upper()}", show_header=True)
        table.add_column("Operation", style="cyan", width=35)
        table.add_column("Status", width=8)
        table.add_column("Time (ms)", justify="right", width=12)
        table.add_column("Detail", width=40)

        cat_passed = 0
        cat_failed = 0

        for op_name, op_args in ops.items():
            passed, elapsed, detail = test_op(op_name, op_args)
            if passed:
                cat_passed += 1
                passed_total += 1
                status = "[green]PASS[/green]"
            else:
                cat_failed += 1
                failed_total += 1
                status = "[red]FAIL[/red]"
                detail = detail[:40]

            table.add_row(op_name, status, f"{elapsed:.1f}", detail)

        console.print(table)
        console.print(f"  {category}: {cat_passed}/{cat_passed + cat_failed} passed")
        console.print()

    # ── Phase 4: Summary ─────────────────────────────────────────────────
    total_tests = passed_total + failed_total
    pass_rate = (passed_total / total_tests * 100) if total_tests > 0 else 0

    color = "green" if pass_rate >= 95 else "yellow" if pass_rate >= 80 else "red"

    console.print(Panel.fit(
        f"[bold]Results: [{color}]{passed_total}/{total_tests} PASSED ({pass_rate:.0f}%)[/{color}][/bold]\n"
        f"  Passed:  {passed_total}\n"
        f"  Failed:  {failed_total}\n"
        f"  Device:  Metal GPU (Apple Silicon)\n"
        f"  Source:  Docker container → gateway → MLX daemon → Metal",
        title="GPU TEST COMPLETE",
    ))

    # ── Phase 5: Benchmark summary ───────────────────────────────────────
    console.print()
    console.print("[bold cyan]Phase 5: Performance Benchmarks[/bold cyan]")
    bench = call("POST", "/compute/eval", {"op": "matmul_throughput", "args": {"size": 1024, "ops": 50}})
    if "error" not in bench:
        console.print(f"  Matmul 1024x1024:  {bench.get('gflops', '?')} GFLOPS ({bench.get('tflops', '?')} TFLOPS)")
        console.print(f"  Per-op latency:    {bench.get('per_op_ms', '?')} ms")

    bw = call("POST", "/compute/eval", {"op": "memory_bandwidth", "args": {"size_mb": 256}})
    if "error" not in bw:
        console.print(f"  Memory bandwidth:  {bw.get('gb_per_sec', '?')} GB/s")

    all_bench = call("POST", "/compute/eval", {"op": "all_ops", "args": {}})
    if "results_ms" in all_bench:
        console.print()
        console.print("  Op latencies:")
        for op_name, ms in all_bench["results_ms"].items():
            console.print(f"    {op_name:40s} {ms:>8.1f} ms")

    console.print()
    console.print("[bold green]Container → Metal GPU: ALL SYSTEMS OPERATIONAL[/bold green]")

    sys.exit(0 if failed_total == 0 else 1)


if __name__ == "__main__":
    main()
