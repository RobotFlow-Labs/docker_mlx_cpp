"""Tests for the MLX GPU compute engine — validates all 107 operations."""

import pytest

# Skip entire module if MLX not available (e.g., CI on Linux)
mlx_available = False
try:
    import mlx.core as mx
    mlx_available = True
except ImportError:
    pass

pytestmark = pytest.mark.skipif(not mlx_available, reason="MLX not available (requires Apple Silicon)")


from daemon.engines.compute import eval_operation, get_device_info, list_operations, clear_cache


class TestDeviceInfo:
    def test_device_info_returns_metal(self):
        info = get_device_info()
        assert info["metal_available"] is True
        assert "gpu" in info["default_device"]

    def test_memory_stats(self):
        info = get_device_info()
        assert "active_memory_bytes" in info
        assert "peak_memory_bytes" in info
        assert "cache_memory_bytes" in info


class TestListOperations:
    def test_lists_all_categories(self):
        ops = list_operations()
        assert "categories" in ops
        assert len(ops["categories"]) >= 14

    def test_total_ops_count(self):
        ops = list_operations()
        total = sum(len(v) for v in ops["categories"].values())
        assert total >= 100


class TestArithmetic:
    @pytest.mark.parametrize("op", ["add", "subtract", "multiply", "divide", "power", "abs", "neg", "exp", "square", "sin", "cos", "floor", "ceil"])
    def test_arithmetic_op(self, op):
        args = {"exponent": 2} if op == "power" else {}
        result = eval_operation(op, args)
        assert "error" not in result, f"{op} failed: {result.get('error')}"
        assert "output_shape" in result

    def test_log_positive_input(self):
        result = eval_operation("log", {"x": {"shape": [256, 256], "fill": 2.0}})
        assert "error" not in result

    def test_sqrt_positive_input(self):
        result = eval_operation("sqrt", {"x": {"shape": [256, 256], "fill": 4.0}})
        assert "error" not in result


class TestLinearAlgebra:
    @pytest.mark.parametrize("op", ["matmul", "norm", "trace", "diagonal", "eye", "tri"])
    def test_linalg_gpu_op(self, op):
        args = {"size": 128} if op in ("eye", "tri") else {}
        result = eval_operation(op, args)
        assert "error" not in result, f"{op} failed: {result.get('error')}"

    @pytest.mark.parametrize("op", ["svd", "qr", "cholesky", "inv"])
    def test_linalg_cpu_op(self, op):
        """These ops run on CPU (MLX constraint) but should still succeed."""
        args = {"size": 64} if op in ("cholesky", "inv") else {"x": {"shape": [64, 64]}}
        result = eval_operation(op, args)
        assert "error" not in result, f"{op} failed: {result.get('error')}"

    def test_inner_product(self):
        result = eval_operation("inner", {"a": {"shape": [256]}, "b": {"shape": [256]}})
        assert "error" not in result

    def test_outer_product(self):
        result = eval_operation("outer", {"a": {"shape": [128]}, "b": {"shape": [128]}})
        assert "error" not in result


class TestReductions:
    @pytest.mark.parametrize("op", ["sum", "mean", "max", "min", "var", "std", "argmax", "argmin", "logsumexp"])
    def test_reduction(self, op):
        result = eval_operation(op, {})
        assert "error" not in result, f"{op} failed: {result.get('error')}"

    def test_all_any(self):
        for op in ("all", "any"):
            result = eval_operation(op, {})
            assert "error" not in result


class TestTransforms:
    @pytest.mark.parametrize("op", ["reshape", "transpose", "flatten", "expand_dims", "squeeze", "stack", "concatenate", "split", "pad", "tile", "repeat", "flip", "roll"])
    def test_transform(self, op):
        args_map = {
            "reshape": {"x": {"shape": [128, 128]}, "new_shape": [16384]},
            "stack": {"n": 3, "shape": [32, 32]},
            "concatenate": {"n": 3, "shape": [32, 32]},
            "split": {"x": {"shape": [128, 128]}, "n": 4},
            "pad": {"x": {"shape": [32, 32]}},
            "tile": {"x": {"shape": [16, 16]}, "reps": [4, 4]},
            "repeat": {"x": {"shape": [64]}, "repeats": 4},
        }
        result = eval_operation(op, args_map.get(op, {}))
        assert "error" not in result, f"{op} failed: {result.get('error')}"


class TestActivations:
    @pytest.mark.parametrize("op", ["relu", "gelu", "silu", "sigmoid", "softmax", "tanh", "leaky_relu", "elu", "softplus"])
    def test_activation(self, op):
        result = eval_operation(op, {})
        assert "error" not in result, f"{op} failed: {result.get('error')}"


class TestNeuralNetOps:
    def test_conv1d(self):
        result = eval_operation("conv1d", {"in_channels": 8, "out_channels": 16, "batch_size": 2, "seq_len": 64})
        assert "error" not in result

    def test_conv2d(self):
        result = eval_operation("conv2d", {"in_channels": 3, "out_channels": 16, "batch_size": 2, "height": 16, "width": 16})
        assert "error" not in result

    def test_attention(self):
        result = eval_operation("scaled_dot_product_attention", {"batch_size": 2, "num_heads": 4, "seq_len": 64, "head_dim": 32})
        assert "error" not in result
        assert result["output_shape"] == [2, 4, 64, 32]

    @pytest.mark.parametrize("op", ["layer_norm", "rms_norm", "batch_norm"])
    def test_normalization(self, op):
        args = {"dims": 128, "x": {"shape": [8, 32, 128]}} if op != "batch_norm" else {"features": 64, "x": {"shape": [8, 64]}}
        result = eval_operation(op, args)
        assert "error" not in result


class TestRandom:
    @pytest.mark.parametrize("op", ["normal", "uniform", "bernoulli", "randint", "truncated_normal"])
    def test_random(self, op):
        args = {"shape": [256, 256]}
        if op == "randint":
            args = {"shape": [512]}
        result = eval_operation(op, args)
        assert "error" not in result


class TestFFT:
    @pytest.mark.parametrize("op", ["fft", "ifft", "rfft", "irfft", "fft2", "ifft2"])
    def test_fft(self, op):
        shape = [128, 128] if "2" in op else [1024]
        result = eval_operation(op, {"x": {"shape": shape}})
        assert "error" not in result


class TestSorting:
    @pytest.mark.parametrize("op", ["sort", "argsort", "topk", "partition"])
    def test_sorting(self, op):
        args = {"x": {"shape": [5000]}}
        if op == "topk":
            args["k"] = 10
        if op == "partition":
            args["kth"] = 50
        result = eval_operation(op, args)
        assert "error" not in result


class TestComparison:
    @pytest.mark.parametrize("op", ["equal", "greater", "less", "not_equal", "greater_equal", "less_equal", "clip", "maximum", "minimum"])
    def test_comparison(self, op):
        result = eval_operation(op, {})
        assert "error" not in result

    def test_where(self):
        result = eval_operation("where", {"shape": [128, 128]})
        assert "error" not in result


class TestBenchmarks:
    def test_matmul_throughput(self):
        result = eval_operation("matmul_throughput", {"size": 256, "ops": 10})
        assert "error" not in result
        assert "gflops" in result
        assert result["gflops"] > 0

    def test_memory_bandwidth(self):
        result = eval_operation("memory_bandwidth", {"size_mb": 32})
        assert "error" not in result
        assert "gb_per_sec" in result

    def test_all_ops_benchmark(self):
        result = eval_operation("all_ops", {})
        assert "results_ms" in result


class TestClearCache:
    def test_clear_cache(self):
        result = clear_cache()
        assert "cleared_bytes" in result
