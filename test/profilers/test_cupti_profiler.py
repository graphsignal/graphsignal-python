import ctypes
import json
import sys
import time
import unittest
from unittest.mock import Mock, patch

import logging
import graphsignal
from graphsignal.profilers.cupti_profiler import CuptiProfiler

logger = logging.getLogger("graphsignal")

class CuptiProfilerTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(api_key="k1", debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    def _has_torch_cuda(self) -> bool:
        try:
            import torch  # type: ignore
        except Exception:
            return False
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def test_setup_skips_on_non_linux(self):
        profiler = CuptiProfiler(profile_name="test_profile", so_path="/tmp/fake.so", debug_mode=True)
        with patch("graphsignal.profilers.cupti_profiler.sys.platform", "darwin"):
            profiler.setup()
        self.assertTrue(profiler._disabled)
        self.assertIsNone(profiler.lib)

    def test_setup_skips_when_libcupti_missing(self):
        profiler = CuptiProfiler(profile_name="test_profile", so_path="/tmp/fake.so", debug_mode=True)

        def _cdll_side_effect(path, *args, **kwargs):
            if path == "libcupti.so":
                raise OSError("not found")
            raise AssertionError("unexpected CDLL load")

        with patch("graphsignal.profilers.cupti_profiler.sys.platform", "linux"), \
             patch("graphsignal.profilers.cupti_profiler.ctypes.CDLL", side_effect=_cdll_side_effect):
            profiler.setup()

        self.assertTrue(profiler._disabled)
        self.assertIsNone(profiler.lib)

    @unittest.skipUnless(sys.platform.startswith("linux"), "CUPTI profiler requires Linux")
    @patch('graphsignal._ticker.update_profile')
    def test_end_to_end_torch_cuda_and_drain(self, mock_update_profile):
        if not self._has_torch_cuda():
            self.skipTest("torch+CUDA not available")

        import torch  # type: ignore

        # Use the SDK's configured CUPTI profiler (avoids initializing CUPTI twice).
        profiler = getattr(graphsignal._ticker, "_cupti_profiler", None)
        if profiler is None or getattr(profiler, "_disabled", True) or getattr(profiler, "lib", None) is None:
            self.fail("CUPTI native library not available / failed to load")

        # Add at least one broad pattern to improve the odds of matching.
        graphsignal.profile_cuda_kernel("cublas", "matmul_gemm")

        # Generate some CUDA activity.
        a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        c = a @ b
        c = torch.relu(c)
        torch.cuda.synchronize()

        # wait enough time for the thread local buckets to be flushed into the global buckets
        time.sleep(1.1) 

        # generate some more activity so that thread local buckets are flushed
        a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
        c = a @ b
        c = torch.relu(c)
        torch.cuda.synchronize()

        # Check that update_profile was called for the CUDA profile.
        any_cuda_profile = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            #logger.debug('PROFILE UPDATE CALL %s %s', args, kwargs)
            name = kwargs.get("name", args[0] if len(args) >= 1 else None)
            profile = kwargs.get("profile", args[1] if len(args) >= 2 else None)
            if name == "profile.cuda" and isinstance(profile, dict) and len(profile) > 0:
                any_cuda_profile = True
                break
        
        # log all fields from metrics_store
        #for profile_field_id, profile_field in graphsignal._ticker._metric_store._profile_fields.items():
        #    logger.debug('FIELD %s %s', profile_field_id, profile_field.field_descriptor)

        self.assertTrue(any_cuda_profile, "expected at least one update_profile call for profile.cuda")

    @unittest.skipUnless(sys.platform.startswith("linux"), "CUPTI profiler requires Linux")
    def test_overhead(self):
        if not self._has_torch_cuda():
            self.skipTest("torch+CUDA not available")

        graphsignal.shutdown()

        import torch  # type: ignore

        # Make the workload more compute-heavy (reduces per-launch overhead %),
        # but keep runtime reasonable for CI.
        SIZE = 4096
        NUM_ITERS = 1000

        a = torch.randn((SIZE, SIZE), device="cuda", dtype=torch.float16)
        b = torch.randn((SIZE, SIZE), device="cuda", dtype=torch.float16)

        def _run_workload() -> int:
            # Ensure previous GPU work doesn't skew timing.
            torch.cuda.synchronize()
            start_ns = time.perf_counter_ns()
            for _ in range(NUM_ITERS):
                c = a @ b
                c = torch.relu(c)
            torch.cuda.synchronize()
            return time.perf_counter_ns() - start_ns

        # Warmup and baseline.
        for _ in range(2):
            _run_workload()
        took_ns_without_profiler = _run_workload()

        # enable graphsignal
        graphsignal.configure(api_key="k1", debug_mode=False)
        graphsignal._ticker.auto_tick = False

        # Enable CUPTI profiler.
        profiler = CuptiProfiler(profile_name="profile.cuda", debug_mode=False)
        profiler.setup()
        if profiler._disabled or profiler.lib is None:
            self.fail("CUPTI native library not available / failed to load")
        graphsignal._ticker._cupti_profiler = profiler

        # Warmup to populate caches and avoid one-time initialization costs.
        graphsignal.profile_cuda_kernel("cublas", "matmul_gemm")
        _run_workload()

        took_ns_with_profiler = _run_workload()

        # Basic sanity: profiler shouldn't add huge overhead to a representative GPU workload.
        # Use a percentage threshold to tolerate machine variability.
        overhead_pct = 100.0 * (took_ns_with_profiler - took_ns_without_profiler) / max(1, took_ns_without_profiler)
        overhead_per_iter_us = (took_ns_with_profiler - took_ns_without_profiler) / NUM_ITERS / 1e3

        logger.setLevel(logging.DEBUG)
        logger.debug("CUPTI overhead=%.2f%%, overhead_per_iter=%.1f us", overhead_pct, overhead_per_iter_us)

        self.assertTrue(overhead_pct < 5.0, f"expected overhead < 5.0%, got {overhead_pct:.2f}%")

        profiler.shutdown()

    def test_drain_native_debug_logs(self):
        profiler = CuptiProfiler(profile_name="test", debug_mode=True)
        data = [
            {"ts": 100, "msg": "graphsignal: line1\n"},
            {"ts": 200, "msg": "graphsignal: line2\n"},
        ]
        json_bytes = json.dumps(data).encode("utf-8")
        buf = ctypes.create_string_buffer(json_bytes)
        profiler.lib = Mock()
        profiler.lib.prof_drain_debug.return_value = ctypes.addressof(buf)
        profiler.lib.prof_free = Mock()
        mock_log_store = Mock()
        with patch.object(graphsignal._ticker, "log_store", return_value=mock_log_store):
            profiler._drain_native_debug_logs()
        self.assertEqual(mock_log_store.log_sdk_message.call_count, 2)
        mock_log_store.log_sdk_message.assert_any_call(
            level="debug",
            message="[cupti] graphsignal: line1\n",
            timestamp_ns=100,
        )
        mock_log_store.log_sdk_message.assert_any_call(
            level="debug",
            message="[cupti] graphsignal: line2\n",
            timestamp_ns=200,
        )
        profiler.lib.prof_free.assert_called_once()
