import ctypes
import json
import logging
import os
import platform
import sys
import time
import re
import threading
from typing import Any, Dict, List, Optional

import graphsignal

logger = logging.getLogger("graphsignal")


class EventFields:
    __slots__ = ['duration_field_id', 'calls_field_id', 'bytes_field_id']

    def __init__(self, duration_field_id=None, calls_field_id=None, bytes_field_id=None):
        self.duration_field_id = duration_field_id
        self.calls_field_id = calls_field_id
        self.bytes_field_id = bytes_field_id


class CuptiProfiler:
    def __init__(self, profile_name, so_path: Optional[str] = None, debug_mode: bool = False):
        self._profile_name = profile_name
        self._so_path = so_path
        self._disabled = True
        self.lib = None

        self._resolution_ns: int = 10_000_000  # 10ms default resolution
        self._activity_window_ns: int = 1000_000_000  # 1 second default activity window
        self._debug_mode: bool = bool(debug_mode)

        self._fields = {}
        self._current_event_id: int = 0
        self._current_event_id_lock = threading.Lock()
        self._drain_stop_event = threading.Event()
        self._drain_timer_thread = None
        self._last_drain_bucket_ts = 0
        self._debug_drain_stop_event = threading.Event()
        self._debug_drain_timer_thread = None

    def set_resolution_ns(self, resolution_ns: int) -> None:
        if resolution_ns < 10_000_000:
            resolution_ns = 10_000_000
        self._resolution_ns = int(resolution_ns)

    def get_resolution_ns(self) -> int:
        return self._resolution_ns

    def set_activity_window_ns(self, activity_window_ns: int) -> None:
        self._activity_window_ns = int(max(0, activity_window_ns))

    def get_activity_window_ns(self) -> int:
        return self._activity_window_ns

    def set_debug_mode(self, enabled: bool) -> None:
        self._debug_mode = enabled

        try:
            if self.lib:
                self.lib.prof_set_debug_mode(1 if self._debug_mode else 0)
        except Exception:
            pass

    def setup(self):
        logger.debug('CUPTI profiler setup started')

        if not sys.platform.startswith("linux"):
            logger.debug("CUPTI profiler not supported on this platform")
            return

        if not _detect_cuda_major():
            logger.debug("CUDA not available, disabling CUPTI profiler")
            return

        loaded = False
        for name in ("libcupti.so", "libcupti.so.13", "libcupti.so.12"):
            try:
                ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
                loaded = True
                break
            except OSError:
                continue
            except Exception:
                continue
        if not loaded:
            logger.warning("libcupti.so not found, disabling CUPTI profiler")
            return

        so_path = self._so_path
        if not so_path:
            _ensure_cuda_injection64_path()
            so_path = _default_cupti_profiler_so()
        
        if not so_path:
            logger.debug("CUPTI profiler shared library not found for path: %s", so_path)
            return

        try:
            _best_effort_load_libcupti()

            self.lib = ctypes.CDLL(so_path)

            self.lib.prof_start.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32]
            self.lib.prof_start.restype = None

            self.lib.prof_stop.argtypes = []
            self.lib.prof_stop.restype = None

            self.lib.prof_drain_json.argtypes = [ctypes.c_uint64, ctypes.c_uint64]
            self.lib.prof_drain_json.restype = ctypes.c_void_p  # char*

            self.lib.prof_free.argtypes = [ctypes.c_void_p]
            self.lib.prof_free.restype = None

            self.lib.prof_add_kernel_pattern.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            self.lib.prof_add_kernel_pattern.restype = None

            self.lib.prof_add_memcpy_kind.argtypes = [ctypes.c_uint64, ctypes.c_char_p]
            self.lib.prof_add_memcpy_kind.restype = None

            self.lib.prof_set_debug_mode.argtypes = [ctypes.c_uint32]
            self.lib.prof_set_debug_mode.restype = None

            self.lib.prof_get_debug_mode.argtypes = []
            self.lib.prof_get_debug_mode.restype = ctypes.c_uint32

            self.lib.prof_drain_debug.argtypes = [ctypes.c_uint32]
            self.lib.prof_drain_debug.restype = ctypes.c_void_p  # char*

            self._start_cupti_profiler()
            self._setup_events()
            self._start_drain_timer()
            self._start_debug_drain_timer()
            self._disabled = False
        except Exception as exc:
            logger.debug("Failed to setup CUPTI profiler", exc_info=True)
            self.lib = None

        logger.debug('CUPTI profiler setup complete')

    def shutdown(self):
        if self._disabled or not self.lib:
            return

        try:
            if self._drain_timer_thread:
                try:
                    self._drain_stop_event.set()
                    self._drain_timer_thread.join(timeout=1.0)
                except Exception:
                    pass
                finally:
                    self._drain_timer_thread = None

            if self._debug_drain_timer_thread:
                try:
                    self._debug_drain_stop_event.set()
                    self._debug_drain_timer_thread.join(timeout=1.0)
                except Exception:
                    pass
                finally:
                    self._debug_drain_timer_thread = None

            self._stop_cupti_profiler()
        except Exception:
            pass
        finally:
            self._disabled = True
            self.lib = None

    def _start_cupti_profiler(self) -> None:
        if not self.lib:
            raise RuntimeError("CUPTI profiler not initialized. Call setup() first.")

        self.lib.prof_start(
            self._resolution_ns,
            self._activity_window_ns,
            1 if self._debug_mode else 0,
        )

    def _stop_cupti_profiler(self) -> None:
        if not self.lib:
            return
        self.lib.prof_stop()

    def _setup_events(self) -> None:
        for event_name, patterns in KERNEL_PATTERNS:
            for pattern in patterns:
                self.add_kernel_pattern(pattern, event_name)

        for memcpy_kind in MEMCPY_KINDS:
            self.add_memcpy_kind(memcpy_kind, memcpy_kind)

    def _next_event_id(self) -> int:
        with self._current_event_id_lock:
            self._current_event_id += 1
            return self._current_event_id

    def add_kernel_pattern(self, pattern: str, event_name: str) -> None:
        descriptor = dict(category="gpu.compute", event_name=event_name, statistic="duration", unit="ns")
        duration_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        descriptor = dict(category="gpu.compute", event_name=event_name, statistic="call_count")
        calls_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        event_id = self._next_event_id()

        try:
            self.lib.prof_add_kernel_pattern(event_id, pattern.encode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to add kernel pattern", exc_info=True)
            return

        self._fields[event_id] = EventFields(
            duration_field_id=duration_field_id,
            calls_field_id=calls_field_id,
            bytes_field_id=None,
        )

    def add_memcpy_kind(self, memcpy_kind: str, event_name: str) -> None:
        descriptor = dict(category="gpu.memory", event_name=event_name, statistic="duration", unit="ns")
        duration_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        descriptor = dict(category="gpu.memory", event_name=event_name, statistic="call_count")
        calls_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        descriptor = dict(category="gpu.memory", event_name=event_name, statistic="bytes", unit="bytes")
        bytes_field_id = graphsignal._ticker.add_counter_profile_field(descriptor=descriptor)

        event_id = self._next_event_id()

        try:
            self.lib.prof_add_memcpy_kind(event_id, memcpy_kind.encode("utf-8"))
        except Exception as exc:
            logger.debug("Failed to add memcpy kind", exc_info=True)
            return

        self._fields[event_id] = EventFields(
            duration_field_id=duration_field_id,
                calls_field_id=calls_field_id,
                bytes_field_id=bytes_field_id,
            )

    def _cupti_activity_drain(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        if not self.lib:
            return {
                "dropped_cupti": 0,
                "buckets": [],
            }

        ptr = self.lib.prof_drain_json(int(start_ts), int(end_ts))
        if not ptr:
            return {
                "dropped_cupti": 0,
                "buckets": [],
            }

        try:
            s = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8")
            return json.loads(s)
        finally:
            self.lib.prof_free(ptr)


    def _start_drain_timer(self):
        try:
            self._last_drain_bucket_ts = _align_down(
                time.time_ns() - self._activity_window_ns,
                self._resolution_ns,
            )
        except Exception:
            self._last_drain_bucket_ts = 0

        self._drain_stop_event = threading.Event()

        def _drain_loop():
            while not self._drain_stop_event.wait(self._activity_window_ns / 1e9 / 10):
                try:
                    now_ns = time.time_ns()
                    from_bucket_ts = self._last_drain_bucket_ts + self._resolution_ns
                    to_bucket_ts = _align_down(
                        now_ns - self._activity_window_ns,
                        self._resolution_ns,
                    )
                    if to_bucket_ts <= from_bucket_ts:
                        continue

                    buckets = self._cupti_activity_drain(from_bucket_ts, to_bucket_ts)
                    self._last_drain_bucket_ts = max(0, to_bucket_ts - self._resolution_ns)

                    self._convert_to_profile(buckets['buckets'])
                except Exception as exc:
                    logger.error('Error in drain timer: %s', exc, exc_info=True)

        self._drain_timer_thread = threading.Thread(target=_drain_loop, daemon=True)
        self._drain_timer_thread.start()

    def _start_debug_drain_timer(self) -> None:
        if not self.lib:
            return

        self._debug_drain_stop_event = threading.Event()

        def _debug_drain_loop() -> None:
            while not self._debug_drain_stop_event.wait(1.0):
                try:
                    if self._debug_mode:
                        self._drain_native_debug_logs()
                except Exception as exc:
                    logger.error("Error in debug drain timer: %s", exc, exc_info=True)

        self._debug_drain_timer_thread = threading.Thread(target=_debug_drain_loop, daemon=True)
        self._debug_drain_timer_thread.start()

    def _convert_to_profile(self, buckets) -> None:
        for bucket_data in buckets:
            profile = {}
            bucket_ts = bucket_data['bucket_ts']
            for event_id, eb in bucket_data['events'].items():
                fields = self._fields.get(int(event_id))
                if not fields:
                    continue

                num_running = int(eb.get("num_running", 0) or 0)
                num_exited = int(eb.get("num_exited", 0) or 0)
                enter_offset_ns = int(eb.get("enter_offset_ns", 0) or 0)
                exit_offset_ns = int(eb.get("exit_offset_ns", 0) or 0)
                bytes_val = int(eb.get("bytes", 0) or 0)

                if num_running > 0 or exit_offset_ns > 0:
                    duration = self._resolution_ns * num_running - enter_offset_ns + exit_offset_ns
                    duration = max(0, duration)
                    if fields.duration_field_id and duration > 0:
                        profile[fields.duration_field_id] = duration
                    num_calls = num_running + num_exited
                    if fields.calls_field_id and num_calls > 0:
                        profile[fields.calls_field_id] = num_calls
                    if fields.bytes_field_id and bytes_val > 0:
                        profile[fields.bytes_field_id] = bytes_val

            if len(profile) > 0:
                graphsignal._ticker.update_profile(name=self._profile_name, profile=profile, measurement_ts=bucket_ts)

    def _drain_native_debug_logs(self) -> None:
        if not self.lib:
            return

        ptr = None
        try:
            ptr = self.lib.prof_drain_debug(200)
            if not ptr:
                return
            s = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", errors="replace")
        finally:
            if ptr:
                try:
                    self.lib.prof_free(ptr)
                except Exception:
                    pass

        if not s:
            return

        try:
            entries = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(entries, list):
            return

        ticker = graphsignal._ticker
        log_store = ticker.log_store()
        for item in entries:
            if not isinstance(item, dict):
                continue
            ts = int(item["ts"])
            msg = item['msg']
            if not msg:
                continue
            msg = str(msg)[:1024]
            log_store.log_sdk_message(
                level="debug",
                message=f"[cupti] {msg}",
                timestamp_ns=ts,
            )


def _align_down(ts_ns: int, resolution_ns: int) -> int:
    return (ts_ns // resolution_ns) * resolution_ns


def _packaged_cupti_so_path() -> Optional[str]:
    try:
        from importlib import resources

        cuda_major = _detect_cuda_major()
        arch = _detect_arch_tag()

        if cuda_major is None:
            return None

        candidate = resources.files("graphsignal").joinpath(
            "_native", f"{arch}-cu{cuda_major}", "libgscuptiprof.so"
        )
        with resources.as_file(candidate) as fp:
            if fp.exists():
                return str(fp)
            else:
                logger.debug("CUPTI profiler shared library not found for path: %s", candidate)
    except Exception:
        pass

    return None


def _detect_arch_tag() -> str:
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):
        return "arm64"
    return "amd64"


def _detect_cuda_major() -> Optional[int]:
    for soname in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11"):
        try:
            cudart = ctypes.CDLL(soname)
            fn = cudart.cudaRuntimeGetVersion
            fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
            fn.restype = ctypes.c_int
            v = ctypes.c_int(0)
            rc = fn(ctypes.byref(v))
            if rc == 0 and v.value > 0:
                return int(v.value // 1000)
        except Exception:
            continue

    try:
        cuda = ctypes.CDLL("libcuda.so.1")
        fn = cuda.cuDriverGetVersion
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        v = ctypes.c_int(0)
        rc = fn(ctypes.byref(v))
        if rc == 0 and v.value > 0:
            return int(v.value // 1000)
    except Exception:
        pass

    try:
        import torch  # type: ignore
        s = getattr(torch.version, "cuda", None)
        if isinstance(s, str):
            m = re.match(r"^(\d+)", s.strip())
            if m:
                return int(m.group(1))
    except Exception:
        pass

    return None


def _ensure_cuda_injection64_path() -> Optional[str]:
    if not sys.platform.startswith("linux"):
        return None

    existing = os.getenv("CUDA_INJECTION64_PATH")
    if existing:
        return existing

    p = _packaged_cupti_so_path()
    if not p:
        return None

    os.environ["CUDA_INJECTION64_PATH"] = p
    return p


def _default_cupti_profiler_so() -> Optional[str]:
    p = os.getenv("CUDA_INJECTION64_PATH")
    if p:
        return p

    p = _packaged_cupti_so_path()
    if p:
        return p

    if os.path.exists("./libgscuptiprof.so"):
        return "./libgscuptiprof.so"

    return None


def _best_effort_load_libcupti() -> None:
    for name in ("libcupti.so", "libcupti.so.13", "libcupti.so.12"):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue

    cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"
    candidates: list[str] = []
    for base in (
        os.path.join(cuda_home, "extras", "CUPTI", "lib64"),
        os.path.join(cuda_home, "extras", "CUPTI", "lib"),
    ):
        candidates.extend([
            os.path.join(base, "libcupti.so"),
            os.path.join(base, "libcupti.so.13"),
            os.path.join(base, "libcupti.so.12"),
        ])
    for p in candidates:
        try:
            ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue


KERNEL_PATTERNS = [
    ("nccl", [
        "nccl",
        "allreduce", 
        "all_reduce",
        "reducescatter", 
        "reduce_scatter",
        "allgather", 
        "all_gather",
        "broadcast",
    ]),
    ("matmul_gemm", [
        "cublas",
        "cublaslt",
        "cublasltmatmul",
        "cutlass",
        "gemm",
    ]),
    ("attention_flash", [
        "flash_attn",
        "flashattn",
        "fmha",
        "paged_attention",
        "flashinfer",
    ]),
    ("qkv_proj", [
        "qkv_proj",
        "fused_qkv",
        "linear_qkv",
    ]),
    ("kv_cache", [
        "reshape_and_cache",
        "paged_kv",
        "cache_kv",
        "write_kv",
        "scatter_kv",
        "gather_kv",
    ]),
    ("layernorm_rmsnorm", [
        "fused_layer_norm",
        "layer_norm",
        "layernorm",
        "rms_norm",
        "rmsnorm",
    ]),
    ("softmax", [
        "scaled_masked_softmax",
        "masked_softmax",
        "softmax",
    ]),
    ("rope_rotary", [
        "rotary_embedding",
        "apply_rotary",
        "rotary",
    ]),
    ("activation", [
        "silu",
        "swish",
        "gelu",
    ]),
    ("elementwise", [
        "vectorized_elementwise",
        "elementwise",
        "pointwise",
        "fused_bias",
        "bias_act",
    ]),
    ("embedding", [
        "index_select",
        "embedding",
    ]),
    ("sampling", [
        "multinomial",
        "top_p",
        "argmax",
        "sampling",
    ]),
    ("triton", [
        "triton",
    ]),
]


MEMCPY_KINDS = [
    "memcpy_host_to_device",
    "memcpy_device_to_host",
    "memcpy_device_to_device",
    "memcpy_host_to_host",
    "memcpy_peer_to_peer",
    "memcpy_host_to_array",
    "memcpy_array_to_host",
    "memcpy_array_to_array",
    "memcpy_array_to_device",
    "memcpy_device_to_array",
    "memcpy_other",
]