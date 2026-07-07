import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import patch

# Configure ROCm env vars (ROCP_TOOL_LIBRARIES, LD_LIBRARY_PATH) at module
# import. The test process itself never loads HIP/ROCm — it spawns a workload
# subprocess that inherits these env vars and loads the rocprofiler-sdk tool
# library (libgsrocmprof.so) there. This mirrors production (graphsignal-run):
# the tool lib is loaded in the workload, the recorder lives in a separate
# process reading /dev/shm.
try:
    from graphsignal.profilers.rocm_profiler import RocmProfiler as _RocmProfilerHelper
    # The native tool lib reads GRAPHSIGNAL_DEBUG directly; set it for the
    # workload subprocess so its ROCm debug output lands in the shm JSON.
    os.environ['GRAPHSIGNAL_DEBUG'] = '1'
    _ROCM_ENV_READY = (
        sys.platform.startswith('linux')
        and _RocmProfilerHelper.setup_env_vars()
    )
except Exception:
    _ROCM_ENV_READY = False

# Set GRAPHSIGNAL_TEST_REQUIRE_ROCM=1 on a real ROCm box to turn the GPU tests'
# skip conditions into hard failures. This is the switch that makes "did the GPU
# tests actually run?" unambiguous: a green run with this flag set proves they
# executed (a missing tool lib / torch would fail loudly instead of skipping).
_REQUIRE_ROCM = os.environ.get('GRAPHSIGNAL_TEST_REQUIRE_ROCM') == '1'


def _clean_probe_env():
    # Strip the injection env vars set suite-wide by module imports —
    # ROCP_TOOL_LIBRARIES (from `import graphsignal`) and CUDA_INJECTION64_PATH
    # (from test_cupti_recorder's import). A bare `import torch` for probing must
    # not init HIP with our tool (or the unrelated CUPTI lib) loaded, or it could
    # exit non-zero for reasons unrelated to real torch+HIP availability.
    env = {**os.environ}
    env.pop('ROCP_TOOL_LIBRARIES', None)
    env.pop('CUDA_INJECTION64_PATH', None)
    return env


def _candidate_pythons():
    # pytest often runs under a poetry/venv that lacks the large, ROCm-specific
    # torch build, while the box's torch lives in a different interpreter (the
    # one importable from the shell). Try, in order: an explicit override, the
    # pytest interpreter, then python3/python on PATH.
    seen = []
    for c in (os.environ.get('GRAPHSIGNAL_TEST_PYTHON'),
              sys.executable,
              shutil.which('python3'),
              shutil.which('python')):
        if c and c not in seen:
            seen.append(c)
    return seen


_WORKLOAD_PYTHON = None  # cached: resolved torch-capable interpreter, or False


def _resolve_workload_python():
    """Return the path of an interpreter that can `import torch`, or None.

    Set GRAPHSIGNAL_TEST_PYTHON to force a specific interpreter (e.g. the ROCm
    torch one) when pytest itself runs under a torch-less venv."""
    global _WORKLOAD_PYTHON
    if _WORKLOAD_PYTHON is not None:
        return _WORKLOAD_PYTHON or None
    env = _clean_probe_env()
    for cand in _candidate_pythons():
        try:
            proc = subprocess.run([cand, '-c', 'import torch'],
                                  timeout=120, capture_output=True, text=True, env=env)
            if proc.returncode == 0:
                _WORKLOAD_PYTHON = cand
                return cand
        except Exception:
            continue
    _WORKLOAD_PYTHON = False
    return None


def _torch_hip_available_in_subprocess():
    # Probe via a subprocess so the test process never imports torch / inits HIP.
    # On ROCm, torch exposes the HIP device through the torch.cuda namespace.
    #
    # Returns (available: bool, detail: str). `detail` explains a False result so
    # a skip/fail is diagnosable on the box (torch missing from the pytest env
    # vs no visible GPU vs probe crash) instead of the opaque "torch+HIP not
    # available".
    py = _resolve_workload_python()
    if not py:
        return False, (
            "torch not importable by any candidate interpreter: %s — set "
            "GRAPHSIGNAL_TEST_PYTHON to a torch-enabled python (pytest is likely "
            "running under a venv without the ROCm torch build)" % _candidate_pythons())

    code = (
        "import sys\n"
        "try:\n"
        "    import torch\n"
        "except Exception as e:\n"
        "    sys.stderr.write('torch import failed: %r' % (e,)); sys.exit(2)\n"
        "hip = getattr(torch.version, 'hip', None)\n"
        "avail = torch.cuda.is_available()\n"
        "sys.stderr.write('torch=%s hip=%s cuda_available=%s device_count=%s' % ("
        "getattr(torch, '__version__', '?'), hip, avail,"
        " (torch.cuda.device_count() if avail else 0)))\n"
        "sys.exit(0 if avail else 1)\n"
    )
    env = _clean_probe_env()
    try:
        proc = subprocess.run([py, '-c', code],
                              timeout=60, capture_output=True, text=True, env=env)
        detail = 'python=%s %s' % (py, (proc.stderr or '').strip())
        if proc.returncode != 0:
            logger.debug("torch+HIP probe rc=%s: %s", proc.returncode, detail)
        return proc.returncode == 0, detail
    except Exception as e:
        return False, 'python=%s %r' % (py, e)


import graphsignal
import graphsignal.sdk
from graphsignal.recorders.rocm_recorder import (
    RocmRecorder, _EventFields, make_op_name, extract_op_name,
    _OP_NAME_FINGERPRINT_SEP, _short_fingerprint)


_WORKLOAD_SCRIPT = textwrap.dedent('''
    import os, sys, time
    import torch

    SIZE = int(os.environ.get('ROCM_TEST_SIZE', '1024'))
    ITERS = int(os.environ.get('ROCM_TEST_ITERS', '20'))

    a = torch.randn((SIZE, SIZE), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE, SIZE), device='cuda', dtype=torch.float16)

    # Warm up so the kernel names are seen.
    for _ in range(2):
        c = a @ b
        c = torch.relu(c)
    torch.cuda.synchronize()

    start_ns = time.perf_counter_ns()
    for _ in range(ITERS):
        c = a @ b
        c = torch.relu(c)
    torch.cuda.synchronize()
    took_ns = time.perf_counter_ns() - start_ns

    print(f'PID={os.getpid()}', flush=True)
    print(f'TOOK_NS={took_ns}', flush=True)
    print('READY', flush=True)

    # Stay alive long enough for the ROCm flush thread to write shm files
    # and for the recorder in the parent to drain them.
    time.sleep(float(os.environ.get('ROCM_TEST_LINGER_SEC', '3.0')))
''')


def _spawn_workload(env_overrides=None, timeout=30.0):
    env = {**os.environ}
    # Keep ROCP_TOOL_LIBRARIES (that's how the profiler is injected into the
    # workload), but drop CUDA_INJECTION64_PATH — it's left on the parent env by
    # test_cupti_recorder's import and is irrelevant/perturbing for a ROCm torch
    # workload.
    env.pop('CUDA_INJECTION64_PATH', None)
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    # Run the workload under a torch-capable interpreter (see
    # _resolve_workload_python); fall back to the pytest interpreter.
    py = _resolve_workload_python() or sys.executable
    proc = subprocess.Popen(
        [py, '-c', _WORKLOAD_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    metadata = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        if line == 'READY':
            return proc, metadata
        m = re.match(r'^(\w+)=(.+)$', line)
        if m:
            metadata[m.group(1)] = m.group(2)
    proc.terminate()
    stderr = proc.stderr.read() if proc.stderr else ''
    raise RuntimeError(f'Workload did not become READY; stderr={stderr!r}')

logger = logging.getLogger('graphsignal')


def _kernel_event(op_name, *, cumtime=1_000_000, ncalls=1, nerrors=0, bytes=0, **extra):
    # Mirrors the per-event JSON the recorder consumes: event_name plus the
    # already-computed stats. `extra` carries optional fields such as
    # cumtime_occupancy / host_sync_wait.
    #
    # The native ROCm lib prefixes kernel event names with `kernel:` (so the
    # recorder detects the kind positively, like memcpy_/sync:). Auto-apply that
    # prefix here for kernel names so kernel test bodies stay concise; names that
    # already carry a recognized non-kernel prefix are passed through.
    event_name = op_name
    if op_name and not op_name.startswith(('kernel:', 'memcpy_', 'sync:')):
        event_name = 'kernel:' + op_name
    event = {
        'event_name': event_name,
        'cumtime': cumtime,
        'ncalls': ncalls,
        'nerrors': nerrors,
        'bytes': bytes,
    }
    event.update(extra)
    return event


class RocmRecorderOpNameTest(unittest.TestCase):
    def test_tensile_gemm_keeps_layout_tile_and_mi(self):
        # Tensile (rocBLAS / hipBLASLt) GEMM kernels keep layout + macro-tile
        # (MT) + matrix-instruction (MI) shape; the scheduling/tuning flag soup
        # is dropped so the op name is short and readable.
        raw = ('Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT128x16x128_'
               'MI16x16x1_SN_LDSB1_ULSGRO0_WG32_4_2')
        self.assertEqual(extract_op_name(raw),
                         'Cijk_Alik_Bljk_MT128x16x128_MI16x16x1')
        op = make_op_name(raw)
        self.assertRegex(op, r'^Cijk_Alik_Bljk_MT128x16x128_MI16x16x1@[0-9a-f]{4}$')

    def test_tensile_flag_variants_share_family(self):
        # Same layout/tile/MI, differing only in a scheduling flag (ULSGRO0 vs
        # ULSGRO1) collapse to the same readable family.
        a = 'Cijk_Alik_Bljk_HA_MT128x16x128_MI16x16x1_ULSGRO0_WG32_4_2'
        b = 'Cijk_Alik_Bljk_HA_MT128x16x128_MI16x16x1_ULSGRO1_WG32_4_2'
        self.assertEqual(extract_op_name(a), 'Cijk_Alik_Bljk_MT128x16x128_MI16x16x1')
        self.assertEqual(extract_op_name(a), extract_op_name(b))
        # No grouping/aggregation: distinct raw symbols still get distinct
        # fingerprints (and thus distinct op names / fields).
        self.assertNotEqual(make_op_name(a), make_op_name(b))

    def test_tensile_distinct_tiles_distinct_family(self):
        # Different macro-tiles are genuinely different kernels -> distinct family.
        a = 'Cijk_Alik_Bljk_HA_MT128x16x128_MI16x16x1_ULSGRO0'
        b = 'Cijk_Alik_Bljk_HA_MT32x16x256_MI16x16x1_ULSGRO0'
        self.assertEqual(extract_op_name(a), 'Cijk_Alik_Bljk_MT128x16x128_MI16x16x1')
        self.assertEqual(extract_op_name(b), 'Cijk_Alik_Bljk_MT32x16x256_MI16x16x1')
        self.assertNotEqual(extract_op_name(a), extract_op_name(b))

    def test_tensile_no_tile_falls_back_to_layout(self):
        # A Cijk symbol without an MT token degrades to just the layout.
        self.assertEqual(extract_op_name('Cijk_Ailk_Bljk_HB'), 'Cijk_Ailk_Bljk')

    def test_itanium_mangled_unwrap_last_identifier(self):
        raw = '_ZN4rocm11busy_kernelEv'
        self.assertEqual(extract_op_name(raw), 'busy_kernel')
        self.assertRegex(make_op_name(raw), r'^busy_kernel@[0-9a-f]{4}$')

    def test_plain_symbol_passthrough_with_fingerprint(self):
        raw = 'naive_conv_nonpacked_fwd'
        op = make_op_name(raw)
        self.assertTrue(op.startswith('naive_conv_nonpacked_fwd' + _OP_NAME_FINGERPRINT_SEP))

    def test_empty_name(self):
        self.assertEqual(extract_op_name(''), '')
        self.assertEqual(make_op_name(''), '')


class RocmRecorderTest(unittest.TestCase):
    def setUp(self):
        graphsignal.sdk.configure(api_key='k1', debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_setup_skips_on_non_linux(self):
        recorder = RocmRecorder(pid=12345)
        with patch('graphsignal.recorders.rocm_recorder.sys.platform', 'darwin'):
            recorder.setup()
        self.assertTrue(recorder._disabled)
        self.assertIsNone(recorder._drain_thread)

    def test_setup_skips_without_pid(self):
        recorder = RocmRecorder(pid=None)
        with patch('graphsignal.recorders.rocm_recorder.sys.platform', 'linux'):
            recorder.setup()
        self.assertTrue(recorder._disabled)
        self.assertIsNone(recorder._drain_thread)

    def test_drain_logs_workload_entries_from_shm_json(self):
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as shm_dir:
            recorder = RocmRecorder(pid=99999)
            with patch.object(recorder, '_shm_dir', return_value=shm_dir):
                payload = {
                    'buckets': [],
                    'log': [
                        {'ts': 1700000000000000000, 'msg': 'graphsignal: hello from rocm\n'},
                        {'ts': 1700000000000000001, 'msg': 'graphsignal: second line\n'},
                    ],
                }
                with open(os.path.join(shm_dir, 'rocm_1.json'), 'w') as f:
                    _json.dump(payload, f)

                with self.assertLogs('graphsignal', level='DEBUG') as cm:
                    recorder._rocm_activity_drain()

        joined = '\n'.join(cm.output)
        self.assertIn('hello from rocm', joined)
        self.assertIn('second line', joined)

    def test_drain_reads_context_from_shm_json(self):
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as shm_dir:
            recorder = RocmRecorder(pid=99998)
            with patch.object(recorder, '_shm_dir', return_value=shm_dir):
                payload = {
                    'buckets': [],
                    'log': [],
                    'context': {'rank': '2', 'local_rank': '0', 'empty': ''},
                }
                with open(os.path.join(shm_dir, 'rocm_1.json'), 'w') as f:
                    _json.dump(payload, f)
                recorder._rocm_activity_drain()

        self.assertEqual(recorder._context.get('rank'), '2')
        self.assertEqual(recorder._context.get('local_rank'), '0')
        self.assertNotIn('empty', recorder._context)

    def test_lazy_field_creation_for_kernel_and_memcpy(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 1000,
            'events': {
                '12345': _kernel_event('Cijk_Ailk_Bljk_HB_MT128x128x16',
                                       cumtime=5_000_000),
                '99999': _kernel_event('memcpy_host_to_device',
                                       cumtime=2_000_000, bytes=1024),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        self.assertIn(12345, recorder._fields)
        self.assertIn(99999, recorder._fields)

        kernel_fields = recorder._fields[12345]
        self.assertIsNotNone(kernel_fields.cumtime_field_id)
        self.assertIsNotNone(kernel_fields.ncalls_field_id)
        self.assertIsNone(kernel_fields.bytes_field_id)
        self.assertIsNotNone(kernel_fields.cumtime_occupancy_field_id)

        memcpy_fields = recorder._fields[99999]
        self.assertIsNotNone(memcpy_fields.bytes_field_id)
        self.assertIsNone(memcpy_fields.cumtime_occupancy_field_id)

        mock_update.assert_called_once()
        call_kwargs = mock_update.call_args[1]
        self.assertEqual(call_kwargs['measurement_ts'], 1000)
        profile = call_kwargs['profile']
        self.assertIn(kernel_fields.cumtime_field_id, profile)
        self.assertIn(memcpy_fields.bytes_field_id, profile)

    def test_skips_event_with_missing_op_name(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 2000,
            'events': {'77777': _kernel_event('')},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        self.assertNotIn(77777, recorder._fields)
        mock_update.assert_not_called()

    def test_skips_unrecognized_prefix(self):
        # The native ROCm lib always prefixes event names (kernel:/memcpy_/sync:).
        # ROCm has no memset/graph event kinds, so such names are unexpected and
        # must be skipped rather than mis-categorized.
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 2100,
            'events': {
                '61001': {'event_name': 'memset_device', 'cumtime': 1_000_000,
                          'ncalls': 1, 'nerrors': 0, 'bytes': 4096},
                '61002': {'event_name': 'graph:kernel[name=_Z6mykernelv,calls=1];',
                          'cumtime': 2_000_000, 'ncalls': 1, 'nerrors': 0, 'bytes': 0},
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        self.assertNotIn(61001, recorder._fields)
        self.assertNotIn(61002, recorder._fields)
        mock_update.assert_not_called()

    def test_reuses_existing_fields(self):
        recorder = RocmRecorder()

        existing = _EventFields(cumtime_field_id=42, ncalls_field_id=43)
        recorder._fields[55555] = existing

        buckets = [{
            'bucket_ts': 3000,
            'events': {'55555': _kernel_event('some_kernel', cumtime=10_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'), \
             patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field') as mock_add_field:
            recorder._convert_to_profile(buckets)

        mock_add_field.assert_not_called()
        self.assertIs(recorder._fields[55555], existing)

    def test_ncalls_emitted(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 5000,
            'events': {'11111': _kernel_event('some_gemm_kernel', ncalls=5)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        profile = mock_update.call_args[1]['profile']
        fields = recorder._fields[11111]
        self.assertEqual(profile[fields.ncalls_field_id], 5)

    def test_top30_kernel_filtering(self):
        recorder = RocmRecorder()

        events = {}
        for i in range(35):
            events[str(10000 + i)] = _kernel_event(f'kernel_{i:02d}',
                                                   cumtime=(i + 1) * 1000)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 6000, 'events': events}])

        self.assertEqual(len(recorder._fields), 30)
        for i in range(5, 35):
            self.assertIn(10000 + i, recorder._fields)
        for i in range(5):
            self.assertNotIn(10000 + i, recorder._fields)

    def test_kernel_descriptor_carries_flat_category_and_kernel_name(self):
        recorder = RocmRecorder()

        raw = 'Cijk_Ailk_Bljk_HB_MT128x128x16_SE'
        events = {
            '11111': _kernel_event(raw, cumtime=5_000_000),
            '22222': _kernel_event('memcpy_host_to_device',
                                   cumtime=1_000_000, bytes=4096),
            '33333': _kernel_event('sync:hipDeviceSynchronize', cumtime=500_000),
        }

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_cf, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 9000, 'events': events}])

        descriptors = [c.kwargs['descriptor'] for c in mock_cf.call_args_list]

        # All kernels carry the flat `rocm.kernel` category; the platform
        # refines it from kernel_name at query time.
        kernel_descs = [d for d in descriptors if d['category'] == 'rocm.kernel']
        self.assertTrue(kernel_descs, 'expected at least one rocm.kernel descriptor')
        for d in kernel_descs:
            self.assertEqual(d['kernel_name'], raw)
            self.assertEqual(d['op_name'], make_op_name(raw))
            self.assertNotIn('kernel:', d['op_name'])

        # Memcpy op_name is its kind string unchanged (already unique).
        memcpy_descs = [d for d in descriptors if d['op_name'] == 'memcpy_host_to_device']
        self.assertTrue(memcpy_descs)
        for d in memcpy_descs:
            self.assertEqual(d['category'], 'rocm.memcpy')
            self.assertNotIn('kernel_name', d)

        # Sync display name has the "sync:" prefix stripped.
        sync_descs = [d for d in descriptors if d['op_name'] == 'hipDeviceSynchronize']
        self.assertTrue(sync_descs)
        for d in sync_descs:
            self.assertEqual(d['category'], 'rocm.sync')
            self.assertNotIn('kernel_name', d)

    def test_memcpy_always_kept(self):
        recorder = RocmRecorder()

        events = {}
        for i in range(35):
            events[str(20000 + i)] = _kernel_event(f'kernel_{i:02d}',
                                                   cumtime=(i + 1) * 1000)
        events['30001'] = _kernel_event('memcpy_host_to_device', cumtime=500, bytes=512)
        events['30002'] = _kernel_event('memcpy_device_to_host', cumtime=300, bytes=256)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile([{'bucket_ts': 7000, 'events': events}])

        self.assertIn(30001, recorder._fields)
        self.assertIn(30002, recorder._fields)
        profile = mock_update.call_args[1]['profile']
        self.assertIn(recorder._fields[30001].bytes_field_id, profile)
        self.assertIn(recorder._fields[30002].bytes_field_id, profile)
        self.assertEqual(len(recorder._fields), 32)  # 30 kernels + 2 memcpy

    def test_persistent_top_n_across_buckets(self):
        recorder = RocmRecorder()

        bucket_a = {str(40000 + i): _kernel_event(f'kernel_{i:02d}',
                                                  cumtime=(i + 1) * 100)
                    for i in range(30)}
        bucket_b = {'49999': _kernel_event('kernel_new_small', cumtime=10)}

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile([{'bucket_ts': 8000, 'events': bucket_a}])
            recorder._convert_to_profile([{'bucket_ts': 8100, 'events': bucket_b}])

        for i in range(30):
            self.assertIn(40000 + i, recorder._top_kernel_ids)
        self.assertNotIn(49999, recorder._top_kernel_ids)
        self.assertNotIn(49999, recorder._fields)
        self.assertEqual(mock_update.call_count, 1)  # only bucket A had eligible events

    def test_sync_category(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 6000,
            'events': {
                '77777': _kernel_event('sync:hipDeviceSynchronize', cumtime=3_000_000),
                '88888': _kernel_event('sync:hipStreamSynchronize', cumtime=1_000_000),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_add, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        categories = {c[1].get('descriptor', {}).get('category') for c in mock_add.call_args_list}
        self.assertIn('rocm.sync', categories)
        self.assertNotIn('rocm.memcpy', categories)

    def test_sync_has_no_bytes_field(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 6100,
            'events': {'77778': _kernel_event('sync:hipEventSynchronize', cumtime=2_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertIsNone(recorder._fields[77778].bytes_field_id)

    def test_sync_always_kept_outside_top_n(self):
        recorder = RocmRecorder()

        events = {str(80000 + i): _kernel_event(f'kernel_{i:02d}',
                                                cumtime=(i + 1) * 1000)
                  for i in range(35)}
        events['90001'] = _kernel_event('sync:hipStreamWaitEvent', cumtime=1)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 7000, 'events': events}])

        self.assertIn(90001, recorder._fields)

    def test_sync_excluded_from_top_n_accumulation(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 7100,
            'events': {'91001': _kernel_event('sync:hipDeviceSynchronize', cumtime=9_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertNotIn(91001, recorder._kernel_cumtime_totals)

    def test_host_sync_wait_emitted_for_kernel(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 9000,
            'events': {
                '11111': _kernel_event('some_gemm_kernel',
                                       cumtime=5_000_000,
                                       host_sync_wait=2_500_000),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        fields = recorder._fields[11111]
        self.assertIsNotNone(fields.host_sync_wait_field_id)
        profile = mock_update.call_args[1]['profile']
        self.assertEqual(profile[fields.host_sync_wait_field_id], 2_500_000)

    def test_host_sync_wait_emitted_for_memcpy(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 9100,
            'events': {
                '22222': _kernel_event('memcpy_host_to_device',
                                       cumtime=1_000_000, bytes=4096,
                                       host_sync_wait=800_000),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        fields = recorder._fields[22222]
        self.assertIsNotNone(fields.host_sync_wait_field_id)
        profile = mock_update.call_args[1]['profile']
        self.assertEqual(profile[fields.host_sync_wait_field_id], 800_000)

    def test_host_sync_wait_not_registered_for_sync_events(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 9200,
            'events': {'33333': _kernel_event('sync:hipDeviceSynchronize', cumtime=500_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        # Sync events don't get a host_sync_wait field — they're the wait source
        # themselves, not something the CPU was blocked on.
        self.assertIsNone(recorder._fields[33333].host_sync_wait_field_id)

    def test_host_sync_wait_omitted_when_zero(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 9300,
            'events': {
                '44444': _kernel_event('some_gemm_kernel', cumtime=5_000_000),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        fields = recorder._fields[44444]
        self.assertIsNotNone(fields.host_sync_wait_field_id)
        profile = mock_update.call_args[1]['profile']
        self.assertNotIn(fields.host_sync_wait_field_id, profile)

    def test_occupancy_emitted(self):
        recorder = RocmRecorder()

        # cumtime_occupancy is computed in the native lib; the recorder passes it
        # through directly onto the cumtime_occupancy field for kernels.
        buckets = [{
            'bucket_ts': 1000,
            'events': {'12345': _kernel_event('some_gemm_kernel',
                                              cumtime=5_000_000,
                                              cumtime_occupancy=2_500_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_counter, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        stats = [c[1].get('descriptor', {}).get('statistic') for c in mock_counter.call_args_list]
        self.assertIn('cumtime_occupancy', stats)

        profile = mock_update.call_args[1]['profile']
        fields = recorder._fields[12345]
        self.assertEqual(profile[fields.cumtime_occupancy_field_id], 2_500_000)

    def test_occupancy_omitted_when_absent(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 1100,
            'events': {'12346': _kernel_event('some_kernel', cumtime=5_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        profile = mock_update.call_args[1]['profile']
        fields = recorder._fields[12346]
        self.assertIsNotNone(fields.cumtime_occupancy_field_id)
        self.assertNotIn(fields.cumtime_occupancy_field_id, profile)

    def test_occupancy_not_emitted_for_memcpy(self):
        recorder = RocmRecorder()

        buckets = [{
            'bucket_ts': 1200,
            'events': {'99991': _kernel_event('memcpy_host_to_device',
                                              cumtime=1_000_000, bytes=1024)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertIsNone(recorder._fields[99991].cumtime_occupancy_field_id)

    def test_profile_tags_include_process_pid(self):
        recorder = RocmRecorder(pid=4242)

        buckets = [{
            'bucket_ts': 1500,
            'events': {'10001': _kernel_event('some_kernel', cumtime=1_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        tags = mock_update.call_args[1]['tags']
        self.assertEqual(tags.get('process.pid'), '4242')

    def test_profile_tags_include_context_overrides(self):
        recorder = RocmRecorder(pid=4242)
        recorder._context = {'rank': '3', 'local_rank': '1', 'unknown_key': 'ignored'}

        buckets = [{
            'bucket_ts': 1600,
            'events': {'10002': _kernel_event('some_kernel', cumtime=1_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        tags = mock_update.call_args[1]['tags']
        self.assertEqual(tags.get('process.rank'), '3')
        self.assertEqual(tags.get('process.local_rank'), '1')
        self.assertNotIn('unknown_key', tags)

    #
    # GPU tests run the torch+HIP workload in a separate subprocess (the
    # rocprofiler-sdk tool lib is loaded into the subprocess via
    # ROCP_TOOL_LIBRARIES, which it inherits from this process's env). The
    # recorder runs in the test process, targeting the subprocess's pid —
    # mirroring graphsignal-run's workload+watcher split.
    #
    # The test process must NOT import torch / init HIP, otherwise the tool lib
    # would also load here.

    def _require_or_skip(self, cond, reason):
        # On a real ROCm box run with GRAPHSIGNAL_TEST_REQUIRE_ROCM=1 so an
        # unmet precondition fails loudly instead of silently skipping — that's
        # what proves the GPU tests actually ran.
        if cond:
            return
        if _REQUIRE_ROCM:
            self.fail(f"GRAPHSIGNAL_TEST_REQUIRE_ROCM=1 but {reason}")
        self.skipTest(reason)

    def test_end_to_end_torch_hip_and_drain(self):
        self._require_or_skip(_ROCM_ENV_READY, "ROCm env not set up (Linux + ROCm required)")
        hip_ok, hip_detail = _torch_hip_available_in_subprocess()
        self._require_or_skip(hip_ok, "torch+HIP not available: %s" % hip_detail)

        proc, meta = _spawn_workload(env_overrides={'ROCM_TEST_LINGER_SEC': '5.0'})
        try:
            workload_pid = int(meta['PID'])
            # Note: don't call setup() — the drain thread would race with our
            # manual drain below and consume buckets first.
            recorder = RocmRecorder(pid=workload_pid)

            # The workload's flush thread runs every ~1s and writes shm files;
            # the C++ side then cleans shm files older than 2 * window (~2s).
            # Drain ~1.2s after activity so the flush has run but the files
            # haven't been swept yet.
            time.sleep(1.2)

            with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
                raw = recorder._rocm_activity_drain()
                buckets = raw.get('buckets', [])
                recorder._convert_to_profile(buckets)

                rocm_profile_calls = 0
                for call in mock_update.call_args_list:
                    kwargs = call.kwargs
                    name = kwargs.get('name', call.args[0] if call.args else None)
                    profile = kwargs.get('profile', call.args[1] if len(call.args) >= 2 else None)
                    if name == 'profile.rocm' and isinstance(profile, dict) and profile:
                        rocm_profile_calls += 1

                logger.debug("rocm e2e (pid=%s): drained %d bucket(s), %d profile.rocm "
                             "update(s), %d event field(s)", workload_pid, len(buckets),
                             rocm_profile_calls, len(recorder._fields))

                if rocm_profile_calls == 0:
                    stderr = proc.stderr.read() if proc.stderr else ''
                    self.fail(
                        f"expected at least one update_profile call for "
                        f"profile.rocm; subprocess stderr={stderr!r}"
                    )
        finally:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)

    def test_overhead(self):
        # Measure the ROCm tool library's overhead inside the workload by
        # comparing two subprocess runs of the same workload:
        #   - baseline: ROCP_TOOL_LIBRARIES unset (tool lib not loaded)
        #   - with profiler: ROCP_TOOL_LIBRARIES set (tool lib loaded)
        self._require_or_skip(_ROCM_ENV_READY, "ROCm env not set up (Linux + ROCm required)")
        hip_ok, hip_detail = _torch_hip_available_in_subprocess()
        self._require_or_skip(hip_ok, "torch+HIP not available: %s" % hip_detail)

        def _run(env_overrides):
            env_overrides = {'ROCM_TEST_SIZE': '4096',
                             'ROCM_TEST_ITERS': '1000',
                             'ROCM_TEST_LINGER_SEC': '0.1',
                             **env_overrides}
            proc, meta = _spawn_workload(env_overrides=env_overrides, timeout=180.0)
            try:
                proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)
                self.fail("Workload subprocess timed out")
            return int(meta['TOOK_NS'])

        took_ns_baseline = _run(env_overrides={'ROCP_TOOL_LIBRARIES': None})
        took_ns_with_profiler = _run(env_overrides={})

        overhead_pct = 100.0 * (took_ns_with_profiler - took_ns_baseline) / max(1, took_ns_baseline)
        overhead_per_iter_us = (took_ns_with_profiler - took_ns_baseline) / 1000 / 1e3

        logger.setLevel(logging.DEBUG)
        logger.debug("ROCm tool overhead=%.2f%% (baseline=%.1fms, with_profiler=%.1fms), "
                     "overhead_per_iter=%.1f us",
                     overhead_pct, took_ns_baseline / 1e6, took_ns_with_profiler / 1e6,
                     overhead_per_iter_us)

        self.assertTrue(overhead_pct < 5.0,
                        f"expected overhead < 5.0%, got {overhead_pct:.2f}%")


if __name__ == '__main__':
    unittest.main()
