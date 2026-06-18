import logging
import os
import re
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import patch

# Configure CUPTI env vars (CUDA_INJECTION64_PATH, LD_LIBRARY_PATH) at module
# import. The test process itself never loads CUDA — it spawns a workload
# subprocess that inherits these env vars and loads the injection library there.
# This mirrors production (graphsignal-run): the injection lib is loaded in the
# workload, the recorder lives in a separate process reading /dev/shm.
try:
    from graphsignal.profilers.cupti_profiler import CuptiProfiler as _CuptiProfilerHelper
    # Native injection lib reads GRAPHSIGNAL_DEBUG directly; set it for the
    # workload subprocess so its CUPTI debug output lands in the shm JSON.
    os.environ['GRAPHSIGNAL_DEBUG'] = '1'
    _CUPTI_ENV_READY = (
        sys.platform.startswith('linux')
        and _CuptiProfilerHelper.setup_env_vars()
    )
except Exception:
    _CUPTI_ENV_READY = False


def _torch_cuda_available_in_subprocess():
    # Probe via a subprocess so the test process never imports torch / inits CUDA.
    code = "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    try:
        proc = subprocess.run([sys.executable, '-c', code],
                              timeout=30, capture_output=True)
        return proc.returncode == 0
    except Exception:
        return False


import graphsignal
import graphsignal.sdk
from graphsignal.recorders.cupti_recorder import (
    CuptiRecorder, _EventFields, _OP_NAME_FINGERPRINT_SEP, _short_fingerprint)


_WORKLOAD_SCRIPT = textwrap.dedent('''
    import os, sys, time
    import torch

    SIZE = int(os.environ.get('CUPTI_TEST_SIZE', '1024'))
    ITERS = int(os.environ.get('CUPTI_TEST_ITERS', '20'))

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

    # Stay alive long enough for the CUPTI flush thread to write shm files
    # and for the recorder in the parent to drain them.
    time.sleep(float(os.environ.get('CUPTI_TEST_LINGER_SEC', '3.0')))
''')


def _spawn_workload(env_overrides=None, timeout=30.0):
    env = {**os.environ}
    if env_overrides:
        for k, v in env_overrides.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
    proc = subprocess.Popen(
        [sys.executable, '-c', _WORKLOAD_SCRIPT],
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
    # The native lib prefixes kernel event names with `kernel:` (so the recorder
    # detects kind positively, like memcpy_/sync_/memset_/graph:). Auto-apply
    # that prefix here for kernel names so existing kernel test bodies stay
    # valid; names that already carry a non-kernel prefix are passed through.
    event_name = op_name
    if op_name and not op_name.startswith((
            'kernel:', 'memcpy_', 'sync_', 'memset_', 'graph:')):
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


class CuptiRecorderTest(unittest.TestCase):
    def setUp(self):
        graphsignal.sdk.configure(api_key='k1', debug_mode=True)
        graphsignal.sdk.sdk()._auto_tick = False

    def tearDown(self):
        graphsignal.sdk.shutdown()

    def test_setup_skips_on_non_linux(self):
        recorder = CuptiRecorder(pid=12345)
        with patch('graphsignal.recorders.cupti_recorder.sys.platform', 'darwin'):
            recorder.setup()
        self.assertTrue(recorder._disabled)
        self.assertIsNone(recorder._drain_thread)

    def test_setup_skips_without_pid(self):
        recorder = CuptiRecorder(pid=None)
        with patch('graphsignal.recorders.cupti_recorder.sys.platform', 'linux'):
            recorder.setup()
        self.assertTrue(recorder._disabled)
        self.assertIsNone(recorder._drain_thread)

    def test_drain_logs_workload_entries_from_shm_json(self):
        import json as _json
        import tempfile

        with tempfile.TemporaryDirectory() as shm_dir:
            recorder = CuptiRecorder(pid=99999)
            with patch.object(recorder, '_shm_dir', return_value=shm_dir):
                payload = {
                    'buckets': [],
                    'log': [
                        {'ts': 1700000000000000000, 'msg': 'graphsignal: hello from cupti\n'},
                        {'ts': 1700000000000000001, 'msg': 'graphsignal: second line\n'},
                    ],
                }
                with open(os.path.join(shm_dir, 'cupti_1.json'), 'w') as f:
                    _json.dump(payload, f)

                with self.assertLogs('graphsignal', level='DEBUG') as cm:
                    recorder._cupti_activity_drain()

        joined = '\n'.join(cm.output)
        self.assertIn('hello from cupti', joined)
        self.assertIn('second line', joined)

    def test_lazy_field_creation_for_kernel_and_memcpy(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 1000,
            'events': {
                '12345': _kernel_event('sm80_xmma_gemm_f16f16', cumtime=5_000_000),
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
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 2000,
            'events': {'77777': _kernel_event('')},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        self.assertNotIn(77777, recorder._fields)
        mock_update.assert_not_called()

    def test_reuses_existing_fields(self):
        recorder = CuptiRecorder()

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
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 5000,
            'events': {'11111': _kernel_event('volta_sgemm_128x64', ncalls=5)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        profile = mock_update.call_args[1]['profile']
        fields = recorder._fields[11111]
        self.assertEqual(profile[fields.ncalls_field_id], 5)

    def test_top30_kernel_filtering(self):
        recorder = CuptiRecorder()

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
        recorder = CuptiRecorder()

        events = {
            '11111': _kernel_event('volta_sgemm_128x64_tn', cumtime=5_000_000),
            '22222': _kernel_event('memcpy_host_to_device',
                                   cumtime=1_000_000, bytes=4096),
            '33333': _kernel_event('sync_context', cumtime=500_000),
        }

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_cf, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 9000, 'events': events}])

        descriptors = [c.kwargs['descriptor'] for c in mock_cf.call_args_list]

        # All kernels carry the flat `cuda.kernel` category; the platform refines
        # it to `cuda.kernel.<sub>` from kernel_name at query time. op_names keep
        # the "<family>@<4 hex>" shape so same-family raw kernels stay distinct.
        sgemm_descs = [d for d in descriptors if d['op_name'].startswith('sgemm@')]
        self.assertTrue(sgemm_descs, 'expected at least one sgemm descriptor')
        for d in sgemm_descs:
            self.assertEqual(d['category'], 'cuda.kernel')
            self.assertEqual(d['kernel_name'], 'volta_sgemm_128x64_tn')
            self.assertRegex(d['op_name'], r'^sgemm@[0-9a-f]{4}$')

        # Memcpy/sync op_names are their kind strings unchanged — no
        # fingerprint suffix because the kind string is already unique.
        memcpy_descs = [d for d in descriptors if d['op_name'] == 'memcpy_host_to_device']
        self.assertTrue(memcpy_descs)
        for d in memcpy_descs:
            self.assertEqual(d['category'], 'cuda.memcpy')
            self.assertNotIn('kernel_name', d)

        sync_descs = [d for d in descriptors if d['op_name'] == 'sync_context']
        self.assertTrue(sync_descs)
        for d in sync_descs:
            self.assertEqual(d['category'], 'cuda.sync')
            self.assertNotIn('kernel_name', d)

    def test_memcpy_always_kept(self):
        recorder = CuptiRecorder()

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
        recorder = CuptiRecorder()

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

    def test_persistent_top_n_evicts_on_accumulation(self):
        recorder = CuptiRecorder()

        bucket_a = {str(50000 + i): _kernel_event(f'kernel_{i:02d}',
                                                  cumtime=(i + 1) * 100)
                    for i in range(30)}
        # huge new kernel + tiny re-emit of k0
        bucket_b = {
            '59999': _kernel_event('kernel_huge', cumtime=5000),
            '50000': _kernel_event('kernel_00', cumtime=1),
        }

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile([{'bucket_ts': 9000, 'events': bucket_a}])
            recorder._convert_to_profile([{'bucket_ts': 9100, 'events': bucket_b}])

        self.assertIn(59999, recorder._top_kernel_ids)
        self.assertNotIn(50000, recorder._top_kernel_ids)

        bucket_b_profile = mock_update.call_args[1]['profile']
        self.assertEqual(mock_update.call_args[1]['measurement_ts'], 9100)
        self.assertIn(recorder._fields[59999].cumtime_field_id, bucket_b_profile)
        k0_fields = recorder._fields[50000]
        self.assertNotIn(k0_fields.cumtime_field_id, bucket_b_profile)

    def test_sync_category(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 6000,
            'events': {
                '77777': _kernel_event('sync_context', cumtime=3_000_000),
                '88888': _kernel_event('sync_stream', cumtime=1_000_000),
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_add, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        categories = {c[1].get('descriptor', {}).get('category') for c in mock_add.call_args_list}
        self.assertIn('cuda.sync', categories)
        self.assertFalse(any(c and c.startswith('cuda.kernel.') for c in categories))
        self.assertNotIn('cuda.memcpy', categories)

    def test_sync_has_no_bytes_field(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 6100,
            'events': {'77778': _kernel_event('sync_event', cumtime=2_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertIsNone(recorder._fields[77778].bytes_field_id)

    def test_host_sync_wait_emitted_for_kernel(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 9000,
            'events': {
                '11111': _kernel_event('volta_sgemm_128x64_tn',
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
        recorder = CuptiRecorder()

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
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 9200,
            'events': {'33333': _kernel_event('sync_context', cumtime=500_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        # Sync events don't get a host_sync_wait field — they're the wait
        # source themselves, not something the CPU was blocked on.
        self.assertIsNone(recorder._fields[33333].host_sync_wait_field_id)

    def test_host_sync_wait_omitted_when_zero(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 9300,
            'events': {
                '44444': _kernel_event('volta_sgemm_128x64_tn',
                                       cumtime=5_000_000),  # no host_sync_wait
            },
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        fields = recorder._fields[44444]
        # Field is registered (kernel category), but not emitted in this bucket.
        self.assertIsNotNone(fields.host_sync_wait_field_id)
        profile = mock_update.call_args[1]['profile']
        self.assertNotIn(fields.host_sync_wait_field_id, profile)

    def test_sync_always_kept_outside_top_n(self):
        recorder = CuptiRecorder()

        events = {str(80000 + i): _kernel_event(f'kernel_{i:02d}',
                                                cumtime=(i + 1) * 1000)
                  for i in range(35)}
        events['90001'] = _kernel_event('sync_stream_wait_event', cumtime=1)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 7000, 'events': events}])

        self.assertIn(90001, recorder._fields)

    def test_sync_excluded_from_top_n_accumulation(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 7100,
            'events': {'91001': _kernel_event('sync_context', cumtime=9_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertNotIn(91001, recorder._kernel_cumtime_totals)

    def test_memset_category_and_bytes(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 8000,
            'events': {'55551': _kernel_event('memset_device', cumtime=1_000_000,
                                              bytes=4096)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_add, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        categories = {c[1].get('descriptor', {}).get('category') for c in mock_add.call_args_list}
        self.assertIn('cuda.memset', categories)
        self.assertIsNotNone(recorder._fields[55551].bytes_field_id)

    def test_memset_excluded_from_top_n_accumulation(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 8200,
            'events': {'94000': _kernel_event('memset_device', cumtime=9_000_000, bytes=8192)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertNotIn(94000, recorder._kernel_cumtime_totals)

    def test_cuda_graph_op_name_is_hashed_signature_with_graph_signature_property(self):
        recorder = CuptiRecorder()

        signature = 'kernel[grid=64,block=128,shmem=0,calls=3];memcpy[grid=0,block=0,shmem=0,calls=1];'
        events = {'70001': _kernel_event('graph:' + signature, cumtime=5_000_000)}

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_cf, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 9400, 'events': events}])

        descriptors = [c.kwargs['descriptor'] for c in mock_cf.call_args_list]
        graph_descs = [d for d in descriptors if d['category'] == 'cuda.graph']
        self.assertTrue(graph_descs, 'expected at least one cuda.graph descriptor')
        expected_op_name = (
            f'cuda_graph{_OP_NAME_FINGERPRINT_SEP}{_short_fingerprint(signature)}')
        for d in graph_descs:
            # op_name collapses to cuda_graph@<hash-of-signature>; the raw
            # signature rides as a descriptor property; no kernel_name.
            self.assertEqual(d['op_name'], expected_op_name)
            self.assertEqual(d['graph_signature'], signature)
            self.assertNotIn('kernel_name', d)

    def test_cuda_graph_no_bytes_no_occupancy_has_host_sync_wait(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 9500,
            'events': {'70002': _kernel_event('graph:kernel[grid=32,block=64,shmem=0,calls=2];',
                                              cumtime=4_000_000,
                                              host_sync_wait=1_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        fields = recorder._fields[70002]
        self.assertIsNone(fields.bytes_field_id)
        self.assertIsNone(fields.cumtime_occupancy_field_id)
        self.assertIsNotNone(fields.host_sync_wait_field_id)
        profile = mock_update.call_args[1]['profile']
        self.assertEqual(profile[fields.host_sync_wait_field_id], 1_000_000)

    def test_cuda_graph_competes_for_top_n(self):
        recorder = CuptiRecorder()

        # 35 kernels with small cumtime plus one graph with the largest cumtime:
        # the graph competes for a top-N slot like a kernel and must be kept.
        events = {str(60000 + i): _kernel_event(f'kernel_{i:02d}',
                                                cumtime=(i + 1) * 1000)
                  for i in range(35)}
        events['95001'] = _kernel_event('graph:kernel[grid=8,block=16,shmem=0,calls=1];', cumtime=10_000_000)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 7200, 'events': events}])

        self.assertIn(95001, recorder._top_kernel_ids)
        self.assertIn(95001, recorder._fields)

    def test_cuda_graph_evicted_from_top_n_when_outranked(self):
        recorder = CuptiRecorder()

        # 35 kernels with large cumtime plus one graph with tiny cumtime: the
        # graph loses the top-N competition and is dropped (no longer always
        # kept like memcpy/sync/memset).
        events = {str(60000 + i): _kernel_event(f'kernel_{i:02d}',
                                                cumtime=(i + 1) * 1_000_000)
                  for i in range(35)}
        events['95001'] = _kernel_event('graph:kernel[grid=8,block=16,shmem=0,calls=1];', cumtime=1)

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 7200, 'events': events}])

        self.assertNotIn(95001, recorder._top_kernel_ids)
        self.assertNotIn(95001, recorder._fields)

    def test_cuda_graph_included_in_top_n_accumulation(self):
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 7300,
            'events': {'95002': _kernel_event('graph:kernel[grid=8,block=16,shmem=0,calls=1];',
                                              cumtime=9_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        # CUDA graphs compete for the top-N slots alongside kernels, so their
        # cumtime accumulates into the ranking.
        self.assertIn(95002, recorder._kernel_cumtime_totals)

    def test_kernel_prefix_stripped_for_kernel_name_and_op_name(self):
        recorder = CuptiRecorder()

        # The native lib stores kernel events as "kernel:<raw symbol>". The
        # recorder must strip that prefix before deriving op_name/kernel_name.
        raw = 'volta_sgemm_128x64_tn'
        events = {'11111': {'event_name': 'kernel:' + raw, 'cumtime': 5_000_000,
                            'ncalls': 1, 'nerrors': 0, 'bytes': 0}}

        with patch.object(graphsignal.sdk.sdk(), 'add_counter_profile_field',
                          wraps=graphsignal.sdk.sdk().add_counter_profile_field) as mock_cf, \
             patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile([{'bucket_ts': 9600, 'events': events}])

        descriptors = [c.kwargs['descriptor'] for c in mock_cf.call_args_list]
        kernel_descs = [d for d in descriptors if d['category'] == 'cuda.kernel']
        self.assertTrue(kernel_descs, 'expected at least one cuda.kernel descriptor')
        from graphsignal.recorders.cupti_recorder import make_op_name
        for d in kernel_descs:
            self.assertEqual(d['kernel_name'], raw)
            self.assertEqual(d['op_name'], make_op_name(raw))
            self.assertNotIn('kernel:', d['op_name'])

    def test_occupancy_emitted(self):
        recorder = CuptiRecorder()

        # cumtime_occupancy is computed in the native lib; the recorder passes
        # it through directly onto the cumtime_occupancy field for kernels.
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
        recorder = CuptiRecorder()

        # Kernel with no cumtime_occupancy (lib hadn't captured SM limits yet):
        # the field is still registered for kernels but not emitted this bucket.
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
        recorder = CuptiRecorder()

        buckets = [{
            'bucket_ts': 1200,
            'events': {'99991': _kernel_event('memcpy_host_to_device',
                                              cumtime=1_000_000, bytes=1024)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile'):
            recorder._convert_to_profile(buckets)

        self.assertIsNone(recorder._fields[99991].cumtime_occupancy_field_id)

    def test_profile_tags_include_process_pid(self):
        recorder = CuptiRecorder(pid=4242)

        buckets = [{
            'bucket_ts': 1500,
            'events': {'10001': _kernel_event('some_kernel', cumtime=1_000_000)},
        }]

        with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
            recorder._convert_to_profile(buckets)

        tags = mock_update.call_args[1]['tags']
        self.assertEqual(tags.get('process.pid'), '4242')

    def test_profile_tags_include_context_overrides(self):
        recorder = CuptiRecorder(pid=4242)
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
    # GPU tests run the torch+CUDA workload in a separate subprocess (the
    # injection library is loaded into the subprocess via CUDA_INJECTION64_PATH,
    # which it inherits from this process's env). The recorder runs in the test
    # process, targeting the subprocess's pid — mirroring graphsignal-run's
    # workload+watcher split.
    #
    # The test process must NOT import torch / init CUDA, otherwise the
    # injection lib would also load here and we'd be back to the in-process
    # crash mode.

    @unittest.skipUnless(_CUPTI_ENV_READY, "CUPTI env not set up (Linux + CUDA required)")
    def test_end_to_end_torch_cuda_and_drain(self):
        if not _torch_cuda_available_in_subprocess():
            self.skipTest("torch+CUDA not available")

        proc, meta = _spawn_workload(env_overrides={'CUPTI_TEST_LINGER_SEC': '5.0'})
        try:
            workload_pid = int(meta['PID'])
            # Note: don't call setup() — the drain thread would race with our
            # manual drain below and consume buckets first.
            recorder = CuptiRecorder(pid=workload_pid)

            # The workload's flush thread runs every 1s and writes shm files;
            # the C++ side then cleans shm files older than 2 * window (2s).
            # Drain ~1.2s after activity so the flush has run but the files
            # haven't been swept yet.
            time.sleep(1.2)

            with patch.object(graphsignal.sdk.sdk(), 'update_profile') as mock_update:
                raw = recorder._cupti_activity_drain()
                recorder._convert_to_profile(raw.get('buckets', []))

                any_cuda_profile = False
                for call in mock_update.call_args_list:
                    kwargs = call.kwargs
                    name = kwargs.get('name', call.args[0] if call.args else None)
                    profile = kwargs.get('profile', call.args[1] if len(call.args) >= 2 else None)
                    if name == 'profile.cuda' and isinstance(profile, dict) and profile:
                        any_cuda_profile = True
                        break

                if not any_cuda_profile:
                    stderr = proc.stderr.read() if proc.stderr else ''
                    self.fail(
                        f"expected at least one update_profile call for "
                        f"profile.cuda; subprocess stderr={stderr!r}"
                    )
        finally:
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)

    @unittest.skipUnless(_CUPTI_ENV_READY, "CUPTI env not set up (Linux + CUDA required)")
    def test_overhead(self):
        # Measure the CUPTI injection library's overhead inside the workload by
        # comparing two subprocess runs of the same workload:
        #   - baseline: CUDA_INJECTION64_PATH unset (injection lib not loaded)
        #   - with profiler: CUDA_INJECTION64_PATH set (injection lib loaded)
        if not _torch_cuda_available_in_subprocess():
            self.skipTest("torch+CUDA not available")

        def _run(env_overrides):
            env_overrides = {'CUPTI_TEST_SIZE': '4096',
                             'CUPTI_TEST_ITERS': '1000',
                             'CUPTI_TEST_LINGER_SEC': '0.1',
                             **env_overrides}
            proc, meta = _spawn_workload(env_overrides=env_overrides, timeout=180.0)
            try:
                proc.wait(timeout=60.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
                proc.wait(timeout=5.0)
                self.fail("Workload subprocess timed out")
            return int(meta['TOOK_NS'])

        took_ns_baseline = _run(env_overrides={'CUDA_INJECTION64_PATH': None})
        took_ns_with_profiler = _run(env_overrides={})

        overhead_pct = 100.0 * (took_ns_with_profiler - took_ns_baseline) / max(1, took_ns_baseline)
        overhead_per_iter_us = (took_ns_with_profiler - took_ns_baseline) / 1000 / 1e3

        logger.setLevel(logging.DEBUG)
        logger.debug("CUPTI injection overhead=%.2f%%, overhead_per_iter=%.1f us",
                     overhead_pct, overhead_per_iter_us)

        self.assertTrue(overhead_pct < 5.0,
                        f"expected overhead < 5.0%, got {overhead_pct:.2f}%")


if __name__ == '__main__':
    unittest.main()
