import unittest
import logging
import sys
import time
from unittest.mock import patch

import graphsignal
from graphsignal.profilers.event_profiler import (
    EventProfiler,
    MAX_EVENT_PROFILER_FIELDS,
    _descriptor_field_key,
)

logger = logging.getLogger('graphsignal')


class EventProfilerTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker._auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_record_event_single_bucket(self, mock_update_profile):
        res = 100_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 10 * res

        desc = {'category': 'custom', 'op_name': 'op_a', 'source': 'test'}
        profiler.record_event(
            desc,
            {'ncalls': 1, 'nerrors': 0},
            base + 10_000_000,
            base + 15_000_000,
        )

        profiler._rollover_buckets(base + 2 * res)

        self.assertGreater(mock_update_profile.call_count, 0)
        found = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name == 'test_profile.events' and profile:
                for v in profile.values():
                    if v == 5_000_000:
                        found = True
                        break
        self.assertTrue(found, 'expected cumtime 5ms in profile payload')

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_record_event_spans_two_buckets(self, mock_update_profile):
        res = 10_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 100 * res

        desc = {'category': 'custom', 'op_name': 'spanning', 'source': 'test'}
        profiler.record_event(
            desc,
            {'ncalls': 1, 'nerrors': 0},
            base,
            base + 15_000_000,
        )

        profiler._rollover_buckets(base + 25_000_000)

        cumtimes = []
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name != 'test_profile.events' or not profile:
                continue
            for v in profile.values():
                if v in (10_000_000, 5_000_000):
                    cumtimes.append(v)

        self.assertIn(10_000_000, cumtimes)
        self.assertIn(5_000_000, cumtimes)

        profiler.shutdown()

    def test_fields_cached_per_descriptor(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()

        desc = {'category': 'x', 'op_name': 'y'}
        stats = {'ncalls': 1}
        m1 = profiler._ensure_descriptor_field_map(desc, stats)
        m2 = profiler._ensure_descriptor_field_map(desc, stats)
        self.assertIs(m1, m2)
        self.assertEqual(len(profiler._fields), 1)
        self.assertEqual(profiler._field_count, 2)
        self.assertEqual(set(m1.keys()), {'cumtime', 'ncalls'})

        profiler.shutdown()

    def test_descriptor_field_key_order_invariant(self):
        a = {'category': 'c', 'op_name': 'o'}
        b = {'op_name': 'o', 'category': 'c'}
        self.assertEqual(_descriptor_field_key(a), _descriptor_field_key(b))

    def test_record_event_requires_descriptor_keys(self):
        profiler = EventProfiler('test_profile.events')
        profiler.setup()
        with self.assertLogs(logger, level='ERROR'):
            profiler.record_event(
                {'op_name': 'only'},
                {'ncalls': 1},
                0,
                10,
            )
        profiler.shutdown()

    def test_record_event_requires_stats(self):
        profiler = EventProfiler('test_profile.events')
        profiler.setup()
        with self.assertLogs(logger, level='ERROR'):
            profiler.record_event(
                {'category': 'c', 'op_name': 'o'},
                {},
                0,
                10,
            )
        profiler.shutdown()

    def test_record_event_without_nerrors(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()
        base = (time.time_ns() // 100_000_000) * 100_000_000 + 20 * 100_000_000
        profiler.record_event(
            {'category': 'c', 'op_name': 'no_errors_key'},
            {'ncalls': 1},
            base + 1_000_000,
            base + 2_000_000,
        )
        profiler._rollover_buckets(base + 3 * 100_000_000)
        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_extra_stat_num_tokens_emitted(self, mock_update_profile):
        res = 50_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 60 * res

        desc = {'category': 'custom', 'op_name': 'tokens_op'}
        profiler.record_event(
            desc,
            {'ncalls': 1, 'num_tokens': 128},
            base + 1_000_000,
            base + 2_000_000,
        )
        profiler._rollover_buckets(base + 2 * res)

        found_tokens = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name != 'test_profile.events' or not profile:
                continue
            if 128 in profile.values():
                found_tokens = True
                break
        self.assertTrue(found_tokens)

        key = _descriptor_field_key(desc)
        self.assertIn('num_tokens', profiler._fields[key])

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_nerrors_emitted(self, mock_update_profile):
        res = 50_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 50 * res

        desc = {'category': 'custom', 'op_name': 'err_op'}
        profiler.record_event(
            desc,
            {'ncalls': 1, 'nerrors': 2},
            base + 1_000_000,
            base + 2_000_000,
        )
        profiler._rollover_buckets(base + 2 * res)

        found_errors = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name != 'test_profile.events' or not profile:
                continue
            if 2 in profile.values():
                found_errors = True
                break
        self.assertTrue(found_errors)

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_late_event_resent_for_old_bucket(self, mock_update_profile):
        res = 10_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 100 * res

        # advance past base; buckets dict is empty so nothing emitted yet
        profiler._rollover_buckets(base + 2 * res)
        mock_update_profile.reset_mock()

        desc = {'category': 'custom', 'op_name': 'late_op'}
        # record a late event whose interval falls entirely in the already-rolled bucket
        profiler.record_event(
            desc,
            {'ncalls': 1},
            base,
            base + res,
        )

        # next rollover should pick up the late bucket and resend it
        profiler._rollover_buckets(base + 3 * res)

        found = False
        for call in mock_update_profile.call_args_list:
            _, kwargs = call
            if (kwargs.get('name') == 'test_profile.events'
                    and kwargs.get('measurement_ts') == base + res):
                profile = kwargs.get('profile', {})
                if any(v == res for v in profile.values()):  # cumtime == full bucket
                    found = True
                    break
        self.assertTrue(found, 'late event should be resent with original bucket measurement_ts')

        profiler.shutdown()

    def test_new_stat_on_second_call_ignored(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()

        desc = {'category': 'c', 'op_name': 'o'}
        base = (time.time_ns() // 100_000_000) * 100_000_000 + 10 * 100_000_000

        profiler.record_event(desc, {'ncalls': 1}, base + 1_000_000, base + 2_000_000)
        key = _descriptor_field_key(desc)
        self.assertNotIn('new_stat', profiler._fields[key])

        profiler.record_event(desc, {'ncalls': 1, 'new_stat': 5}, base + 3_000_000, base + 4_000_000)
        self.assertNotIn('new_stat', profiler._fields[key])

        profiler.shutdown()

    def test_field_limit_blocks_new_descriptors(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()
        profiler._field_count = MAX_EVENT_PROFILER_FIELDS
        profiler.record_event(
            {'category': 'c', 'op_name': 'new'},
            {'ncalls': 1, 'nerrors': 0},
            100,
            200,
        )
        self.assertEqual(len(profiler._fields), 0)
        profiler.shutdown()
