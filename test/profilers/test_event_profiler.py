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

        # 5ms wall-clock interval inside one bucket
        profiler.record_event(
            op_name='op_a',
            category='custom',
            meta_info={'source': 'test'},
            start_ns=base + 10_000_000,
            end_ns=base + 15_000_000,
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
        self.assertTrue(found, 'expected cumtime 5ms (wall clock) in profile payload')

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_record_event_wall_clock_derived(self, mock_update_profile):
        res = 100_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 10 * res

        # 1ms wall-clock interval; cumtime is now wall clock only
        profiler.record_event(
            op_name='op_wallclock',
            category='custom',
            start_ns=base + 10_000_000,
            end_ns=base + 11_000_000,
        )

        profiler._rollover_buckets(base + 2 * res)

        found_1ms = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name != 'test_profile.events' or not profile:
                continue
            for v in profile.values():
                if v == 1_000_000:
                    found_1ms = True

        self.assertTrue(found_1ms, 'expected 1ms wall-clock cumtime in profile')

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_record_event_end_ns_none_places_in_start_bucket(self, mock_update_profile):
        res = 100_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 10 * res

        profiler.record_event(
            op_name='op_no_end',
            category='custom',
            start_ns=base + 5_000_000,
        )

        profiler._rollover_buckets(base + 2 * res)

        found = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name == 'test_profile.events' and profile:
                # active_ns = exit_offset_ns - enter_offset_ns = 1ns (end = start + 1)
                for v in profile.values():
                    if v > 0:
                        found = True
                        break
        self.assertTrue(found, 'event with end_ns=None should still be recorded')

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_record_event_spans_two_buckets(self, mock_update_profile):
        res = 10_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 100 * res

        # starts at base (bucket boundary), ends 15ms later → spans two buckets
        profiler.record_event(
            op_name='spanning',
            category='custom',
            meta_info={'source': 'test'},
            start_ns=base,
            end_ns=base + 15_000_000,
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

        # start bucket: num_running=1, enter=0 → active = 1*10ms - 0 + 0 = 10ms
        # terminal bucket: num_running=0, exit=5ms → active = 0 - 0 + 5ms = 5ms
        self.assertIn(10_000_000, cumtimes)
        self.assertIn(5_000_000, cumtimes)

        profiler.shutdown()

    def test_fields_cached_per_descriptor(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()

        desc = {'category': 'x', 'op_name': 'y'}
        m1 = profiler._ensure_descriptor_field_map(desc)
        m2 = profiler._ensure_descriptor_field_map(desc)
        self.assertIs(m1, m2)
        self.assertEqual(len(profiler._fields), 1)
        self.assertEqual(profiler._field_count, 3)
        self.assertEqual(set(m1.keys()), {'cumtime', 'ncalls', 'nerrors'})

        profiler.shutdown()

    def test_meta_info_included_in_descriptor(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()

        base = (time.time_ns() // 100_000_000) * 100_000_000 + 10 * 100_000_000
        profiler.record_event(
            op_name='op',
            category='cat',
            meta_info={'model': 'gpt4', 'region': 'us'},
            start_ns=base + 1_000_000,
            end_ns=base + 2_000_000,
        )

        expected_key = _descriptor_field_key(
            {'op_name': 'op', 'category': 'cat', 'model': 'gpt4', 'region': 'us'})
        self.assertIn(expected_key, profiler._fields)

        profiler.shutdown()

    def test_descriptor_field_key_order_invariant(self):
        a = {'category': 'c', 'op_name': 'o'}
        b = {'op_name': 'o', 'category': 'c'}
        self.assertEqual(_descriptor_field_key(a), _descriptor_field_key(b))

    def test_record_event_requires_op_name(self):
        profiler = EventProfiler('test_profile.events')
        profiler.setup()
        with self.assertLogs(logger, level='ERROR'):
            profiler.record_event(
                op_name='',
                category='c',
                start_ns=0,
            )
        profiler.shutdown()

    def test_record_event_requires_category(self):
        profiler = EventProfiler('test_profile.events')
        profiler.setup()
        with self.assertLogs(logger, level='ERROR'):
            profiler.record_event(
                op_name='op',
                category='',
                start_ns=0,
            )
        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_nerrors_emitted(self, mock_update_profile):
        res = 50_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 50 * res

        profiler.record_event(
            op_name='err_op',
            category='custom',
            has_error=True,
            start_ns=base + 1_000_000,
            end_ns=base + 2_000_000,
        )
        profiler._rollover_buckets(base + 2 * res)

        found_errors = False
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            name = kwargs.get('name', args[0] if args else None)
            profile = kwargs.get('profile', args[1] if len(args) > 1 else None)
            if name != 'test_profile.events' or not profile:
                continue
            if 1 in profile.values():
                found_errors = True
                break
        self.assertTrue(found_errors)

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_nerrors_zero_not_emitted(self, mock_update_profile):
        res = 50_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 50 * res

        profiler.record_event(
            op_name='ok_op',
            category='custom',
            start_ns=base + 1_000_000,
            end_ns=base + 2_000_000,
        )
        profiler._rollover_buckets(base + 2 * res)

        key = _descriptor_field_key({'op_name': 'ok_op', 'category': 'custom'})
        field_map = profiler._fields.get(key, {})
        nerrors_fid = field_map.get('nerrors')
        for call in mock_update_profile.call_args_list:
            args, kwargs = call
            profile = kwargs.get('profile', {})
            self.assertNotIn(nerrors_fid, profile)

        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_late_event_resent_for_old_bucket(self, mock_update_profile):
        res = 10_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 100 * res

        profiler._rollover_buckets(base + 2 * res)
        mock_update_profile.reset_mock()

        # full-bucket event: enter=0, exit=res → active = res
        profiler.record_event(
            op_name='late_op',
            category='custom',
            start_ns=base,
            end_ns=base + res,
        )

        profiler._rollover_buckets(base + 3 * res)

        found = False
        for call in mock_update_profile.call_args_list:
            _, kwargs = call
            if (kwargs.get('name') == 'test_profile.events'
                    and kwargs.get('measurement_ts') == base + res):
                profile = kwargs.get('profile', {})
                if any(v == res for v in profile.values()):
                    found = True
                    break
        self.assertTrue(found, 'late event should be resent with original bucket measurement_ts')

        profiler.shutdown()

    def test_field_limit_blocks_new_descriptors(self):
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(100_000_000)
        profiler.setup()
        profiler._field_count = MAX_EVENT_PROFILER_FIELDS - 2
        profiler.record_event(
            op_name='new',
            category='c',
            start_ns=100,
            end_ns=200,
        )
        self.assertEqual(len(profiler._fields), 0)
        profiler.shutdown()

    @patch('graphsignal._ticker.update_profile')
    def test_ncalls_equals_num_running_plus_num_exited(self, mock_update_profile):
        res = 10_000_000
        profiler = EventProfiler('test_profile.events')
        profiler.set_resolution_ns(res)
        profiler.setup()

        base = (time.time_ns() // res) * res + 100 * res

        # spans two buckets: start bucket gets num_running=1, terminal gets num_exited=1
        profiler.record_event(
            op_name='op_span',
            category='custom',
            start_ns=base,
            end_ns=base + 15_000_000,
        )
        profiler._rollover_buckets(base + 25_000_000)

        # both buckets should emit ncalls=1 (num_running=1 or num_exited=1)
        ncalls_found = []
        key = _descriptor_field_key({'op_name': 'op_span', 'category': 'custom'})
        field_map = profiler._fields.get(key, {})
        ncalls_fid = field_map.get('ncalls')
        for call in mock_update_profile.call_args_list:
            _, kwargs = call
            if kwargs.get('name') == 'test_profile.events':
                profile = kwargs.get('profile', {})
                if ncalls_fid in profile:
                    ncalls_found.append(profile[ncalls_fid])

        self.assertTrue(all(v == 1 for v in ncalls_found))
        self.assertEqual(len(ncalls_found), 2)

        profiler.shutdown()
