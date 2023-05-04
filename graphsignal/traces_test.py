import unittest
import logging
import sys
import time
import random
import numpy as np
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.traces import Trace, TraceOptions, DEFAULT_OPTIONS
from graphsignal.recorders.process_recorder import ProcessRecorder
from graphsignal.trace_sampler import TraceSampler
from graphsignal.uploader import Uploader


logger = logging.getLogger('graphsignal')


class TraceTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            tags={'k1': 'v1'},
            record_data_samples=True,
            upload_on_shutdown=False,
            debug_mode=True)
        graphsignal._agent.hostname = 'h1'

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_trace')
    def test_start_stop(self, mocked_upload_trace, mocked_process_on_trace_stop):
        graphsignal.set_tag('k2', 'v2')

        graphsignal.set_context_tag('k3', 'v3')
        graphsignal.set_context_tag('k4', 'v4')

        for i in range(10):
            trace = Trace(
                operation='ep1',
                tags={'k4': 4.0})
            trace.set_tag('k5', 'v5')
            trace.set_param('p1', 'v1')
            trace.set_data('input', np.asarray([[1, 2],[3, 4]]), counts=dict(c1=1, c2=2))
            time.sleep(0.01)
            trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertEqual(trace.labels, ['root'])
        self.assertTrue(trace.process_usage.start_ms > 0)
        self.assertEqual(trace.tags[0].key, 'deployment')
        self.assertEqual(trace.tags[0].value, 'd1')
        self.assertEqual(trace.tags[1].key, 'operation')
        self.assertEqual(trace.tags[1].value, 'ep1')
        self.assertEqual(trace.tags[2].key, 'hostname')
        self.assertIsNotNone(trace.tags[2].value)
        self.assertEqual(trace.tags[3].key, 'k1')
        self.assertEqual(trace.tags[3].value, 'v1')
        self.assertEqual(trace.tags[4].key, 'k2')
        self.assertEqual(trace.tags[4].value, 'v2')
        self.assertEqual(trace.tags[5].key, 'k3')
        self.assertEqual(trace.tags[5].value, 'v3')
        self.assertEqual(trace.tags[6].key, 'k4')
        self.assertEqual(trace.tags[6].value, '4.0')
        self.assertEqual(trace.tags[7].key, 'k5')
        self.assertEqual(trace.tags[7].value, 'v5')
        self.assertEqual(trace.params[0].name, 'p1')
        self.assertEqual(trace.params[0].value, 'v1')
        self.assertEqual(trace.data_profile[0].data_name, 'input')
        self.assertEqual(trace.data_profile[0].shape, [2, 2])
        self.assertEqual(trace.data_profile[0].counts[-2].name, 'c1')
        self.assertEqual(trace.data_profile[0].counts[-2].count, 1)
        self.assertEqual(trace.data_profile[0].counts[-1].name, 'c2')
        self.assertEqual(trace.data_profile[0].counts[-1].count, 2)
        self.assertEqual(trace.data_samples[0].data_name, 'input')
        self.assertEqual(trace.data_samples[0].content_type, 'application/json')
        self.assertEqual(trace.data_samples[0].content_bytes, b'[[1, 2], [3, 4]]')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2', 'k3': 'v3', 'k4': 4.0, 'k5': 'v5'}
        key = store.metric_key('performance', 'latency', metric_tags)
        self.assertTrue(len(store._metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'call_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 10)

        data_tags = metric_tags.copy()
        data_tags['data'] = 'input'
        key = store.metric_key('data', 'element_count', data_tags)
        self.assertEqual(store._metrics[key].counter, 40)
        key = store.metric_key('data', 'c1', data_tags)
        self.assertEqual(store._metrics[key].counter, 10)
        key = store.metric_key('data', 'c2', data_tags)
        self.assertEqual(store._metrics[key].counter, 20)

    @patch.object(ProcessRecorder, 'on_trace_start')
    @patch.object(Uploader, 'upload_trace')
    def test_start_stop_nested(self, mocked_upload_trace, mocked_process_on_trace_start):
        with Trace(operation='ep1') as trace1:
            with Trace(operation='ep2') as trace2:
                pass

        t1 = mocked_upload_trace.call_args_list[1][0][0]
        t2 = mocked_upload_trace.call_args_list[0][0][0]

        self.assertEqual(t1.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(t1.start_us > 0)
        self.assertTrue(t1.end_us > 0)
        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(t2.span.parent_trace_id, t1.trace_id)
        self.assertEqual(t2.span.root_trace_id, t1.trace_id)

        self.assertEqual(t2.sampling_type, signals_pb2.Trace.SamplingType.PARENT_SAMPLING)
        self.assertTrue(t2.start_us > 0)
        self.assertTrue(t2.end_us > 0)
        self.assertEqual(t2.labels, [])


    @patch.object(ProcessRecorder, 'on_trace_start')
    @patch.object(Uploader, 'upload_trace')
    def test_start_exception(self, mocked_upload_trace, mocked_process_on_trace_start):
        mocked_process_on_trace_start.side_effect = Exception('ex1')

        store = graphsignal._agent.log_store()
        store.clear()

        trace = Trace(operation='ep1')
        trace.stop()
        trace = mocked_upload_trace.call_args[0][0]

        self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.labels, ['root'])
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_trace')
    def test_stop_exception(self, mocked_upload_trace, mocked_process_on_trace_stop):
        mocked_process_on_trace_stop.side_effect = Exception('ex1')

        store = graphsignal._agent.log_store()
        store.clear()

        trace = Trace(operation='ep1')
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(Uploader, 'upload_trace')
    def test_operation_exception(self, mocked_upload_trace):
        for _ in range(2):
            try:
                with Trace(operation='ep1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_trace.call_count, 2)
        trace = mocked_upload_trace.call_args[0][0]

        self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(trace.trace_id != '')
        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.labels, ['root', 'exception'])
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex1')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 2)

    @patch.object(Uploader, 'upload_trace')
    def test_set_exception(self, mocked_upload_trace):
        trace = Trace(operation='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(ex)
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.labels, ['root', 'exception'])
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex2')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)


    @patch.object(Uploader, 'upload_trace')
    def test_set_exception_true(self, mocked_upload_trace):
        trace = Trace(operation='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(exc_info=True)
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.labels, ['root', 'exception'])
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex2')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_trace')
    def test_outlier(self, mocked_upload_trace):
        for _ in range(500):
            with Trace(operation='ep1'):
                time.sleep(0.00001)
        with Trace(operation='ep1'):
            time.sleep(0.01)

        has_outliers = False
        for call_args in mocked_upload_trace.call_args_list:
            trace = call_args[0][0]
            if 'latency-outlier' in trace.labels:
                has_outliers = True
                self.assertEqual(trace.sampling_type, signals_pb2.Trace.SamplingType.ERROR_SAMPLING)
                break
        self.assertTrue(has_outliers)

    @patch.object(Uploader, 'upload_trace')
    def test_set_data(self, mocked_upload_trace):
        with Trace(operation='ep1') as trace:
            trace.set_data('d1', {'c1': 100, 'c2': None}, check_missing_values=True)

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.labels, ['root', 'missing-values'])

    @patch.object(Uploader, 'upload_trace')
    def test_spans(self, mocked_upload_trace):
        with Trace(operation='ep1'):
            with Trace(operation='ep2'):
                with Trace(operation='ep3'):
                    pass
            with Trace(operation='ep4'):
                pass

        t3 = mocked_upload_trace.call_args_list[0][0][0]
        t2 = mocked_upload_trace.call_args_list[1][0][0]
        t4 = mocked_upload_trace.call_args_list[2][0][0]
        t1 = mocked_upload_trace.call_args_list[3][0][0]

        self.assertEqual(t1.span.parent_trace_id, '')
        self.assertEqual(t1.span.root_trace_id, t1.trace_id)

        self.assertEqual(t2.span.parent_trace_id, t1.trace_id)
        self.assertEqual(t2.span.root_trace_id, t1.trace_id)

        self.assertEqual(t3.span.parent_trace_id, t2.trace_id)
        self.assertEqual(t3.span.root_trace_id, t1.trace_id)

        self.assertEqual(t4.span.parent_trace_id, t1.trace_id)
        self.assertEqual(t4.span.root_trace_id, t1.trace_id)

    @patch.object(Uploader, 'upload_trace')
    def test_overhead(self, mocked_upload_trace):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._agent.debug_mode = False

        calls = 10000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Trace(operation='ep1') as trace:
                trace.set_tag('k1', 'v1')
                trace.set_param('k2', 'v2')
                trace.set_data('test', 'test string data', check_missing_values=True)
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace


def find_log_entry(store, text):
    for entry in store._logs:
        if entry.message and text in entry.message:
            return entry
        if entry.exception and text in entry.exception:
            return entry