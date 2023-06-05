import unittest
import logging
import sys
import time
import random
import numpy as np
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.spans import Span, TraceOptions, DEFAULT_OPTIONS, get_current_span
from graphsignal.recorders.process_recorder import ProcessRecorder
from graphsignal.uploader import Uploader


logger = logging.getLogger('graphsignal')


class SpansTest(unittest.TestCase):
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
        graphsignal._tracer.hostname = 'h1'

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    def test_start_stop(self, mocked_upload_span, mocked_process_on_span_stop):
        graphsignal.set_tag('k2', 'v2')

        graphsignal.set_context_tag('k3', 'v3')
        graphsignal.set_context_tag('k4', 'v4')

        for i in range(10):
            span = Span(
                operation='ep1',
                tags={'k4': 4.0})
            span.set_tag('k5', 'v5')
            span.set_param('p1', 'v1')
            span.set_data('input', np.asarray([[1, 2],[3, 4]]), counts=dict(c1=1, c2=2))
            time.sleep(0.01)
            self.assertEqual(get_current_span(), span)
            span.stop()
            self.assertIsNone(get_current_span())

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertEqual(span.labels, ['root'])
        self.assertTrue(span.process_usage.start_ms > 0)
        self.assertEqual(span.tags[0].key, 'deployment')
        self.assertEqual(span.tags[0].value, 'd1')
        self.assertEqual(span.tags[1].key, 'operation')
        self.assertEqual(span.tags[1].value, 'ep1')
        self.assertEqual(span.tags[2].key, 'hostname')
        self.assertIsNotNone(span.tags[2].value)
        self.assertEqual(span.tags[3].key, 'k1')
        self.assertEqual(span.tags[3].value, 'v1')
        self.assertEqual(span.tags[4].key, 'k2')
        self.assertEqual(span.tags[4].value, 'v2')
        self.assertEqual(span.tags[5].key, 'k3')
        self.assertEqual(span.tags[5].value, 'v3')
        self.assertEqual(span.tags[6].key, 'k4')
        self.assertEqual(span.tags[6].value, '4.0')
        self.assertEqual(span.tags[7].key, 'k5')
        self.assertEqual(span.tags[7].value, 'v5')
        self.assertEqual(span.params[0].name, 'p1')
        self.assertEqual(span.params[0].value, 'v1')
        self.assertEqual(span.data_profile[0].data_name, 'input')
        self.assertEqual(span.data_profile[0].shape, [2, 2])
        self.assertEqual(span.data_profile[0].counts[-2].name, 'c1')
        self.assertEqual(span.data_profile[0].counts[-2].count, 1)
        self.assertEqual(span.data_profile[0].counts[-1].name, 'c2')
        self.assertEqual(span.data_profile[0].counts[-1].count, 2)
        self.assertEqual(span.data_samples[0].data_name, 'input')
        self.assertEqual(span.data_samples[0].content_type, 'application/json')
        self.assertEqual(span.data_samples[0].content_bytes, b'[[1, 2], [3, 4]]')

        store = graphsignal._tracer.metric_store()
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

    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_stop_nested(self, mocked_upload_span, mocked_process_on_span_start):
        with Span(operation='ep1'):
            with Span(operation='ep2'):
                pass

        t1 = mocked_upload_span.call_args_list[1][0][0]
        t2 = mocked_upload_span.call_args_list[0][0][0]

        self.assertEqual(t1.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(t1.start_us > 0)
        self.assertTrue(t1.end_us > 0)
        self.assertEqual(t1.labels, ['root'])
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)

        self.assertEqual(t2.sampling_type, signals_pb2.Span.SamplingType.PARENT_SAMPLING)
        self.assertTrue(t2.start_us > 0)
        self.assertTrue(t2.end_us > 0)
        self.assertEqual(t2.labels, [])


    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_exception(self, mocked_upload_span, mocked_process_on_span_start):
        mocked_process_on_span_start.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='ep1')
        span.stop()
        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.labels, ['root'])
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    def test_stop_exception(self, mocked_upload_span, mocked_process_on_span_stop):
        mocked_process_on_span_stop.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='ep1')
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(Uploader, 'upload_span')
    def test_operation_exception(self, mocked_upload_span):
        for _ in range(2):
            try:
                with Span(operation='ep1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_span.call_count, 2)
        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(span.span_id != '')
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.labels, ['root'])
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex1')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 2)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception(self, mocked_upload_span):
        span = Span(operation='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(ex)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.labels, ['root'])
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception_true(self, mocked_upload_span):
        span = Span(operation='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(exc_info=True)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertEqual(span.sampling_type, signals_pb2.Span.SamplingType.RANDOM_SAMPLING)
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.labels, ['root'])
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'ep1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_set_data(self, mocked_upload_span):
        with Span(operation='ep1') as span:
            span.set_data('d1', None)

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.labels, ['root'])

    @patch.object(Uploader, 'upload_span')
    def test_subspans(self, mocked_upload_span):
        with Span(operation='ep1'):
            with Span(operation='ep2'):
                with Span(operation='ep3'):
                    pass
            with Span(operation='ep4'):
                pass

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t4 = mocked_upload_span.call_args_list[2][0][0]
        t1 = mocked_upload_span.call_args_list[3][0][0]

        self.assertEqual(t1.context.parent_span_id, '')
        self.assertEqual(t1.context.root_span_id, t1.span_id)

        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)

        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)

        self.assertEqual(t4.context.parent_span_id, t1.span_id)
        self.assertEqual(t4.context.root_span_id, t1.span_id)

    @patch.object(Uploader, 'upload_span')
    def test_propagation(self, mocked_upload_span):
        with Span(operation='ep1') as t1:
            t1._is_sampling = False
            t1._proto = None
            with Span(operation='ep2') as t2:
                t2._is_sampling = False
                t2._proto = None
                with Span(operation='ep3') as t3:
                    t3.add_exception(Exception('ex2'))
            with Span(operation='ep4'):
                pass

        t3 = mocked_upload_span.call_args_list[0][0][0]
        t2 = mocked_upload_span.call_args_list[1][0][0]
        t4 = mocked_upload_span.call_args_list[2][0][0]
        t1 = mocked_upload_span.call_args_list[3][0][0]

        self.assertEqual(t1.context.parent_span_id, '')
        self.assertEqual(t1.context.root_span_id, t1.span_id)
        self.assertEqual(t1.sampling_type, signals_pb2.Span.SamplingType.CHILD_SAMPLING)

        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)
        self.assertEqual(t2.sampling_type, signals_pb2.Span.SamplingType.CHILD_SAMPLING)

        self.assertEqual(t3.context.parent_span_id, t2.span_id)
        self.assertEqual(t3.context.root_span_id, t1.span_id)
        self.assertEqual(t3.sampling_type, signals_pb2.Span.SamplingType.ERROR_SAMPLING)

        self.assertEqual(t4.context.parent_span_id, t1.span_id)
        self.assertEqual(t4.context.root_span_id, t1.span_id)
        self.assertEqual(t4.sampling_type, signals_pb2.Span.SamplingType.PARENT_SAMPLING)


    @patch.object(Uploader, 'upload_span')
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._tracer.debug_mode = False

        calls = 10000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Span(operation='ep1') as span:
                span.set_tag('k1', 'v1')
                span.set_param('k2', 'v2')
                span.set_data('test', 'test string data')
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