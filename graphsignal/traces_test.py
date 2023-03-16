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
            tags={'k1': 'v1', 'k2': 'v2'},
            upload_on_shutdown=False,
            debug_mode=True)
        graphsignal._agent.hostname = 'h1'

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_trace')
    def test_start_stop(self, mocked_upload_trace, mocked_process_on_trace_stop):
        graphsignal.set_tag('k3', 'v3')
        graphsignal.log_param('p1', 'v1')

        for i in range(10):
            trace = Trace(
                endpoint='ep1',
                tags={'k3': 'v33', 'k4': 4.0})
            trace.set_tag('k5', 'v5')
            trace.set_param('p2', 'v2')
            trace.set_data('input', np.asarray([[1, 2],[3, 4]]), counts=dict(c1=1, c2=2))
            time.sleep(0.01)
            trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.SAMPLE_TRACE)
        self.assertTrue(trace.process_usage.start_ms > 0)
        self.assertEqual(trace.tags[0].key, 'deployment')
        self.assertEqual(trace.tags[0].value, 'd1')
        self.assertEqual(trace.tags[1].key, 'endpoint')
        self.assertEqual(trace.tags[1].value, 'ep1')
        self.assertEqual(trace.tags[2].key, 'hostname')
        self.assertIsNotNone(trace.tags[2].value)
        self.assertEqual(trace.tags[3].key, 'k1')
        self.assertEqual(trace.tags[3].value, 'v1')
        self.assertEqual(trace.tags[4].key, 'k2')
        self.assertEqual(trace.tags[4].value, 'v2')
        self.assertEqual(trace.tags[5].key, 'k3')
        self.assertEqual(trace.tags[5].value, 'v33')
        self.assertEqual(trace.tags[6].key, 'k4')
        self.assertEqual(trace.tags[6].value, '4.0')
        self.assertEqual(trace.tags[7].key, 'k5')
        self.assertEqual(trace.tags[7].value, 'v5')
        self.assertEqual(trace.params[0].name, 'p1')
        self.assertEqual(trace.params[0].value, 'v1')
        self.assertEqual(trace.params[1].name, 'p2')
        self.assertEqual(trace.params[1].value, 'v2')
        self.assertTrue(trace.trace_info.latency_us > 0)
        self.assertEqual(trace.data_profile[0].data_name, 'input')
        self.assertEqual(trace.data_profile[0].shape, [2, 2])
        self.assertEqual(trace.data_profile[0].counts[-2].name, 'c1')
        self.assertEqual(trace.data_profile[0].counts[-2].count, 1)
        self.assertEqual(trace.data_profile[0].counts[-1].name, 'c2')
        self.assertEqual(trace.data_profile[0].counts[-1].count, 2)

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'endpoint': 'ep1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2', 'k3': 'v33', 'k4': 4.0, 'k5': 'v5'}
        key = store.metric_key('performance', 'latency', metric_tags)
        self.assertTrue(len(store.metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'call_count', metric_tags)
        self.assertEqual(store.metrics[key].counter, 10)

        data_tags = metric_tags.copy()
        data_tags['data'] = 'input'
        key = store.metric_key('data', 'element_count', data_tags)
        self.assertEqual(store.metrics[key].counter, 40)
        key = store.metric_key('data', 'c1', data_tags)
        self.assertEqual(store.metrics[key].counter, 10)
        key = store.metric_key('data', 'c2', data_tags)
        self.assertEqual(store.metrics[key].counter, 20)

    @patch.object(ProcessRecorder, 'on_trace_start')
    @patch.object(Uploader, 'upload_trace')
    def test_start_exception(self, mocked_upload_trace, mocked_process_on_trace_start):
        mocked_process_on_trace_start.side_effect = Exception('ex1')
        trace = Trace(endpoint='ep1')
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.SAMPLE_TRACE)
        self.assertEqual(trace.agent_errors[0].message, 'ex1')
        self.assertNotEqual(trace.agent_errors[0].stack_trace, '')

    @patch.object(ProcessRecorder, 'on_trace_stop')
    @patch.object(Uploader, 'upload_trace')
    def test_agent_exception(self, mocked_upload_trace, mocked_process_on_trace_stop):
        mocked_process_on_trace_stop.side_effect = Exception('ex1')
        trace = Trace(
            endpoint='ep1')
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.agent_errors[0].message, 'ex1')
        self.assertNotEqual(trace.agent_errors[0].stack_trace, '')

    @patch.object(Uploader, 'upload_trace')
    def test_inference_exception(self, mocked_upload_trace):
        
        for _ in range(2):
            try:
                with Trace(endpoint='ep1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_trace.call_count, 2)
        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.trace_id != '')
        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.EXCEPTION_TRACE)
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex1')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'endpoint': 'ep1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store.metrics[key].counter, 2)

    @patch.object(Uploader, 'upload_trace')
    def test_set_exception(self, mocked_upload_trace):
        trace = Trace(endpoint='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(ex)
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.EXCEPTION_TRACE)
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex2')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'endpoint': 'ep1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store.metrics[key].counter, 1)


    @patch.object(Uploader, 'upload_trace')
    def test_set_exception_true(self, mocked_upload_trace):
        trace = Trace(endpoint='ep1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            trace.set_exception(exc_info=True)
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.EXCEPTION_TRACE)
        self.assertEqual(trace.exceptions[0].exc_type, 'Exception')
        self.assertEqual(trace.exceptions[0].message, 'ex2')
        self.assertNotEqual(trace.exceptions[0].stack_trace, '')

        store = graphsignal._agent.metric_store()
        metric_tags =  {'deployment': 'd1', 'endpoint': 'ep1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store.metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_trace')
    def test_outlier(self, mocked_upload_trace):
        for _ in range(500):
            with Trace(endpoint='ep1'):
                time.sleep(0.00001)
        with Trace(endpoint='ep1'):
            time.sleep(0.01)

        has_outliers = False
        for call_args in mocked_upload_trace.call_args_list:
            trace = call_args[0][0]
            if trace.trace_type == signals_pb2.TraceType.LATENCY_OUTLIER_TRACE:
                has_outliers = True
                break
        self.assertTrue(has_outliers)

    @patch.object(Uploader, 'upload_trace')
    def test_set_data(self, mocked_upload_trace):
        with Trace(endpoint='ep1') as trace:
            trace.set_data('d1', {'c1': 100, 'c2': None}, check_missing_values=True)

        trace = mocked_upload_trace.call_args[0][0]

        self.assertTrue(trace.start_us > 0)
        self.assertTrue(trace.end_us > 0)
        self.assertEqual(trace.trace_type, signals_pb2.TraceType.MISSING_VALUES_TRACE)

    @patch.object(Uploader, 'upload_trace')
    def test_spans(self, mocked_upload_trace):
        trace = Trace(endpoint='ep1')
        trace2 = Trace(endpoint='ep2')
        trace3 = Trace(endpoint='ep3')
        trace3.stop()
        trace2.stop()
        trace4 = Trace(endpoint='ep4')
        trace4.stop()
        trace.stop()

        trace = mocked_upload_trace.call_args[0][0]

        self.assertEqual(trace.root_span.name, 'ep1')
        self.assertTrue(trace.root_span.start_ns > 0)
        self.assertTrue(trace.root_span.end_ns > 0)

        self.assertEqual(trace.root_span.spans[0].name, 'ep2')
        self.assertTrue(trace.root_span.spans[0].is_endpoint)
        self.assertTrue(trace.root_span.spans[0].start_ns > trace.root_span.start_ns)
        self.assertTrue(trace.root_span.spans[0].end_ns < trace.root_span.end_ns)

        self.assertEqual(trace.root_span.spans[0].spans[0].name, 'ep3')
        self.assertTrue(trace.root_span.spans[0].spans[0].is_endpoint)
        self.assertTrue(trace.root_span.spans[0].spans[0].start_ns > trace.root_span.spans[0].start_ns)
        self.assertTrue(trace.root_span.spans[0].spans[0].end_ns < trace.root_span.spans[0].end_ns)

        self.assertEqual(trace.root_span.spans[1].name, 'ep4')
        self.assertTrue(trace.root_span.spans[1].start_ns > trace.root_span.spans[0].end_ns)
        self.assertTrue(trace.root_span.spans[1].end_ns < trace.root_span.end_ns)

    @patch.object(Uploader, 'upload_trace')
    @patch.object(TraceSampler, 'lock', return_value=False)
    def test_overhead(self, mocked_lock, mocked_upload_trace):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        calls = 10000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Trace(endpoint='ep1') as trace:
                trace.set_tag('k1', 'v1')
                trace.set_param('k2', 'v2')
                trace.set_data('test', 'test string data', check_missing_values=True)
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace
