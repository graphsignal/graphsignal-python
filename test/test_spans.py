import unittest
import logging
import sys
import time
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.spans import Span, get_current_span, encode_data_payload
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
            record_payloads=True,
            upload_on_shutdown=False,
            debug_mode=True)
        graphsignal._tracer.hostname = 'h1'

    def tearDown(self):
        graphsignal.shutdown()

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    @patch.object(Uploader, 'upload_score')
    def test_start_stop(self, mocked_upload_score, mocked_upload_span, mocked_process_on_span_stop):
        graphsignal.set_tag('k2', 'v2')

        graphsignal.set_context_tag('k3', 'v3')
        graphsignal.set_context_tag('k4', 'v4')

        for i in range(10):
            span = Span(
                operation='op1',
                tags={'k4': 4.0})
            span.set_tag('k5', 'v5')
            span.set_payload('input', [[1, 2],[3, 4]], usage=dict(c1=1, c2=2))
            span.set_usage('c3', 3)
            time.sleep(0.01)
            self.assertEqual(get_current_span(), span)
            span.first_token()
            span.score(name='test-score', score=0.5, severity=3, comment='c1')
            time.sleep(0.01)
            span.stop()
            self.assertIsNone(get_current_span())

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertTrue(span.context.end_ns > 0)
        self.assertTrue(span.context.start_ns > 0)
        self.assertTrue(span.context.first_token_ns > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)
        self.assertEqual(span.tags[0].key, 'deployment')
        self.assertEqual(span.tags[0].value, 'd1')
        self.assertEqual(span.tags[1].key, 'operation')
        self.assertEqual(span.tags[1].value, 'op1')
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
        self.assertEqual(span.usage[0].payload_name, 'input')
        self.assertEqual(span.usage[0].name, 'c1')
        self.assertEqual(span.usage[0].value, 1)
        self.assertEqual(span.usage[1].payload_name, 'input')
        self.assertEqual(span.usage[1].name, 'c2')
        self.assertEqual(span.usage[1].value, 2)
        self.assertEqual(span.usage[2].payload_name, '')
        self.assertEqual(span.usage[2].name, 'c3')
        self.assertEqual(span.usage[2].value, 3)
        self.assertEqual(span.payloads[0].name, 'input')
        self.assertEqual(span.payloads[0].content_type, 'application/json')
        self.assertEqual(span.payloads[0].content_bytes, b'[[1, 2], [3, 4]]')
        self.assertEqual(span.config[0].key, 'graphsignal.library.version')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'op1', 'hostname': 'h1', 'k1': 'v1', 'k2': 'v2', 'k3': 'v3', 'k4': 4.0, 'k5': 'v5'}
        key = store.metric_key('performance', 'latency', metric_tags)
        self.assertTrue(len(store._metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'first_token', metric_tags)
        self.assertTrue(len(store._metrics[key].histogram) > 0)
        key = store.metric_key('performance', 'call_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 10)

        data_tags = metric_tags.copy()
        key = store.metric_key('data', 'c3', data_tags)
        self.assertEqual(store._metrics[key].counter, 30)

        data_tags['payload'] = 'input'
        key = store.metric_key('data', 'c1', data_tags)
        self.assertEqual(store._metrics[key].counter, 10)
        key = store.metric_key('data', 'c2', data_tags)
        self.assertEqual(store._metrics[key].counter, 20)

        score = mocked_upload_score.call_args[0][0]

        self.assertTrue(score.score_id is not None and score.score_id != '')
        self.assertEqual(score.span_id, span.span_id)
        self.assertEqual(score.name, 'test-score')
        self.assertEqual(score.tags[0].key, 'deployment')
        self.assertEqual(score.tags[0].value, 'd1')
        self.assertEqual(score.tags[1].key, 'operation')
        self.assertEqual(score.tags[1].value, 'op1')
        self.assertEqual(score.tags[2].key, 'hostname')
        self.assertIsNotNone(score.tags[2].value)
        self.assertEqual(score.tags[3].key, 'k1')
        self.assertEqual(score.tags[3].value, 'v1')
        self.assertEqual(score.tags[4].key, 'k2')
        self.assertEqual(score.tags[4].value, 'v2')
        self.assertEqual(score.tags[5].key, 'k3')
        self.assertEqual(score.tags[5].value, 'v3')
        self.assertEqual(score.tags[6].key, 'k4')
        self.assertEqual(score.tags[6].value, '4.0')
        self.assertEqual(score.tags[7].key, 'k5')
        self.assertEqual(score.tags[7].value, 'v5')
        self.assertEqual(score.score, 0.5)
        self.assertEqual(score.severity, 3)
        self.assertEqual(score.comment, 'c1')
        self.assertTrue(score.create_ts > 0)



    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_stop_nested(self, mocked_upload_span, mocked_process_on_span_start):
        with Span(operation='op1'):
            with Span(operation='op2'):
                pass

        t1 = mocked_upload_span.call_args_list[1][0][0]
        t2 = mocked_upload_span.call_args_list[0][0][0]

        self.assertTrue(t1.start_us > 0)
        self.assertTrue(t1.end_us > 0)
        self.assertEqual(t1.context.root_span_id, t1.span_id)
        self.assertEqual(t2.context.parent_span_id, t1.span_id)
        self.assertEqual(t2.context.root_span_id, t1.span_id)

        self.assertTrue(t2.start_us > 0)
        self.assertTrue(t2.end_us > 0)


    @patch.object(ProcessRecorder, 'on_span_start')
    @patch.object(Uploader, 'upload_span')
    def test_start_exception(self, mocked_upload_span, mocked_process_on_span_start):
        mocked_process_on_span_start.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='op1')
        span.stop()
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)
        self.assertEqual(find_log_entry(store, 'ex1').tags['deployment'], 'd1')
        self.assertIsNotNone(find_log_entry(store, 'ex1').message)
        self.assertIsNotNone(find_log_entry(store, 'ex1').exception)

    @patch.object(ProcessRecorder, 'on_span_stop')
    @patch.object(Uploader, 'upload_span')
    def test_stop_exception(self, mocked_upload_span, mocked_process_on_span_stop):
        mocked_process_on_span_stop.side_effect = Exception('ex1')

        store = graphsignal._tracer.log_store()
        store.clear()

        span = Span(operation='op1')
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
                with Span(operation='op1'):
                    raise Exception('ex1')
            except Exception as ex:
                if str(ex) != 'ex1':
                    raise ex

        self.assertEqual(mocked_upload_span.call_count, 2)
        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.span_id != '')
        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex1')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'op1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 2)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception(self, mocked_upload_span):
        span = Span(operation='op1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(ex)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'op1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_add_exception_true(self, mocked_upload_span):
        span = Span(operation='op1')
        try:
            raise Exception('ex2')
        except Exception as ex:
            span.add_exception(exc_info=True)
        span.stop()

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)
        self.assertEqual(span.exceptions[0].exc_type, 'Exception')
        self.assertEqual(span.exceptions[0].message, 'ex2')
        self.assertNotEqual(span.exceptions[0].stack_trace, '')

        store = graphsignal._tracer.metric_store()
        metric_tags =  {'deployment': 'd1', 'operation': 'op1', 'hostname': 'h1', 'k1': 'v1'}
        key = store.metric_key('performance', 'exception_count', metric_tags)
        self.assertEqual(store._metrics[key].counter, 1)

    @patch.object(Uploader, 'upload_span')
    def test_set_payload(self, mocked_upload_span):
        with Span(operation='op1') as span:
            span.set_payload('d1', None)

        span = mocked_upload_span.call_args[0][0]

        self.assertTrue(span.start_us > 0)
        self.assertTrue(span.end_us > 0)
        self.assertEqual(span.context.root_span_id, span.span_id)

    @patch.object(Uploader, 'upload_span')
    def test_subspans(self, mocked_upload_span):
        with Span(operation='op1'):
            with Span(operation='op2'):
                with Span(operation='op3'):
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
    def test_overhead(self, mocked_upload_span):
        #import cProfile, pstats
        #profiler = cProfile.Profile()
        #profiler.enable()

        graphsignal._tracer.debug_mode = False

        calls = 10000
        start_ns = time.perf_counter_ns()
        for _ in range(calls):
            with Span(operation='op1') as span:
                span.set_tag('k1', 'v1')
                span.set_payload('test', 'test string data')
        took_ns = time.perf_counter_ns() - start_ns

        #stats = pstats.Stats(profiler).sort_stats('time')
        #stats.print_stats()

        self.assertTrue(took_ns / calls < 200 * 1e3) # less than 200 microseconds per trace

    def test_encode_data_payload(self):
        content_type, content_bytes = encode_data_payload(['text\n', 2.0, float('nan')])
        self.assertEqual(content_type, 'application/json')
        self.assertEqual(content_bytes, b'["text\\n", 2.0, NaN]')


def find_log_entry(store, text):
    for entry in store._logs:
        if entry.message and text in entry.message:
            return entry
        if entry.exception and text in entry.exception:
            return entry
