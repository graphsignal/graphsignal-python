import unittest
import sys
import time
import logging

from unittest.mock import patch, Mock

import graphsignal
from graphsignal.core.signal_uploader import SignalUploader
from graphsignal.proto import signals_pb2
from test.http_server import HttpTestServer

logger = logging.getLogger('graphsignal')


class SignalUploaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False
        graphsignal._ticker.signal_uploader().clear()

    def tearDown(self):
        graphsignal._ticker.signal_uploader().clear()
        graphsignal.shutdown()

    @patch.object(SignalUploader, '_post')
    def test_flush(self, mocked_post):
        span_pb = signals_pb2.Span()
        span_pb.span_id = 's1'
        span_pb.trace_id = 't1'
        span_pb.start_ts = 0
        span_pb.end_ts = 0
        span_pb.name = 'op1'
        
        graphsignal._ticker.signal_uploader().upload_span(span_pb)
        graphsignal._ticker.signal_uploader().flush()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._ticker.signal_uploader()._buffer), 0)

    @patch.object(SignalUploader, '_post')
    def test_flush_fail(self, mocked_post):
        def side_effect(*args):
            raise Exception("Ex1")
        mocked_post.side_effect = side_effect

        span_pb = signals_pb2.Span()
        span_pb.span_id = 's1'
        span_pb.trace_id = 't1'
        span_pb.start_ts = 0
        span_pb.end_ts = 0
        span_pb.name = 'op1'
        
        graphsignal._ticker.signal_uploader().upload_span(span_pb)
        graphsignal._ticker.signal_uploader().upload_span(span_pb)
        graphsignal._ticker.signal_uploader().flush()

        self.assertEqual(len(graphsignal._ticker.signal_uploader()._buffer), 2)

    def test_upload_signals(self):
        graphsignal._ticker.api_url = 'http://localhost:5005'

        server = HttpTestServer(5005)
        server.set_response_data(b'')
        server.start()
        server.wait_ready()

        span_pb = signals_pb2.Span()
        span_pb.span_id = 's1'
        span_pb.trace_id = 't1'
        span_pb.start_ts = 0
        span_pb.end_ts = 1000
        span_pb.name = 'span1'

        metric_pb = signals_pb2.Metric()
        metric_pb.name = 'metric1'
        metric_pb.type = signals_pb2.Metric.MetricType.GAUGE_METRIC
        datapoint = metric_pb.datapoints.add()
        datapoint.gauge = 42.0
        datapoint.measurement_ts = int(time.time())

        log_batch_pb = signals_pb2.LogBatch()
        # Attach logger name as a tag and add a single log entry.
        log_entry_pb = log_batch_pb.log_entries.add()
        log_entry_pb.level = signals_pb2.LogEntry.LogLevel.INFO_LEVEL
        log_entry_pb.message = 'test log'
        log_entry_pb.log_ts = time.time_ns()

        graphsignal._ticker.signal_uploader().upload_span(span_pb)
        graphsignal._ticker.signal_uploader().upload_metric(metric_pb)
        graphsignal._ticker.signal_uploader().upload_log_batch(log_batch_pb)
        graphsignal._ticker.signal_uploader().flush()

        request_data = server.get_request_data()
        upload_request = signals_pb2.UploadRequest()
        upload_request.ParseFromString(request_data)

        server.join(timeout=2.0)

        self.assertEqual(len(upload_request.spans), 1)
        self.assertEqual(upload_request.spans[0].span_id, 's1')

        self.assertEqual(len(upload_request.metrics), 1)
        self.assertEqual(upload_request.metrics[0].name, 'metric1')
        self.assertEqual(len(upload_request.metrics[0].datapoints), 1)
        self.assertEqual(upload_request.metrics[0].datapoints[0].gauge, 42.0)
        
        self.assertEqual(len(upload_request.log_batches), 1)
        batch = upload_request.log_batches[0]
        self.assertEqual(len(batch.log_entries), 1)
        self.assertEqual(batch.log_entries[0].level, signals_pb2.LogEntry.LogLevel.INFO_LEVEL)
