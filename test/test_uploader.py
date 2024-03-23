import unittest
import sys
import time
import logging
import threading
import json
import gzip
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler

from unittest.mock import patch, Mock

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal import client

logger = logging.getLogger('graphsignal')


class UploaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(
            api_key='k1',
            deployment='d1',
            upload_on_shutdown=False,
            debug_mode=True)
        graphsignal._tracer.uploader().clear()

    def tearDown(self):
        graphsignal._tracer.uploader().clear()
        graphsignal.shutdown()

    @patch.object(client.DefaultApi, 'upload_spans')
    def test_flush(self, mocked_upload_spans):
        model = client.Span(span_id='1', start_us=0, end_us=0)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().flush()

        mocked_upload_spans.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(client.DefaultApi, 'upload_spans')
    def test_flush_in_thread(self, mocked_upload_spans):
        graphsignal._tracer.uploader().FLUSH_DELAY_SEC = 0.01
        model = client.Span(span_id='1', start_us=0, end_us=0)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().flush_in_thread()
        time.sleep(0.1)

        mocked_upload_spans.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(client.DefaultApi, 'upload_spans')
    def test_flush_in_thread_cancelled(self, mocked_upload_spans):
        graphsignal._tracer.uploader().FLUSH_DELAY_SEC = 5
        model = client.Span(span_id='1', start_us=0, end_us=0)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().flush_in_thread()
        self.assertIsNotNone(graphsignal._tracer.uploader()._flush_timer)
        graphsignal._tracer.uploader().flush()
        self.assertIsNone(graphsignal._tracer.uploader()._flush_timer)

        mocked_upload_spans.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(client.DefaultApi, 'upload_spans')
    def test_flush_fail(self, mocked_upload_spans):
        def side_effect(*args):
            raise Exception("Ex1")
        mocked_upload_spans.side_effect = side_effect

        model = client.Span(span_id='1', start_us=0, end_us=0)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().flush()

        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 2)

    def test_upload_spans(self):
        graphsignal._tracer.api_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.set_response_data(b'{}')
        server.start()

        model = client.Span(span_id='s1', start_us=0, end_us=0)
        graphsignal._tracer.uploader().upload_span(model)
        graphsignal._tracer.uploader().flush()

        self.assertEqual(json.loads(server.get_request_data())[0]['span_id'], 's1')

        server.join()

class TestServer(threading.Thread):
    def __init__(self, port, delay=None, handler_func=None):
        self.port = port
        RequestHandler.delay = delay
        RequestHandler.handler_func = [handler_func]
        threading.Thread.__init__(self)
        self.server = HTTPServer(('localhost', self.port), RequestHandler)

    def get_request_data(self):
        return RequestHandler.request_data

    def set_response_data(self, response_data):
        RequestHandler.response_data = response_data

    def set_response_code(self, response_code):
        RequestHandler.response_code = response_code

    def run(self):
        self.server.handle_request()


class RequestHandler(BaseHTTPRequestHandler):
    delay = None
    handler_func = None
    request_data = None
    response_data = None
    response_code = 200
    response_type = 'application/octet-stream'

    def do_GET(self):
        if self.delay:
            time.sleep(self.delay)

        if RequestHandler.handler_func:
            RequestHandler.handler_func[0]()

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', RequestHandler.response_type)
        self.end_headers()
        self.wfile.write(RequestHandler.response_data)

    def do_POST(self):
        if self.delay:
            time.sleep(self.delay)

        self.request_url = self.path
        content_len = int(self.headers.get('content-length'))

        if self.headers.get('content-encoding') == 'gzip':
            decompressed_data = gzip.GzipFile(
                fileobj=BytesIO(self.rfile.read(content_len))).read()
            RequestHandler.request_data = decompressed_data
        else:
            RequestHandler.request_data = self.rfile.read(content_len)

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', RequestHandler.response_type)
        self.end_headers()
        self.wfile.write(RequestHandler.response_data)