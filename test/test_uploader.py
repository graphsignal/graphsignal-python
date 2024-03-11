import unittest
import sys
import json
import threading
import time
import gzip
import logging
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import urlopen
from urllib.request import Request
from urllib.parse import urlencode
from urllib.error import URLError


from unittest.mock import patch, Mock

import graphsignal
from graphsignal.uploader import Uploader
from graphsignal.proto import signals_pb2

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

    @patch.object(Uploader, '_post', return_value=signals_pb2.UploadResponse().SerializeToString())
    def test_flush(self, mocked_post):
        proto = signals_pb2.Span()
        graphsignal._tracer.uploader().upload_span(proto)
        graphsignal._tracer.uploader().flush()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(Uploader, '_post', return_value=signals_pb2.UploadResponse().SerializeToString())
    def test_flush_in_thread(self, mocked_post):
        graphsignal._tracer.uploader().FLUSH_DELAY_SEC = 0.01
        proto = signals_pb2.Span()
        graphsignal._tracer.uploader().upload_span(proto)
        graphsignal._tracer.uploader().flush_in_thread()
        time.sleep(0.1)

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(Uploader, '_post', return_value=signals_pb2.UploadResponse().SerializeToString())
    def test_flush_in_thread_cancelled(self, mocked_post):
        graphsignal._tracer.uploader().FLUSH_DELAY_SEC = 5
        proto = signals_pb2.Span()
        graphsignal._tracer.uploader().upload_span(proto)
        graphsignal._tracer.uploader().flush_in_thread()
        self.assertIsNotNone(graphsignal._tracer.uploader()._flush_timer)
        graphsignal._tracer.uploader().flush()
        self.assertIsNone(graphsignal._tracer.uploader()._flush_timer)

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 0)

    @patch.object(Uploader, '_post', return_value=signals_pb2.UploadResponse().SerializeToString())
    def test_flush_fail(self, mocked_post):
        def side_effect(*args):
            raise URLError("Ex1")
        mocked_post.side_effect = side_effect

        proto = signals_pb2.Span()
        graphsignal._tracer.uploader().upload_span(proto)
        graphsignal._tracer.uploader().upload_span(proto)
        graphsignal._tracer.uploader().flush()

        self.assertEqual(len(graphsignal._tracer.uploader()._buffer), 2)

    def test_post(self):
        graphsignal._tracer.api_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.set_response_data(
            signals_pb2.UploadResponse().SerializeToString())
        server.start()

        proto = signals_pb2.Span()
        proto.span_id = 't1'
        upload_request = signals_pb2.UploadRequest()
        upload_request.spans.append(proto)
        upload_request.upload_ms = 123
        graphsignal._tracer.uploader()._post(
            'signals', upload_request.SerializeToString())

        received_upload_request = signals_pb2.UploadRequest()
        received_upload_request.ParseFromString(server.get_request_data())
        self.assertEqual(
            received_upload_request.spans[0].span_id, 't1')
        self.assertEqual(received_upload_request.upload_ms, 123)

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

        decompressed_data = gzip.GzipFile(
            fileobj=BytesIO(self.rfile.read(content_len))).read()
        RequestHandler.request_data = decompressed_data

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', RequestHandler.response_type)
        self.end_headers()
        self.wfile.write(RequestHandler.response_data)
