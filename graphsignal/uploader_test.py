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
            debug_mode=True)
        graphsignal._agent.uploader.clear()

    def tearDown(self):
        graphsignal._agent.uploader.clear()
        graphsignal.shutdown()

    @patch.object(Uploader, '_post')
    def test_flush(self, mocked_post):
        signal = signals_pb2.MLSignal()
        graphsignal._agent.uploader.upload_signal(signal)
        graphsignal._agent.uploader.flush()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._agent.uploader.buffer), 0)

    @patch.object(Uploader, '_post',
                  return_value=signals_pb2.UploadResponse().SerializeToString())
    def test_flush_in_thread(self, mocked_post):
        signal = signals_pb2.MLSignal()
        graphsignal._agent.uploader.upload_signal(signal)
        graphsignal._agent.uploader.flush_in_thread()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._agent.uploader.buffer), 0)

    @patch.object(Uploader, '_post')
    def test_flush_fail(self, mocked_post):
        def side_effect(*args):
            raise URLError("Ex1")
        mocked_post.side_effect = side_effect

        signal = signals_pb2.MLSignal()
        graphsignal._agent.uploader.upload_signal(signal)
        graphsignal._agent.uploader.upload_signal(signal)
        graphsignal._agent.uploader.flush()

        self.assertEqual(len(graphsignal._agent.uploader.buffer), 2)

    def test_post(self):
        graphsignal._agent.uploader.agent_api_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.set_response_data(
            signals_pb2.UploadResponse().SerializeToString())
        server.start()

        signal = signals_pb2.MLSignal()
        signal.model_name = 'm1'
        upload_request = signals_pb2.UploadRequest()
        upload_request.ml_signals.append(signal)
        upload_request.upload_ms = 123
        graphsignal._agent.uploader._post(
            'signals', upload_request.SerializeToString())

        received_upload_request = signals_pb2.UploadRequest()
        received_upload_request.ParseFromString(server.get_request_data())
        self.assertEqual(
            received_upload_request.ml_signals[0].model_name, 'm1')
        self.assertEqual(received_upload_request.upload_ms, 123)

        server.join()

    def test_post(self):
        graphsignal._agent.uploader.agent_api_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.set_response_data(
            signals_pb2.UploadResponse().SerializeToString())
        server.start()

        signal = signals_pb2.MLSignal()
        signal.model_name = 'm1'
        upload_request = signals_pb2.UploadRequest()
        upload_request.ml_signals.append(signal)
        upload_request.upload_ms = 123
        graphsignal._agent.uploader._post(
            'signals', upload_request.SerializeToString())

        received_upload_request = signals_pb2.UploadRequest()
        received_upload_request.ParseFromString(server.get_request_data())
        self.assertEqual(
            received_upload_request.ml_signals[0].model_name, 'm1')
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
