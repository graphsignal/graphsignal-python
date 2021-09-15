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
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')


class UploaderTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal._get_uploader().clear()
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal._get_uploader().clear()
        graphsignal.shutdown()

    @patch.object(Uploader, '_post')
    def test_flush(self, mocked_post):
        window = metrics_pb2.PredictionWindow()
        graphsignal._get_uploader().upload_window(window)
        graphsignal._get_uploader().flush()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._get_uploader().buffer), 0)

    @patch.object(Uploader, '_post',
                  return_value=metrics_pb2.UploadResponse().SerializeToString())
    def test_flush_in_thread(self, mocked_post):
        window = metrics_pb2.PredictionWindow()
        graphsignal._get_uploader().upload_window(window)
        graphsignal._get_uploader().flush_in_thread()
        graphsignal.tick()

        mocked_post.assert_called_once()
        self.assertEqual(len(graphsignal._get_uploader().buffer), 0)

    @patch.object(Uploader, '_post')
    def test_flush_fail(self, mocked_post):
        def side_effect(*args):
            raise URLError("Ex1")
        mocked_post.side_effect = side_effect

        window = metrics_pb2.PredictionWindow()
        graphsignal._get_uploader().upload_window(window)
        graphsignal._get_uploader().upload_window(window)
        graphsignal._get_uploader().flush()

        self.assertEqual(len(graphsignal._get_uploader().buffer), 2)

    def test_post(self):
        graphsignal._get_uploader().collector_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.set_response_data(
            metrics_pb2.UploadResponse().SerializeToString())
        server.start()

        window = metrics_pb2.PredictionWindow()
        window.model.deployment_name = 'd1'
        upload_request = metrics_pb2.UploadRequest()
        upload_request.windows.append(window)
        upload_request.upload_ts = 123
        graphsignal._get_uploader()._post('metrics', upload_request.SerializeToString())

        received_upload_request = metrics_pb2.UploadRequest()
        received_upload_request.ParseFromString(server.get_request_data())
        self.assertEqual(
            received_upload_request.windows[0].model.deployment_name, 'd1')
        self.assertEqual(received_upload_request.upload_ts, 123)

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
