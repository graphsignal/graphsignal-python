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

logger = logging.getLogger('graphsignal')


class UploaderTest(unittest.TestCase):
    def setUp(self):
        logger.setLevel(logging.DEBUG)
        graphsignal._get_uploader().clear()
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal._get_uploader().clear()
        graphsignal.shutdown()

    @patch.object(Uploader, 'post_json')
    def test_flush(self, mocked_post_json):
        graphsignal._get_uploader().upload_window({'m1': 1})
        graphsignal._get_uploader().flush()

        mocked_post_json.assert_called_once_with('windows', [{'m1': 1}])
        self.assertEqual(len(graphsignal._get_uploader().buffer['windows']), 0)

    @patch.object(Uploader, 'post_json')
    def test_flush_in_thread(self, mocked_post_json):
        graphsignal._get_uploader().upload_window({'m1': 1})
        graphsignal._get_uploader().flush_in_thread()
        graphsignal.tick()

        mocked_post_json.assert_called_once_with('windows', [{'m1': 1}])
        self.assertEqual(len(graphsignal._get_uploader().buffer['windows']), 0)

    @patch.object(Uploader, 'post_json')
    def test_flush_fail(self, mocked_post_json):
        def side_effect(*args):
            raise URLError("Ex1")
        mocked_post_json.side_effect = side_effect

        graphsignal._get_uploader().upload_window({'m1': 1})
        graphsignal._get_uploader().upload_window({'m2': 2})
        graphsignal._get_uploader().flush()

        self.assertEqual(len(graphsignal._get_uploader().buffer['windows']), 2)

    def test_post_json(self):
        graphsignal._get_uploader().collector_url = 'http://localhost:5005'

        server = TestServer(5005)
        server.start()

        graphsignal._get_uploader().post_json('metrics', [{'m1': 1}])

        request_data = json.loads(server.get_request_data())
        self.assertEqual(request_data[0]['m1'], 1)

        server.join()


class TestServer(threading.Thread):
    def __init__(self, port, delay=None, handler_func=None):
        self.port = port
        RequestHandler.delay = delay
        RequestHandler.handler_func = [handler_func]
        RequestHandler.response_data = '{}'
        RequestHandler.response_code = 200
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


def timestamp():
    return int(time.time())


class RequestHandler(BaseHTTPRequestHandler):
    delay = None
    handler_func = None
    request_data = None
    response_data = '{}'
    response_code = 200

    def do_GET(self):
        if self.delay:
            time.sleep(self.delay)

        if RequestHandler.handler_func:
            RequestHandler.handler_func[0]()

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(RequestHandler.response_data.encode('utf-8'))

    def do_POST(self):
        if self.delay:
            time.sleep(self.delay)

        self.request_url = self.path
        content_len = int(self.headers.get('content-length'))

        decompressed_data = gzip.GzipFile(
            fileobj=BytesIO(self.rfile.read(content_len))).read()
        RequestHandler.request_data = decompressed_data.decode('utf-8')

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(RequestHandler.response_data.encode('utf-8'))
