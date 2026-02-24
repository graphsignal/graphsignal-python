import time
import threading
import socket
import gzip
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler


class HttpTestServer(threading.Thread):
    def __init__(self, port=None, delay=None, handler_func=None):
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                s.listen(1)
                port = s.getsockname()[1]
        
        self.port = port
        RequestHandler.delay = delay
        RequestHandler.handler_func = [handler_func]
        threading.Thread.__init__(self, daemon=True)
        self.server = HTTPServer(('localhost', self.port), RequestHandler)
        self._ready = threading.Event()

    def get_port(self):
        return self.port

    def get_request_data(self):
        return RequestHandler.request_data

    def set_response_data(self, response_data):
        RequestHandler.response_data = response_data

    def set_response_code(self, response_code):
        RequestHandler.response_code = response_code

    def run(self):
        self._ready.set()
        self.server.handle_request()

    def wait_ready(self, timeout=1.0):
        self._ready.wait(timeout=timeout)


class RequestHandler(BaseHTTPRequestHandler):
    delay = None
    request_data = None
    response_data = None
    response_code = 200
    response_type = 'application/json'

    def do_GET(self):
        if self.delay:
            time.sleep(self.delay)

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', RequestHandler.response_type)
        self.end_headers()
        if RequestHandler.response_data:
            self.wfile.write(RequestHandler.response_data)
            self.wfile.flush()

    def do_POST(self):
        if self.delay:
            time.sleep(self.delay)

        self.request_url = self.path
        content_len = int(self.headers.get('content-length', 0))

        if content_len > 0:
            if self.headers.get('content-encoding') == 'gzip':
                decompressed_data = gzip.GzipFile(
                    fileobj=BytesIO(self.rfile.read(content_len))).read()
                RequestHandler.request_data = decompressed_data
            else:
                RequestHandler.request_data = self.rfile.read(content_len)

        self.send_response(RequestHandler.response_code)
        self.send_header('Content-Type', RequestHandler.response_type)
        self.end_headers()
        if RequestHandler.response_data:
            self.wfile.write(RequestHandler.response_data)
            self.wfile.flush()

    def log_message(self, format, *args):
        pass

