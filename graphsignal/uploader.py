import os
import time
import json
import gzip
import sys
import threading
import base64
import logging
from io import BytesIO
from urllib.request import urlopen
from urllib.request import Request
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.error import HTTPError

import graphsignal

logger = logging.getLogger('graphsignal')


class Uploader(object):
    MAX_BUFFER_SIZE = 2500

    def __init__(self):
        self.collector_url = 'https://log-api.graphsignal.ai'
        self.buffer = {}
        self.buffer_lock = threading.Lock()
        self.flush_lock = threading.Lock()

    def configure(self):
        if 'GRAPHSIGNAL_LOG_API_URL' in os.environ:
            self.collector_url = os.environ['GRAPHSIGNAL_LOG_API_URL']

    def clear(self):
        with self.buffer_lock:
            self.buffer = {}

    def upload_window(self, window):
        self.upload_json('windows', [window])

    def upload_json(self, endpoint, data):
        with self.buffer_lock:
            if endpoint not in self.buffer:
                self.buffer[endpoint] = []
            self.buffer[endpoint].extend(data)
            if len(self.buffer[endpoint]) > self.MAX_BUFFER_SIZE:
                self.buffer[endpoint] = self.buffer[endpoint][-self.MAX_BUFFER_SIZE:]

    def flush_in_thread(self):
        threading.Thread(target=self.flush).start()

    def flush(self):
        with self.flush_lock:
            for endpoint, outgoing in self.buffer.items():
                with self.buffer_lock:
                    if len(outgoing) == 0:
                        continue
                    self.buffer[endpoint] = []
                try:
                    upload_start = time.time()
                    self.post_json(endpoint, outgoing)
                    logger.debug(
                        'Upload to %s/%s took %.3f sec', self.collector_url, endpoint, time.time() - upload_start)
                except URLError:
                    logger.debug(
                        'Failed uploading to %s/%s', self.collector_url, endpoint, exc_info=True)
                    with self.buffer_lock:
                        self.buffer[endpoint][:0] = outgoing
                except Exception:
                    logger.debug(
                        'Exception when uploading to %s/%s', self.collector_url, endpoint, exc_info=True)

    def post_json(self, endpoint, data):
        logger.debug('Posting data to %s/%s: %s',
                     self.collector_url, endpoint, data)

        api_key_64 = _base64_encode(
            graphsignal._get_config().api_key + ':').replace('\n', '')
        headers = {
            'Accept-Encoding': 'gzip',
            'Authorization': "Basic %s" % api_key_64,
            'Content-Type': 'application/json',
            'Content-Encoding': 'gzip'
        }

        gzip_out = BytesIO()
        with gzip.GzipFile(fileobj=gzip_out, mode="w") as out_file:
            out_file.write(json.dumps(data).encode('utf-8'))
            out_file.close()

        gzip_out_val = gzip_out.getvalue()
        if isinstance(gzip_out_val, str):
            data_gzip = bytearray(gzip_out.getvalue())
        else:
            data_gzip = gzip_out.getvalue()
        request = Request(
            url=self.collector_url + '/' + endpoint,
            data=data_gzip,
            headers=headers)

        try:
            resp = urlopen(request, timeout=10)
            result_data = resp.read()
            if resp.info():
                content_type = resp.info().get('Content-Encoding')
                if content_type == 'gzip':
                    result_data = gzip.GzipFile(
                        '', 'r', 0, BytesIO(result_data)).read()
            resp.close()
            return json.loads(result_data.decode('utf-8'))
        except HTTPError as herr:
            logger.debug(
                'Failed uploading to %s/%s', self.collector_url, endpoint)
            logger.debug('Error message from server: %s',
                         herr.read().decode('utf-8'))
            return {}


def _timestamp():
    return int(time.time())


def _base64_encode(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')
