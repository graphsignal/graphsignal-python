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
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class Uploader:
    MAX_BUFFER_SIZE = 10000
    FLUSH_DELAY_SEC = 5

    def __init__(self):
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._flush_timer = None

    def setup(self):
        pass

    def clear(self):
        with self._buffer_lock:
            self._buffer = []

    def upload_span(self, span):
        self.upload_signal(span)

    def upload_score(self, score):
        self.upload_signal(score)

    def upload_metric(self, metric):
        self.upload_signal(metric)

    def upload_log_entry(self, log_entry):
        self.upload_signal(log_entry)

    def upload_signal(self, signal):
        with self._buffer_lock:
            self._buffer.append(signal)
            if len(self._buffer) > self.MAX_BUFFER_SIZE:
                self._buffer = self._buffer[-self.MAX_BUFFER_SIZE:]

    def flush_in_thread(self):
        if self._flush_timer is None:
            self._flush_timer = threading.Timer(self.FLUSH_DELAY_SEC, self.flush)
            self._flush_timer.start()

    def flush(self):
        with self._flush_lock:
            if self._flush_timer is not None:
                self._flush_timer.cancel()
            self._flush_timer = None

            with self._buffer_lock:
                if len(self._buffer) == 0:
                    return
                outgoing = self._buffer
                self._buffer = []
            try:
                upload_start = time.time()
                payload = _create_upload_request(outgoing)
                content = self._post('signals', payload)
                _create_upload_response(content)
                logger.debug('Upload took %.3f sec (%dB)', time.time() - upload_start, len(payload))
            except URLError:
                logger.debug('Failed uploading signals, will retry', exc_info=True)
                with self._buffer_lock:
                    self._buffer[:0] = outgoing
            except Exception:
                logger.error('Error uploading signals', exc_info=True)

    def _post(self, endpoint, data):
        logger.debug('Posting data to %s/%s', graphsignal._tracer.api_url, endpoint)

        api_key_64 = _base64_encode(graphsignal._tracer.api_key + ':').replace('\n', '')
        headers = {
            'Accept-Encoding': 'gzip',
            'Authorization': "Basic %s" % api_key_64,
            'Content-Type': 'application/octet-stream',
            'Content-Encoding': 'gzip'
        }

        data_gzip = _gzip_data(data)

        request = Request(
            url=graphsignal._tracer.api_url + '/' + endpoint,
            data=data_gzip,
            headers=headers)

        try:
            resp = urlopen(request, timeout=10)
            content = resp.read()
            if resp.info():
                content_type = resp.info().get('Content-Encoding')
                if content_type == 'gzip':
                    content = _gunzip_data(content)
            resp.close()
            return content
        except HTTPError as herr:
            logger.debug('Error message from server: %s',
                         herr.read().decode('utf-8'))
            raise herr


def _create_upload_request(outgoing):
    upload_request = signals_pb2.UploadRequest()
    for signal in outgoing:
        if isinstance(signal, signals_pb2.Span):
            upload_request.spans.append(signal)
        elif isinstance(signal, signals_pb2.Score):
            upload_request.scores.append(signal)
        elif isinstance(signal, signals_pb2.Metric):
            upload_request.metrics.append(signal)
        elif isinstance(signal, signals_pb2.LogEntry):
            upload_request.log_entries.append(signal)
    upload_request.upload_ms = int(time.time() * 1e3)
    return upload_request.SerializeToString()


def _create_upload_response(content):
    upload_response = signals_pb2.UploadResponse()
    upload_response.ParseFromString(content)
    return upload_response


def _gzip_data(data):
    gzip_out = BytesIO()
    with gzip.GzipFile(fileobj=gzip_out, mode="w") as out_file:
        out_file.write(data)
        out_file.close()

    gzip_out_val = gzip_out.getvalue()
    if isinstance(gzip_out_val, str):
        data_gzip = bytearray(gzip_out.getvalue())
    else:
        data_gzip = gzip_out.getvalue()

    return data_gzip


def _gunzip_data(data):
    return gzip.GzipFile('', 'r', 0, BytesIO(data)).read()


def _base64_encode(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')
