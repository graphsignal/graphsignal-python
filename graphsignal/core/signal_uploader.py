import time
import threading
import logging
import gzip
from io import BytesIO

import requests
import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class SignalUploader:
    MAX_BUFFER_SIZE = 10000
    FLUSH_DELAY_SEC = 5

    def __init__(self):
        self._buffer = []
        self._buffer_lock = threading.Lock()
        self._flush_lock = threading.Lock()

    def setup(self):
        pass

    def clear(self):
        with self._buffer_lock:
            self._buffer = []

    def upload_span(self, span):
        self.upload_signal(span)

    def upload_metric(self, metric):
        self.upload_signal(metric)

    def upload_log_batch(self, log_batch):
        self.upload_signal(log_batch)

    def upload_signal(self, signal):
        with self._buffer_lock:
            self._buffer.append(signal)
            if len(self._buffer) > self.MAX_BUFFER_SIZE:
                self._buffer = self._buffer[-self.MAX_BUFFER_SIZE:]

    def flush(self):
        with self._flush_lock:
            with self._buffer_lock:
                if len(self._buffer) == 0:
                    return
                outgoing = self._buffer
                self._buffer = []

            try:
                upload_start = time.time()

                upload_request = self._create_upload_request(outgoing)
                self._post('api/v1/signals/', upload_request)

                logger.debug('Upload took %.3f sec', time.time() - upload_start)
            except Exception:
                logger.debug('Failed uploading signals, will retry', exc_info=True)
                with self._buffer_lock:
                    self._buffer[:0] = outgoing

    def _post(self, endpoint, data):
        logger.debug('Posting data to %s/%s', graphsignal._ticker.api_url, endpoint)

        url = f"{graphsignal._ticker.api_url}/{endpoint}"
        data_gzip = self._gzip_data(data)

        headers = {
            'X-API-Key': graphsignal._ticker.api_key,
            'Content-Type': 'application/octet-stream',
            'Content-Encoding': 'gzip'
        }

        try:
            resp = requests.post(
                url,
                data=data_gzip,
                headers=headers,
                timeout=10
            )
            resp.raise_for_status()
            # requests automatically decompresses gzipped responses
            return resp.content
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg = e.response.text
            logger.debug('Error message from server: %s', error_msg)
            raise

    def _create_upload_request(self, outgoing):
        upload_request = signals_pb2.UploadRequest()

        for signal in outgoing:
            if isinstance(signal, signals_pb2.Span):
                upload_request.spans.append(signal)
            elif isinstance(signal, signals_pb2.Metric):
                upload_request.metrics.append(signal)
            elif isinstance(signal, signals_pb2.LogBatch):
                upload_request.log_batches.append(signal)

        upload_request.upload_ts = time.time_ns()
        return upload_request.SerializeToString()

    def _gzip_data(self, data):
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

