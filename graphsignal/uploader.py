import time
import threading
import logging

import graphsignal
from graphsignal import client
from graphsignal.client.rest import ApiException

logger = logging.getLogger('graphsignal')


class Uploader:
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

    def upload_error(self, error):
        self.upload_signal(error)

    def upload_metric(self, metric):
        self.upload_signal(metric)

    def upload_log_entry(self, log_entry):
        self.upload_signal(log_entry)

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

                configuration = client.Configuration(
                    host = graphsignal._tracer.api_url,   
                )
                configuration.api_key['APIKeyHeader'] = graphsignal._tracer.api_key
                with client.ApiClient(configuration) as api_client:
                    api_instance = client.DefaultApi(api_client)
    
                    spans = [signal for signal in outgoing if isinstance(signal, client.Span)]
                    if len(spans) > 0:
                        api_instance.upload_spans(spans)

                    errors = [signal for signal in outgoing if isinstance(signal, client.Error)]
                    if len(errors) > 0:
                        api_instance.upload_errors(errors)

                    metrics = [signal for signal in outgoing if isinstance(signal, client.Metric)]
                    if len(metrics) > 0:
                        api_instance.upload_metrics(metrics)

                    log_entries = [signal for signal in outgoing if isinstance(signal, client.LogEntry)]
                    if len(log_entries) > 0:
                        api_instance.upload_logs(log_entries)

                logger.debug('Upload took %.3f sec', time.time() - upload_start)
            except Exception:
                logger.debug('Failed uploading signals, will retry', exc_info=True)
                with self._buffer_lock:
                    self._buffer[:0] = outgoing
