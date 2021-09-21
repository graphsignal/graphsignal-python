import time
import logging
import threading
import sys
import traceback
import platform
from google.protobuf.json_format import MessageToDict

import graphsignal
from graphsignal import statistics
from graphsignal.uploader import Uploader
from graphsignal.predictions import Prediction
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')

MAX_METADATA_SIZE = 10

MAX_BUFFER_SIZE = 100
MIN_WINDOW_length = 600

_session_index = {}
_session_index_lock = threading.Lock()


class Session(object):
    __slots__ = [
        '_deployment_name',
        '_metadata',
        '_current_window',
        '_prediction_buffer',
        '_buffer_size',
        '_metric_updaters',
        '_update_lock',
        '_is_updated',
        '_upload_timer'
    ]

    def __init__(self, deployment_name):
        self._deployment_name = deployment_name
        self._metadata = {}
        self._update_lock = threading.Lock()
        self._reset_current_window()
        self._current_window.model.deployment_name = deployment_name

    def _reset_current_window(self):
        self._current_window = metrics_pb2.PredictionWindow()
        self._prediction_buffer = []
        self._buffer_size = 0
        self._metric_updaters = {}
        self._is_updated = False

    def _set_updated(self):
        self._is_updated = True
        if self._upload_window():
            graphsignal._get_uploader().flush_in_thread()

    def set_metadata(self, key=None, value=None):
        '''
        Set model metadata.

        Args:
            key (:obj:`str`):
                Metadata entry key.
            value (:obj:`str` or :obj:`int` or :obj:`float`):
                Metadata entry value.
        '''

        if not isinstance(key, str) or len(key) > 250:
            logger.error('invalid metadata entry key format')
            return
        if not isinstance(value, (str, int, float)) or len(str(value)) > 2500:
            logger.error(
                'invalid metadata entry value format for key: {0}'.format(key))
            return

        if len(self._current_window.model.metadata) >= MAX_METADATA_SIZE:
            logger.error(
                'too many metadata entries, max={0}'.format(MAX_METADATA_SIZE))
            return

        self._metadata[key] = str(value)

    def log_prediction(
            self,
            input_data=None,
            input_columns=None,
            output_data=None,
            output_columns=None,
            actual_timestamp=None):
        '''
        Log single or batch model prediction.

        See `Supported Data Formats <https://graphsignal.com/docs/integrations/python/supported-data-formats/>`_
        for detailed description of data types and formats.

        Computed data statistics are computed and uploaded for time windows at certain intervals and on process exit.
        No raw data is uploaded.

        Args:
            input_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Input data instances.
            input_columns (:obj:`list`, optional):
                A list of input data column names. If not provided, column names are inferred from `input_data`.
            output_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Output data instances.
            output_columns (:obj:`list`, optional):
                A list of output data column names. If not provided, column names are inferred from `output_data`.
            actual_timestamp (:obj:`int`, optional, default is current timestamp):
                Actual unix timestamp of the measurement, when different from current timestamp.
        '''

        data_size = max(
            statistics.estimate_size(input_data),
            statistics.estimate_size(output_data))
        if data_size == 0:
            logger.debug('Logged empty data')
            return

        self._buffer_size += data_size
        self._current_window.num_predictions += data_size

        timestamp = actual_timestamp if actual_timestamp is not None else int(
            time.time())
        if self._current_window.start_ts == 0:
            self._current_window.start_ts = timestamp
        self._current_window.end_ts = timestamp

        with self._update_lock:
            self._prediction_buffer.append(Prediction(
                input_data=input_data,
                input_columns=input_columns,
                output_data=output_data,
                output_columns=output_columns))

        if self._buffer_size > graphsignal._get_config().buffer_size:
            self._merge_buffer()

        self._set_updated()

    def _merge_buffer(self):
        if self._buffer_size > 0:
            with self._update_lock:
                current_buffer = self._prediction_buffer
                self._prediction_buffer = []
                self._buffer_size = 0
            try:
                statistics.update_metrics(
                    self._metric_updaters, self._current_window, current_buffer)
            except BaseException:
                logger.error('Error updating metrics', exc_info=True)

    def _upload_window(self, force=False):
        if not self._is_updated:
            return False

        if not force:
            if self._current_window.end_ts - \
                    self._current_window.start_ts < graphsignal._get_config().window_seconds:
                return False

        self._merge_buffer()
        for metric_updater in self._metric_updaters.values():
            metric_updater.finalize()

        self._current_window.model.deployment_name = self._deployment_name
        for key, value in self._metadata.items():
            self._current_window.model.metadata[key] = value

        with self._update_lock:
            window = self._current_window
            self._reset_current_window()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Uploading window:')
            logger.debug(MessageToDict(window))

        graphsignal._get_uploader().upload_window(window)
        return True


def get_session(deployment_name):
    if not deployment_name or len(deployment_name) > 250:
        raise ValueError('invalid deployment_name format')

    with _session_index_lock:
        if deployment_name in _session_index:
            return _session_index[deployment_name]
        else:
            sess = Session(deployment_name)
            _session_index[deployment_name] = sess
            return sess


def reset_all():
    with _session_index_lock:
        _session_index.clear()


def upload_all(force=False):
    session_list = None
    with _session_index_lock:
        session_list = _session_index.values()

    uploaded = False
    for session in session_list:
        if session._upload_window(force=force):
            uploaded = True

    return uploaded
