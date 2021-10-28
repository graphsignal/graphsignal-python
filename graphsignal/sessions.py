import time
import logging
import threading
import sys
import traceback
import platform
from google.protobuf.json_format import MessageToDict

import graphsignal
from graphsignal import statistics
from graphsignal.windows import PredictionRecord, ExceptionRecord, GroundTruthRecord, WindowUpdater
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')

MAX_METADATA_SIZE = 10
MAX_EXTRA_INFO_SIZE = 10
MAX_SEGMENT_LENGTH = 50

_session_index = {}
_session_index_lock = threading.Lock()


class Session(object):
    __slots__ = [
        '_deployment_name',
        '_metadata',
        '_update_lock',
        '_current_window_updater',
        '_current_window_timestamp'
    ]

    def __init__(self, deployment_name):
        self._deployment_name = deployment_name
        self._metadata = {}
        self._update_lock = threading.Lock()
        self._current_window_updater = None
        self._current_window_timestamp = None

    def _tumble_window(self, timestamp, force=False, flush=True):
        window_seconds = graphsignal._get_config().window_seconds
        time_bucket = int(timestamp / window_seconds) * window_seconds

        if not force and time_bucket == self._current_window_timestamp:
            # window matches, no need to tumble/create
            return True

        if self._current_window_timestamp and time_bucket < self._current_window_timestamp:
            logger.error(
                'Cannot tumble time windows backwards. Please log in chronological order when providing timestamps explicitely.')
            return False

        if self._current_window_updater and not self._current_window_updater.is_empty():
            with self._update_lock:
                self._current_window_updater.finalize()
                window = self._current_window_updater.window()
                self._current_window_updater = WindowUpdater()
                self._current_window_timestamp = time_bucket

            window.model.deployment_name = self._deployment_name
            for key, value in self._metadata.items():
                window.model.metadata[key] = value

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Uploading window:')
                logger.debug(MessageToDict(window))

            graphsignal._get_uploader().upload_window(window)
            if flush:
                graphsignal._get_uploader().flush_in_thread()
        else:
            with self._update_lock:
                self._current_window_updater = WindowUpdater()
                self._current_window_timestamp = time_bucket

        return True

    def log_metadata(self, key=None, value=None):
        if not isinstance(key, str) or len(key) > 250:
            logger.error('Invalid metadata entry key format')
            return
        if not isinstance(value, (str, int, float)) or len(str(value)) > 2500:
            logger.error(
                'Invalid metadata entry value format for key: {0}'.format(key))
            return

        if len(self._metadata) >= MAX_METADATA_SIZE:
            logger.error(
                'Too many metadata entries, max={0}'.format(MAX_METADATA_SIZE))
            return

        self._metadata[key] = str(value)

    def log_prediction(
            self,
            features=None,
            prediction=None,
            actual_timestamp=None):
        if not isinstance(features, dict):
            logger.error(
                'features (dict) must be provided')
            return

        if not isinstance(prediction, (bool, int, float, str, tuple, list)):
            logger.error(
                'prediction (bool, int, float, str, tuple, list) must be provided')
            return

        if isinstance(prediction, tuple):
            prediction = list(prediction)

        if not isinstance(prediction, list):
            prediction = [prediction]

        self.log_prediction_batch(
            features=[list(features.values())
                      ] if features is not None else None,
            feature_names=list(
                features.keys()) if features is not None else None,
            predictions=[prediction] if prediction is not None else None,
            actual_timestamp=actual_timestamp
        )

    def log_prediction_batch(
            self,
            features=None,
            feature_names=None,
            predictions=None,
            actual_timestamp=None):
        if actual_timestamp is not None and not _check_timestamp(
                actual_timestamp):
            logger.error(
                'actual_timestamp (int) must be a unix timestamp in seconds')
            return

        batch_size = max(
            statistics.estimate_size(features),
            statistics.estimate_size(predictions))
        if batch_size == 0:
            logger.debug('Logged empty data')
            return

        timestamp = actual_timestamp if actual_timestamp else _timestamp()

        if not self._tumble_window(timestamp):
            return

        self._current_window_updater.add_prediction(PredictionRecord(
            features=features,
            feature_names=feature_names,
            predictions=predictions,
            timestamp=timestamp), batch_size)

    def log_exception(
            self,
            message=None,
            extra_info=None,
            exc_info=None,
            actual_timestamp=None):
        if not message and not exc_info:
            logger.error('Eigher of message or exc_info must be provided')
            return

        if message:
            if not isinstance(message, str):
                message = repr(message)
            if len(message) > 250:
                message = message[:250] + '...'

        stack_trace = None
        if exc_info:
            if exc_info == True:
                exc_info = sys.exc_info()
            if len(
                    exc_info) == 3 and exc_info[0] and exc_info[1] and exc_info[2]:
                exception_part = traceback.format_exception_only(
                    exc_info[0], exc_info[1])
                if len(exception_part) > 0:
                    message = str(exception_part[0]).rstrip()
                stack_trace_part = traceback.format_tb(exc_info[2])
                if len(stack_trace_part) > 0:
                    stack_trace = ''.join(stack_trace_part)

        if not message:
            logger.error('Cannot extract exception message')
            return

        if actual_timestamp is not None and not _check_timestamp(
                actual_timestamp):
            logger.error(
                'actual_timestamp (int) must be a unix timestamp in seconds')
            return

        filtered_extra_info = {}
        if extra_info and isinstance(extra_info, dict):
            for name in list(extra_info.keys())[:MAX_EXTRA_INFO_SIZE]:
                value = extra_info[name]
                name = str(name)
                if len(name) > 250:
                    name = name[:250] + '...'
                value = str(value)
                if len(value) > 2500:
                    value = value[:2500] + '...'
                filtered_extra_info[name] = value

        timestamp = actual_timestamp if actual_timestamp else _timestamp()

        if not self._tumble_window(timestamp):
            return

        self._current_window_updater.add_exception(ExceptionRecord(
            message=message,
            extra_info=filtered_extra_info if extra_info else None,
            stack_trace=stack_trace,
            timestamp=timestamp))

    def log_ground_truth(
            self,
            label,
            prediction,
            prediction_timestamp=None,
            segments=None):

        if label is None or not isinstance(
                label, (str, bool, int, float)):
            logger.error('label (str|bool|int|float) must be provided')
            return

        if prediction is None or not isinstance(
                prediction, (str, bool, int, float)):
            logger.error(
                'prediction (str|bool|int|float) must be provided')
            return

        if not isinstance(label, type(prediction)):
            logger.error(
                'label and prediction must have the same type (str|bool|int|float)')
            return

        if prediction_timestamp is not None and not _check_timestamp(
                prediction_timestamp):
            logger.error(
                'prediction_timestamp (int) must be a unix timestamp in seconds')
            return

        if segments is not None and not isinstance(segments, list):
            logger.error('segments (list) must be provided')
            return

        segments = [segment[:MAX_SEGMENT_LENGTH] for segment in segments
                    if isinstance(segment, str)]
        timestamp = prediction_timestamp if prediction_timestamp else _timestamp()

        if not self._tumble_window(timestamp):
            return

        self._current_window_updater.add_ground_truth(GroundTruthRecord(
            label=label,
            prediction=prediction,
            prediction_timestamp=timestamp,
            segments=segments))


def get_session(deployment_name):
    if not deployment_name or len(deployment_name) > 250:
        raise ValueError('Invalid deployment_name format')

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

    for session in session_list:
        session._tumble_window(_timestamp(), force=force, flush=False)


def _check_timestamp(timestamp):
    now = _timestamp()
    return (isinstance(timestamp, int) and
            timestamp > now - 31536000 and  # one year in the past
            timestamp < now + 86400)  # one day in the future


def _timestamp():
    return int(time.time())
