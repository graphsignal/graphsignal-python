import time
import logging
import threading
import sys
import traceback
import platform
from google.protobuf.json_format import MessageToDict
import numpy as np
import pandas as pd

import graphsignal
from graphsignal import statistics
from graphsignal.windows import PredictionRecord, ExceptionRecord, EvaluationRecord, WindowUpdater
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.flush()

    def _tumble_window(self, to_timestamp, force=False, flush=True):
        if to_timestamp is not None:
            window_seconds = graphsignal._get_config().window_seconds
            time_bucket = int(to_timestamp / window_seconds) * window_seconds
        elif self._current_window_timestamp:
            # makes sense only if force provided
            time_bucket = self._current_window_timestamp
        else:
            # no timestamp to tumble to
            return

        if not force and time_bucket == self._current_window_timestamp:
            # window matches, no need to tumble/create
            return

        if self._current_window_timestamp and time_bucket < self._current_window_timestamp:
            raise ValueError(
                'Cannot tumble time windows backwards. Please log in chronological order when providing timestamps explicitely.')

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

    def log_metadata(self, key=None, value=None):
        if not isinstance(key, str) or key == '':
            raise ValueError('Metadata entry key (str) must be provided')
        if not isinstance(value, (str, int, float)):
            raise ValueError(
                'Metadata entry value (str|int|float) for key: {0} must be provided'.format(key))

        if len(self._metadata) >= MAX_METADATA_SIZE:
            raise ValueError(
                'Too many metadata entries, max={0}'.format(MAX_METADATA_SIZE))

        value = str(value)

        if len(key) > 250:
            value[:250] + '...'

        if len(str(value)) > 2500:
            value[:2500] + '...'

        self._metadata[key] = value

    def log_prediction(
            self,
            features=None,
            output=None,
            actual_timestamp=None):
        if features is None and output is None:
            raise ValueError(
                'features or output or both must be provided')

        if features is not None and not isinstance(features, dict):
            raise ValueError(
                'features must be (dict)')

        if output is not None and not isinstance(
                output, (bool, int, float, str, tuple, list, dict)):
            raise ValueError(
                'output must be (bool|int|float|str|tuple|list|dict)')

        if isinstance(output, tuple):
            output = list(output)

        if isinstance(output, (bool, int, float, str)):
            output = [output]

        if isinstance(output, dict):
            output = list(output.values())

        self.log_prediction_batch(
            features=[list(features.values())
                      ] if features is not None else None,
            feature_names=list(
                features.keys()) if features is not None else None,
            outputs=[output] if output is not None else None,
            output_names=list(
                output.keys()) if isinstance(output, dict) else None,
            actual_timestamp=actual_timestamp
        )

    def log_prediction_batch(
            self,
            features=None,
            feature_names=None,
            outputs=None,
            output_names=None,
            actual_timestamp=None):
        if actual_timestamp is not None and not _check_timestamp(
                actual_timestamp):
            raise ValueError(
                'actual_timestamp (int) must be a unix timestamp in seconds')

        if features is None and outputs is None:
            raise ValueError(
                'features or outputs or both must be provided')

        if features is not None and not isinstance(
                features, (list, dict, np.ndarray, pd.DataFrame)):
            raise ValueError(
                'features must be (list|dict|np.ndarray|pd.DataFrame)')

        if feature_names is not None and not isinstance(feature_names, list):
            raise ValueError(
                'feature_names must be (list)')

        if outputs is not None and not isinstance(
                outputs, (list, dict, np.ndarray, pd.DataFrame)):
            raise ValueError(
                'outputs must be (list|dict|np.ndarray|pd.DataFrame)')

        if output_names is not None and not isinstance(output_names, list):
            raise ValueError(
                'output_names must be (list)')

        batch_size = max(
            statistics.estimate_size(features),
            statistics.estimate_size(outputs))
        if batch_size == 0:
            logger.debug('Logged empty data')
            return

        timestamp = actual_timestamp if actual_timestamp else _timestamp()

        self._tumble_window(timestamp)

        self._current_window_updater.add_prediction(PredictionRecord(
            features=features,
            feature_names=feature_names,
            outputs=outputs,
            output_names=output_names,
            timestamp=timestamp), batch_size)

    def log_evaluation(
            self,
            prediction,
            label,
            segments=None,
            actual_timestamp=None):

        if prediction is None or not isinstance(
                prediction, (str, bool, int, float)):
            raise ValueError(
                'prediction (str|bool|int|float) must be provided')

        if label is None or not isinstance(
                label, (str, bool, int, float)):
            raise ValueError('label (str|bool|int|float) must be provided')

        if not isinstance(label, type(prediction)):
            raise ValueError(
                'label and prediction must have the same type (str|bool|int|float)')

        if actual_timestamp is not None and not _check_timestamp(
                actual_timestamp):
            raise ValueError(
                'actual_timestamp (int) must be a unix timestamp in seconds')

        if segments is not None and not isinstance(segments, list):
            raise ValueError('segments must be (list)')

        segments = [segment[:MAX_SEGMENT_LENGTH] for segment in segments
                    if isinstance(segment, str)]
        timestamp = actual_timestamp if actual_timestamp else _timestamp()

        self._tumble_window(timestamp)

        self._current_window_updater.add_evaluation(EvaluationRecord(
            prediction=prediction,
            label=label,
            segments=segments,
            timestamp=timestamp))

    def log_evaluation_batch(
            self,
            predictions,
            labels,
            segments=None,
            actual_timestamp=None):

        if predictions is None or not isinstance(
                prediction, (list, np.ndarray)):
            raise ValueError(
                'predictions (list|numpy.ndarray) must be provided')

        if labels is None or not isinstance(
                label, (list, np.ndarray)):
            raise ValueError('labels (list|numpy.ndarray) must be provided')

        if isinstance(predictions, np.ndarray):
            if predictions.ndim > 2:
                raise ValueError('predictions has too many (>2) dimensions')
            predictions = predictions.tolist()

        if isinstance(labels, np.ndarray):
            if labels.ndim > 2:
                raise ValueError('labels has too many (>2) dimensions')
            labels = labels.tolist()

        if len(labels) != len(predictions):
            raise ValueError('Number of labels and predictions must be equal')

        if segments is not None:
            if not isinstance(segments, list):
                raise ValueError('segments must be (list of list)')
            if len(segments) != len(predictions):
                raise ValueError(
                    'Number of predictions, labels and segments must be equal')

        if segments is not None:
            for prediction, label, segments in zip(
                    predictions, labels, segments):
                log_evaluation(
                    prediction=prediction,
                    label=label,
                    segments=segments,
                    actual_timestamp=actual_timestamp)
        else:
            for prediction, label in zip(predictions, labels):
                log_evaluation(
                    prediction=prediction,
                    label=label,
                    actual_timestamp=actual_timestamp)

    def flush(self):
        self._tumble_window(None, force=True, flush=False)
        graphsignal._get_uploader().flush()


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


def upload_all():
    session_list = None
    with _session_index_lock:
        session_list = _session_index.values()

    for session in session_list:
        session._tumble_window(None, force=True, flush=False)


def _check_timestamp(timestamp):
    now = _timestamp()
    return (isinstance(timestamp, int) and
            timestamp > now - 31536000 and  # one year in the past
            timestamp < now + 86400)  # one day in the future


def _timestamp():
    return int(time.time())
