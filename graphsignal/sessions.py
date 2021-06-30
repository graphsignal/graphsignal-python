import time
import logging
import threading
import sys
import traceback
import platform

import graphsignal
from graphsignal import statistics
from graphsignal.uploader import Uploader
from graphsignal.predictions import Prediction
from graphsignal.windows import Window, Model, Metric, Event

logger = logging.getLogger('graphsignal')

MAX_TAGS = 10
MAX_EVENTS = 50

MIN_WINDOW_SIZE = 50
MIN_WINDOW_DURATION = 120
MAX_WINDOW_DURATION = 600

_session_index = {}
_session_index_lock = threading.Lock()


class Session(object):
    __slots__ = [
        '_deployment_name',
        '_tags',
        '_prediction_window',
        '_event_window',
        '_window_start_time',
        '_window_size',
        '_update_lock',
        '_is_updated',
        '_upload_timer'
    ]

    def __init__(self, deployment_name):
        self._deployment_name = deployment_name
        self._tags = {}
        self._update_lock = threading.Lock()
        self._reset_window()

    def _reset_window(self):
        self._prediction_window = []
        self._event_window = []
        self._window_start_time = time.time()
        self._window_size = 0
        self._is_updated = False

    def _set_updated(self):
        self._is_updated = True
        if self._upload_window():
            graphsignal._get_uploader().flush_in_thread()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and exc_val and exc_tb:
            message = traceback.format_exception_only(exc_type, exc_val)
            stack_trace = traceback.format_tb(exc_tb)

            attributes = {}
            if isinstance(message, list) and len(message) > 0:
                attributes['Message'] = str('\n'.join(message))
            if isinstance(stack_trace, list) and len(stack_trace) > 0:
                attributes['Stack trace'] = str('\n'.join(stack_trace))

            self.log_event(
                description='Prediction exception',
                attributes=attributes,
                is_error=True
            )

    def set_tag(self, name=None, value=None):
        '''
        Set model deployment tags.

        Args:
            name (:obj:`str`):
                Tag name.
            value (:obj:`int` or :obj:`float`):
                Tag value.
        '''

        if not isinstance(name, str) or len(name) > 250:
            logger.error('invalid tag name format')
            return
        if not isinstance(value, (str, int, float)) or len(str(value)) > 2500:
            logger.error(
                'invalid tag value format for name: {0}'.format(name))
            return

        if len(self._tags) >= MAX_TAGS:
            logger.error(
                'too many tags, max={0}'.format(MAX_TAGS))
            return

        self._tags[name] = value

    def log_prediction(
            self,
            input_data=None,
            output_data=None,
            actual_timestamp=None):
        '''
        Log single or batch model prediction.

        See `Supported Data Formats <https://graphsignal.ai/docs/integrations/python/supported-data-formats/>`_
        for detailed description of data types and formats.

        Computed data statistics are uploaded at certain intervals and on process exit. No raw data is uploaded.

        Args:
            input_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Input data instances.
            output_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Output data instances.
            actual_timestamp (:obj:`int`, optional, default is current timestamp):
                Actual timestamp of the measurement, when different from current timestamp.
        '''

        self._window_size += max(
            statistics.estimate_size(input_data),
            statistics.estimate_size(output_data))

        with self._update_lock:
            self._prediction_window.append(Prediction(
                input_data=input_data,
                output_data=output_data,
                timestamp=actual_timestamp))

        self._set_updated()

    def log_event(
            self,
            description=None,
            attributes=None,
            is_error=False,
            actual_timestamp=None):
        '''
        Log arbitrary event or exception.

        Args:
            description (:obj:`str`):
                Event description.
            attributes (:obj:`dict`, optional):
                Event attributes.
            is_error (:obj:`bool`, optional):
                Set error type.
            actual_timestamp (:obj:`int`, optional, default is current timestamp):
                Actual timestamp of the measurement, when different from current timestamp.
        '''

        if not description or not isinstance(
                description, str) or len(description) > 250:
            logger.error('invalid format for description')
            return

        if attributes is not None:
            if isinstance(attributes, dict):
                for name, value in attributes.items():
                    if not isinstance(name, str) or len(name) > 250:
                        logger.error('invalid attribute name format')
                        return
                    if not isinstance(value, (str, int, float)):
                        logger.error(
                            'invalid attribute value format for attribute name {0}'.format(name))
                        return
            else:
                logger.error('invalid attributes format, expecting dict')
                return

        if len(self._event_window) >= MAX_EVENTS:
            logger.error('too many events, max={0}'.format(MAX_EVENTS))
            return

        type_name = Event.TYPE_INFO
        if is_error:
            type_name = Event.TYPE_ERROR
            event_name = Event.NAME_ERROR

        with self._update_lock:
            event = Event(
                type=type_name,
                name=event_name,
                description=description,
                timestamp=actual_timestamp)
            for name, value in attributes.items():
                event.add_attribute(name, value)
            self._event_window.append(event)

        self._set_updated()

    def _upload_window(self, force=False):
        if not self._is_updated:
            return False

        # check if current window should be uploaded
        if not force:
            window_duration = time.time() - self._window_start_time
            if window_duration < MIN_WINDOW_DURATION:
                return False
            if (self._window_size < MIN_WINDOW_SIZE and
                    window_duration < MAX_WINDOW_DURATION):
                return False

        # reset
        with self._update_lock:
            prediction_window = self._prediction_window
            events_window = self._event_window
            self._reset_window()

        # initialize window object
        window = Window()

        # set model
        window.model = Model(
            deployment=self._deployment_name)
        if self._tags is not None:
            for name, value in self._tags.items():
                window.model.add_tag(name, value)

        # add prediction count metric
        last_timestamp = max([p.timestamp for p in prediction_window if p]) if len(
            prediction_window) > 0 else None
        prediction_count_metric = Metric(
            dataset='model_statistics',
            name='prediction_count',
            aggregation=Metric.AGGREGATION_SUM,
            timestamp=last_timestamp)
        prediction_count_metric.set_gauge(len(prediction_window))
        window.add_metric(prediction_count_metric)

        # add computed data metrics
        try:
            data_metrics = statistics.compute_metrics(
                prediction_window)
            if data_metrics is not None and len(data_metrics) > 0:
                for metric in data_metrics:
                    window.add_metric(metric)
        except Exception:
            logger.error(
                'Unable to compute data statistics', exc_info=True)

        # add events
        for event in events_window:
            window.add_event(event)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Uploading window:')
            logger.debug(window)

        graphsignal._get_uploader().upload_window(window.to_dict())
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
