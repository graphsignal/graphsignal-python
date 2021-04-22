import time
import logging
import threading
import sys
import traceback
import platform

import graphsignal
from graphsignal import statistics
from graphsignal import system
from graphsignal.uploader import Uploader
from graphsignal.predictions import Prediction
from graphsignal.windows import Window, Model, Metric, Event
from graphsignal.spans import Span

logger = logging.getLogger('graphsignal')

MAX_MODEL_ATTRIBUTES = 10
MAX_METRICS = 50
MAX_EVENTS = 50

MIN_WINDOW_SIZE = 50
MIN_WINDOW_DURATION = 120
MAX_WINDOW_DURATION = 600

_session_index = {}
_session_index_lock = threading.Lock()


class Session(object):
    __slots__ = [
        '_model_name',
        '_model_deployment',
        '_model_attributes',
        '_metric_index',
        '_prediction_window',
        '_event_window',
        '_window_start_time',
        '_window_size',
        '_update_lock',
        '_is_updated',
        '_upload_timer'
    ]

    def __init__(self, model_name, deployment_name=None):
        self._model_name = model_name
        self._model_deployment = deployment_name
        self._model_attributes = {}
        self._update_lock = threading.Lock()
        self._reset_window()
        self._add_system_attributes()

    def _reset_window(self):
        self._metric_index = {}
        self._prediction_window = []
        self._event_window = []
        self._window_start_time = time.time()
        self._window_size = 0
        self._is_updated = False

    def _set_updated(self):
        self._is_updated = True
        if self._upload_window():
            graphsignal._get_uploader().flush_in_thread()

    def set_attribute(self, name=None, value=None):
        '''
        Set model attributes. Only last set value is kept.

        Args:
            name (:obj:`str`):
                Model attribute name.
            value (:obj:`int` or :obj:`float`):
                Model attribute value.
        '''

        if not isinstance(name, str) or len(name) > 250:
            logger.error('invalid model attribute name format')
            return
        if not isinstance(value, (str, int, float)) or len(str(value)) > 2500:
            logger.error(
                'invalid model attribute value format for name: {0}'.format(name))
            return

        if len(self._model_attributes) >= MAX_MODEL_ATTRIBUTES:
            logger.error(
                'too many model attributes, max={0}'.format(MAX_MODEL_ATTRIBUTES))
            return

        self._model_attributes[name] = value

    def log_prediction(
            self,
            input_data=None,
            input_type='tabular',
            output_data=None,
            output_type='tabular',
            context_data=None,
            actual_timestamp=None):
        '''
        Log single or batch model prediction.

        See `Supported Data Formats <https://graphsignal.ai/docs/integrations/python/supported-data-formats/>`_
        for detailed description of data types and formats.

        Computed data statistics such as feature and class distributions are uploaded
        at certain intervals and on process exit. Additionally, some prediction samples
        and outliers may be uploaded. This can be disabled by setting ``log_instances`` option
        to ``False`` when configuring the logger.

        Args:
            input_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Input data instances.
            input_type (:obj:`str`, optional, default  is ``tabular``):
                Type of the provided input data.
            output_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Output data instances.
            output_type (:obj:`str`, optional, default  is ``tabular``):
                Type of the provided output data.
            context_data (:obj:`list` or :obj:`dict` or :obj:`numpy.ndarray` or :obj:`pandas.DataFrame`, optional):
                Context data for each prediction instance, such as feature ID or any other prediction related information.
                Context data is not monitored, it is only included in prediction samples and/or outliers.
                The number of rows in `context_data` should match `input_data` and/or `output_data` instances.
            actual_timestamp (:obj:`int`, optional, default is current timestamp):
                Actual timestamp of the measurement, when different from current timestamp.
        '''

        if input_type != 'tabular':
            logger.error('Data type \'%s\' is not supported', input_type)
            return

        if output_type != 'tabular':
            logger.error('Data type \'%s\' is not supported', output_type)
            return

        self._window_size += max(
            statistics.estimate_size(input_data),
            statistics.estimate_size(output_type))

        with self._update_lock:
            self._prediction_window.append(Prediction(
                input_data=input_data,
                input_type=Prediction.data_type(input_type),
                output_data=output_data,
                output_type=Prediction.data_type(output_type),
                context_data=context_data,
                timestamp=actual_timestamp))

        self._set_updated()

    def log_metric(self, name=None, value=None, actual_timestamp=None):
        '''
        Logs metric value. The last logged value is used
        when metric is aggregated.

        Args:
            name (:obj:`str`):
                Metric name.
            value (:obj:`int` or :obj:`float`):
                Metric value.
            actual_timestamp (:obj:`int`, optional, default is current timestamp):
                Actual timestamp of the measurement, when different from current timestamp.
        '''

        if not isinstance(name, str) or len(name) > 250:
            logger.error('invalid metric name format')
            return
        if value is not None and not isinstance(
                value, (int, float)) or len(str(value)) > 2500:
            logger.error(
                'invalid metric value format for "{0}": {1}'.format(name, type(value)))
            return

        if len(self._metric_index) >= MAX_METRICS:
            logger.error('too many metrics, max={0}'.format(MAX_METRICS))
            return

        with self._update_lock:
            metric = None
            if name in self._metric_index:
                metric = self._metric_index[name]
            else:
                metric = self._metric_index[name] = Metric(
                    dataset=Metric.DATASET_USER_DEFINED,
                    name=name,
                    timestamp=actual_timestamp)
            metric.set_gauge(value)

        self._set_updated()

    def log_event(
            self,
            description=None,
            attributes=None,
            is_error=False,
            exc_info=None,
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
            exc_info (:obj:`bool` or :obj:`tuple`, optional):
                If the event should represents an exception, pass exception tuple
                or ``True`` to automatically read it from ``sys.exc_info()``.
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
        if is_error or exc_info:
            type_name = Event.TYPE_ERROR

        if exc_info:
            event_name = 'exception'
            if exc_info == True:
                exc_info = sys.exc_info()
            if len(
                    exc_info) == 3 and exc_info[0] and exc_info[1] and exc_info[2]:
                attributes['error_message'] = traceback.format_exception_only(
                    exc_info[0], exc_info[1])
                attributes['stack_trace'] = traceback.format_tb(exc_info[2])
        else:
            event_name = 'user_defined_event'

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

    def measure_latency(self):
        '''
        Measure prediction/inference call latency.

        When used as a context manager, will also record any exceptions.

        Otherwise, ``start()`` and ``stop()`` methods should be called
        on returned ``Span`` object to start and stop latency measurement.

        Returns:
            :obj:`Span` - span object for measuring latency.
        '''

        return Span(self)

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
            metric_index = self._metric_index
            prediction_window = self._prediction_window
            events_window = self._event_window
            self._reset_window()

        # initialize window object
        window = Window()

        # set model
        window.model = Model(
            name=self._model_name,
            deployment=self._model_deployment)
        if self._model_attributes is not None:
            for name, value in self._model_attributes.items():
                window.model.add_attribute(name, value)

        # add user defined metrics
        user_metrics = metric_index.values()
        for metric in user_metrics:
            window.add_metric(metric)

        # add prediction count metric
        last_timestamp = max([p.timestamp for p in prediction_window if p]) if len(prediction_window) > 0 else None
        prediction_count_metric = Metric(
            dataset=Metric.DATASET_SYSTEM,
            name='prediction_count',
            timestamp=last_timestamp)
        prediction_count_metric.set_gauge(len(prediction_window))
        window.add_metric(prediction_count_metric)

        # add computed data metrics
        try:
            data_metrics, data_samples = statistics.compute_metrics(
                prediction_window)
            if data_metrics is not None and len(data_metrics) > 0:
                for metric in data_metrics:
                    window.add_metric(metric)
            if data_samples is not None and len(data_samples) > 0:
                for sample in data_samples:
                    window.add_sample(sample)
        except Exception:
            logger.error(
                'Unable to compute data statistics', exc_info=True)

        # add system metrics
        if graphsignal._get_config().log_system_metrics:
            vm_rss = system.vm_rss()
            if vm_rss is not None:
                vm_rss_metric = Metric(
                    dataset=Metric.DATASET_SYSTEM,
                    name='process_memory_usage')
                vm_rss_metric.set_gauge(vm_rss, unit=Metric.UNIT_KILOBYTE)
                window.add_metric(vm_rss_metric)

            vm_size = system.vm_size()
            if vm_size is not None:
                vm_size_metric = Metric(
                    dataset=Metric.DATASET_SYSTEM,
                    name='process_virtual_memory')
                vm_size_metric.set_gauge(vm_size, unit=Metric.UNIT_KILOBYTE)
                window.add_metric(vm_size_metric)

        # finalize metrics
        for metric in window.metrics:
            metric.finalize()

        # add events
        for event in events_window:
            window.add_event(event)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Uploading window:')
            logger.debug(window)

        graphsignal._get_uploader().upload_window(window.to_dict())
        return True

    def _add_system_attributes(self):
        self.set_attribute('Python', platform.python_version())
        self.set_attribute('OS', platform.system())


def get_session(model_name, deployment_name=None):
    if not isinstance(model_name, str) or len(model_name) > 250:
        raise ValueError('invalid model name format')

    if deployment_name is not None and (not isinstance(deployment_name, str) or len(deployment_name) > 250):
        raise ValueError('invalid deployment_name format')

    model_key = model_name
    if deployment_name is not None:
        model_key += model_name

    with _session_index_lock:
        if model_key in _session_index:
            return _session_index[model_key]
        else:
            sess = Session(model_name, deployment_name)
            _session_index[model_key] = sess
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
