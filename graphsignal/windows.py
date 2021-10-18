import logging
import threading
import hashlib
from functools import lru_cache

import graphsignal
from graphsignal.sketches.kll import KLLSketch
from graphsignal import statistics
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')

MAX_EXCEPTIONS = 10


class PredictionRecord(object):
    __slots__ = [
        'features',
        'feature_names',
        'predictions',
        'timestamp'
    ]

    def __init__(
            self,
            features=None,
            feature_names=None,
            predictions=None,
            timestamp=None):
        self.features = features
        self.feature_names = feature_names
        self.predictions = predictions
        self.timestamp = timestamp


class ExceptionRecord(object):
    __slots__ = [
        'message',
        'extra_info',
        'stack_trace',
        'timestamp'
    ]

    def __init__(
            self,
            message=None,
            extra_info=None,
            stack_trace=None,
            timestamp=None):
        self.message = message
        self.extra_info = extra_info
        self.stack_trace = stack_trace
        self.timestamp = timestamp


class GroundTruthRecord(object):
    __slots__ = [
        'label',
        'prediction',
        'prediction_timestamp',
        'segments'
    ]

    def __init__(
            self,
            label=None,
            prediction=None,
            prediction_timestamp=None,
            segments=None):
        self.label = label
        self.prediction = prediction
        self.prediction_timestamp = prediction_timestamp
        self.segments = segments


class WindowUpdater(object):
    __slots__ = [
        '_window_proto',
        '_update_lock',
        '_is_empty',
        '_metric_updaters',
        '_prediction_records',
        '_prediction_batch_size',
        '_ground_truth_records',
    ]

    def __init__(self):
        self._window_proto = metrics_pb2.PredictionWindow()
        self._update_lock = threading.Lock()
        self._is_empty = True
        self._metric_updaters = {}
        self._prediction_records = []
        self._prediction_batch_size = 0
        self._ground_truth_records = []

    def window(self):
        return self._window_proto

    def is_empty(self):
        return self._is_empty

    def add_prediction(self, prediction, batch_size):
        with self._update_lock:
            self._prediction_records.append(prediction)
            self._prediction_batch_size += batch_size
            self._is_empty = False

        if self._prediction_batch_size >= graphsignal._get_config().buffer_size:
            self._update_predictions()

    def add_exception(self, exception):
        with self._update_lock:
            self._window_proto.num_exceptions += 1

            if len(self._window_proto.exceptions) < MAX_EXCEPTIONS:
                exception_proto = self._window_proto.exceptions.add()
                exception_proto.message = exception.message
                if exception.extra_info is not None:
                    for name, value in exception.extra_info.items():
                        exception_proto.extra_info[name] = value
                if exception.stack_trace is not None:
                    exception_proto.stack_trace = exception.stack_trace
                exception_proto.create_ts = exception.timestamp
            self._is_empty = False

    def add_ground_truth(self, ground_truth):
        with self._update_lock:
            self._ground_truth_records.append(ground_truth)
            self._is_empty = False

        if len(self._ground_truth_records) >= graphsignal._get_config().buffer_size:
            self._update_ground_truth()

    def _update_predictions(self):
        if len(self._prediction_records) > 0:
            with self._update_lock:
                if self._window_proto.start_ts == 0:
                    self._window_proto.start_ts = self._prediction_records[0].timestamp
                else:
                    self._window_proto.start_ts = min(
                        self._window_proto.start_ts, self._prediction_records[0].timestamp)
                self._window_proto.end_ts = max(
                    self._window_proto.end_ts, self._prediction_records[-1].timestamp)
                self._window_proto.num_predictions += self._prediction_batch_size

                try:
                    statistics.update_data_metrics(
                        self._metric_updaters, self._window_proto, self._prediction_records)
                except BaseException:
                    logger.error('Error updating data metrics', exc_info=True)

                self._prediction_batch_size = 0
                self._prediction_records = []

    def _update_ground_truth(self):
        if len(self._ground_truth_records) > 0:
            with self._update_lock:
                if self._window_proto.start_ts == 0:
                    self._window_proto.start_ts = self._ground_truth_records[0].prediction_timestamp
                else:
                    self._window_proto.start_ts = min(
                        self._window_proto.start_ts, self._ground_truth_records[0].prediction_timestamp)
                self._window_proto.end_ts = max(
                    self._window_proto.end_ts, self._ground_truth_records[-1].prediction_timestamp)
                self._window_proto.num_ground_truths += len(
                    self._ground_truth_records)

                try:
                    statistics.update_performance_metrics(
                        self._metric_updaters, self._window_proto, self._ground_truth_records)
                except BaseException:
                    logger.error('Error updating data metrics', exc_info=True)

                self._ground_truth_records = []

    def finalize(self):
        self._update_predictions()
        self._update_ground_truth()

        with self._update_lock:
            for metric_updater in self._metric_updaters.values():
                metric_updater.finalize()


class MetricUpdater(object):
    __slots__ = [
        '_metric_proto',
        '_sketch'
    ]

    def __init__(self, metric_proto, name, dimensions=None):
        self._metric_proto = metric_proto
        self._sketch = None

        self._metric_proto.name = name
        if dimensions is not None:
            for name, value in dimensions.items():
                self._metric_proto.dimensions[name] = value

    def update_gauge(self, value):
        if self._metric_proto.type == self._metric_proto.NOT_INITIALIZED:
            self._metric_proto.type = self._metric_proto.GAUGE
        self._metric_proto.gauge_value.gauge = value

    def update_counter(self, value):
        if self._metric_proto.type == self._metric_proto.NOT_INITIALIZED:
            self._metric_proto.type = self._metric_proto.COUNTER
            self._metric_proto.counter_value.counter = 0
        self._metric_proto.counter_value.counter += value

    def update_ratio(self, value, total):
        if self._metric_proto.type == self._metric_proto.NOT_INITIALIZED:
            self._metric_proto.type = self._metric_proto.RATIO
            self._metric_proto.ratio_value.counter = 0
            self._metric_proto.ratio_value.total = 0
        self._metric_proto.ratio_value.counter += value
        self._metric_proto.ratio_value.total += total

    def update_distribution(self, values):
        if len(values) == 0:
            return

        if self._metric_proto.type == self._metric_proto.NOT_INITIALIZED:
            self._metric_proto.type = self._metric_proto.DISTRIBUTION
            self._metric_proto.distribution_value.sketch_impl = self._metric_proto.distribution_value.KLL10
            k_val = 128 if isinstance(values[0], (int, float)) else 10
            self._sketch = KLLSketch(k=k_val)

        for value in values:
            self._sketch.update(value)

    def finalize(self):
        if self._metric_proto.type == self._metric_proto.DISTRIBUTION:
            self._sketch.to_proto(
                self._metric_proto.distribution_value.sketch_kll10)


def get_data_stream(window_proto, data_source):
    data_stream = window_proto.data_streams[str(data_source)]
    if data_stream.data_source == data_stream.DataSource.NOT_INITIALIZED:
        data_stream.data_source = data_source
    return data_stream


def get_metric_updater(
        metric_updaters, data_stream_proto, name, dimensions=None):
    metric_key = '{0}:{1}:{2}'.format(
        data_stream_proto.data_source,
        name,
        str(dimensions) if dimensions is not None else '')

    if metric_key not in metric_updaters:
        metric_id = _sha1('{0}:{1}:{2}'.format(
            data_stream_proto.data_source,
            name,
            canonical_string(dimensions) if dimensions is not None else ''),
            size=12)
        metric = metric_updaters[metric_key] = MetricUpdater(
            data_stream_proto.metrics[metric_id], name, dimensions)
        return metric
    else:
        return metric_updaters[metric_key]


def canonical_string(obj):
    if isinstance(obj, str):
        return obj
    elif isinstance(obj, list):
        return ','.join([str(v) for v in obj])
    elif isinstance(obj, dict):
        return ','.join(['{0}={1}'.format(k, v)
                        for k, v in dict(sorted(obj.items())).items()])
    else:
        raise ValueError('Type not supported')


@lru_cache(maxsize=2500)
def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]
