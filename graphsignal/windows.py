import copy
import logging
import math
import time
import hashlib
import random
import numpy as np

logger = logging.getLogger('graphsignal')


RESERVOIR_SIZE = 100
MIN_HISTOGRAM_BIN_COUNT = 10
MAX_HISTOGRAM_BIN_COUNT = 50


class Window(object):
    __slots__ = [
        'model',
        'metrics',
        'events',
        'timestamp'
    ]

    def __init__(
            self,
            model=None,
            timestamp=None):
        self.model = model
        self.metrics = None
        self.events = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def add_metric(self, metric):
        if self.metrics is None:
            self.metrics = []
        self.metrics.append(metric)

    def add_event(self, event):
        if self.events is None:
            self.events = []
        self.events.append(event)

    def to_dict(self):
        metric_dicts = [metric.to_dict()
                        for metric in self.metrics] if self.metrics else None
        event_dicts = [event.to_dict()
                       for event in self.events] if self.events else None

        window_dict = {
            'model': self.model.to_dict(),
            'metrics': metric_dicts,
            'events': event_dicts,
            'timestamp': self.timestamp
        }

        for key in list(window_dict):
            if window_dict[key] is None:
                del window_dict[key]

        return window_dict

    def __str__(self):
        report = []

        report.append('Model')
        if self.model:
            report.append('    deployment: {0}'.format(self.model.deployment))
            if self.model.tags is not None:
                report.append(
                    '    tags: {0}'.format(
                        self.model.tags))

        if self.metrics is not None:
            report.append('Metrics ({0})'.format(len(self.metrics)))
            for metric in self.metrics:
                report.append('    {0}:{1}:{2}: {3} (type: {4})'.format(
                    metric.dataset, metric.dimension, metric.name, metric.measurement, metric.type))

        if self.events is not None:
            report.append('Events ({0})'.format(len(self.events)))
            for event in self.events:
                report.append('    {0}: {1}'.format(
                    event.type, event.description))
                if event.attributes is not None:
                    for attribute in event.attributes:
                        report.append(
                            '        attribute: {0}: {1}'.format(attribute.name, attribute.value))

        return '\n'.join(report)


class Tag(object):
    __slots__ = ['name', 'value']

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_dict(self):
        return {
            'name': self.name,
            'value': self.value
        }


class Attribute(object):
    __slots__ = ['name', 'value']

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def to_dict(self):
        return {
            'name': self.name,
            'value': self.value
        }


class Model(object):
    __slots__ = [
        'deployment',
        'tags',
        'timestamp'
    ]

    def __init__(
            self,
            deployment=None,
            timestamp=None):
        self.deployment = deployment
        self.tags = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def add_tag(self, name, value):
        if self.tags is None:
            self.tags = []
        self.tags.append(Tag(name, str(value)))

    def to_dict(self):
        tag_dicts = [tag.to_dict(
        ) for tag in self.tags] if self.tags else None

        model_dict = {
            'deployment': self.deployment,
            'tags': tag_dicts,
            'timestamp': self.timestamp
        }

        for key in list(model_dict):
            if model_dict[key] is None:
                del model_dict[key]

        return model_dict


class Metric(object):
    __slots__ = [
        'dataset',
        'dimension',
        'name',
        'type',
        'aggregation',
        'unit',
        'measurement',
        'timestamp'
    ]

    TYPE_GAUGE = 'gauge'
    TYPE_STATISTIC = 'statistic'
    TYPE_HISTOGRAM = 'histogram'

    AGGREGATION_LAST = 'last'
    AGGREGATION_SUM = 'sum'
    AGGREGATION_MERGE = 'merge'

    UNIT_NONE = ''
    UNIT_PERCENT = '%'
    UNIT_MILLISECOND = 'ms'
    UNIT_KILOBYTE = 'KB'
    UNIT_CATEGORY_HASH = '#'

    def __init__(
            self,
            dataset=None,
            dimension=None,
            name=None,
            aggregation=AGGREGATION_LAST,
            unit=UNIT_NONE,
            timestamp=None):
        self.dataset = dataset
        self.dimension = dimension
        self.name = name
        self.type = None
        self.aggregation = aggregation
        self.unit = unit
        self.measurement = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def set_gauge(self, value):
        self.type = Metric.TYPE_GAUGE
        self.measurement = [value]

    def set_statistic(self, statistic, sample_size):
        self.type = Metric.TYPE_STATISTIC
        self.measurement = [statistic, sample_size]

    def compute_histogram(self, values):
        if len(values) == 0:
            return None

        self.type = Metric.TYPE_HISTOGRAM
        self.aggregation = Metric.AGGREGATION_MERGE
        self.measurement = []

        unique_values, counts = np.unique(values, return_counts=True)
        if len(unique_values) < MAX_HISTOGRAM_BIN_COUNT:
            self.measurement.append(0)
            for value, count in zip(unique_values.tolist(), counts.tolist()):
                self.measurement.extend([value, count])
            return

        # probably not categorical value, fallback to log10 bins
        m_max = max(values)
        m_range = m_max - min(values)
        bin_size = 10 ** math.floor(math.log10(m_range /
                                               float(MIN_HISTOGRAM_BIN_COUNT)))

        bin_precision = math.floor(math.log10(bin_size))
        if bin_precision > 0:
            bin_precision = 0
        else:
            bin_precision = int(abs(bin_precision))

        hist = {}
        for m in values:
            bin = None
            if bin_size > 0:
                # round to avoid approximation
                bin = math.floor(round(m / bin_size, 10)) * bin_size
            else:
                bin = m

            if bin in hist:
                hist[bin] += 1
            else:
                hist[bin] = 1

        self.measurement.append(bin_size)
        for bin, count in hist.items():
            self.measurement.append(round(bin, bin_precision))
            self.measurement.append(count)

    def compute_categorical_histogram(self, values):
        if len(values) == 0:
            return None

        self.type = Metric.TYPE_HISTOGRAM
        self.aggregation = Metric.AGGREGATION_MERGE
        self.measurement = []

        n_components = 100
        counts = {}
        for value in values:
            h = _category_hash(value, n_components=n_components)
            if h in counts:
                counts[h] += 1
            else:
                counts[h] = 1

        self.measurement = [0]
        for i in range(n_components):
            if i in counts:
                self.measurement.append(i)
                self.measurement.append(counts[i])

    def to_dict(self):
        metric_dict = {
            'dataset': self.dataset,
            'dimension': self.dimension,
            'name': self.name,
            'type': self.type,
            'aggregation': self.aggregation,
            'unit': self.unit,
            'measurement': self.measurement,
            'timestamp': self.timestamp
        }

        for key in list(metric_dict):
            if metric_dict[key] is None:
                del metric_dict[key]

        return metric_dict


def _category_hash(category, n_components=256):
    # hashing trick
    h = hashlib.md5(category.encode('utf-8')).hexdigest()
    return int(h, 16) % n_components


class Event(object):
    __slots__ = [
        'type',
        'score',
        'name',
        'description',
        'attributes',
        'timestamp'
    ]

    TYPE_INFO = 'info'
    TYPE_ERROR = 'error'

    NAME_INFO = 'info'
    NAME_ERROR = 'error'
    NAME_ANOMALY = 'anomaly'

    def __init__(
            self,
            type=None,
            score=None,
            name=None,
            description=None,
            timestamp=None):
        self.type = type
        self.score = score if score is not None else 1
        self.name = name
        self.description = description
        self.attributes = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def add_attribute(self, name, value):
        if self.attributes is None:
            self.attributes = []
        self.attributes.append(Attribute(name, str(value)))

    def to_dict(self):
        attribute_dicts = [attribute.to_dict(
        ) for attribute in self.attributes] if self.attributes else None

        event_dict = {
            'type': self.type,
            'score': self.score,
            'name': self.name,
            'description': self.description,
            'attributes': attribute_dicts,
            'timestamp': self.timestamp
        }

        for key in list(event_dict):
            if event_dict[key] is None:
                del event_dict[key]

        return event_dict


def _now():
    return int(time.time())
