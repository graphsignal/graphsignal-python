import copy
import logging
import math
import time
import hashlib
import random
import numpy as np

logger = logging.getLogger('graphsignal')


RESERVOIR_SIZE = 100
HISTOGRAM_BIN_COUNT = 10


class Window(object):
    __slots__ = [
        'model',
        'metrics',
        'samples',
        'events',
        'timestamp'
    ]

    def __init__(
            self,
            model=None,
            timestamp=None):
        self.model = model
        self.metrics = None
        self.samples = None
        self.events = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def add_metric(self, metric):
        if self.metrics is None:
            self.metrics = []
        self.metrics.append(metric)

    def add_sample(self, sample):
        if self.samples is None:
            self.samples = []
        self.samples.append(sample)

    def add_event(self, event):
        if self.events is None:
            self.events = []
        self.events.append(event)

    def to_dict(self):
        metric_dicts = [metric.to_dict()
                        for metric in self.metrics] if self.metrics else None
        sample_dicts = [sample.to_dict()
                        for sample in self.samples] if self.samples else None
        event_dicts = [event.to_dict()
                       for event in self.events] if self.events else None

        window_dict = {
            'model': self.model.to_dict(),
            'metrics': metric_dicts,
            'samples': sample_dicts,
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
            report.append('    name: {0}'.format(self.model.name))
            report.append('    deployment: {0}'.format(self.model.deployment))
            if self.model.attributes is not None:
                report.append(
                    '    attributes: {0}'.format(
                        self.model.attributes))

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

        if self.samples is not None:
            report.append('Samples ({0})'.format(len(self.samples)))
            for sample in self.samples:
                report.append('    {0}'.format(sample.name))
                for part in sample.parts:
                    report.append(
                        '        dataset: {0}'.format(
                            part.dataset))
                    report.append('        format: {0}'.format(part.format))
                    report.append(
                        '        data size: {0}'.format(len(part.data)))

        return '\n'.join(report)


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
        'name',
        'deployment',
        'attributes',
        'timestamp'
    ]

    def __init__(
            self,
            name=None,
            deployment=None,
            timestamp=None):
        self.name = name
        self.deployment = deployment
        self.attributes = None
        self.timestamp = timestamp if timestamp is not None else _now()

    def add_attribute(self, name, value):
        if self.attributes is None:
            self.attributes = []
        self.attributes.append(Attribute(name, str(value)))

    def to_dict(self):
        attribute_dicts = [attribute.to_dict(
        ) for attribute in self.attributes] if self.attributes else None

        model_dict = {
            'name': self.name,
            'deployment': self.deployment,
            'attributes': attribute_dicts,
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
        'unit',
        'measurement',
        'timestamp',
        '_reservoir',
        '_percent'
    ]

    DATASET_INPUT = 'input'
    DATASET_OUTPUT = 'output'
    DATASET_SYSTEM = 'system'
    DATASET_USER_DEFINED = 'user_defined'

    TYPE_GAUGE = 'gauge'
    TYPE_STATISTIC = 'statistic'
    TYPE_HISTOGRAM = 'histogram'

    UNIT_NONE = ''
    UNIT_PERCENT = '%'
    UNIT_MILLISECOND = 'ms'
    UNIT_KILOBYTE = 'KB'

    def __init__(
            self,
            dataset=None,
            dimension=None,
            name=None,
            timestamp=None):
        self.dataset = dataset
        self.dimension = dimension
        self.name = name
        self.type = None
        self.unit = None
        self.measurement = None
        self.timestamp = timestamp if timestamp is not None else _now()
        self._reservoir = None
        self._percent = None

    def set_gauge(self, value, unit=UNIT_NONE):
        self.type = Metric.TYPE_GAUGE
        self.measurement = [value]
        self.unit = unit

    def set_statistic(self, statistic, sample_size, unit=UNIT_NONE):
        self.type = Metric.TYPE_STATISTIC
        self.measurement = [statistic, sample_size]
        self.unit = unit

    def update_percentile(self, value, percent, unit=UNIT_NONE):
        if not self.type:
            self.type = Metric.TYPE_STATISTIC
            self.unit = unit
            self.measurement = [None, 0]
            self._reservoir = []
            self._percent = percent

        if len(self._reservoir) < RESERVOIR_SIZE:
            self._reservoir.append(value)
        else:
            self._reservoir[random.randint(0, RESERVOIR_SIZE - 1)] = value

        self.measurement[1] += 1

    def compute_histogram(self, values, unit=UNIT_NONE):
        if len(values) == 0:
            return None

        self.type = Metric.TYPE_HISTOGRAM
        self.unit = unit
        self.measurement = []

        m_max = max(values)
        m_range = m_max - min(values)

        if m_range == 0 or len(values) == 1:
            self.measurement.append(0)
            self.measurement.extend([values[0], len(values)])
            return

        bin_size = 10 ** math.floor(math.log10(m_range /
                                               float(HISTOGRAM_BIN_COUNT)))

        bin_exp = math.floor(math.log10(bin_size))
        if bin_exp > 0:
            bin_exp = 0
        else:
            bin_exp = int(abs(bin_exp))

        hist = {}
        for m in values:
            bin = None
            if bin_size > 0:
                bin = math.floor(m / bin_size) * bin_size
            else:
                bin = m

            if bin in hist:
                hist[bin] += 1
            else:
                hist[bin] = 1

        self.measurement.append(bin_size)
        for bin in sorted(hist):
            self.measurement.append(round(bin, bin_exp))
            self.measurement.append(hist[bin])

    def compute_categorical_histogram(self, values, unit=UNIT_NONE):
        if len(values) == 0:
            return None

        self.type = Metric.TYPE_HISTOGRAM
        self.unit = unit
        self.measurement = []

        n_components = 100
        counts = {}
        for value in values:
            h = _category_hash(value, n_components=n_components)
            if h in counts:
                counts[h] += 1
            else:
                counts[h] = 1

        self.measurement = [1]
        for i in range(n_components):
            if i in counts:
                self.measurement.append(i)
                self.measurement.append(counts[i])

    def finalize(self):
        if self._reservoir is not None:
            size = len(self._reservoir)
            index = int(math.ceil((size * self._percent) / 100)) - 1
            self.measurement[0] = sorted(self._reservoir)[index]

    def to_dict(self):
        metric_dict = {
            'dataset': self.dataset,
            'dimension': self.dimension,
            'name': self.name,
            'type': self.type,
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


class SamplePart(object):
    __slots__ = [
        'dataset',
        'format',
        'data']

    FORMAT_CSV = 'csv'

    def __init__(self, dataset, format_, data):
        self.dataset = dataset
        self.format = format_
        self.data = data

    def to_dict(self):
        return {
            'dataset': self.dataset,
            'format': self.format,
            'data': self.data
        }


class Sample(object):
    __slots__ = [
        'name',
        'size',
        'parts',
        'timestamp'
    ]

    def __init__(self, name=None, size=None, timestamp=None):
        self.name = name
        self.size = size
        self.parts = []
        self.timestamp = timestamp if timestamp is not None else _now()

    def set_size(self, size):
        self.size = size

    def add_part(self, dataset, format_, data, insert_at=None):
        if insert_at:
            self.parts.insert(insert_at, SamplePart(dataset, format_, data))
        else:
            self.parts.append(SamplePart(dataset, format_, data))

    def to_dict(self):
        part_dicts = [part.to_dict()
                         for part in self.parts] if self.parts else None

        sample_dict = {
            'name': self.name,
            'size': self.size,
            'parts': part_dicts,
            'timestamp': self.timestamp
        }

        for key in list(sample_dict):
            if sample_dict[key] is None:
                del sample_dict[key]

        return sample_dict


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
