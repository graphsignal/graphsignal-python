import logging
import math
import xxhash

import graphsignal
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')


class ProfileField:
    def __init__(self, field_id, field_type, field_descriptor):
        self.field_id = field_id
        self.field_type = field_type
        self.field_descriptor = field_descriptor

class MetricStore:
    MAX_PROFILE_FIELDS = 10000

    def __init__(self):
        self._metrics = {}
        self._profile_fields = {}

    def metric_key(self, name, tags):
        return (name, frozenset(tags.items()))

    def _get_metric(self, metric_type, name, tags=None, unit=None):
        if name is None:
            raise ValueError('Name cannot be None')
        if metric_type is None:
            raise ValueError('Metric type cannot be None')

        all_tags = graphsignal._ticker.tags.copy()
        if tags is not None:
            all_tags.update(tags)

        metric_key = (name, frozenset(all_tags.items()))
        if metric_key not in self._metrics:
            metric = signals_pb2.Metric()
            metric.type = metric_type
            metric.name = name
            for key, value in all_tags.items():
                tag = metric.tags.add()
                tag.key = str(key)[:50]
                tag.value = str(value)[:250]
            if unit is not None:
                metric.unit = unit
            self._metrics[metric_key] = metric
            return metric
        else:
            return self._metrics[metric_key]

    def _get_datapoint(self, metric, aggregate=False):
        if len(metric.datapoints) == 0:
            return metric.datapoints.add()
        else:
            if aggregate:
                return metric.datapoints[-1]
            else:
                return metric.datapoints.add()

    def set_gauge(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if value is None:
            raise ValueError('Gauge value cannot be None')

        metric = self._get_metric(signals_pb2.Metric.MetricType.GAUGE_METRIC, name, tags=tags, unit=unit)

        dp = self._get_datapoint(metric, aggregate)
        dp.gauge = value
        dp.measurement_ts = measurement_ts

    def inc_counter(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if value is None:
            raise ValueError('Counter value cannot be None')
        
        metric = self._get_metric(signals_pb2.Metric.MetricType.COUNTER_METRIC, name, tags=tags, unit=unit)

        dp = self._get_datapoint(metric, aggregate)
        dp.total += value
        dp.measurement_ts = measurement_ts

    def update_summary(self, name, count, sum_val, sum2_val, measurement_ts, unit=None, aggregate=False, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if count is None:
            raise ValueError('Summary count cannot be None')
        if sum_val is None:
            raise ValueError('Summary sum cannot be None')
        if sum2_val is None:
            raise ValueError('Summary sum2 cannot be None')
        
        metric = self._get_metric(signals_pb2.Metric.MetricType.SUMMARY_METRIC, name, tags=tags, unit=unit)

        dp = self._get_datapoint(metric, aggregate)

        summary = dp.summary
        summary.count += count
        summary.sum += sum_val
        if sum2_val is not None:
            summary.sum2 += sum2_val

        dp.measurement_ts = measurement_ts

    def update_histogram(self, name, value, measurement_ts, unit=None, aggregate=False, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if value is None:
            raise ValueError('Histogram value cannot be None')
        
        metric = self._get_metric(signals_pb2.Metric.MetricType.HISTOGRAM_METRIC, name, tags=tags, unit=unit)

        dp = self._get_datapoint(metric, aggregate)

        bin = _get_value_bin(value)
        for i in range(len(dp.histogram.bins)):
            if dp.histogram.bins[i] == bin:
                dp.histogram.counts[i] += 1
                break
        else:
            dp.histogram.bins.append(bin)
            dp.histogram.counts.append(1)

        dp.measurement_ts = measurement_ts

    def add_gauge_profile_field(self, descriptor):
        return self._add_profile_field(signals_pb2.ProfileField.FieldType.GAUGE_FIELD, descriptor)

    def add_counter_profile_field(self, descriptor):
        return self._add_profile_field(signals_pb2.ProfileField.FieldType.COUNTER_FIELD, descriptor)

    def _add_profile_field(self, field_type, descriptor):
        descriptor = {key: str(value) for key, value in descriptor.items()}
        descriptor_str = ';'.join(sorted([f'{key}:{value}' for key, value in descriptor.items()]))

        xxh = xxhash.xxh64()
        xxh.update(descriptor_str.encode('utf-8'))
        field_id = xxh.intdigest()

        if len(self._profile_fields) < self.MAX_PROFILE_FIELDS:
            self._profile_fields[field_id] = ProfileField(field_id, field_type, descriptor)
            #logger.debug('Added profile field: %s, %s', field_id, descriptor_str)
        else:
            logger.debug('Max profile fields reached, skipping profile field: %s', descriptor_str)

        return field_id

    def update_profile(self, name, profile, measurement_ts, unit=None, tags=None):
        if name is None:
            raise ValueError('Metric name cannot be None')
        if profile is None:
            raise ValueError('Profile cannot be None')
        
        metric = self._get_metric(signals_pb2.Metric.MetricType.PROFILE_METRIC, name, tags=tags, unit=unit)

        dp = self._get_datapoint(metric)
        for field_id, value in profile.items():
            if field_id in self._profile_fields:
                dp.profile.field_ids.append(field_id)
                dp.profile.values.append(value)
            else:
                logger.debug('Profile field not found, skipping: %s', field_id)

        dp.measurement_ts = measurement_ts

    def has_unexported(self):
        return len(self._metrics) > 0

    def export(self):
        metrics = list(self._metrics.values())
        self._metrics.clear()

        for metric in metrics:
            metric_profile_fields = {}
            for dp in metric.datapoints:
                for field_id in dp.profile.field_ids:
                    if field_id not in metric_profile_fields:
                        metric_profile_fields[field_id] = self._profile_fields[field_id]
            for profile_field in metric_profile_fields.values():
                field = metric.fields.add()
                field.field_id = profile_field.field_id
                field.type = profile_field.field_type
                if profile_field.field_descriptor:
                    for key, value in profile_field.field_descriptor.items():
                        field.descriptor[key] = value

        return metrics

    def clear(self):
        self._metrics.clear()


def _get_value_bin(value):
    bin_size = max(10 ** (int(math.log(value, 10)) - 1), 1)
    bin = int(value / bin_size) * bin_size
    return bin
