import logging
import time

try:
    from prometheus_client import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    REGISTRY = None

import graphsignal

logger = logging.getLogger('graphsignal')


class PrometheusAdapter():
    def __init__(self, registry=None, name_map_func=None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Prometheus metrics will not be collected.")
            return

        if registry is None:
            self._registry = REGISTRY
        else:
            self._registry = registry
        self._last_values = {}

        if not name_map_func:
            raise ValueError('name_map_func is required')
        self._name_map_func = name_map_func

    def setup(self):
        pass

    def collect(self):
        if not PROMETHEUS_AVAILABLE:
            return

        ticker = graphsignal._ticker

        now_ns = time.time_ns()

        try:
            metrics_collected = 0

            for metric in self._registry.collect():
                target_name = self._name_map_func(metric.name)
                if not target_name:
                    continue
                
                metrics_collected += 1
                
                sample_tags = {}
                
                # Get labels from the first sample (they are consistent across sample group)
                if metric.samples:
                    for label_name, label_value in metric.samples[0].labels.items():
                        if label_name not in ['le', 'quantile']:
                            sample_tags[label_name] = label_value

                sample_groups = {}
                for sample in metric.samples:
                    group_key = frozenset(sample.labels.items()) if sample.labels else frozenset()
                    if group_key not in sample_groups:
                        sample_groups[group_key] = {}
                    sample_groups[group_key][sample.name] = sample
                
                for sample_map in sample_groups.values():
                    if metric.type == 'gauge':
                        sample = sample_map.get(metric.name, None)
                        if sample:
                            ticker.set_gauge(
                                name=target_name, 
                                tags=sample_tags, 
                                value=sample.value, 
                                measurement_ts=now_ns)
                    
                    elif metric.type == 'counter':
                        sample = sample_map.get(f'{metric.name}_total', None)
                        if sample:
                            metric_key = (target_name, tuple(sorted(sample_tags.items())))
                            current_value = sample.value
                            
                            if metric_key in self._last_values:
                                last_value = self._last_values[metric_key]
                                delta = current_value - last_value
                                if delta >= 0:
                                    ticker.inc_counter(
                                        name=target_name, 
                                        tags=sample_tags, 
                                        value=delta, 
                                        measurement_ts=now_ns)
                            else:
                                # First time seeing this counter, store initial value
                                self._last_values[metric_key] = current_value
                            
                            # Update the last value for next delta calculation
                            self._last_values[metric_key] = current_value

                    elif metric.type == 'histogram':
                        count_sample = sample_map.get(f'{metric.name}_count', None)
                        sum_sample = sample_map.get(f'{metric.name}_sum', None)
                        if count_sample and sum_sample:
                            ticker.update_summary(
                                name=target_name, 
                                tags=sample_tags, 
                                count=int(count_sample.value), 
                                sum_val=sum_sample.value,
                                sum2_val=sum_sample.value * sum_sample.value,  # Approximate sum2
                                measurement_ts=now_ns)

                    elif metric.type == 'summary':
                        count_sample = sample_map.get(f'{metric.name}_count', None)
                        sum_sample = sample_map.get(f'{metric.name}_sum', None)
                        if count_sample and sum_sample:
                            ticker.update_summary(
                                name=target_name, 
                                tags=sample_tags, 
                                count=int(count_sample.value), 
                                sum_val=sum_sample.value,
                                sum2_val=sum_sample.value * sum_sample.value,  # Approximate sum2
                                measurement_ts=now_ns)

        except Exception as e:
            logger.error(f"Failed to collect Prometheus metrics: {e}", exc_info=True)
