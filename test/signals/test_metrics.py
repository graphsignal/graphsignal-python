import unittest
import logging
import sys
import os
import time
from unittest.mock import patch, Mock
import pprint
import random

import graphsignal
from graphsignal.proto import signals_pb2
from test.test_utils import find_tag

logger = logging.getLogger('graphsignal')

class MetricStoreTest(unittest.TestCase):
    def setUp(self):
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._ticker.auto_tick = False

    def tearDown(self):
        graphsignal.shutdown()

    @patch('time.time', return_value=1)
    def test_update_and_export(self, mocked_time):
        store = graphsignal._ticker.metric_store()
        store.set_gauge(name='m1', tags={'t1': '1'}, value=1, measurement_ts=10, unit='u1')
        store.set_gauge(name='m1', tags={'t1': '1'}, value=2, measurement_ts=20, unit='u1')
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(find_tag(protos[0], 't1'), '1')
        self.assertEqual(protos[0].unit, 'u1')
        # New behavior: each call adds a separate datapoint
        self.assertEqual(len(protos[0].datapoints), 2)
        self.assertEqual(protos[0].datapoints[0].gauge, 1)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 10)
        self.assertEqual(protos[0].datapoints[1].gauge, 2)
        self.assertEqual(protos[0].datapoints[1].measurement_ts, 20)

        store.clear()

        store.inc_counter(name='m1', tags={'t1': '1'}, value=1, measurement_ts=10, unit='u1')
        store.inc_counter(name='m1', tags={'t1': '1'}, value=2, measurement_ts=20, unit='u1')
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(find_tag(protos[0], 't1'), '1')
        self.assertEqual(protos[0].unit, 'u1')
        # New behavior: each call adds a separate datapoint
        self.assertEqual(len(protos[0].datapoints), 2)
        self.assertEqual(protos[0].datapoints[0].total, 1)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 10)
        self.assertEqual(protos[0].datapoints[1].total, 2)
        self.assertEqual(protos[0].datapoints[1].measurement_ts, 20)

        store.clear()

        store.update_summary(name='m1', tags={'t1': '1'}, count=1, sum_val=10, sum2_val=100, measurement_ts=10, unit='u1')
        store.update_summary(name='m1', tags={'t1': '1'}, count=2, sum_val=20, sum2_val=400, measurement_ts=20, unit='u1')
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(find_tag(protos[0], 't1'), '1')
        self.assertEqual(protos[0].unit, 'u1')
        # New behavior: each call adds a separate datapoint
        self.assertEqual(len(protos[0].datapoints), 2)
        self.assertEqual(protos[0].datapoints[0].summary.count, 1)
        self.assertEqual(protos[0].datapoints[0].summary.sum, 10)
        self.assertEqual(protos[0].datapoints[0].summary.sum2, 100)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 10)
        self.assertEqual(protos[0].datapoints[1].summary.count, 2)
        self.assertEqual(protos[0].datapoints[1].summary.sum, 20)
        self.assertEqual(protos[0].datapoints[1].summary.sum2, 400)
        self.assertEqual(protos[0].datapoints[1].measurement_ts, 20)

        store.clear()

        store.update_histogram(name='m1', tags={'t1': '1'}, value=1, measurement_ts=10, unit='u1')
        store.update_histogram(name='m1', tags={'t1': '1'}, value=2, measurement_ts=20, unit='u1')
        protos = store.export()
        self.assertEqual(len(protos), 1)
        self.assertEqual(protos[0].name, 'm1')
        self.assertEqual(find_tag(protos[0], 't1'), '1')
        self.assertEqual(protos[0].unit, 'u1')
        # New behavior: each call adds a separate datapoint
        self.assertEqual(len(protos[0].datapoints), 2)
        self.assertEqual(len(protos[0].datapoints[0].histogram.bins), 1)
        self.assertEqual(len(protos[0].datapoints[0].histogram.counts), 1)
        self.assertEqual(protos[0].datapoints[0].histogram.counts[0], 1)
        self.assertEqual(protos[0].datapoints[0].measurement_ts, 10)
        self.assertEqual(len(protos[0].datapoints[1].histogram.bins), 1)
        self.assertEqual(len(protos[0].datapoints[1].histogram.counts), 1)
        self.assertEqual(protos[0].datapoints[1].histogram.counts[0], 1)
        self.assertEqual(protos[0].datapoints[1].measurement_ts, 20)

    def test_has_unexported(self):
        store = graphsignal._ticker.metric_store()
        self.assertFalse(store.has_unexported())
        store.set_gauge(name='m1', tags={'t1': '1'}, value=1, measurement_ts=10, unit='u1')
        self.assertTrue(store.has_unexported())

    def test_set_gauge(self):
        store = graphsignal._ticker.metric_store()
        
        # Test basic gauge functionality
        store.set_gauge(name='cpu_usage', tags={'host': 'server1'}, value=75.5, measurement_ts=1000, unit='percent')
        store.set_gauge(name='cpu_usage', tags={'host': 'server1'}, value=80.2, measurement_ts=2000, unit='percent')
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'cpu_usage')
        self.assertEqual(metric.type, signals_pb2.Metric.MetricType.GAUGE_METRIC)
        self.assertEqual(metric.unit, 'percent')
        self.assertEqual(find_tag(metric, 'host'), 'server1')
        
        # Verify multiple datapoints are added
        self.assertEqual(len(metric.datapoints), 2)
        self.assertEqual(metric.datapoints[0].gauge, 75.5)
        self.assertEqual(metric.datapoints[0].measurement_ts, 1000)
        self.assertEqual(metric.datapoints[1].gauge, 80.2)
        self.assertEqual(metric.datapoints[1].measurement_ts, 2000)
        
        # Test validation
        with self.assertRaises(ValueError):
            store.set_gauge(name=None, tags={}, value=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.set_gauge(name='test', tags={}, value=None, measurement_ts=1000)
        
        # Test without tags and unit
        store.set_gauge(name='simple_gauge', tags=None, value=42.0, measurement_ts=3000, unit=None)
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0].unit, '')
        
        # Test aggregate=True - should update last datapoint
        store.clear()
        store.set_gauge(name='cpu_usage', tags={'host': 'server1'}, value=75.5, measurement_ts=1000, unit='percent', aggregate=True)
        store.set_gauge(name='cpu_usage', tags={'host': 'server1'}, value=80.2, measurement_ts=2000, unit='percent', aggregate=True)
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'cpu_usage')
        # Should have only 1 datapoint (updated, not new)
        self.assertEqual(len(metric.datapoints), 1)
        # Value should be updated to the new value
        self.assertEqual(metric.datapoints[0].gauge, 80.2)
        # measurement_ts should be updated
        self.assertEqual(metric.datapoints[0].measurement_ts, 2000)

    def test_inc_counter(self):
        store = graphsignal._ticker.metric_store()
        
        # Test basic counter functionality
        store.inc_counter(name='request_count', tags={'method': 'GET'}, value=1, measurement_ts=1000, unit='count')
        store.inc_counter(name='request_count', tags={'method': 'GET'}, value=2, measurement_ts=2000, unit='count')
        store.inc_counter(name='request_count', tags={'method': 'POST'}, value=1, measurement_ts=3000, unit='count')
        
        metrics = store.export()
        # Should have 2 metrics (different tags)
        self.assertEqual(len(metrics), 2)
        
        # Find the GET metric
        get_metric = next(m for m in metrics if any(tag.key == 'method' and tag.value == 'GET' for tag in m.tags))
        self.assertEqual(get_metric.name, 'request_count')
        self.assertEqual(get_metric.type, signals_pb2.Metric.MetricType.COUNTER_METRIC)
        self.assertEqual(get_metric.unit, 'count')
        
        # Verify multiple datapoints
        self.assertEqual(len(get_metric.datapoints), 2)
        self.assertEqual(get_metric.datapoints[0].total, 1)
        self.assertEqual(get_metric.datapoints[0].measurement_ts, 1000)
        self.assertEqual(get_metric.datapoints[1].total, 2)
        self.assertEqual(get_metric.datapoints[1].measurement_ts, 2000)
        
        # Find the POST metric
        post_metric = next(m for m in metrics if any(tag.key == 'method' and tag.value == 'POST' for tag in m.tags))
        self.assertEqual(len(post_metric.datapoints), 1)
        self.assertEqual(post_metric.datapoints[0].total, 1)
        
        # Test validation
        with self.assertRaises(ValueError):
            store.inc_counter(name=None, tags={}, value=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.inc_counter(name='test', tags={}, value=None, measurement_ts=1000)
        
        # Test aggregate=True - should update last datapoint
        store.clear()
        store.inc_counter(name='request_count', tags={'method': 'GET'}, value=1, measurement_ts=1000, unit='count', aggregate=True)
        store.inc_counter(name='request_count', tags={'method': 'GET'}, value=2, measurement_ts=2000, unit='count', aggregate=True)
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'request_count')
        # Should have only 1 datapoint (updated, not new)
        self.assertEqual(len(metric.datapoints), 1)
        # Total should be incremented (1 + 2 = 3)
        self.assertEqual(metric.datapoints[0].total, 3)
        # measurement_ts should be updated
        self.assertEqual(metric.datapoints[0].measurement_ts, 2000)

    def test_update_summary(self):
        store = graphsignal._ticker.metric_store()
        
        # Test basic summary functionality
        store.update_summary(name='response_time', tags={'service': 'api'}, 
                            count=10, sum_val=100.5, sum2_val=1500.25, measurement_ts=1000, unit='ms')
        store.update_summary(name='response_time', tags={'service': 'api'}, 
                            count=20, sum_val=250.8, sum2_val=4000.5, measurement_ts=2000, unit='ms')
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'response_time')
        self.assertEqual(metric.type, signals_pb2.Metric.MetricType.SUMMARY_METRIC)
        self.assertEqual(metric.unit, 'ms')
        
        # Verify multiple datapoints
        self.assertEqual(len(metric.datapoints), 2)
        self.assertEqual(metric.datapoints[0].summary.count, 10)
        self.assertEqual(metric.datapoints[0].summary.sum, 100.5)
        self.assertEqual(metric.datapoints[0].summary.sum2, 1500.25)
        self.assertEqual(metric.datapoints[0].measurement_ts, 1000)
        
        self.assertEqual(metric.datapoints[1].summary.count, 20)
        self.assertEqual(metric.datapoints[1].summary.sum, 250.8)
        self.assertEqual(metric.datapoints[1].summary.sum2, 4000.5)
        self.assertEqual(metric.datapoints[1].measurement_ts, 2000)
        
        # Test validation
        with self.assertRaises(ValueError):
            store.update_summary(name=None, tags={}, count=1, sum_val=1, sum2_val=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.update_summary(name='test', tags={}, count=None, sum_val=1, sum2_val=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.update_summary(name='test', tags={}, count=1, sum_val=None, sum2_val=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.update_summary(name='test', tags={}, count=1, sum_val=1, sum2_val=None, measurement_ts=1000)
        
        # Test aggregate=True - should update last datapoint
        store.clear()
        store.update_summary(name='response_time', tags={'service': 'api'}, 
                            count=10, sum_val=100.5, sum2_val=1500.25, measurement_ts=1000, unit='ms', aggregate=True)
        store.update_summary(name='response_time', tags={'service': 'api'}, 
                            count=20, sum_val=250.8, sum2_val=4000.5, measurement_ts=2000, unit='ms', aggregate=True)
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'response_time')
        # Should have only 1 datapoint (updated, not new)
        self.assertEqual(len(metric.datapoints), 1)
        # Values should be incremented (10 + 20 = 30, 100.5 + 250.8 = 351.3, 1500.25 + 4000.5 = 5500.75)
        self.assertEqual(metric.datapoints[0].summary.count, 30)
        self.assertEqual(metric.datapoints[0].summary.sum, 351.3)
        self.assertEqual(metric.datapoints[0].summary.sum2, 5500.75)
        # measurement_ts should be updated
        self.assertEqual(metric.datapoints[0].measurement_ts, 2000)

    def test_update_histogram(self):
        store = graphsignal._ticker.metric_store()
        
        # Test basic histogram functionality
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=10.5, measurement_ts=1000, unit='ms')
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=25.3, measurement_ts=2000, unit='ms')
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=10.5, measurement_ts=3000, unit='ms')
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'latency')
        self.assertEqual(metric.type, signals_pb2.Metric.MetricType.HISTOGRAM_METRIC)
        self.assertEqual(metric.unit, 'ms')
        
        # Verify multiple datapoints with bins and counts
        self.assertEqual(len(metric.datapoints), 3)
        
        # First datapoint
        self.assertEqual(len(metric.datapoints[0].histogram.bins), 1)
        self.assertEqual(len(metric.datapoints[0].histogram.counts), 1)
        self.assertEqual(metric.datapoints[0].histogram.counts[0], 1)
        self.assertEqual(metric.datapoints[0].measurement_ts, 1000)
        
        # Second datapoint
        self.assertEqual(len(metric.datapoints[1].histogram.bins), 1)
        self.assertEqual(len(metric.datapoints[1].histogram.counts), 1)
        self.assertEqual(metric.datapoints[1].histogram.counts[0], 1)
        self.assertEqual(metric.datapoints[1].measurement_ts, 2000)
        
        # Third datapoint
        self.assertEqual(len(metric.datapoints[2].histogram.bins), 1)
        self.assertEqual(len(metric.datapoints[2].histogram.counts), 1)
        self.assertEqual(metric.datapoints[2].histogram.counts[0], 1)
        self.assertEqual(metric.datapoints[2].measurement_ts, 3000)
        
        # Test validation
        with self.assertRaises(ValueError):
            store.update_histogram(name=None, tags={}, value=1, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.update_histogram(name='test', tags={}, value=None, measurement_ts=1000)
        
        # Test aggregate=True - should update last datapoint
        store.clear()
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=10.5, measurement_ts=1000, unit='ms', aggregate=True)
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=10.5, measurement_ts=2000, unit='ms', aggregate=True)
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'latency')
        # Should have only 1 datapoint (updated, not new)
        self.assertEqual(len(metric.datapoints), 1)
        # Histogram should have the same bin with count incremented (1 + 1 = 2)
        self.assertEqual(len(metric.datapoints[0].histogram.bins), 1)
        self.assertEqual(len(metric.datapoints[0].histogram.counts), 1)
        self.assertEqual(metric.datapoints[0].histogram.counts[0], 2)
        # measurement_ts should be updated
        self.assertEqual(metric.datapoints[0].measurement_ts, 2000)
        
        # Test aggregate=True with different value - should add new bin
        store.clear()
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=10.5, measurement_ts=1000, unit='ms', aggregate=True)
        store.update_histogram(name='latency', tags={'region': 'us-east'}, value=25.3, measurement_ts=2000, unit='ms', aggregate=True)
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        # Should have only 1 datapoint (updated, not new)
        self.assertEqual(len(metric.datapoints), 1)
        # Histogram should have 2 bins (one for each value)
        self.assertEqual(len(metric.datapoints[0].histogram.bins), 2)
        self.assertEqual(len(metric.datapoints[0].histogram.counts), 2)
        # Each bin should have count of 1
        self.assertEqual(metric.datapoints[0].histogram.counts[0], 1)
        self.assertEqual(metric.datapoints[0].histogram.counts[1], 1)

    def test_update_profile(self):
        store = graphsignal._ticker.metric_store()
        
        # Add profile fields first - required before calling update_profile
        cpu_field_id = store.add_gauge_profile_field({'field_name': 'cpu'})
        memory_field_id = store.add_gauge_profile_field({'field_name': 'memory'})
        disk_field_id = store.add_gauge_profile_field({'field_name': 'disk'})
        
        # Test basic profile functionality
        profile1 = {cpu_field_id: 50.5, memory_field_id: 1024.0, disk_field_id: 500.0}
        store.update_profile(name='resource_usage', tags={'env': 'prod'}, 
                            profile=profile1, measurement_ts=1000, unit='bytes')
        
        profile2 = {cpu_field_id: 75.2, memory_field_id: 2048.0, disk_field_id: 750.0}
        store.update_profile(name='resource_usage', tags={'env': 'prod'}, 
                            profile=profile2, measurement_ts=2000, unit='bytes')
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'resource_usage')
        self.assertEqual(metric.type, signals_pb2.Metric.MetricType.PROFILE_METRIC)
        self.assertEqual(metric.unit, 'bytes')
        
        # Verify profile fields are exported
        self.assertEqual(len(metric.fields), 3)
        field_id_to_name = {field.field_id: field.descriptor.get('field_name', '') for field in metric.fields}
        field_names = set(field_id_to_name.values())
        self.assertEqual(field_names, {'cpu', 'memory', 'disk'})
        
        # Verify multiple datapoints
        self.assertEqual(len(metric.datapoints), 2)
        
        # First datapoint
        dp1 = metric.datapoints[0]
        self.assertEqual(len(dp1.profile.field_ids), 3)
        self.assertEqual(len(dp1.profile.values), 3)
        self.assertEqual(dp1.measurement_ts, 1000)
        
        # Verify field_ids match field_names
        dp1_values = {field_id_to_name[fid]: val for fid, val in zip(dp1.profile.field_ids, dp1.profile.values)}
        self.assertEqual(dp1_values['cpu'], 50.5)
        self.assertEqual(dp1_values['memory'], 1024.0)
        self.assertEqual(dp1_values['disk'], 500.0)
        
        # Second datapoint
        dp2 = metric.datapoints[1]
        self.assertEqual(len(dp2.profile.field_ids), 3)
        self.assertEqual(len(dp2.profile.values), 3)
        self.assertEqual(dp2.measurement_ts, 2000)
        
        dp2_values = {field_id_to_name[fid]: val for fid, val in zip(dp2.profile.field_ids, dp2.profile.values)}
        self.assertEqual(dp2_values['cpu'], 75.2)
        self.assertEqual(dp2_values['memory'], 2048.0)
        self.assertEqual(dp2_values['disk'], 750.0)
        
        # Test validation
        with self.assertRaises(ValueError):
            store.update_profile(name=None, tags={}, profile={}, measurement_ts=1000)
        with self.assertRaises(ValueError):
            store.update_profile(name='test', tags={}, profile=None, measurement_ts=1000)
        
        # Test that update_profile skips fields that weren't added
        store.clear()
        unknown_field_id = 999999
        profile_with_unknown = {cpu_field_id: 50.5, unknown_field_id: 100.0}
        store.update_profile(name='resource_usage', tags={'env': 'prod'}, 
                            profile=profile_with_unknown, measurement_ts=3000, unit='bytes')
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        # Should only have 1 field_id (cpu), unknown_field_id should be skipped
        self.assertEqual(len(metric.datapoints[0].profile.field_ids), 1)
        self.assertEqual(len(metric.datapoints[0].profile.values), 1)
        self.assertEqual(metric.datapoints[0].profile.field_ids[0], cpu_field_id)
        self.assertEqual(metric.datapoints[0].profile.values[0], 50.5)
        
        # Test counter profile fields
        store.clear()
        request_count_field_id = store.add_counter_profile_field({'field_name': 'request_count'})
        error_count_field_id = store.add_counter_profile_field({'field_name': 'error_count'})
        
        profile_counter = {request_count_field_id: 100.0, error_count_field_id: 5.0}
        store.update_profile(name='api_metrics', tags={'service': 'api'}, 
                            profile=profile_counter, measurement_ts=4000, unit='count')
        
        metrics = store.export()
        self.assertEqual(len(metrics), 1)
        metric = metrics[0]
        self.assertEqual(metric.name, 'api_metrics')
        self.assertEqual(metric.type, signals_pb2.Metric.MetricType.PROFILE_METRIC)
        self.assertEqual(len(metric.fields), 2)
        
        # Verify field types
        field_types = {field.field_id: field.type for field in metric.fields}
        self.assertEqual(field_types[request_count_field_id], signals_pb2.ProfileField.FieldType.COUNTER_FIELD)
        self.assertEqual(field_types[error_count_field_id], signals_pb2.ProfileField.FieldType.COUNTER_FIELD)
        
        # Verify datapoint
        self.assertEqual(len(metric.datapoints), 1)
        dp = metric.datapoints[0]
        self.assertEqual(len(dp.profile.field_ids), 2)
        self.assertEqual(len(dp.profile.values), 2)
        self.assertEqual(dp.measurement_ts, 4000)
