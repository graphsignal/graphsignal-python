import unittest
import logging
import sys
import time

from prometheus_client import Counter, Gauge, Histogram, Summary, REGISTRY

import graphsignal
from graphsignal.recorders.prometheus_adapter import PrometheusAdapter

logger = logging.getLogger('graphsignal')


class PrometheusAdapterIntegrationTest(unittest.TestCase):
    def setUp(self):
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        # Enable debug logging
        logger.setLevel(logging.DEBUG)
        
        graphsignal.configure(
            api_key='k1',
            debug_mode=True)
        graphsignal._tracer.auto_export = False
        
        # Clear any existing metrics
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()
        
        # Clear the metric store
        graphsignal._tracer.metric_store().clear()

    def tearDown(self):
        graphsignal.shutdown()
        # Clean up metrics
        REGISTRY._collector_to_names.clear()
        REGISTRY._names_to_collectors.clear()

    def test_gauge_metric(self):
        """Test gauge metric collection"""
        # Expose a gauge metric
        gauge = Gauge('test_gauge', 'Test gauge', ['label1'])
        gauge.labels('value1').set(42.5)
        
        # Collect the metric
        adapter = PrometheusAdapter()
        adapter.setup({'test_gauge': 'mapped_gauge'})
        adapter.collect()
        
        # Assert the metric was collected correctly
        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags.update({'label1': 'value1'})
        key = store.metric_key('mapped_gauge', metric_tags)
        
        self.assertIn(key, store._metrics)
        self.assertEqual(store._metrics[key].gauge, 42.5)

    def test_counter_metric(self):
        """Test counter metric delta calculation"""
        # Expose a counter metric
        counter = Counter('test_counter', 'Test counter', ['label1'])
        counter.labels('value1').inc(10)
        
        # Collect the metric
        adapter = PrometheusAdapter()
        adapter.setup({'test_counter': 'mapped_counter'})
        
        # First collection - should not increment (just stores initial value)
        adapter.collect()
        
        # Increment the counter more
        counter.labels('value1').inc(5)
        
        # Second collection - should increment by the delta (5)
        adapter.collect()
        
        # Assert the counter was incremented by the delta
        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags.update({'label1': 'value1'})
        key = store.metric_key('mapped_counter', metric_tags)
        
        self.assertIn(key, store._metrics)
        # The counter should have been incremented by 5 (the delta from 10 to 15)
        self.assertEqual(store._metrics[key].total, 5)

    def test_histogram_metric(self):
        """Test histogram metric collection"""
        # Expose a histogram metric
        histogram = Histogram('test_histogram', 'Test histogram', ['label1'])
        histogram.labels('value1').observe(1.5)
        
        # Collect the metric
        adapter = PrometheusAdapter()
        adapter.setup({'test_histogram': 'mapped_histogram'})
        adapter.collect()
        
        # Assert the histogram was collected (check for rate metric)
        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags.update({'label1': 'value1'})
        
        # Histograms now create rate metrics using _count and _sum
        key = store.metric_key('mapped_histogram', metric_tags)
        
        # Check that the rate metric was collected
        self.assertIn(key, store._metrics)

    def test_summary_metric(self):
        """Test summary metric collection"""
        # Expose a summary metric
        summary = Summary('test_summary', 'Test summary', ['label1'])
        summary.labels('value1').observe(2.5)
        
        # Collect the metric
        adapter = PrometheusAdapter()
        adapter.setup({'test_summary': 'mapped_summary'})
        adapter.collect()
        
        # Assert the summary was collected (check for rate metric)
        store = graphsignal._tracer.metric_store()
        metric_tags = graphsignal._tracer.tags.copy()
        metric_tags.update({'label1': 'value1'})
        
        # Summaries now create rate metrics using _count and _sum
        key = store.metric_key('mapped_summary', metric_tags)
        
        # Check that the rate metric was collected
        self.assertIn(key, store._metrics)


if __name__ == '__main__':
    unittest.main()
