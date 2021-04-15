import unittest
import logging
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.windows import Metric

logger = logging.getLogger('graphsignal')


class WindowsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_update_percentile(self):
        data = [1, 2, 3, 4, 5] * 10
        data.append(100)
        data.append(1000)

        metric = Metric()
        for sample in data:
            metric.update_percentile(sample, 95)

        metric.finalize()

        self.assertEqual(
            metric.measurement,
            [5, 52]
        )

    def test_compute_histogram(self):
        data = [-0.22, -0.001, 0, 1, 1.1, 2, 2.00345]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0.1, -0.3, 1, -0.1, 1, 0.0, 1, 1.0, 1, 1.1, 1, 2.0, 2]
        )

    def test_compute_histogram_one_measurement(self):
        data = [212]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0, 212, 1]
        )

    def test_compute_histogram_same_measurement(self):
        data = [0.1111, 0.1111, 0.1111]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0, 0.1111, 3]
        )

    def test_compute_histogram_many_measurements(self):
        data = [i for i in range(15, 799)]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(len(metric.measurement), 159)

    def test_compute_histogram_big_range(self):
        data = [-50032, 2000456]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [100000, -100000, 1, 2000000, 1]
        )

    def test_compute_histogram_small_range(self):
        data = [-0.00000012, 0.000009]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0.0000001, -0.0000002, 1, 0.000009, 1]
        )

    def test_compute_categorical_histogram(self):
        data = ['a', 'b', 'b', 'c', 'd']

        metric = Metric()
        metric.compute_categorical_histogram(data)

        self.assertEqual(
            metric.measurement,
            [1, 43, 3, 53, 1, 97, 1]
        )
