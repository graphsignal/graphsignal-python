import unittest
import logging
from unittest.mock import patch, Mock

import graphsignal
from graphsignal.windows import Metric

logger = logging.getLogger('graphsignal')


class WindowsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_compute_histogram_0_bin(self):
        data = [1.1, 1.1, 2, 2, 3, 3, 3, 4.0001, 4.0001]

        metric = Metric()
        metric.compute_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0, 1.1, 2, 2.0, 2, 3.0, 3, 4.0001, 2]
        )

    def test_compute_histogram_log_bin(self):
        data = [i for i in range(-150, 1350)]

        metric = Metric()
        metric.compute_histogram(data)
        self.assertEqual(
            metric.measurement,
            [100, -200, 50, -100, 100, 0, 100, 100, 100, 200, 100, 300, 100, 400, 100, 500, 100,
                600, 100, 700, 100, 800, 100, 900, 100, 1000, 100, 1100, 100, 1200, 100, 1300, 50]
        )

    def test_compute_categorical_histogram(self):
        data = ['a', 'b', 'b', 'c', 'd']

        metric = Metric()
        metric.compute_categorical_histogram(data)

        self.assertEqual(
            metric.measurement,
            [0, 43, 3, 53, 1, 97, 1]
        )
