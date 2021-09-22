import unittest
import logging
import time

import graphsignal
from graphsignal.sketches.kll import KLLSketch
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')


class KLLSketchTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_kll_update(self):
        k1 = KLLSketch()
        k1.update(1)
        k1.update(2)
        k1.update(2)
        k1.update(3.333)
        k1.update(1000)
        k1.update(0.0001)
        self.assertEqual(k1.count(), 6)
        self.assertEqual(
            k1.ranks(), [
                (0.0001, 1), (1, 2), (2, 3), (2, 4), (3.333, 5), (1000, 6)])
        self.assertEqual(k1.cdf(), [(0.0001, 0.16666666666666666), (1, 0.3333333333333333), (
            2, 0.5), (2, 0.6666666666666666), (3.333, 0.8333333333333334), (1000, 1.0)])

    def test_kll_update_str(self):
        k1 = KLLSketch()
        k1.update('1')
        k1.update('2')
        k1.update('2')
        k1.update('3')
        self.assertEqual(k1.count(), 4)
        self.assertEqual(k1.ranks(), [('1', 1), ('2', 2), ('2', 3), ('3', 4)])
        self.assertEqual(
            k1.cdf(), [
                ('1', 0.25), ('2', 0.5), ('2', 0.75), ('3', 1.0)])

    def test_kll_merge(self):
        k1 = KLLSketch()

        k2 = KLLSketch()
        k2.update(1)
        k2.update(2)
        k1.merge(k2)

        k3 = KLLSketch()
        k3.update(1)
        k3.update(2)
        k3.update(12)
        k1.merge(k3)

        self.assertEqual(
            k1.cdf(), [
                (1, 0.2), (1, 0.4), (2, 0.6), (2, 0.8), (12, 1.0)])

    def test_kll_distribution(self):
        k1 = KLLSketch()
        for i in range(5):
            k1.update(1)
        k1.update(2)
        for i in range(100):
            k1.update(4)

        self.assertEqual(k1.distribution(), [[1, 5], [2, 1], [4, 100]])

    def test_kll_proto(self):
        k1 = KLLSketch()
        k1.update(1)
        k1.update(2)
        k1.update(3)

        window = metrics_pb2.PredictionWindow()
        proto = window.data_streams['1'].metrics['1'].distribution_value.sketch_kll10

        k1.to_proto(proto)

        self.assertEqual(proto.item_type, proto.ItemType.DOUBLE)

        k2 = KLLSketch()
        k2.from_proto(proto)

        self.assertEqual(k2._k, k1._k)
        self.assertEqual(k2._c, k1._c)
        self.assertEqual(k2._H, k1._H)
        self.assertEqual(k2._size, k1._size)
        self.assertEqual(k2._max_size, k1._max_size)
        self.assertEqual(k2.count(), k1.count())
        self.assertEqual(k2.ranks(), k1.ranks())
        self.assertEqual(k2.cdf(), k1.cdf())

    def test_kll_update_perf(self):
        s = KLLSketch()

        start = time.time()

        #import cProfile
        #from pstats import Stats, SortKey

        # with cProfile.Profile() as pr:
        for i in range(1000):
            s.update(i)

        #stats = Stats(pr)
        # stats.sort_stats(SortKey.CUMULATIVE).print_stats(25)

        took = time.time() - start
        print('KLL update (1000) took: ', took)
        #print('CDF', s.cdf())
        self.assertTrue(took < 1)

    def test_kll_merge_perf(self):
        import random
        sketches = []
        for i in range(1000):
            s = KLLSketch()
            s.update(random.randint(1, 100))
            sketches.append(s)

        start = time.time()

        ns = KLLSketch()
        for s in sketches:
            ns.merge(s)

        took = time.time() - start
        print('kll merge (1000) took: ', time.time() - start)
        #print('kll size: ', len(ns.serialize()))
        self.assertTrue(took < 1)
