import time
import unittest
import logging
from unittest.mock import patch, Mock
import sys
import numpy as np
import pandas as pd
import pprint
from google.protobuf.json_format import MessageToDict

import graphsignal
from graphsignal import statistics
from graphsignal.predictions import Prediction
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')


class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        if len(logger.handlers) == 0:
            logger.addHandler(logging.StreamHandler(sys.stdout))
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_data_size(self):
        self.assertEqual(statistics.estimate_size([[1], [2]]), 2)
        self.assertEqual(statistics.estimate_size(np.array([[1], [2]])), 2)
        self.assertEqual(statistics.estimate_size(
            {'a': [1, 2, 3], 'b': [4, 5, 6]}), 3)
        self.assertEqual(statistics.estimate_size(
            {'a': np.asarray([1, 2, 3]), 'b': np.asarray([4, 5, 6])}), 3)
        self.assertEqual(
            statistics.estimate_size(
                pd.DataFrame(
                    data=[
                        [1],
                        [2]])),
            2)

    def test_convert_to_numpy_dict_of_scalars(self):
        d1 = {'c1': 1, 'c2': 2.1}
        d2 = {'c1': 3, 'c2': 5.1}
        columns, data = statistics._convert_to_numpy([d1, d2])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['c1', 'c2'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 3])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([2.1, 5.1])))

    def test_convert_to_numpy_dict_of_lists(self):
        d1 = {'c1': [1, 2], 'c2': [3.1, 4.1]}
        d2 = {'c1': [3], 'c2': [5.1]}
        columns, data = statistics._convert_to_numpy([d1, d2])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['c1', 'c2'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2, 3])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1, 5.1])))
        self.assertEqual(data[0].dtype, 'int64')
        self.assertEqual(data[1].dtype, 'float64')

    def test_convert_to_numpy_list_of_ndarray(self):
        d1 = [np.asarray([1, 2]), np.asarray([3.1, 4.1])]
        d2 = [np.asarray([3, 4]), np.asarray([5.1, 6.1])]
        columns, data = statistics._convert_to_numpy(
            [d1, d2], columns=['A', 'B'])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['A', 'B'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2, 3, 4])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1, 5.1, 6.1])))

    def test_convert_to_numpy_dict_of_ndarray(self):
        d1 = {'c1': np.asarray([1, 2]), 'c2': np.asarray([3.1, 4.1])}
        d2 = {'c1': np.asarray([5]), 'c2': np.asarray([6.1])}
        columns, data = statistics._convert_to_numpy([d1, d2])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['c1', 'c2'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2, 5])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1, 6.1])))

    def test_convert_to_numpy_dataframe_with_columns(self):
        d1 = pd.DataFrame(data={'c1': [1, 2], 'c2': [3.1, 4.1]})
        d2 = pd.DataFrame(data={'c1': [3], 'c2': [5.1]})
        columns, data = statistics._convert_to_numpy([d1, d2])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['c1', 'c2'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2, 3])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1, 5.1])))
        self.assertEqual(data[0].dtype, 'int64')
        self.assertEqual(data[1].dtype, 'float64')

    def test_convert_to_numpy_dataframe_wo_columns(self):
        d = pd.DataFrame(data=[[1, 3.1], [2, 4.1]])
        columns, data = statistics._convert_to_numpy([d])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['0', '1'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1])))

    def test_convert_to_numpy_ndarray(self):
        d1 = np.asarray([[1, 3.1], [2, 4.1]])
        d2 = np.asarray([[8, 12]])
        columns, data = statistics._convert_to_numpy([d1, d2])
        self.assertIsNotNone(data)
        self.assertEqual(columns, ['0', '1'])
        self.assertTrue(np.array_equal(
            data[0], np.asarray([1, 2, 8])))
        self.assertTrue(np.array_equal(
            data[1], np.asarray([3.1, 4.1, 12])))
        self.assertEqual(data[0].dtype, 'float64')
        self.assertEqual(data[1].dtype, 'float64')

    def test_convert_to_numpy_ndarray_3d(self):
        d = np.full((2, 5, 3), [1, 2, 3.3])
        with self.assertRaises(ValueError):
            statistics._convert_to_numpy([d])

    def test_update_metrics_empty(self):
        window = metrics_pb2.PredictionWindow()
        statistics.update_metrics({}, window, [])

        self.assertEqual(len(window.data_streams[str(
            metrics_pb2.DataStream.DataSource.MODEL_INPUT)].metrics), 0)
        self.assertEqual(len(window.data_streams[str(
            metrics_pb2.DataStream.DataSource.MODEL_OUTPUT)].metrics), 0)

    def test_update_metrics(self):
        window = metrics_pb2.PredictionWindow()
        metric_updaters = {}

        input_data = {
            'f1': [1, 1, 2, 0],
            'f2': [3.5, 4.5, 5.5, 0],
            'f3': ['a', 'b', 'c', 'c'],
            'f4': [0, float('nan'), float('inf'), 2]}
        output_data = [0.1, 0.2, 0.1, 0.4]

        statistics.update_metrics(
            metric_updaters, window, [
                Prediction(
                    input_data=input_data, output_data=output_data)])

        for metric_updater in metric_updaters.values():
            metric_updater.finalize()

        window_dict = MessageToDict(window)

        input_metrics_json = window_dict['dataStreams'][str(
            metrics_pb2.DataStream.DataSource.MODEL_INPUT)]['metrics']
        output_metrics_json = window_dict['dataStreams'][str(
            metrics_pb2.DataStream.DataSource.MODEL_OUTPUT)]['metrics']

        for metric in input_metrics_json.values():
            if 'distributionValue' in metric:
                del metric['distributionValue']['sketchKll10']

        for metric in output_metrics_json.values():
            if 'distributionValue' in metric:
                del metric['distributionValue']['sketchKll10']

        #pp = pprint.PrettyPrinter()
        # pp.pprint(input_metrics_json)
        # pp.pprint(output_metrics_json)

        self.assertEqual(input_metrics_json,
                         {'1c359883cf3d': {'dimensions': {'column': 'f1'},
                                           'distributionValue': {'sketchImpl': 'KLL10'},
                                           'name': 'distribution',
                                           'type': 'DISTRIBUTION'},
                             '3aa8ce8a745e': {'counterValue': {'counter': 4.0},
                                              'name': 'instance_count',
                                              'type': 'COUNTER'},
                             '401c985323db': {'dimensions': {'column': 'f1'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             '43689dc15d24': {'dimensions': {'column': 'f1'},
                                              'name': 'integer_values',
                                              'ratioValue': {'counter': 4.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             '60258f4378d5': {'dimensions': {'column': 'f4'},
                                              'name': 'missing_values',
                                              'ratioValue': {'counter': 2.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             '682f9d56838c': {'dimensions': {'column': 'f4'},
                                              'name': 'integer_values',
                                              'ratioValue': {'counter': 2.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             '6add4609813c': {'gaugeValue': {'gauge': 4.0},
                                              'name': 'column_count',
                                              'type': 'GAUGE'},
                             '6c077d9e5a61': {'dimensions': {'column': 'f4'},
                                              'name': 'zero_values',
                                              'ratioValue': {'counter': 1.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             '9cac7145da6a': {'dimensions': {'column': 'f2'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             '9df17486f062': {'dimensions': {'column': 'f3'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             'a0e8a20c8920': {'dimensions': {'column': 'f3'},
                                              'name': 'empty_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             'b930a73df792': {'dimensions': {'column': 'f1'},
                                              'name': 'zero_values',
                                              'ratioValue': {'counter': 1.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             'cff27a80f0b7': {'dimensions': {'column': 'f1'},
                                              'name': 'float_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             'd96d26ea23b1': {'dimensions': {'column': 'f4'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             'dd7c7096ed49': {'dimensions': {'column': 'f4'},
                                              'name': 'float_values',
                                              'ratioValue': {'total': 4.0},
                                              'type': 'RATIO'},
                             'e5073f3ae895': {'dimensions': {'column': 'f2'},
                                              'name': 'integer_values',
                                              'ratioValue': {'counter': 1.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             'e55eece578e6': {'dimensions': {'column': 'f3'},
                                              'name': 'string_values',
                                              'ratioValue': {'counter': 4.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             'ebad37241fb2': {'dimensions': {'column': 'f2'},
                                              'name': 'float_values',
                                              'ratioValue': {'counter': 3.0, 'total': 4.0},
                                              'type': 'RATIO'},
                             'f0c86a2c6f9d': {'dimensions': {'column': 'f3'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             'f8179a7753dc': {'dimensions': {'column': 'f2'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             'ff8da228450b': {'dimensions': {'column': 'f2'},
                                              'name': 'zero_values',
                                              'ratioValue': {'counter': 1.0, 'total': 4.0},
                                              'type': 'RATIO'}})
        self.assertEqual(output_metrics_json,
                         {'05c94b6ef3dd': {'dimensions': {'column': '3'},
                                           'name': 'missing_values',
                                           'ratioValue': {'total': 1.0},
                                           'type': 'RATIO'},
                             '08bdcbcb78e5': {'dimensions': {'column': '1'},
                                              'name': 'float_values',
                                              'ratioValue': {'counter': 1.0, 'total': 1.0},
                                              'type': 'RATIO'},
                             '094ce9a486c0': {'gaugeValue': {'gauge': 4.0},
                                              'name': 'column_count',
                                              'type': 'GAUGE'},
                             '0a030c2aa2b1': {'dimensions': {'column': '1'},
                                              'name': 'integer_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '0f0ab90137aa': {'dimensions': {'column': '2'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             '1f3c3ac80b3a': {'dimensions': {'column': '0'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '32143615ee5a': {'dimensions': {'column': '1'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             '41ed21ff8956': {'dimensions': {'column': '2'},
                                              'name': 'float_values',
                                              'ratioValue': {'counter': 1.0, 'total': 1.0},
                                              'type': 'RATIO'},
                             '5c917c4a138c': {'dimensions': {'column': '3'},
                                              'name': 'zero_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '685542684133': {'dimensions': {'column': '0'},
                                              'name': 'integer_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '73c707ce113c': {'dimensions': {'column': '2'},
                                              'name': 'zero_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '785cfd27ee94': {'dimensions': {'column': '3'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             '81dcca1fbb5c': {'dimensions': {'column': '2'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             '8e1bc8013472': {'counterValue': {'counter': 1.0},
                                              'name': 'instance_count',
                                              'type': 'COUNTER'},
                             '903e0399f098': {'dimensions': {'column': '3'},
                                              'name': 'float_values',
                                              'ratioValue': {'counter': 1.0, 'total': 1.0},
                                              'type': 'RATIO'},
                             'c07ffa6709a7': {'dimensions': {'column': '2'},
                                              'name': 'integer_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             'c20d28d3bc8e': {'dimensions': {'column': '1'},
                                              'name': 'zero_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             'dd7593fd9cae': {'dimensions': {'column': '0'},
                                              'name': 'float_values',
                                              'ratioValue': {'counter': 1.0, 'total': 1.0},
                                              'type': 'RATIO'},
                             'de5757c0d4d2': {'dimensions': {'column': '1'},
                                              'name': 'missing_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             'e3f8dbe7ba73': {'dimensions': {'column': '3'},
                                              'name': 'integer_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'},
                             'ee6941f991dd': {'dimensions': {'column': '0'},
                                              'distributionValue': {'sketchImpl': 'KLL10'},
                                              'name': 'distribution',
                                              'type': 'DISTRIBUTION'},
                             'f038d97c5e5a': {'dimensions': {'column': '0'},
                                              'name': 'zero_values',
                                              'ratioValue': {'total': 1.0},
                                              'type': 'RATIO'}})

    def test_update_metrics_perf(self):
        window = metrics_pb2.PredictionWindow()
        metric_updaters = {}

        prediction_buffer = []
        for i in range(100):
            input_data = {}
            for j in range(50):
                input_data['f1_' + str(j)] = i
                input_data['f2_' + str(j)] = i + 3.53242342
                input_data['f3_' + str(j)] = 100 * 'abc' + str(i)
                input_data['f4_' + str(j)] = float('nan')
                input_data['f4_' + str(j)] = float('+inf')
            output_data = [0.1, 0.2, 0.1, 0.4]
            prediction_buffer.append(
                Prediction(
                    input_data=input_data,
                    output_data=output_data))

        #import cProfile
        #from pstats import Stats, SortKey

        start_ts = time.time()

        # with cProfile.Profile() as pr:
        for k in range(10):
            statistics.update_metrics(
                metric_updaters, window, prediction_buffer)
        #stats = Stats(pr)
        # stats.sort_stats(SortKey.CUMULATIVE).print_stats(25)

        for metric_updater in metric_updaters.values():
            metric_updater.finalize()

        took = time.time() - start_ts

        print('update_metrics took (sec): ', round(took, 6))
        self.assertTrue(len(window.data_streams[str(
            metrics_pb2.DataStream.DataSource.MODEL_INPUT)].metrics) > 0)
        self.assertTrue(len(window.data_streams[str(
            metrics_pb2.DataStream.DataSource.MODEL_OUTPUT)].metrics) > 0)
        self.assertTrue(took < 1)

    def test_truncate_strings(self):
        truncated = statistics._truncate_strings(
            ['abc', '1234567890abcdefgh', '1234567890abcdefghi'])
        self.assertEqual(
            truncated, [
                'abc', '1234567890abcdefgh', '1234567890...efghi'])
