import time
import unittest
import logging
from unittest.mock import patch, Mock
import numpy as np
import pandas as pd
import pprint

import graphsignal
from graphsignal import statistics as ds
from graphsignal.windows import Metric
from graphsignal.predictions import Prediction

logger = logging.getLogger('graphsignal')


class StatisticsTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        graphsignal.configure(api_key='k1', debug_mode=True)

    def tearDown(self):
        graphsignal.shutdown()

    def test_data_size(self):
        self.assertEqual(ds.estimate_size([[1], [2]]), 2)
        self.assertEqual(ds.estimate_size(np.array([[1], [2]])), 2)
        self.assertEqual(ds.estimate_size({'a': [1, 2, 3], 'b': [4, 5, 6]}), 3)
        self.assertEqual(ds.estimate_size(
            {'a': np.asarray([1, 2, 3]), 'b': np.asarray([4, 5, 6])}), 3)
        self.assertEqual(ds.estimate_size(pd.DataFrame(data=[[1], [2]])), 2)

    def test_convert_to_2d_dict_of_lists(self):
        d = {'c1': [1, 2], 'c2': [3.1, 4.1]}
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), ['c1', 'c2'])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'int64', 'float64'])

    def test_convert_to_2d_list_of_dicts(self):
        d = [{'c1': 1, 'c2': 2.1}, {'c1': 3, 'c2': 4.1}]
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 2.1], [3, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), ['c1', 'c2'])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'int64', 'float64'])

    def test_convert_to_2d_list_of_nested_dicts(self):
        d = [{'c1': 1, 'c2': {'c3': 2.1}}, {'c1': 3, 'c2': {'c3': 4.1}}]
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1], [3]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), ['c1'])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], ['int64'])

    def test_convert_to_2d_list_of_ndarray(self):
        d = [np.asarray([1, 3.1]), np.asarray([2, 4.1])]
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), [0, 1])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()],
                         ['float64', 'float64'])

    def test_convert_to_2d_dict_of_ndarray(self):
        d = {'c1': np.asarray([1, 2]), 'c2': np.asarray([3.1, 4.1])}
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), ['c1', 'c2'])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'int64', 'float64'])

    def test_convert_to_2d_dataframe_with_columns(self):
        df = pd.DataFrame(data={'c1': [1, 2], 'c2': [3.1, 4.1]})
        data2d = ds._convert_to_2d(df)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), ['c1', 'c2'])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'int64', 'float64'])

    def test_convert_to_2d_dataframe_wo_columns(self):
        df = pd.DataFrame(data=[[1, 3.1], [2, 4.1]])
        data2d = ds._convert_to_2d(df)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.shape[0], 2)
        self.assertEqual(data2d.columns.values.tolist(), [0, 1])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'int64', 'float64'])

    def test_convert_to_2d_ndarray(self):
        d = np.asarray([[1, 3.1], [2, 4.1]])
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(
            data2d.to_numpy(), np.asarray([[1, 3.1], [2, 4.1]])))
        self.assertEqual(data2d.columns.values.tolist(), [0, 1])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()],
                         ['float64', 'float64'])

    def test_convert_to_2d_ndarray_3d(self):
        d = np.full((2, 5, 3), [1, 2, 3.3])
        data2d = ds._convert_to_2d(d)
        self.assertIsNotNone(data2d)
        self.assertTrue(np.array_equal(data2d.to_numpy(), np.asarray([
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3],
            [1, 2, 3.3]
        ])))
        self.assertEqual(data2d.shape[0], 10)
        self.assertEqual(data2d.columns.values.tolist(), [0, 1, 2])
        self.assertEqual([str(t) for t in data2d.dtypes.tolist()], [
                         'float64', 'float64', 'float64'])

    def test_convert_window_to_2d(self):
        d1 = np.asarray([[1, 3.1], [2, 4.1]])
        d2 = np.asarray([[2, 3.2], [3, 5.1]])

        data2d = ds._convert_window_to_2d([(d1, 1), (d2, 2)])
        self.assertTrue(np.array_equal(data2d[0].to_numpy(), np.concatenate([d1, d2])))
        self.assertTrue(np.array_equal(data2d[1], np.concatenate([[1, 1], [2, 2]])))


    @patch('time.time', return_value=1)
    def test_compute_metrics_empty(self, mocked_time):
        metrics, samples = ds.compute_metrics(
            [Prediction(input_data=[], output_data=[])])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(metric_dicts,
                         []
                         )
        self.assertEqual(samples, [])

    @patch('time.time', return_value=1)
    def test_compute_metrics_simple(self, mocked_time):
        input_data = pd.DataFrame(
            data={'f1': [1, 1, 2, 0], 'f2': [3.5, 4.5, 5.5, 0]})
        output_data = pd.DataFrame(
            data={
                'c1': [
                    0.1, 0.2, 0.1, 0.4], 'c2': [
                    5, 10, 7, 12]})
        metrics, _ = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'dataset': 'input',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [25.0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [1.0, 4],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.816496580927726, 4],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.0, 4],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [1.5, 4],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.1, 0.0, 1, 1.0, 2, 2.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [25.0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [4, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [3.375, 4],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [2.3935677693908453, 4],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [-1.333118339791635, 4],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [1.9096859504132233, 4],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f2',
              'measurement': [0.1, 0.0, 1, 3.5, 1, 4.5, 1, 5.5, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.2, 4],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.14142135623730953, 4],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [1.4142135623730951, 4],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [1.5000000000000018, 4],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.01, 0.1, 2, 0.2, 1, 0.4, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [0.0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [4, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [8.5, 4],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [3.1091263510296048, 4],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [0.0, 4],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [-2.4328180737217604, 4],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c2',
              'measurement': [0.1, 5.0, 1, 7.0, 1, 10.0, 1, 12.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''}]
        )

    @patch('time.time', return_value=1)
    def test_compute_metrics_nan_inf(self, mocked_time):
        input_data = pd.DataFrame(data={'f1': [1, float('nan')]})
        output_data = pd.DataFrame(data={'c1': [float('inf'), 2]})
        metrics, _ = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'dataset': 'input',
              'measurement': [2, 2],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'measurement': [1, 2],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [50.0, 2],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.0, 2],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [2, 2],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [1.0, 2],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.0, 2],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0.0, 2],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [-3.0, 2],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': 'f1',
              'measurement': [0, 1.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [2, 2],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [1, 2],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [50.0, 2],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.0, 2],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [2, 2],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [2.0, 2],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.0, 2],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0.0, 2],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [-3.0, 2],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': 'c1',
              'measurement': [0, 2.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''}]
        )

    @patch('time.time', return_value=1)
    def test_compute_metrics_categorical(self, mocked_time):
        input_data = pd.DataFrame(
            data=[['a', 1], ['b', 2], ['b', 2], ['c', 3]])
        output_data = pd.DataFrame(data=['d', 'e', 'e', 'f', 'f', None, ''])
        metrics, _ = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'dataset': 'input',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '0',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': '0',
              'measurement': [0.0, 4],
              'name': 'empty_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': '0',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '0',
              'measurement': [1, 43, 3, 97, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [0.0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [0.0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [2.0, 4],
              'name': 'mean',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [0.816496580927726, 4],
              'name': 'standard_deviation',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [0.0, 4],
              'name': 'skewness',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [1.5, 4],
              'name': 'kurtosis',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'input',
              'dimension': '1',
              'measurement': [0.1, 1.0, 1, 2.0, 2, 3.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [7, 7],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'measurement': [1, 7],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': '0',
              'measurement': [14.285714285714285, 7],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': '0',
              'measurement': [14.285714285714285, 7],
              'name': 'empty_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': '%'},
             {'dataset': 'output',
              'dimension': '0',
              'measurement': [4, 7],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'dataset': 'output',
              'dimension': '0',
              'measurement': [1, 53, 1, 59, 2, 66, 3],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''}]
        )

    @patch('graphsignal.statistics._random_index', return_value=[10, 20, 30])
    @patch('time.time', return_value=1)
    def test_compute_metrics_samples(self, mocked_time, mocked_random_index):
        r = []
        r.extend(list(range(100)))
        r.append(1001)
        r.extend(list(range(100)))
        r.append(1002)
        c = ['abc' + str(i) for i in r]
        input_data = pd.DataFrame(data={'f1': r, 'f2': r})
        output_data = pd.DataFrame(data={'c1': r})
        context_data = pd.DataFrame(data={'ctx1': c})
        _, samples = ds.compute_metrics([Prediction(
            input_data=input_data, output_data=output_data, context_data=context_data)])
        samples_dict = [sample.to_dict() for sample in samples]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(samples_dict)
        self.assertEqual(
            samples_dict,
            [{'name': 'random_sample',
              'parts': [{'data': 'f1,f2\n10,10\n20,20\n30,30',
                            'dataset': 'input',
                            'format': 'csv'},
                           {'data': 'c1\n10\n20\n30',
                            'dataset': 'output',
                            'format': 'csv'},
                           {'data': 'ctx1,prediction_timestamp\nabc10,1\nabc20,1\nabc30,1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 3,
              'timestamp': 1},
             {'name': 'input_outliers',
              'parts': [{'data': 'f1,f2\n1002,1002\n1001,1001',
                            'dataset': 'input',
                            'format': 'csv'},
                           {'data': 'c1\n1002\n1001',
                            'dataset': 'output',
                            'format': 'csv'},
                           {'data': 'ctx1,prediction_timestamp\nabc1002,1\nabc1001,1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 2,
              'timestamp': 1},
             {'name': 'output_outliers',
              'parts': [{'data': 'c1\n1002\n1001',
                            'dataset': 'output',
                            'format': 'csv'},
                           {'data': 'f1,f2\n1002,1002\n1001,1001',
                            'dataset': 'input',
                            'format': 'csv'},
                           {'data': 'ctx1,prediction_timestamp\nabc1002,1\nabc1001,1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 2,
              'timestamp': 1}]
        )

    @patch('time.time', return_value=1)
    def test_compute_metrics_sample_single(self, mocked_time):
        input_data = pd.DataFrame(data={'f1': [1], 'f2': [2]})
        output_data = pd.DataFrame(data={'c1': [3]})
        _, samples = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        sample_dict = [sample.to_dict() for sample in samples]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(sample_dict)
        self.assertEqual(
            sample_dict,
            [{'name': 'random_sample',
              'parts': [{'data': 'f1,f2\n1,2', 'dataset': 'input', 'format': 'csv'},
                           {'data': 'c1\n3', 'dataset': 'output', 'format': 'csv'},
                           {'data': 'prediction_timestamp\n1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 1,
              'timestamp': 1}]
        )

    @patch('graphsignal.statistics._random_index', return_value=[0])
    @patch('time.time', return_value=1)
    def test_compute_metrics_sample_no_outliers(
            self, mocked_time, mocked_random_index):
        r = [1] * 100
        input_data = pd.DataFrame(data={'f1': r, 'f2': r})
        output_data = pd.DataFrame(data={'c1': r})
        _, samples = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        sample_dict = [sample.to_dict() for sample in samples]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(sample_dict)
        self.assertEqual(
            sample_dict,
            [{'name': 'random_sample',
              'parts': [{'data': 'f1,f2\n1,1', 'dataset': 'input', 'format': 'csv'},
                           {'data': 'c1\n1', 'dataset': 'output', 'format': 'csv'},
                           {'data': 'prediction_timestamp\n1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 1,
              'timestamp': 1}]
        )

    @patch('graphsignal.statistics._random_index', return_value=[0])
    @patch('time.time', return_value=1)
    def test_compute_metrics_sample_missing_values(
            self, mocked_time, mocked_random_index):
        r = [[None, np.nan, np.inf, float('nan'), float('inf')]] * 100
        input_data = pd.DataFrame(data=r)
        output_data = pd.DataFrame(data=r)
        _, samples = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        sample_dict = [sample.to_dict() for sample in samples]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(sample_dict)
        self.assertEqual(
            sample_dict,
            [{'name': 'random_sample',
              'parts': [{'data': '0,1,2,3,4\nNone,nan,inf,nan,inf',
                            'dataset': 'input',
                            'format': 'csv'},
                           {'data': '0,1,2,3,4\nNone,nan,inf,nan,inf',
                            'dataset': 'output',
                            'format': 'csv'},
                           {'data': 'prediction_timestamp\n1',
                            'dataset': 'context',
                            'format': 'csv'}],
              'size': 1,
              'timestamp': 1}]
        )
