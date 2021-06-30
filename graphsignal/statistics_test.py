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

    def test_concat_prediction_data(self):
        d1 = np.asarray([[1, 3.1], [2, 4.1]])
        d2 = np.asarray([[2, 3.2], [3, 5.1]])

        data_window = ds._concat_prediction_data(
            [(d1, 1), (d2, 2)])
        self.assertTrue(np.array_equal(
            data_window.data.to_numpy(), np.concatenate([d1, d2])))
        self.assertTrue(np.array_equal(
            data_window.timestamp, np.concatenate([[1, 1], [2, 2]])))

    @patch('time.time', return_value=1)
    def test_compute_metrics_empty(self, mocked_time):
        metrics = ds.compute_metrics(
            [Prediction(input_data=[], output_data=[])])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(metric_dicts,
                         []
                         )

    @patch('time.time', return_value=1)
    def test_compute_metrics_simple(self, mocked_time):
        input_data = pd.DataFrame(
            data={'f1': [1, 1, 2, 0], 'f2': [3.5, 4.5, 5.5, 0]})
        output_data = pd.DataFrame(
            data={
                'c1': [
                    0.1, 0.2, 0.1, 0.4], 'c2': [
                    5, 10, 7, 12]})
        metrics = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'aggregation': 'sum',
              'dataset': 'model_inputs',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_inputs',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [1, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [0, 0, 1, 1, 2, 2, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f2',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f2',
              'measurement': [1, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f2',
              'measurement': [4, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_inputs',
              'dimension': 'f2',
              'measurement': [0, 0.0, 1, 3.5, 1, 4.5, 1, 5.5, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_outputs',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [0, 0.1, 2, 0.2, 1, 0.4, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c2',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c2',
              'measurement': [0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c2',
              'measurement': [4, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_outputs',
              'dimension': 'c2',
              'measurement': [0, 5, 1, 7, 1, 10, 1, 12, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''}]
        )

    @patch('time.time', return_value=1)
    def test_compute_metrics_nan_inf(self, mocked_time):
        input_data = pd.DataFrame(data={'f1': [1, float('nan')]})
        output_data = pd.DataFrame(data={'c1': [float('inf'), 2]})
        metrics = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'aggregation': 'sum',
              'dataset': 'model_inputs',
              'measurement': [2, 2],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_inputs',
              'measurement': [1, 2],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [1, 2],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [0, 2],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [2, 2],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_inputs',
              'dimension': 'f1',
              'measurement': [0, 1.0, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'measurement': [2, 2],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_outputs',
              'measurement': [1, 2],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [1, 2],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [0, 2],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': 'c1',
              'measurement': [2, 2],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_outputs',
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
        metrics = ds.compute_metrics(
            [Prediction(input_data=input_data, output_data=output_data)])
        metric_dicts = [metric.to_dict() for metric in metrics]
        #pp = pprint.PrettyPrinter()
        # pp.pprint(metric_dicts)
        self.assertEqual(
            metric_dicts,
            [{'aggregation': 'sum',
              'dataset': 'model_inputs',
              'measurement': [4, 4],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_inputs',
              'measurement': [2, 4],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '0',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '0',
              'measurement': [0, 4],
              'name': 'empty_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '0',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_inputs',
              'dimension': '0',
              'measurement': [0, 43, 3, 97, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': '#'},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '1',
              'measurement': [0, 4],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '1',
              'measurement': [0, 4],
              'name': 'zero_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_inputs',
              'dimension': '1',
              'measurement': [3, 4],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_inputs',
              'dimension': '1',
              'measurement': [0, 1, 1, 2, 2, 3, 1],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'measurement': [7, 7],
              'name': 'instance_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'last',
             'dataset': 'model_outputs',
              'measurement': [1, 7],
              'name': 'dimension_count',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': '0',
              'measurement': [1, 7],
              'name': 'missing_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': '0',
              'measurement': [1, 7],
              'name': 'empty_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'sum',
             'dataset': 'model_outputs',
              'dimension': '0',
              'measurement': [4, 7],
              'name': 'unique_values',
              'timestamp': 1,
              'type': 'statistic',
              'unit': ''},
             {'aggregation': 'merge',
             'dataset': 'model_outputs',
              'dimension': '0',
              'measurement': [0, 53, 1, 59, 2, 66, 3],
              'name': 'distribution',
              'timestamp': 1,
              'type': 'histogram',
              'unit': '#'}]
        )
