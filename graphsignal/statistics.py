import re
import unicodedata
import time
import logging
import numpy as np
import pandas as pd

import graphsignal
from graphsignal.predictions import Prediction
from graphsignal.metrics import *
from graphsignal import metrics_pb2

logger = logging.getLogger('graphsignal')
_rand = np.random.RandomState(int(time.time()))

MAX_COLUMNS = 250


def estimate_size(data):
    if data is None:
        return 0

    if isinstance(data, list):
        return len(data)
    elif isinstance(data, dict):
        if len(data) > 0:
            first_elem = next(iter(data.values()))
            if isinstance(first_elem, list):
                return len(first_elem)
            elif isinstance(first_elem, np.ndarray):
                return first_elem.shape[0] if first_elem.ndim > 0 else 1
            else:
                return 1
    elif isinstance(data, np.ndarray):
        return data.shape[0] if data.ndim > 0 else 1
    elif isinstance(data, pd.DataFrame):
        return data.shape[0]
    return 0


def update_metrics(metric_updaters, window_proto, prediction_buffer):
    if len(prediction_buffer) == 0:
        return

    start_ts = time.time()
    input_buffer = [p.input_data
                    for p in prediction_buffer if p.input_data is not None]
    input_columns = prediction_buffer[-1].input_columns
    input_columns, input_arr = _convert_to_numpy(
        input_buffer, columns=input_columns)

    output_buffer = [p.output_data
                     for p in prediction_buffer if p.output_data is not None]
    output_columns = prediction_buffer[-1].output_columns
    output_columns, output_arr = _convert_to_numpy(
        output_buffer, columns=output_columns)

    if input_arr is None and output_arr is None:
        return

    if input_arr is not None:
        data_stream = get_data_stream(
            window_proto,
            metrics_pb2.DataStream.DataSource.MODEL_INPUT,
            metrics_pb2.DataStream.DataType.TABULAR)
        _update_tabular_metrics(
            metric_updaters,
            data_stream,
            input_columns,
            input_arr)

    if output_arr is not None:
        data_stream = get_data_stream(
            window_proto,
            metrics_pb2.DataStream.DataSource.MODEL_OUTPUT,
            metrics_pb2.DataStream.DataType.TABULAR)
        _update_tabular_metrics(
            metric_updaters,
            data_stream,
            output_columns,
            output_arr)

    logger.debug('Computing data metrics took %.3f sec',
                 time.time() - start_ts)


def _update_tabular_metrics(metric_updaters, data_stream_proto, columns, data):
    if len(columns) != len(data):
        raise ValueError(
            'Error processing tabular data: columns and values do not match')

    instance_count = data[0].shape[0]
    metric_updater = get_metric_updater(
        metric_updaters, data_stream_proto, 'instance_count')
    metric_updater.update_counter(instance_count)

    metric_updater = get_metric_updater(
        metric_updaters, data_stream_proto, 'column_count')
    metric_updater.update_gauge(len(columns))

    for column_name, column_values in zip(
            columns[:MAX_COLUMNS], data[:MAX_COLUMNS]):
        if isinstance(column_name, str) and column_name.lower() in (
                'timestamp', 'time', 'datetime', 'date'):
            continue
        column_type = column_values.dtype
        if np.issubdtype(column_type, np.datetime64):
            continue
        value_count = column_values.shape[0]

        dimensions = {
            'column': str(column_name)[:50]
        }

        if np.issubdtype(column_type, np.number):
            missing_count = np.count_nonzero(np.isnan(column_values))
            missing_count += np.count_nonzero(np.isinf(column_values))
            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'missing_values', dimensions)
            metric_updater.update_ratio(missing_count, value_count)

            zero_count = np.count_nonzero(np.equal(column_values, 0))
            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'zero_values', dimensions)
            metric_updater.update_ratio(zero_count, value_count)

            finite_column_values = column_values[np.isfinite(column_values)]
            finite_value_count = finite_column_values.shape[0]

            if np.issubdtype(column_type, np.integer):
                integer_count = finite_value_count
                float_count = 0
            elif np.issubdtype(column_type, np.floating):
                integer_count = np.count_nonzero(
                    np.equal(np.mod(finite_column_values, 1), 0))
                float_count = finite_value_count - integer_count
            else:
                integer_count = 0
                float_count = 0

            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'integer_values', dimensions)
            metric_updater.update_ratio(integer_count, value_count)

            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'float_values', dimensions)
            metric_updater.update_ratio(float_count, value_count)

            if finite_column_values.shape[0] > 0:
                finite_column_values_list = finite_column_values.tolist()

                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'distribution', dimensions)
                metric_updater.update_distribution(finite_column_values_list)
        else:
            string_column_values = [
                m for m in column_values.tolist() if isinstance(m, str)]
            if len(string_column_values) > 0:
                missing_count = value_count - len(string_column_values)
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'missing_values', dimensions)
                metric_updater.update_ratio(missing_count, value_count)

                empty_count = string_column_values.count('')
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'empty_values', dimensions)
                metric_updater.update_ratio(empty_count, value_count)

                string_count = len(string_column_values)
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'string_values', dimensions)
                metric_updater.update_ratio(string_count, value_count)

                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'distribution', dimensions)
                metric_updater.update_distribution(
                    _truncate_strings(string_column_values))


def _convert_to_numpy(data_buffer, columns=None):
    if len(data_buffer) == 0:
        return None, None

    data = None

    first_item = data_buffer[0]

    # list
    if isinstance(first_item, list):
        keys = list(range(len(first_item)))
        data_index = {key: [] for key in keys}
        for item in data_buffer:
            for key in keys:
                data_index[key].append(item[key])
        if columns is None:
            columns = [str(key) for key in keys]
        data = []
        for arrays in data_index.values():
            flat_array = _flatten_and_convert_arrays(arrays)
            if flat_array is None:
                data = None
                break
            data.append(flat_array)

    # dict
    elif isinstance(first_item, dict):
        keys = list(first_item.keys())
        data_index = {key: [] for key in keys}
        for item in data_buffer:
            for key in keys:
                data_index[key].append(item[key])
        columns = [str(key) for key in keys]
        data = []
        for arrays in data_index.values():
            flat_array = _flatten_and_convert_arrays(arrays)
            if flat_array is None:
                data = None
                break
            data.append(flat_array)

    # numpy.ndarray
    elif isinstance(first_item, np.ndarray):
        if first_item.ndim == 2:
            if columns is None:
                columns = [str(column)
                           for column in range(first_item.shape[1])]
            data_arr = np.concatenate(data_buffer)
            data = [
                col_arr.ravel() for col_arr in np.hsplit(
                    data_arr, data_arr.shape[1])]

    # pandas.DataFrame
    elif isinstance(first_item, pd.DataFrame):
        data_df = pd.concat(data_buffer, ignore_index=True)
        columns = data_df.columns.values.tolist()
        data = []
        for column in columns:
            data.append(data_df[column].to_numpy())
        columns = [str(column) for column in columns]

    if data is None:
        raise ValueError(
            'Unsupported data format: please use one or two-dimensional list, dict, numpy.  ndarray or pandas.DataFrame')

    return columns, data


def _flatten_and_convert_arrays(arrays):
    flat_array = None
    first_array = arrays[0]
    if isinstance(first_array, list):
        flat_array = np.asarray([elem for array in arrays for elem in array])
    elif isinstance(first_array, np.ndarray):
        if first_array.ndim == 1:
            flat_array = np.concatenate(arrays)
    else:
        flat_array = np.asarray(arrays)  # scalars

    return flat_array


def _is_2d(data):
    if isinstance(data, list):
        if len(data) > 0:
            if not _is_1d(data[0]):
                return False
        return True
    elif isinstance(data, dict):
        if len(data) > 0:
            if not _is_1d(list(data.values())[0]):
                return False
        return True
    elif isinstance(data, np.ndarray):
        return data.ndim == 2
    return False


def _is_1d(data):
    if isinstance(data, list):
        if len(data) > 0:
            if not _is_scalar(data[0]):
                return False
        return True
    if isinstance(data, dict):
        if len(data) > 0:
            if not _is_scalar(list(data.values())[0]):
                return False
        return True
    elif isinstance(data, np.ndarray):
        return data.ndim == 1
    return False


def _is_scalar(data):
    return isinstance(data, (int, float, str))


def _truncate_strings(values, max_size=18, front_size=10, tail_size=5):
    truncated_values = []
    for value in values:
        if len(value) > max_size:
            truncated_values.append(
                value[:front_size] + '...' + value[-tail_size:])
        else:
            truncated_values.append(value)
    return truncated_values
