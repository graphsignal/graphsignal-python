import re
import unicodedata
import time
import logging
import numpy as np
import pandas as pd

import graphsignal
from graphsignal.predictions import Prediction, DataWindow
from graphsignal.windows import Metric

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
            elem = next(iter(data.values()))
            if isinstance(elem, list):
                return len(elem)
            elif isinstance(elem, np.ndarray):
                return elem.shape[0] if elem.ndim > 0 else 1
    elif isinstance(data, np.ndarray):
        return data.shape[0] if data.ndim > 0 else 1
    elif isinstance(data, pd.DataFrame):
        return data.shape[0]
    return 0


def compute_metrics(prediction_window):
    start_ts = time.time()
    metrics = []

    if len(prediction_window) == 0:
        return metrics

    last_timestamp = max([p.timestamp for p in prediction_window if p])

    prediction_inputs = [(p.input_data, p.timestamp)
                         for p in prediction_window if p.input_data is not None]
    input_window = _concat_prediction_data(prediction_inputs)

    prediction_outputs = [(p.output_data, p.timestamp)
                          for p in prediction_window if p.output_data is not None]
    output_window = _concat_prediction_data(prediction_outputs)

    if input_window is None and output_window is None:
        logger.warning('Provided empty data, nothing to compute')
        return metrics

    # compute metrics
    if input_window is not None:
        metrics.extend(
            _compute_tabular_metrics(
                input_window,
                'model_inputs',
                last_timestamp))

    if output_window is not None:
        metrics.extend(
            _compute_tabular_metrics(
                output_window,
                'model_outputs',
                last_timestamp))

    logger.debug('Computing metrics took %.3f sec',
                 time.time() - start_ts)

    return metrics


def _compute_tabular_metrics(data_window, dataset, timestamp):
    metrics = []

    instance_count = data_window.data.shape[0]

    metric = Metric(
        dataset=dataset,
        name='instance_count',
        aggregation=Metric.AGGREGATION_SUM,
        timestamp=timestamp)
    metric.set_statistic(instance_count, instance_count)
    metrics.append(metric)

    metric = Metric(
        dataset=dataset,
        name='dimension_count',
        aggregation=Metric.AGGREGATION_LAST,
        timestamp=timestamp)
    metric.set_statistic(len(data_window.data.columns), instance_count)
    metrics.append(metric)

    types = data_window.data.dtypes.tolist()
    columns = _format_names(data_window.data.columns.values.tolist())

    for column_index in range(min(data_window.data.shape[1], MAX_COLUMNS)):
        column_name = columns[column_index]
        if column_name.lower() in ('timestamp', 'time', 'datetime', 'date'):
            continue
        column_type = types[column_index]
        if np.issubdtype(column_type, np.datetime64):
            continue
        column_values = data_window.data[data_window.data.columns[column_index]].to_numpy(
        )

        if np.issubdtype(column_type, np.number):
            missing_count = np.count_nonzero(np.isnan(column_values))
            missing_count += np.count_nonzero(np.isinf(column_values))
            metric = Metric(
                dataset=dataset,
                dimension=column_name,
                name='missing_values',
                aggregation=Metric.AGGREGATION_SUM,
                timestamp=timestamp)
            metric.set_statistic(missing_count, instance_count)
            metrics.append(metric)

            zero_count = np.count_nonzero(column_values == 0)
            metric = Metric(
                dataset=dataset,
                dimension=column_name,
                name='zero_values',
                aggregation=Metric.AGGREGATION_SUM,
                timestamp=timestamp)
            metric.set_statistic(zero_count, instance_count)
            metrics.append(metric)

            finite_column_values = column_values[np.isfinite(column_values)]
            if finite_column_values.shape[0] > 0:
                unique_count = np.unique(column_values).size
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='unique_values',
                    aggregation=Metric.AGGREGATION_SUM,
                    timestamp=timestamp)
                metric.set_statistic(unique_count, instance_count)
                metrics.append(metric)

                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='distribution',
                    timestamp=timestamp)
                metric.compute_histogram(finite_column_values.tolist())
                metrics.append(metric)
        else:
            string_column_values = [
                m for m in column_values.tolist() if isinstance(m, str)]
            if len(string_column_values) > 0:
                missing_count = column_values.shape[0] - \
                    len(string_column_values)
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='missing_values',
                    aggregation=Metric.AGGREGATION_SUM,
                    timestamp=timestamp)
                metric.set_statistic(missing_count, instance_count)
                metrics.append(metric)

                empty_count = string_column_values.count('')
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='empty_values',
                    aggregation=Metric.AGGREGATION_SUM,
                    timestamp=timestamp)
                metric.set_statistic(empty_count, instance_count)
                metrics.append(metric)

                unique_count = len(set(string_column_values))
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='unique_values',
                    aggregation=Metric.AGGREGATION_SUM,
                    timestamp=timestamp)
                metric.set_statistic(unique_count, instance_count)
                metrics.append(metric)

                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='distribution',
                    unit=Metric.UNIT_CATEGORY_HASH,
                    timestamp=timestamp)
                metric.compute_categorical_histogram(string_column_values)
                metrics.append(metric)

    return metrics


def _concat_prediction_data(prediction_data):
    if prediction_data is None:
        return None

    data2d_window = []
    for data, timestamp in prediction_data:
        data2d = _convert_to_2d(data)
        if data2d is None:
            return None
        data2d_window.append((
            data2d,
            np.full((data2d.shape[0],), timestamp)))

    if len(data2d_window) > 0:
        data2d_df = pd.concat(
            [data2d for data2d, _ in data2d_window], ignore_index=True)
        timestamp_arr = np.concatenate(
            [timestamp for _, timestamp in data2d_window])
        if not data2d_df.empty:
            return DataWindow(
                data=data2d_df,
                timestamp=timestamp_arr)

    return None


def _convert_to_2d(data):
    data2d = None

    # list or dict
    if isinstance(data, (list, dict)):
        if _is_1d(data):  # 1-d list or dict
            data2d = pd.DataFrame(data)
        elif _is_2d(data):  # 2-d list or dict
            data2d = pd.DataFrame(data)
        elif isinstance(data, dict):  # N-d dict
            logger.error('Only dict with max 2 dimensions is supported')
            return None
        # list with N-d ndarray
        elif len(data) > 0 and isinstance(data[0], np.ndarray):
            data = np.asarray(data)
            if data.ndim > 2:
                data = np.reshape(data, (-1, data.shape[-1]))
            data2d = pd.DataFrame(data)
        else:  # N-d list of lists and/or dicts
            data2d = pd.DataFrame(_filter_2d(data))

    # numpy ndarray
    elif isinstance(data, np.ndarray):
        if data.ndim > 2:
            data = np.reshape(data, (-1, data.shape[-1]))
        data2d = pd.DataFrame(data)

    # pandas DataFrame
    elif isinstance(data, pd.DataFrame):
        data2d = data.copy(deep=False)

    if data2d is None:
        logger.error(
            'Unsupported data format: {0}. Please use list, dict, numpy.ndarray, pandas.DataFrame'.format(type(data)))

    return data2d


def _is_scalar(data):
    return isinstance(data, (int, float, str))


def _is_1d(data):
    if isinstance(data, list):
        for elem in data:
            if not _is_scalar(elem):
                return False
        return True
    if isinstance(data, dict):
        for elem in data.values():
            if not _is_scalar(elem):
                return False
        return True
    elif isinstance(data, np.ndarray):
        return data.ndim == 1
    return False


def _is_2d(data):
    if isinstance(data, list):
        for elem in data:
            if not _is_1d(elem):
                return False
        return True
    elif isinstance(data, dict):
        for elem in data.values():
            if not _is_1d(elem):
                return False
        return True
    return False


def _filter_2d(data):
    filtered = []
    for instance in data:
        if isinstance(instance, list):
            filtered.append([v for v in instance if _is_scalar(v)])
        elif isinstance(instance, dict):
            filtered.append(
                {k: v for k, v in instance.items() if _is_scalar(v)})
        elif isinstance(instance, (int, float, str)):
            filtered.append([instance])
    return filtered


def _format_names(names):
    return [str(n)[:50] for n in names]


def _format_values(values):
    return [_format_value(value) for value in values]


def _format_value(value):
    if isinstance(value, str):
        return value[0:250]
    else:
        return str(value)
