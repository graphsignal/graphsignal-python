import re
import unicodedata
import time
import logging
import hashlib
from functools import lru_cache
import numpy as np
import pandas as pd

import graphsignal
from graphsignal.windows import PredictionRecord, GroundTruthRecord, get_data_stream, get_metric_updater, canonical_string
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


def update_data_metrics(metric_updaters, window_proto, prediction_records):
    if len(prediction_records) == 0:
        return

    start_ts = time.time()
    features_buffer = [p.features
                       for p in prediction_records if p.features is not None]
    feature_names = prediction_records[-1].feature_names
    feature_names, features_arr = _convert_to_numpy(
        features_buffer, columns=feature_names)

    predictions_buffer = [p.predictions
                          for p in prediction_records if p.predictions is not None]
    prediction_outputs, predictions_arr = _convert_to_numpy(
        predictions_buffer)

    if features_arr is None and predictions_arr is None:
        return

    if features_arr is not None:
        data_stream = get_data_stream(
            window_proto,
            metrics_pb2.DataStream.DataSource.FEATURES)
        _update_tabular_metrics(
            metric_updaters,
            data_stream,
            'feature',
            feature_names,
            features_arr)

    if predictions_arr is not None:
        data_stream = get_data_stream(
            window_proto,
            metrics_pb2.DataStream.DataSource.PREDICTIONS)
        _update_tabular_metrics(
            metric_updaters,
            data_stream,
            'output',
            prediction_outputs,
            predictions_arr)

    logger.debug('Computing data metrics took %.3f sec',
                 time.time() - start_ts)


def _update_tabular_metrics(
        metric_updaters, data_stream_proto, column_label, columns, data):
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

        dimensions = {}
        dimensions[column_label] = str(column_name)[:50]

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
            missing_column_values = [
                m for m in column_values.tolist() if m is None]
            missing_count = len(missing_column_values)
            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'missing_values', dimensions)
            metric_updater.update_ratio(missing_count, value_count)

            string_column_values = [
                m for m in column_values.tolist() if isinstance(m, str)]
            if len(string_column_values) > 0:
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

            bool_column_values = [
                str(m) for m in column_values.tolist() if isinstance(m, bool)]
            if len(bool_column_values) > 0:
                bool_count = len(bool_column_values)
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'boolean_values', dimensions)
                metric_updater.update_ratio(bool_count, value_count)

                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'distribution', dimensions)
                metric_updater.update_distribution(
                    _truncate_strings(bool_column_values))


def _convert_to_numpy(data_buffer, columns=None):
    if len(data_buffer) == 0:
        return None, None

    data = None
    first_item = data_buffer[0]

    # dict of list
    if isinstance(first_item, dict):
        keys = list(first_item.keys())
        data_index = {key: [] for key in keys}
        for item in data_buffer:
            for key in keys:
                data_index[key].append(item[key])
        columns = [str(key) for key in keys]
        data = []
        for arrays in data_index.values():
            flat_array = np.asarray(
                [elem for array in arrays for elem in array])
            if flat_array is None:
                data = None
                break
            data.append(flat_array)

    # list of list
    elif isinstance(first_item, list):
        if isinstance(first_item[0], list):
            if columns is None:
                columns = [str(column)
                           for column in range(len(first_item[0]))]
            data_arr = np.concatenate(data_buffer)
            data = [
                col_arr.ravel() for col_arr in np.hsplit(
                    data_arr, data_arr.shape[1])]

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
            'Unsupported data format: please use one or two-dimensional list, dict, numpy.ndarray or pandas.DataFrame')

    return columns, data


def _truncate_strings(values, max_size=18, front_size=10, tail_size=5):
    truncated_values = []
    for value in values:
        if len(value) > max_size:
            truncated_values.append(
                value[:front_size] + '...' + value[-tail_size:])
        else:
            truncated_values.append(value)
    return truncated_values


def update_performance_metrics(
        metric_updaters, window_proto, ground_truth_records):

    if len(ground_truth_records) == 0:
        return

    for ground_truth in ground_truth_records:
        is_binary = isinstance(ground_truth.label, bool)
        is_categorical = isinstance(ground_truth.label, (str, list))
        is_numeric = isinstance(ground_truth.label, (int, float))

        if is_binary:
            data_stream_proto = get_data_stream(
                window_proto,
                metrics_pb2.DataStream.DataSource.GROUND_TRUTH_BINARY)
        elif is_categorical:
            data_stream_proto = get_data_stream(
                window_proto,
                metrics_pb2.DataStream.DataSource.GROUND_TRUTH_CATEGORICAL)
        elif is_numeric:
            data_stream_proto = get_data_stream(
                window_proto,
                metrics_pb2.DataStream.DataSource.GROUND_TRUTH_NUMERIC)
        else:
            continue

        # accuracy for binary and categorical
        if is_binary or is_categorical:
            label_match = (ground_truth.label ==
                           ground_truth.prediction)

            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'accuracy')
            if label_match:
                metric_updater.update_ratio(1, 1)
            else:
                metric_updater.update_ratio(0, 1)

            if ground_truth.segments is not None:
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'segment_totals')
                metric_updater.update_distribution(ground_truth.segments)
                if label_match:
                    metric_updater = get_metric_updater(
                        metric_updaters, data_stream_proto, 'segment_matches')
                    metric_updater.update_distribution(ground_truth.segments)

        # confusion matrix for binary
        if is_binary:
            # true positive
            if ground_truth.label and ground_truth.prediction:
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'binary_true_positives')
                metric_updater.update_counter(1)

            # true negative
            elif not ground_truth.label and not ground_truth.prediction:
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'binary_true_negatives')
                metric_updater.update_counter(1)

            # false positive
            elif not ground_truth.label and ground_truth.prediction:
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'binary_false_positives')
                metric_updater.update_counter(1)

            # false negative
            elif ground_truth.label and not ground_truth.prediction:
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'binary_false_negatives')
                metric_updater.update_counter(1)

        # confusion matrix for categorical
        elif is_categorical:
            label_hash = _sha1(
                canonical_string(
                    ground_truth.label), size=8)
            # totals
            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'total')
            metric_updater.update_counter(1)

            if ground_truth.label == ground_truth.prediction:
                # true positives
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'class_true_positives')
                metric_updater.update_distribution([label_hash])
            else:
                prediction_hash = _sha1(
                    canonical_string(ground_truth.prediction), size=8)

                # false positives for ground_truth.prediction
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'class_false_positives')
                metric_updater.update_distribution([prediction_hash])

                # false negatives for ground_truth.label
                metric_updater = get_metric_updater(
                    metric_updaters, data_stream_proto, 'class_false_negatives')
                metric_updater.update_distribution([label_hash])

        # mse and mae for regression
        elif is_numeric:
            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'mse_n')
            metric_updater.update_counter(1)

            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'mse_sum')
            metric_updater.update_counter(
                (ground_truth.label - ground_truth.prediction) ** 2)

            metric_updater = get_metric_updater(
                metric_updaters, data_stream_proto, 'mae_sum')
            metric_updater.update_counter(
                abs(ground_truth.label - ground_truth.prediction))


@lru_cache(maxsize=250)
def _sha1(text, size=-1):
    sha1_hash = hashlib.sha1()
    sha1_hash.update(text.encode('utf-8'))
    return sha1_hash.hexdigest()[0:size]
