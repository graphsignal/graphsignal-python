import re
import unicodedata
import time
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.neighbors import LocalOutlierFactor

import graphsignal
from graphsignal.windows import Metric, Sample, SamplePart

logger = logging.getLogger('graphsignal')
_rand = np.random.RandomState(int(time.time()))

MAX_COLUMNS = 250
RANDOM_SAMPLE_SIZE = 10
OUTLIER_SAMPLE_SIZE = 10
MIN_INSTANCES_FOR_OUTLIER_DETECTION = 30


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
    samples = []

    if len(prediction_window) == 0:
        return metrics, samples

    last_timestamp = max([p.timestamp for p in prediction_window if p])

    # convert data to 2d
    input_data_window = [(p.input_data, p.timestamp)
                        for p in prediction_window if p.input_data is not None]
    input_data2d, input_timestamps = _convert_window_to_2d(input_data_window)

    output_data_window = [(p.output_data, p.timestamp)
                         for p in prediction_window if p.output_data is not None]
    output_data2d, output_timestamps = _convert_window_to_2d(output_data_window)

    context_data_window = [(p.context_data, p.timestamp)
                          for p in prediction_window if p.context_data is not None]
    context_data2d, _ = _convert_window_to_2d(context_data_window)

    if input_data2d is None and output_data2d is None:
        logger.warning('Provided empty data, nothing to compute')
        return metrics, samples

    # add timestamp to context
    timestamps = input_timestamps if input_timestamps is not None else output_timestamps
    if timestamps is not None:
        if context_data2d is None:
            context_data2d = pd.DataFrame(
                data={'prediction_timestamp': timestamps})
        elif timestamps.shape[0] == context_data2d.shape[0]:
            context_data2d['prediction_timestamp'] = timestamps

    # compute metrics
    if input_data2d is not None:
        metrics.extend(
            _compute_tabular_metrics(
                input_data2d,
                Metric.DATASET_INPUT,
                last_timestamp))

    if output_data2d is not None:
        metrics.extend(
            _compute_tabular_metrics(
                output_data2d,
                Metric.DATASET_OUTPUT,
                last_timestamp))

    # compute samples
    if graphsignal._get_config().log_instances:
        random_sample = _compute_random_sample(
            input_data2d, output_data2d, context_data2d, last_timestamp)
        if random_sample:
            samples.append(random_sample)
        outlier_samples = _compute_outlier_samples(
            input_data2d, output_data2d, context_data2d, last_timestamp)
        if outlier_samples is not None:
            samples.extend(outlier_samples)

    logger.debug('Computing metrics and samples took %.3f sec',
                 time.time() - start_ts)

    return metrics, samples


def _compute_tabular_metrics(data2d, dataset, timestamp):
    metrics = []

    instance_count = data2d.shape[0]

    metric = Metric(
        dataset=dataset,
        name='instance_count',
        timestamp=timestamp)
    metric.set_statistic(instance_count, instance_count)
    metrics.append(metric)

    metric = Metric(
        dataset=dataset,
        name='dimension_count',
        timestamp=timestamp)
    metric.set_statistic(len(data2d.columns), instance_count)
    metrics.append(metric)

    types = data2d.dtypes.tolist()
    columns = _format_names(data2d.columns.values.tolist())

    for column_index in range(min(data2d.shape[1], MAX_COLUMNS)):
        column_name = columns[column_index]
        if column_name.lower() in ('timestamp', 'time', 'datetime', 'date'):
            continue
        column_type = types[column_index]
        if np.issubdtype(column_type, np.datetime64):
            continue
        column_values = data2d[data2d.columns[column_index]].to_numpy()

        if np.issubdtype(column_type, np.number):
            missing_count = np.count_nonzero(np.isnan(column_values))
            missing_count += np.count_nonzero(np.isinf(column_values))
            metric = Metric(
                dataset=dataset,
                dimension=column_name,
                name='missing_values',
                timestamp=timestamp)
            metric.set_statistic(
                missing_count / instance_count * 100,
                instance_count,
                Metric.UNIT_PERCENT)
            metrics.append(metric)

            zero_count = np.count_nonzero(column_values == 0)
            metric = Metric(
                dataset=dataset,
                dimension=column_name,
                name='zero_values',
                timestamp=timestamp)
            metric.set_statistic(
                zero_count / instance_count * 100,
                instance_count,
                Metric.UNIT_PERCENT)
            metrics.append(metric)

            finite_column_values = column_values[np.isfinite(column_values)]
            if finite_column_values.shape[0] > 0:
                unique_count = np.unique(column_values).size
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='unique_values',
                    timestamp=timestamp)
                metric.set_statistic(unique_count, instance_count)
                metrics.append(metric)

                mean_value = np.mean(finite_column_values)
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='mean',
                    timestamp=timestamp)
                metric.set_statistic(mean_value, instance_count)
                metrics.append(metric)

                std_value = np.std(
                    finite_column_values,
                    ddof=min(1, finite_column_values.shape[0] - 1))
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='standard_deviation',
                    timestamp=timestamp)
                metric.set_statistic(std_value, instance_count)
                metrics.append(metric)

                skewness_value = skew(finite_column_values, bias=False)
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='skewness',
                    timestamp=timestamp)
                metric.set_statistic(skewness_value, instance_count)
                metrics.append(metric)

                kurtosis_value = kurtosis(finite_column_values, bias=False)
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='kurtosis',
                    timestamp=timestamp)
                metric.set_statistic(kurtosis_value, instance_count)
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
                    timestamp=timestamp)
                metric.set_statistic(
                    missing_count / instance_count * 100,
                    instance_count,
                    Metric.UNIT_PERCENT)
                metrics.append(metric)

                empty_count = string_column_values.count('')
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='empty_values',
                    timestamp=timestamp)
                metric.set_statistic(
                    empty_count / instance_count * 100,
                    instance_count,
                    Metric.UNIT_PERCENT)
                metrics.append(metric)

                unique_count = len(set(string_column_values))
                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='unique_values',
                    timestamp=timestamp)
                metric.set_statistic(unique_count, instance_count)
                metrics.append(metric)

                metric = Metric(
                    dataset=dataset,
                    dimension=column_name,
                    name='distribution',
                    timestamp=timestamp)
                metric.compute_categorical_histogram(string_column_values)
                metrics.append(metric)

    return metrics


def _compute_random_sample(
        input_data2d, output_data2d, context_data2d, timestamp):
    sample = Sample(name='random_sample', timestamp=timestamp)
    if input_data2d is not None:
        sample_idx = _random_index(input_data2d)
        sample.set_size(len(sample_idx))
        sample.add_part(
            dataset='input',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=input_data2d.iloc[sample_idx, :].to_numpy().tolist(),
                columns=_format_names(input_data2d.columns.values)))
        if output_data2d is not None and input_data2d.shape[0] == output_data2d.shape[0]:
            sample.add_part(
                dataset='output',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=output_data2d.iloc[sample_idx, :].to_numpy().tolist(),
                    columns=_format_names(output_data2d.columns.values)))
        if context_data2d is not None and input_data2d.shape[0] == context_data2d.shape[0]:
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_data2d.iloc[sample_idx,
                                             :].to_numpy().tolist(),
                    columns=_format_names(context_data2d.columns.values)))
    elif output_data2d is not None:
        sample_idx = _random_index(output_data2d)
        sample.set_size(len(sample_idx))
        sample.add_part(
            dataset='output',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=output_data2d.iloc[sample_idx, :].to_numpy().tolist(),
                columns=_format_names(output_data2d.columns.values)))
        if context_data2d is not None and output_data2d.shape[0] == context_data2d.shape[0]:
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_data2d.iloc[sample_idx,
                                             :].to_numpy().tolist(),
                    columns=_format_names(context_data2d.columns.values)))

    return sample


def _random_index(data2d):
    sample_size = min(data2d.shape[0], RANDOM_SAMPLE_SIZE)
    return _rand.choice(data2d.shape[0], sample_size, replace=False)


def _compute_outlier_samples(
        input_data2d, output_data2d, context_data2d, timestamp):
    samples = []

    if input_data2d is not None:
        sample = Sample(name='input_outliers', timestamp=timestamp)
        sample_idx = _outlier_index(input_data2d)
        if sample_idx is not None and sample_idx.shape[0] > 0:
            sample.set_size(sample_idx.shape[0])
            sample.add_part(
                dataset='input',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=input_data2d.iloc[sample_idx, :].to_numpy().tolist(),
                    columns=_format_names(input_data2d.columns.values)))
            if output_data2d is not None and input_data2d.shape[0] == output_data2d.shape[0]:
                sample.add_part(
                    dataset='output',
                    format_=SamplePart.FORMAT_CSV,
                    data=_create_csv(
                        data=output_data2d.iloc[sample_idx,
                                                :].to_numpy().tolist(),
                        columns=_format_names(output_data2d.columns.values)))
            if context_data2d is not None and input_data2d.shape[0] == context_data2d.shape[0]:
                sample.add_part(
                    dataset='context',
                    format_=SamplePart.FORMAT_CSV,
                    data=_create_csv(
                        data=context_data2d.iloc[sample_idx,
                                                 :].to_numpy().tolist(),
                        columns=_format_names(context_data2d.columns.values)))
            samples.append(sample)

    if output_data2d is not None:
        sample = Sample(name='output_outliers', timestamp=timestamp)
        sample_idx = _outlier_index(output_data2d)
        if sample_idx is not None and sample_idx.shape[0] > 0:
            sample.set_size(sample_idx.shape[0])
            sample.add_part(
                dataset='output',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=output_data2d.iloc[sample_idx, :].to_numpy().tolist(),
                    columns=_format_names(output_data2d.columns.values)))
            if input_data2d is not None and output_data2d.shape[0] == input_data2d.shape[0]:
                sample.add_part(
                    dataset='input',
                    format_=SamplePart.FORMAT_CSV,
                    data=_create_csv(
                        data=input_data2d.iloc[sample_idx,
                                               :].to_numpy().tolist(),
                        columns=_format_names(input_data2d.columns.values)),
                    insert_at=0)
            if context_data2d is not None and output_data2d.shape[0] == context_data2d.shape[0]:
                sample.add_part(
                    dataset='context',
                    format_=SamplePart.FORMAT_CSV,
                    data=_create_csv(
                        data=context_data2d.iloc[sample_idx,
                                                 :].to_numpy().tolist(),
                        columns=_format_names(context_data2d.columns.values)))
            samples.append(sample)

    return samples


def _outlier_index(data2d):
    if data2d.shape[0] < MIN_INSTANCES_FOR_OUTLIER_DETECTION:
        return None

    numeric_data2d = data2d.select_dtypes(include=np.number).to_numpy()
    if numeric_data2d.shape[1] == 0:
        return None

    finite_numeric_data2d = numeric_data2d[np.isfinite(
        numeric_data2d).all(axis=1)]
    if finite_numeric_data2d.shape[0] < MIN_INSTANCES_FOR_OUTLIER_DETECTION:
        return None

    try:
        n_neighbors = min(finite_numeric_data2d.shape[0] - 1, 20)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        scores = lof.fit_predict(finite_numeric_data2d)
        factors = lof.negative_outlier_factor_

        top_factor_idx = np.argsort(factors)[0:OUTLIER_SAMPLE_SIZE]
        top_scores = scores[top_factor_idx]
        return top_factor_idx[top_scores == -1]
    except BaseException:
        logger.error('Error detecting outliers', exc_info=True)

    return None


def _create_csv(data, columns):
    rows = []
    rows.append(','.join(_format_values(columns)))
    for instance in data:
        rows.append(','.join(_format_values(instance)))
    return '\n'.join(rows)


def _convert_window_to_2d(data_window):
    if data_window is None:
        return None

    data2d_window = []
    for data, timestamp in data_window:
        data2d = _convert_to_2d(data)
        if data2d is None:
            return None
        data2d_window.append((data2d, np.full((data2d.shape[0],), timestamp)))

    if len(data2d_window) > 0:
        data2d_df = pd.concat(
            [data2d for data2d, _ in data2d_window], ignore_index=True)
        timestamps_arr = np.concatenate(
            [timestamps for _, timestamps in data2d_window])
        if not data2d_df.empty:
            return data2d_df, timestamps_arr

    return None, None


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
