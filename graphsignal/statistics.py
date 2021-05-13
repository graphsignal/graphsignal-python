import re
import unicodedata
import time
import logging
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.neighbors import LocalOutlierFactor

import graphsignal
from graphsignal.predictions import Prediction, DataWindow
from graphsignal.windows import Metric, Sample, SamplePart

logger = logging.getLogger('graphsignal')
_rand = np.random.RandomState(int(time.time()))

MAX_COLUMNS = 250
RANDOM_SAMPLE_SIZE = 10
OUTLIER_SAMPLE_SIZE = 10
USER_DEFINED_SAMPLE_SIZE = 10
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

    prediction_inputs = [(p.input_data, p.ensure_sample, p.timestamp)
                         for p in prediction_window if p.input_data is not None]
    input_window = _concat_prediction_data(prediction_inputs)

    prediction_outputs = [(p.output_data, p.ensure_sample, p.timestamp)
                          for p in prediction_window if p.output_data is not None]
    output_window = _concat_prediction_data(prediction_outputs)

    prediction_context = [(p.context_data, p.ensure_sample, p.timestamp)
                          for p in prediction_window if p.context_data is not None]
    context_window = _concat_prediction_data(prediction_context)

    if input_window is None and output_window is None:
        logger.warning('Provided empty data, nothing to compute')
        return metrics, samples

    # add timestamps to context
    if context_window is not None:
        if input_window is not None and context_window.size() == input_window.size():
            context_window.data['prediction_timestamp'] = input_window.timestamp
        elif output_window is not None and context_window.size() == output_window.size():
            context_window.data['prediction_timestamp'] = output_window.timestamp
    else:
        if input_window:
            context_window = DataWindow(data=pd.DataFrame(
                data={'prediction_timestamp': input_window.timestamp}))
        elif output_window:
            context_window = DataWindow(data=pd.DataFrame(
                data={'prediction_timestamp': output_window.timestamp}))

    # compute metrics
    if input_window is not None:
        metrics.extend(
            _compute_tabular_metrics(
                input_window,
                Metric.DATASET_INPUT,
                last_timestamp))

    if output_window is not None:
        metrics.extend(
            _compute_tabular_metrics(
                output_window,
                Metric.DATASET_OUTPUT,
                last_timestamp))

    # compute samples
    if graphsignal._get_config().log_instances:
        random_sample = _compute_random_sample(
            input_window, output_window, context_window, last_timestamp)
        if random_sample:
            samples.append(random_sample)

        input_outlier_sample, input_outlier_metric = _compute_input_outlier_sample(
            input_window, output_window, context_window, last_timestamp)
        if input_outlier_sample is not None:
            samples.append(input_outlier_sample)
        if input_outlier_metric is not None:
            metrics.append(input_outlier_metric)

        output_outlier_sample, output_outlier_metric = _compute_output_outlier_sample(
            input_window, output_window, context_window, last_timestamp)
        if output_outlier_sample is not None:
            samples.append(output_outlier_sample)
        if output_outlier_metric is not None:
            metrics.append(output_outlier_metric)

        user_defined_sample = _compute_user_defined_sample(
            input_window, output_window, context_window, last_timestamp)
        if user_defined_sample is not None:
            samples.append(user_defined_sample)

    logger.debug('Computing metrics and samples took %.3f sec',
                 time.time() - start_ts)

    return metrics, samples


def _compute_tabular_metrics(data_window, dataset, timestamp):
    metrics = []

    instance_count = data_window.data.shape[0]

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
        column_values = data_window.data[data_window.data.columns[column_index]].to_numpy()

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
        input_window, output_window, context_window, timestamp):
    sample = Sample(name='random_sample', timestamp=timestamp)
    if input_window is not None:
        sample_idx = _random_index(input_window.data)
        sample.set_size(sample_idx.shape[0])

        sample.add_part(
            dataset='input',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=input_window.data.iloc[sample_idx, :].to_numpy().tolist(),
                columns=_format_names(input_window.data.columns.values)))

        if output_window is not None and input_window.size() == output_window.size():
            sample.add_part(
                dataset='output',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=output_window.data.iloc[sample_idx, :].to_numpy(
                    ).tolist(),
                    columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == input_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                  :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))
    elif output_window is not None:
        sample_idx = _random_index(output_window.data)
        sample.set_size(sample_idx.shape[0])

        sample.add_part(
            dataset='output',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=output_window.data.iloc[sample_idx,
                                             :].to_numpy().tolist(),
                columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == output_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                  :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))

    return sample


def _random_index(data):
    sample_size = min(data.shape[0], RANDOM_SAMPLE_SIZE)
    return _rand.choice(data.shape[0], sample_size, replace=False)


def _compute_input_outlier_sample(
        input_window, output_window, context_window, timestamp):
    if input_window is None:
        return None, None

    metric = Metric(
        dataset=Metric.DATASET_INPUT,
        name='outlier_count',
        timestamp=timestamp)

    sample_idx, outlier_count = _outlier_index(input_window.data)
    if sample_idx is not None and sample_idx.shape[0] > 0:
        sample = Sample(name='input_outliers', timestamp=timestamp)
        sample.set_size(sample_idx.shape[0])

        sample.add_part(
            dataset='input',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=input_window.data.iloc[sample_idx, :].to_numpy(
                ).tolist(),
                columns=_format_names(input_window.data.columns.values)))

        if output_window is not None and output_window.size() == input_window.size():
            sample.add_part(
                dataset='output',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=output_window.data.iloc[sample_idx,
                                                    :].to_numpy().tolist(),
                    columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == input_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                    :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))

        metric.set_statistic(outlier_count, input_window.size())
        return sample, metric
    else:
        metric.set_statistic(0, input_window.size())
        return None, metric


def _compute_output_outlier_sample(
        input_window, output_window, context_window, timestamp):
    if output_window is None:
        return None, None

    metric = Metric(
        dataset=Metric.DATASET_OUTPUT,
        name='outlier_count',
        timestamp=timestamp)

    sample_idx, outlier_count = _outlier_index(output_window.data)
    if sample_idx is not None and sample_idx.shape[0] > 0:
        sample = Sample(name='output_outliers', timestamp=timestamp)
        sample.set_size(sample_idx.shape[0])

        if input_window is not None and input_window.size() == output_window.size():
            sample.add_part(
                dataset='input',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=input_window.data.iloc[sample_idx,
                                                :].to_numpy().tolist(),
                    columns=_format_names(input_window.data.columns.values)),
                insert_at=0)

        sample.add_part(
            dataset='output',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=output_window.data.iloc[sample_idx, :].to_numpy(
                ).tolist(),
                columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == output_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                    :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))

        metric.set_statistic(outlier_count, output_window.size())
        return sample, metric
    else:
        metric.set_statistic(0, input_window.size())
        return None, metric


def _outlier_index(data):
    if data.shape[0] < MIN_INSTANCES_FOR_OUTLIER_DETECTION:
        return None, 0

    numeric_data = data.select_dtypes(include=np.number).to_numpy()
    if numeric_data.shape[1] == 0:
        return None, 0

    finite_numeric_data = numeric_data[np.isfinite(
        numeric_data).all(axis=1)]
    if finite_numeric_data.shape[0] < MIN_INSTANCES_FOR_OUTLIER_DETECTION:
        return None, 0

    try:
        n_neighbors = min(finite_numeric_data.shape[0] - 1, 20)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        scores = lof.fit_predict(finite_numeric_data)
        factors = lof.negative_outlier_factor_

        outlier_count = np.count_nonzero(scores == -1)

        top_factor_idx = np.argsort(factors)[0:OUTLIER_SAMPLE_SIZE]
        top_scores = scores[top_factor_idx]

        return top_factor_idx[top_scores == -1], outlier_count
    except BaseException:
        logger.error('Error detecting outliers', exc_info=True)

    return None, 0


def _compute_user_defined_sample(
        input_window, output_window, context_window, timestamp):
    sample = Sample(name='user_defined_sample', timestamp=timestamp)
    if input_window is not None:
        sample_idx = _user_defined_index(
            input_window.data, input_window.ensure_sample)
        if sample_idx.shape[0] == 0:
            return None
        sample.set_size(sample_idx.shape[0])

        sample.add_part(
            dataset='input',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=input_window.data.iloc[sample_idx, :].to_numpy().tolist(),
                columns=_format_names(input_window.data.columns.values)))

        if output_window is not None and output_window.size() == input_window.size():
            sample.add_part(
                dataset='output',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=output_window.data.iloc[sample_idx, :].to_numpy(
                    ).tolist(),
                    columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == input_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                  :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))
    elif output_window is not None:
        sample_idx = _user_defined_index(
            output_window.data, output_window.ensure_sample)
        if sample_idx.shape[0] == 0:
            return None
        sample.set_size(sample_idx.shape[0])

        sample.add_part(
            dataset='output',
            format_=SamplePart.FORMAT_CSV,
            data=_create_csv(
                data=output_window.data.iloc[sample_idx,
                                             :].to_numpy().tolist(),
                columns=_format_names(output_window.data.columns.values)))

        if context_window is not None and context_window.size() == output_window.size():
            sample.add_part(
                dataset='context',
                format_=SamplePart.FORMAT_CSV,
                data=_create_csv(
                    data=context_window.data.iloc[sample_idx,
                                                  :].to_numpy().tolist(),
                    columns=_format_names(context_window.data.columns.values)))

    return sample


def _user_defined_index(data, ensure_sample):
    return data[ensure_sample][0:USER_DEFINED_SAMPLE_SIZE].index.values


def _create_csv(data, columns):
    rows = []
    rows.append(','.join(_format_values(columns)))
    for instance in data:
        rows.append(','.join(_format_values(instance)))
    return '\n'.join(rows)


def _concat_prediction_data(prediction_data):
    if prediction_data is None:
        return None

    data2d_window = []
    for data, ensure_sample, timestamp in prediction_data:
        data2d = _convert_to_2d(data)
        if data2d is None:
            return None
        data2d_window.append((
            data2d,
            np.full((data2d.shape[0],), ensure_sample),
            np.full((data2d.shape[0],), timestamp)))

    if len(data2d_window) > 0:
        data2d_df = pd.concat(
            [data2d for data2d, _, _ in data2d_window], ignore_index=True)
        ensure_sample_arr = np.concatenate(
            [ensure_sample for _, ensure_sample, _ in data2d_window])
        timestamp_arr = np.concatenate(
            [timestamp for _, _, timestamp in data2d_window])
        if not data2d_df.empty:
            return DataWindow(
                data=data2d_df,
                ensure_sample=ensure_sample_arr,
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
