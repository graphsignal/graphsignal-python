from typing import Union, Any, Optional, Dict
import logging
import sys
import time
import traceback

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.data import compute_counts, build_stats

logger = logging.getLogger('graphsignal')

SAMPLE_TRACES = {1, 10, 100, 1000}

class EndpointTrace:
    MAX_TAGS = 10
    MAX_PARAMS = 10
    MAX_DATA_OBJECTS = 10

    __slots__ = [
        '_trace_sampler',
        '_metric_store',
        '_mv_detector',
        '_agent',
        '_endpoint',
        '_tags',
        '_is_sampling',
        '_context',
        '_signal',
        '_is_stopped',
        '_start_counter',
        '_exc_info',
        '_params',
        '_data',
        '_has_missing_values'
    ]

    def __init__(self, endpoint, tags=None):
        if not endpoint:
            raise ValueError('endpoint is required')
        if not isinstance(endpoint, str):
            raise ValueError('endpoint must be string')
        if len(endpoint) > 50:
            raise ValueError('endpoint is too long (>50)')
        if tags is not None:
            if not isinstance(tags, dict):
                raise ValueError('tags must be dict')
            if len(tags) > EndpointTrace.MAX_TAGS:
                raise ValueError('too many tags (>{0})'.format(EndpointTrace.MAX_TAGS))

        self._endpoint = endpoint
        if tags is not None:
            self._tags = dict(tags)
        else:
            self._tags = None

        self._agent = graphsignal._agent
        self._trace_sampler = None
        self._metric_store = None
        self._mv_detector = None
        self._is_stopped = False
        self._is_sampling = False
        self._context = False
        self._signal = None
        self._exc_info = None
        self._params = None
        self._data = None
        self._has_missing_values = False

        try:
            self._start()
        except Exception as ex:
            if self._is_sampling:
                self._is_stopped = True
                self._trace_sampler.unlock()
            raise ex

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._exc_info = exc_info
        self.stop()
        return False

    def _init_sampling(self):
        self._is_sampling = True
        self._signal = self._agent.create_signal()
        self._context = {}

    def _start(self):
        if self._is_stopped:
            return

        self._trace_sampler = self._agent.trace_sampler(self._endpoint)
        self._metric_store = self._agent.metric_store(self._endpoint)
        self._mv_detector = self._agent.mv_detector()

        if self._trace_sampler.lock('samples', include_trace_idx=SAMPLE_TRACES):
            self._init_sampling()

            # emit start event
            try:
                self._agent.emit_trace_start(self._signal, self._context)
            except Exception as exc:
                logger.error('Error in trace start event handlers', exc_info=True)
                self._add_agent_exception(exc)

        self._start_counter = time.perf_counter()

    def _stop(self) -> None:
        stop_counter = time.perf_counter()
        duration_us = int((stop_counter - self._start_counter) * 1e6)
        end_us = _timestamp_us()

        if self._is_stopped:
            return
        self._is_stopped = True

        if self._is_sampling:
            # emit stop event
            try:
                self._agent.emit_trace_stop(self._signal, self._context)
            except Exception as exc:
                logger.error('Error in trace stop event handlers', exc_info=True)
                self._add_agent_exception(exc)

        # if exception, but the trace is not being recorded, try to start tracing
        if not self._is_sampling and self._exc_info and self._exc_info[0]:
            if self._trace_sampler.lock('exceptions'):
                self._init_sampling()

        # update time and counters
        self._metric_store.add_time(duration_us)
        self._metric_store.inc_call_count(1, end_us)

        # update data counters and check for missing values
        if self._data is not None:
            for data_name, data in self._data.items():
                data_counts = compute_counts(data)

                for count_name, count in data_counts.items():
                    self._metric_store.inc_data_counter(
                        data_name, count_name, count, end_us)

                if self._mv_detector.detect(data_name, data_counts):
                    self._has_missing_values = True

        # if missing values detected, but the trace is not being recorded, try to start tracing
        if not self._is_sampling and self._has_missing_values:
            if self._trace_sampler.lock('missing-values'):
                self._init_sampling()

        # update exception counter
        if self._exc_info and self._exc_info[0]:
            self._metric_store.inc_exception_count(1, end_us)

        # fill and upload signal
        if self._is_sampling:
            # emit read event
            try:
                self._agent.emit_trace_read(self._signal, self._context)
            except Exception as exc:
                logger.error('Error in trace read event handlers', exc_info=True)
                self._add_agent_exception(exc)

            # copy data to signal
            self._signal.endpoint_name = self._endpoint
            self._signal.start_us = end_us - duration_us
            self._signal.end_us = end_us
            if self._exc_info and self._exc_info[0]:
                self._signal.signal_type = signals_pb2.SignalType.EXCEPTION_SIGNAL
            elif self._has_missing_values:
                self._signal.signal_type = signals_pb2.SignalType.MISSING_VALUES_SIGNAL
            else:
                self._signal.signal_type = signals_pb2.SignalType.SAMPLE_SIGNAL
            self._signal.process_usage.start_ms = self._agent._process_start_ms

            # copy tags
            if self._tags is not None:
                for key, value in self._tags.items():
                    tag = self._signal.tags.add()
                    tag.key = key[:50]
                    tag.value = str(value)[:50]

            # copy metrics
            self._metric_store.finalize(end_us)
            if self._metric_store.latency_us:
                self._signal.trace_metrics.latency_us.CopyFrom(
                    self._metric_store.latency_us)
            if self._metric_store.call_count:
                self._signal.trace_metrics.call_count.CopyFrom(
                    self._metric_store.call_count)
            if self._metric_store.exception_count:
                self._signal.trace_metrics.exception_count.CopyFrom(
                    self._metric_store.exception_count)
            for counter in self._metric_store.data_counters.values():
                self._signal.data_metrics.append(counter)
            self._agent.reset_metric_store(self._endpoint)

            # copy trace measurements
            self._signal.trace_sample.trace_idx = self._trace_sampler.current_trace_idx()
            self._signal.trace_sample.latency_us = duration_us

            # copy exception
            if self._exc_info and self._exc_info[0]:
                exception = self._signal.exceptions.add()
                if self._exc_info[0] and hasattr(self._exc_info[0], '__name__'):
                    exception.exc_type = str(self._exc_info[0].__name__)
                if self._exc_info[1]:
                    exception.message = str(self._exc_info[1])
                if self._exc_info[2]:
                    frames = traceback.format_tb(self._exc_info[2])
                    if len(frames) > 0:
                        exception.stack_trace = ''.join(frames)

            # copy params
            if self._params is not None:
                for name, value in self._params.items():
                    param = self._signal.trace_sample.params.add()
                    param.name = name[:50]
                    param.value = str(value)[:50]

            # copy data stats
            if self._data is not None:
                for name, data in self._data.items():
                    try:
                        data_stats_proto = build_stats(data)
                        data_stats_proto.data_name = name
                        if data_stats_proto:
                            self._signal.data_stats.append(data_stats_proto)
                    except Exception as exc:
                        logger.error('Error computing data stats', exc_info=True)
                        self._add_agent_exception(exc)

            # queue signal for upload
            self._agent.uploader().upload_signal(self._signal)
            self._agent.tick()

    def stop(self) -> None:
        try:
            self._stop()
        finally:
            if self._is_sampling:
                self._is_stopped = True
                self._trace_sampler.unlock()

    def is_sampling(self):
        return self._is_sampling

    def set_tag(self, key: str, value: str) -> None:
        if not key:
            raise ValueError('set_tag: key must be provided')
        if value is None:
            raise ValueError('set_tag: value must be provided')

        if self._tags is None:
            self._tags = {}

        if len(self._tags) > EndpointTrace.MAX_TAGS:
            raise ValueError('set_tag: too many tags (>{0})'.format(EndpointTrace.MAX_TAGS))

        self._tags[key] = value

    def set_param(self, name: str, value: str) -> None:
        if not name:
            raise ValueError('set_param: name must be provided')
        if value is None:
            raise ValueError('set_param: value must be provided')

        if self._params is None:
            self._params = {}

        if len(self._params) > EndpointTrace.MAX_PARAMS:
            raise ValueError('set_param: too many params (>{0})'.format(EndpointTrace.MAX_PARAMS))

        self._params[name] = value

    def set_exception(self, exc: Optional[Exception] = None, exc_info: Optional[bool] = None) -> None:
        if exc is not None and not isinstance(exc, Exception):
            raise ValueError('set_exception: exc must be instance of Exception')

        if exc_info is not None and not isinstance(exc_info, bool):
            raise ValueError('set_exception: exc_info must be bool')

        if exc:
            self._exc_info = (exc.__class__, str(exc), exc.__traceback__)
        elif exc_info == True:
            self._exc_info = sys.exc_info()

    def set_data(self, name: str, obj: Any) -> None:
        if self._data is None:
            self._data = {}

        if len(self._data) > EndpointTrace.MAX_DATA_OBJECTS:
            raise ValueError('set_data: too many data objects (>{0})'.format(EndpointTrace.MAX_DATA_OBJECTS))

        if name and not isinstance(name, str):
            raise ValueError('set_data: name must be string')

        self._data[name] = obj

    def _add_agent_exception(self, exc):
        agent_error = self._signal.agent_errors.add()
        agent_error.message = str(exc)
        if exc.__traceback__:
            frames = traceback.format_tb(exc.__traceback__)
            if len(frames) > 0:
                agent_error.stack_trace = ''.join(frames)


def _timestamp_us():
    return int(time.time() * 1e6)
