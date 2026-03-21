import logging
import os

import sglang

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.utils import sanitize_str, sha1
from graphsignal.otel.otel_collector import OTELCollector
from graphsignal.otel.prometheus_adapter import PrometheusAdapter
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class SGLangRecorder(BaseRecorder):
    MAX_TRACE_SAMPLING_DECISIONS = 10000

    def __init__(self):
        self._prometheus_adapter = None
        self._otel_collector = None
        self._otel_endpoint = None
        self._library_version = None
        self._startup_options = {}
        self._trace_sampling_decisions = {}
        self._trace_sampling_order = []

    def setup(self):
        if not graphsignal._ticker.auto_instrument:
            return

        self._library_version = sglang.__version__
        ticker = graphsignal._ticker
        ticker.set_tag('inference.engine.name', 'sglang')
        ticker.set_tag('inference.engine.version', self._library_version)

        self._setup_otel_collector()
        self._patch_sglang_args()

        for category, function_path in PROFILED_PATHS:
            ticker.profile_function_path(function_path, category=category)

        registry = self._create_prometheus_registry()
        if not registry:
            logger.warning('SGLang Prometheus metrics are not available.')
            return

        def metric_name_map_func(name):
            if name.startswith('sglang:'):
                return f'sglang.{name[7:]}'
            if name.startswith('sglang_'):
                return f'sglang.{name[7:]}'
            return None

        self._prometheus_adapter = PrometheusAdapter(
            registry=registry,
            name_map_func=metric_name_map_func)
        self._prometheus_adapter.setup()

    def _setup_otel_collector(self):
        if self._otel_collector:
            return

        def otel_export_callback(otel_spans):
            for otel_span in otel_spans:
                self._convert_otel_span(otel_span)
            graphsignal._ticker.tick()

        self._otel_collector = OTELCollector(export_callback=otel_export_callback)
        self._otel_collector.setup()

        port = self._otel_collector.get_port()
        if port:
            self._otel_endpoint = f'localhost:{port}'
        else:
            endpoint = self._otel_collector.get_endpoint()
            if endpoint and endpoint.startswith('grpc://'):
                endpoint = endpoint[len('grpc://'):]
            self._otel_endpoint = endpoint

        if self._otel_endpoint:
            logger.debug('SGLang OTEL endpoint configured: %s', self._otel_endpoint)

    def _patch_sglang_args(self):
        try:
            import sglang.srt.server_args as server_args_mod
            import sglang.launch_server as launch_server_mod
        except Exception:
            logger.debug('SGLang server modules are not available for patching.', exc_info=True)
            return

        def _apply_server_args(server_args):
            if server_args is None:
                return
            if hasattr(server_args, 'enable_metrics') and not server_args.enable_metrics:
                server_args.enable_metrics = True
            if hasattr(server_args, 'enable_trace') and not server_args.enable_trace:
                server_args.enable_trace = True
            if self._otel_endpoint and hasattr(server_args, 'otlp_traces_endpoint'):
                server_args.otlp_traces_endpoint = self._otel_endpoint
            self._capture_startup_options(server_args)

        def after_prepare_server_args(args, kwargs, ret, exc, context):
            _apply_server_args(ret)

        patch_method(server_args_mod, 'prepare_server_args', after_func=after_prepare_server_args)

        def before_run_server(args, kwargs):
            server_args = args[0] if args else kwargs.get('server_args')
            _apply_server_args(server_args)

        patch_method(launch_server_mod, 'run_server', before_func=before_run_server)

    def _capture_startup_options(self, server_args):
        for option_name in STARTUP_PERF_OPTIONS:
            if not hasattr(server_args, option_name):
                continue
            option_value = getattr(server_args, option_name)
            if option_value is None:
                continue
            self._startup_options[option_name] = option_value

    def _create_prometheus_registry(self):
        try:
            from prometheus_client import CollectorRegistry, REGISTRY, multiprocess
        except Exception:
            return None

        if os.environ.get('PROMETHEUS_MULTIPROC_DIR'):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return registry

        return REGISTRY

    def on_tick(self):
        if self._prometheus_adapter:
            self._prometheus_adapter.collect()

    def _convert_otel_span(self, otel_span):
        if not otel_span.name:
            logger.error('Invalid Open Telemetry span: name=%s', otel_span.name)
            return

        if not (otel_span.start_time > 0 and otel_span.end_time > 0):
            logger.error('Invalid Open Telemetry span: start_time=%s, end_time=%s', otel_span.start_time, otel_span.end_time)
            return

        trace_id = _otel_id_str(getattr(otel_span, 'trace_id', None))
        span_id = _otel_id_str(getattr(otel_span, 'span_id', None))
        if not trace_id or not span_id:
            logger.debug(
                'Invalid Open Telemetry span IDs: trace_id=%s span_id=%s',
                getattr(otel_span, 'trace_id', None),
                getattr(otel_span, 'span_id', None),
            )
            return
        parent_span_id = _otel_id_str(getattr(otel_span, 'parent_span_id', None))
        if not self._should_sample_trace_span(otel_span.name, trace_id, parent_span_id):
            return

        span = signals_pb2.Span()
        span.span_id = sha1(span_id, size=12)
        span.trace_id = sha1(trace_id, size=12)
        if _has_parent_span_id(parent_span_id):
            span.parent_span_id = sha1(parent_span_id, size=12)
        span.start_ts = otel_span.start_time
        span.end_ts = otel_span.end_time
        span.name = f'sglang.{otel_span.name}'
        _add_counter(span, 'span.duration', span.end_ts - span.start_ts)

        if graphsignal._ticker.tags:
            for tag_key, tag_value in graphsignal._ticker.tags.items():
                _add_tag(span, tag_key, tag_value)

        attributes = otel_span.attributes if otel_span.attributes else {}
        _add_tag(span, 'sampling.reason', 'sglang.otel')

        if 'gen_ai.request.id' in attributes:
            _add_tag(span, 'sglang.request.id', attributes['gen_ai.request.id'])
        if 'gen_ai.response.model' in attributes:
            _add_tag(span, 'sglang.response.model', attributes['gen_ai.response.model'])
            _add_attribute(span, 'sglang.response.model', attributes['gen_ai.response.model'])
        if 'gen_ai.request.temperature' in attributes:
            _add_attribute(span, 'sglang.request.temperature', attributes['gen_ai.request.temperature'])
        if 'gen_ai.request.top_p' in attributes:
            _add_attribute(span, 'sglang.request.top_p', attributes['gen_ai.request.top_p'])
        if 'gen_ai.request.top_k' in attributes:
            _add_attribute(span, 'sglang.request.top_k', attributes['gen_ai.request.top_k'])
        if 'gen_ai.request.max_tokens' in attributes:
            _add_attribute(span, 'sglang.request.max_tokens', attributes['gen_ai.request.max_tokens'])
        if 'gen_ai.request.n' in attributes:
            _add_attribute(span, 'sglang.request.n', attributes['gen_ai.request.n'])

        if 'gen_ai.usage.prompt_tokens' in attributes:
            _add_counter(span, 'sglang.usage.prompt_tokens', attributes['gen_ai.usage.prompt_tokens'])
        if 'gen_ai.usage.cached_tokens' in attributes:
            _add_counter(span, 'sglang.usage.cached_tokens', attributes['gen_ai.usage.cached_tokens'])
        if 'gen_ai.usage.completion_tokens' in attributes:
            _add_counter(span, 'sglang.usage.completion_tokens', attributes['gen_ai.usage.completion_tokens'])

        if 'gen_ai.latency.time_in_queue' in attributes:
            _add_counter(span, 'sglang.latency.time_in_queue', attributes['gen_ai.latency.time_in_queue'], sec_to_ns=True)
        if 'gen_ai.latency.time_to_first_token' in attributes:
            _add_counter(span, 'sglang.latency.time_to_first_token', attributes['gen_ai.latency.time_to_first_token'], sec_to_ns=True)
        e2e_latency = attributes.get('gen_ai.latency.e2e')
        if e2e_latency is None:
            e2e_latency = attributes.get('gen_ai.latency.time_in_request')
        if e2e_latency is not None:
            _add_counter(span, 'sglang.latency.e2e', e2e_latency, sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_prefill' in attributes:
            _add_counter(span, 'sglang.latency.time_in_model_prefill', attributes['gen_ai.latency.time_in_model_prefill'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_decode' in attributes:
            _add_counter(span, 'sglang.latency.time_in_model_decode', attributes['gen_ai.latency.time_in_model_decode'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_inference' in attributes:
            _add_counter(span, 'sglang.latency.time_in_model_inference', attributes['gen_ai.latency.time_in_model_inference'], sec_to_ns=True)

        for option_name, option_value in self._startup_options.items():
            _add_attribute(span, f'sglang.startup.{option_name}', option_value)

        graphsignal._ticker.signal_uploader().upload_span(span)

    def _should_sample_trace_span(self, span_name, trace_id, parent_span_id):
        if not _has_parent_span_id(parent_span_id):
            sampled = graphsignal._ticker.should_trace((f'sglang.{span_name}', 'random'))
            self._remember_trace_sampling_decision(trace_id, sampled)
            return sampled
        if not trace_id:
            return False
        return self._trace_sampling_decisions.get(trace_id, False)

    def _remember_trace_sampling_decision(self, trace_id, sampled):
        if not trace_id:
            return
        if trace_id not in self._trace_sampling_decisions:
            self._trace_sampling_order.append(trace_id)
        self._trace_sampling_decisions[trace_id] = sampled
        while len(self._trace_sampling_order) > self.MAX_TRACE_SAMPLING_DECISIONS:
            old_trace_id = self._trace_sampling_order.pop(0)
            self._trace_sampling_decisions.pop(old_trace_id, None)

    def shutdown(self):
        if self._otel_collector:
            self._otel_collector.shutdown()
            self._otel_collector = None


def _add_tag(span, key, value):
    tag = span.tags.add()
    tag.key = sanitize_str(key, max_len=50)
    tag.value = sanitize_str(value, max_len=250)


def _otel_id_str(value, max_len=64):
    if value is None:
        return None
    normalized = sanitize_str(value, max_len=max_len).strip().lower()
    return normalized or None


def _has_parent_span_id(parent_span_id):
    return bool(parent_span_id) and parent_span_id.strip('0') != ''


def _add_attribute(span, name, value):
    attr = span.attributes.add()
    attr.name = sanitize_str(name, max_len=50)
    attr.value = sanitize_str(value, max_len=2500)


def _add_counter(span, name, value, sec_to_ns=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if sec_to_ns:
        value = int(value * 1e9)
    counter = span.counters.add()
    counter.name = name
    counter.value = float(value)


STARTUP_PERF_OPTIONS = (
    'tp_size',
    'pp_size',
    'dp_size',
    'moe_dp_size',
    'context_length',
    'dtype',
    'quantization',
    'kv_cache_dtype',
    'mem_fraction_static',
    'max_running_requests',
    'max_total_tokens',
    'chunked_prefill_size',
    'schedule_policy',
    'stream_interval',
    'attention_backend',
    'decode_attention_backend',
    'prefill_attention_backend',
    'sampling_backend',
    'cuda_graph_max_bs',
    'disable_cuda_graph',
    'disable_cuda_graph_padding',
    'enable_mixed_chunk',
    'disable_piecewise_cuda_graph',
    'piecewise_cuda_graph_max_tokens',
)


PROFILED_PATHS = [
    ('sglang.e2e', 'sglang.srt.entrypoints.http_server.generate_request'),
    ('sglang.e2e', 'sglang.srt.entrypoints.http_server.openai_v1_chat_completions'),
    ('sglang.e2e', 'sglang.srt.entrypoints.http_server.openai_v1_completions'),
    ('sglang.e2e', 'sglang.srt.entrypoints.openai.serving_base.OpenAIServingBase.handle_request'),
    ('sglang.tokenizer', 'sglang.srt.managers.tokenizer_manager.TokenizerManager.generate_request'),
    ('sglang.tokenizer', 'sglang.srt.managers.tokenizer_manager.TokenizerManager._tokenize_one_request'),
    ('sglang.tokenizer', 'sglang.srt.managers.tokenizer_manager.TokenizerManager._send_one_request'),
    ('sglang.tokenizer', 'sglang.srt.managers.tokenizer_manager.TokenizerManager._handle_batch_request'),
    ('sglang.scheduler', 'sglang.srt.managers.scheduler.Scheduler.recv_requests'),
    ('sglang.scheduler', 'sglang.srt.managers.scheduler.Scheduler.process_input_requests'),
    ('sglang.scheduler', 'sglang.srt.managers.scheduler.Scheduler.get_next_batch_to_run'),
    ('sglang.scheduler', 'sglang.srt.managers.scheduler.Scheduler.run_batch'),
    ('sglang.scheduler', 'sglang.srt.managers.scheduler.Scheduler.process_batch_result'),
]
