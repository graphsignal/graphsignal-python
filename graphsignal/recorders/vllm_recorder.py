import logging

import vllm

import graphsignal
from graphsignal.proto import signals_pb2
from graphsignal.utils import sanitize_str, sha1
from graphsignal.otel.otel_collector import OTELCollector
from graphsignal.otel.prometheus_adapter import PrometheusAdapter
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import patch_method

logger = logging.getLogger('graphsignal')


class VLLMRecorder(BaseRecorder):
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

        self._library_version = vllm.__version__
        ticker = graphsignal._ticker
        ticker.set_tag('inference.engine.name', 'vllm')
        ticker.set_tag('inference.engine.version', self._library_version)

        self._setup_otel_collector()
        self._patch_vllm_args()

        for category, function_path in PROFILED_PATHS:
            ticker.profile_function_path(function_path, category=category)

        prometheus_registry = None
        try:
            from vllm.v1.metrics.prometheus import get_prometheus_registry
            prometheus_registry = get_prometheus_registry()
        except ImportError:
            logger.warning('vLLM Prometheus metrics are not available.')
            return

        def metric_name_map_func(name):
            if name.startswith('vllm:'):
                return f'vllm.{name[5:]}'
            return None

        self._prometheus_adapter = PrometheusAdapter(
            registry=prometheus_registry,
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
            logger.debug('vLLM OTEL endpoint configured: %s', self._otel_endpoint)

    def _patch_vllm_args(self):
        def _apply_engine_args(engine_args):
            if engine_args is None:
                return
            if hasattr(engine_args, 'disable_log_stats'):
                engine_args.disable_log_stats = False
            if self._otel_endpoint and hasattr(engine_args, 'otlp_traces_endpoint'):
                engine_args.otlp_traces_endpoint = self._otel_endpoint
            self._capture_startup_options(engine_args)

        def _apply_vllm_config(vllm_config):
            if vllm_config is None:
                return
            if self._otel_endpoint and hasattr(vllm_config, 'observability_config'):
                obs = vllm_config.observability_config
                if hasattr(obs, 'otlp_traces_endpoint'):
                    obs.otlp_traces_endpoint = self._otel_endpoint
            self._capture_startup_options_from_config(vllm_config)

        try:
            from vllm.v1.engine.async_llm import AsyncLLM

            def before_from_engine_args(args, kwargs):
                engine_args = args[0] if args else kwargs.get('engine_args')
                _apply_engine_args(engine_args)

            def before_from_vllm_config(args, kwargs):
                vllm_config = args[0] if args else kwargs.get('vllm_config')
                _apply_vllm_config(vllm_config)

            patch_method(AsyncLLM, 'from_engine_args', before_func=before_from_engine_args)
            patch_method(AsyncLLM, 'from_vllm_config', before_func=before_from_vllm_config)
        except Exception:
            logger.debug('vLLM AsyncLLM not available for patching.', exc_info=True)

        try:
            def before_llm_init(args, kwargs):
                kwargs['disable_log_stats'] = False
                if self._otel_endpoint:
                    kwargs['otlp_traces_endpoint'] = self._otel_endpoint

            patch_method(vllm.LLM, '__init__', before_func=before_llm_init)
        except Exception:
            logger.debug('vLLM LLM not available for patching.', exc_info=True)

    def _capture_startup_options(self, engine_args):
        for option_name in STARTUP_PERF_OPTIONS:
            if not hasattr(engine_args, option_name):
                continue
            option_value = getattr(engine_args, option_name)
            if option_value is None:
                continue
            self._startup_options[option_name] = option_value

    def _capture_startup_options_from_config(self, vllm_config):
        if vllm_config is None:
            return
        for config_path, option_name in VLLM_CONFIG_OPTIONS:
            try:
                obj = vllm_config
                for part in config_path.split('.'):
                    obj = getattr(obj, part)
                if obj is not None:
                    self._startup_options[option_name] = obj
            except (AttributeError, TypeError):
                continue

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
        span.name = f'vllm.{otel_span.name}'
        _add_counter(span, 'span.duration', span.end_ts - span.start_ts)

        if graphsignal._ticker.tags:
            for tag_key, tag_value in graphsignal._ticker.tags.items():
                _add_tag(span, tag_key, tag_value)

        attributes = otel_span.attributes if otel_span.attributes else {}
        _add_tag(span, 'sampling.reason', 'vllm.otel')

        if 'gen_ai.request.id' in attributes:
            _add_tag(span, 'vllm.request.id', attributes['gen_ai.request.id'])
        if 'gen_ai.response.model' in attributes:
            _add_tag(span, 'vllm.response.model', attributes['gen_ai.response.model'])
            _add_attribute(span, 'vllm.response.model', attributes['gen_ai.response.model'])
        if 'gen_ai.request.temperature' in attributes:
            _add_attribute(span, 'vllm.request.temperature', attributes['gen_ai.request.temperature'])
        if 'gen_ai.request.top_p' in attributes:
            _add_attribute(span, 'vllm.request.top_p', attributes['gen_ai.request.top_p'])
        if 'gen_ai.request.max_tokens' in attributes:
            _add_attribute(span, 'vllm.request.max_tokens', attributes['gen_ai.request.max_tokens'])
        if 'gen_ai.request.n' in attributes:
            _add_attribute(span, 'vllm.request.n', attributes['gen_ai.request.n'])

        if 'gen_ai.usage.num_sequences' in attributes:
            _add_counter(span, 'vllm.usage.num_sequences', attributes['gen_ai.usage.num_sequences'])
        if 'gen_ai.usage.prompt_tokens' in attributes:
            _add_counter(span, 'vllm.usage.prompt_tokens', attributes['gen_ai.usage.prompt_tokens'])
        if 'gen_ai.usage.completion_tokens' in attributes:
            _add_counter(span, 'vllm.usage.completion_tokens', attributes['gen_ai.usage.completion_tokens'])

        if 'gen_ai.latency.time_in_queue' in attributes:
            _add_counter(span, 'vllm.latency.time_in_queue', attributes['gen_ai.latency.time_in_queue'], sec_to_ns=True)
        if 'gen_ai.latency.time_to_first_token' in attributes:
            _add_counter(span, 'vllm.latency.time_to_first_token', attributes['gen_ai.latency.time_to_first_token'], sec_to_ns=True)
        if 'gen_ai.latency.e2e' in attributes:
            _add_counter(span, 'vllm.latency.e2e', attributes['gen_ai.latency.e2e'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_scheduler' in attributes:
            _add_counter(span, 'vllm.latency.time_in_scheduler', attributes['gen_ai.latency.time_in_scheduler'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_forward' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_forward', attributes['gen_ai.latency.time_in_model_forward'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_execute' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_execute', attributes['gen_ai.latency.time_in_model_execute'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_prefill' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_prefill', attributes['gen_ai.latency.time_in_model_prefill'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_decode' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_decode', attributes['gen_ai.latency.time_in_model_decode'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_inference' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_inference', attributes['gen_ai.latency.time_in_model_inference'], sec_to_ns=True)

        for option_name, option_value in self._startup_options.items():
            _add_attribute(span, f'vllm.startup.{option_name}', option_value)

        graphsignal._ticker.signal_uploader().upload_span(span)

    def _should_sample_trace_span(self, span_name, trace_id, parent_span_id):
        if not _has_parent_span_id(parent_span_id):
            sampled = graphsignal._ticker.should_trace((f'vllm.{span_name}', 'random'))
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
    'model',
    'tokenizer',
    'dtype',
    'kv_cache_dtype',
    'quantization',
    'tensor_parallel_size',
    'pipeline_parallel_size',
    'data_parallel_size',
    'enforce_eager',
    'enable_prefix_caching',
    'enable_chunked_prefill',
    'max_num_seqs',
    'max_num_batched_tokens',
    'max_model_len',
    'gpu_memory_utilization',
    'scheduling_policy',
    'speculative_config',
    'block_size',
    'swap_space',
    'num_scheduler_steps',
)

VLLM_CONFIG_OPTIONS = (
    ('model_config.model', 'model'),
    ('model_config.tokenizer', 'tokenizer'),
    ('model_config.dtype', 'dtype'),
    ('model_config.quantization', 'quantization'),
    ('model_config.enforce_eager', 'enforce_eager'),
    ('model_config.max_model_len', 'max_model_len'),
    ('cache_config.cache_dtype', 'kv_cache_dtype'),
    ('cache_config.enable_prefix_caching', 'enable_prefix_caching'),
    ('cache_config.block_size', 'block_size'),
    ('cache_config.swap_space_bytes', 'swap_space'),
    ('parallel_config.tensor_parallel_size', 'tensor_parallel_size'),
    ('parallel_config.pipeline_parallel_size', 'pipeline_parallel_size'),
    ('parallel_config.data_parallel_size', 'data_parallel_size'),
    ('scheduler_config.max_num_seqs', 'max_num_seqs'),
    ('scheduler_config.max_num_batched_tokens', 'max_num_batched_tokens'),
    ('scheduler_config.chunked_prefill_enabled', 'enable_chunked_prefill'),
    ('scheduler_config.policy', 'scheduling_policy'),
    ('scheduler_config.num_scheduler_steps', 'num_scheduler_steps'),
)


PROFILED_PATHS = [
    ('vllm.e2e', "vllm.entrypoints.llm.LLM.generate"),
    ('vllm.e2e', "vllm.v1.engine.async_llm.AsyncLLM.generate"),
    ('vllm.e2e', "vllm.entrypoints.api_server.generate"),
    ('vllm.e2e', "vllm.entrypoints.openai.chat_completion.api_router.create_chat_completion"),
    ('vllm.e2e', "vllm.entrypoints.openai.completion.api_router.create_completion"),
    ('vllm.e2e', "vllm.entrypoints.openai.responses.api_router.create_responses"),
    ('vllm.e2e', "vllm.entrypoints.openai.chat_completion.serving.OpenAIServingChat.create_chat_completion"),
    ('vllm.e2e', "vllm.entrypoints.openai.completion.serving.OpenAIServingCompletion.create_completion"),
    ('vllm.e2e', "vllm.entrypoints.openai.responses.serving.OpenAIServingResponses.create_responses"),

    ('vllm.engine', "vllm.v1.engine.llm_engine.LLMEngine.add_request"),
    ('vllm.engine', "vllm.v1.engine.llm_engine.LLMEngine.step"),
    ('vllm.engine', "vllm.v1.engine.core.EngineCore.step"),
    ('vllm.engine', "vllm.v1.engine.core.EngineCore.step_with_batch_queue"),
    ('vllm.engine', "vllm.v1.engine.core.EngineCore.post_step"),

    ('vllm.engine', "vllm.v1.engine.input_processor.InputProcessor.process_inputs"),

    ('vllm.engine', "vllm.v1.core.sched.scheduler.Scheduler.schedule"),
    ('vllm.engine', "vllm.v1.core.sched.scheduler.Scheduler.update_from_output"),
    ('vllm.engine', "vllm.v1.core.sched.scheduler.Scheduler.get_grammar_bitmask"),

    ('vllm.model_exec', "vllm.v1.executor.abstract.Executor.execute_model"),
    ('vllm.model_exec', "vllm.v1.executor.abstract.Executor.sample_tokens"),
    ('vllm.model_exec', "vllm.v1.worker.worker_base.WorkerBase.execute_model"),
    ('vllm.model_exec', "vllm.v1.worker.gpu_worker.Worker.execute_model"),

    ('vllm.model_exec', "vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model"),
    ('vllm.model_exec', "vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward"),
    ('vllm.model_exec', "vllm.v1.worker.gpu_model_runner.GPUModelRunner._prepare_inputs"),
    ('vllm.model_exec', "vllm.v1.worker.gpu_model_runner.GPUModelRunner._update_states"),
    ('vllm.model_exec', "vllm.v1.worker.gpu_model_runner.GPUModelRunner.sample_tokens"),

    ('vllm.spec_decode', "vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer.propose"),
    ('vllm.spec_decode', "vllm.v1.spec_decode.eagle.SpecDecodeBaseProposer.propose_tree"),
    ('vllm.spec_decode', "vllm.v1.spec_decode.ngram_proposer.NgramProposer.propose"),
    ('vllm.spec_decode', "vllm.v1.spec_decode.ngram_proposer_gpu.NgramProposerGPU.propose"),

    ('vllm.attention', "vllm.model_executor.layers.attention.attention.Attention.forward"),
    ('vllm.attention', "vllm.model_executor.layers.attention.mla_attention.MLAAttention.forward"),

    ('vllm.kv_cache', "vllm.v1.core.kv_cache_manager.KVCacheManager.allocate_slots"),
    ('vllm.kv_cache', "vllm.v1.core.kv_cache_manager.KVCacheManager.free"),
    ('vllm.kv_cache', "vllm.v1.core.kv_cache_manager.KVCacheManager.get_computed_blocks"),
    ('vllm.kv_cache', "vllm.v1.core.kv_cache_manager.KVCacheManager.cache_blocks"),
    ('vllm.kv_cache', "vllm._custom_ops.reshape_and_cache_flash"),
    ('vllm.kv_cache', "vllm._custom_ops.swap_blocks"),

    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.all_reduce"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.all_gather"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.reduce_scatter"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.broadcast"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.broadcast_tensor_dict"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.send_tensor_dict"),
    ('vllm.comm', "vllm.distributed.parallel_state.GroupCoordinator.recv_tensor_dict"),
    ('vllm.comm', "vllm.distributed.communication_op.tensor_model_parallel_all_reduce"),
    ('vllm.comm', "vllm.distributed.communication_op.tensor_model_parallel_all_gather"),
    ('vllm.comm', "vllm.distributed.communication_op.tensor_model_parallel_reduce_scatter"),

    ('vllm.output', "vllm.v1.engine.output_processor.OutputProcessor.process_outputs"),
    ('vllm.output', "vllm.v1.engine.detokenizer.BaseIncrementalDetokenizer.update"),
]
