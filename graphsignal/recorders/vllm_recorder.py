import logging
import time
import vllm

import graphsignal
from graphsignal import client
from graphsignal.utils import uuid_sha1, sanitize_str
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import trace_method, patch_method, parse_semver, compare_semver
from graphsignal.recorders.prometheus_adapter import PrometheusAdapter
from graphsignal.recorders.otel_adapter import OTELAdapter

logger = logging.getLogger('graphsignal')

class VLLMRecorder(BaseRecorder):
    def __init__(self):
        self._profiling = False
        self._otel_adapter = None

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        version = vllm.__version__
        self._library_version = version
        parsed_version = parse_semver(version)

        tracer = graphsignal._tracer
        tracer.set_tag('inference.engine.name', 'vllm')
        tracer.set_tag('inference.engine.version', version)

        if compare_semver(parsed_version, (0, 8, 0)) >= 0:
            # LLM
            def read_kwarg(store, kwargs, key, new_key=None, default=None):
                if new_key is None:
                    new_key = key
                if key in kwargs:
                    store[new_key] = str(kwargs[key])
                elif default is not None:
                    store[new_key] = str(default)
            
            def before_llm_init(args, kwargs):
                kwargs['disable_log_stats'] = False
                # kwargs['otlp_traces_endpoint'] = 'grpc://localhost:4317'

            def after_llm_init(args, kwargs, ret, exc, context):
                llm_obj = args[0]
                llm_tags = {}
                llm_params = {}

                # do not enable otel adapter until supported by vllm v1
                # otherwise vllm falls back to v0
                '''if self._otel_adapter is None:
                    # Define the export callback function
                    def otel_export_callback(otel_spans):
                        for otel_span in otel_spans:
                            self._convert_otel_span(otel_span)
                        graphsignal._tracer.tick()

                    self._otel_adapter = OTELAdapter(export_callback=otel_export_callback)
                    self._otel_adapter.setup()'''

                model = None
                if len(args) > 1 and args[1] is not None:
                    model = args[1]
                read_kwarg(llm_tags, kwargs, 'model', 'vllm.model.name', default=model)
                read_kwarg(llm_tags, kwargs, 'tokenizer', 'vllm.tokenizer.name')
                read_kwarg(llm_tags, kwargs, 'dtype', 'vllm.dtype')
                read_kwarg(llm_tags, kwargs, 'quantization', 'vllm.quantization')
                read_kwarg(llm_tags, kwargs, 'enforce_eager', 'vllm.enforce_eager')
                read_kwarg(llm_tags, kwargs, 'tensor_parallel_size', 'vllm.tensor_parallel_size')
                read_kwarg(llm_tags, kwargs, 'enforce_eager', 'vllm.enforce_eager')

                def trace_generate(span, args, kwargs, ret, exc):
                    for tag_name, tag_value in llm_tags.items():
                        span.set_tag(tag_name, tag_value)
                    for param_name, param_value in llm_params.items():
                        span.set_attribute(param_name, param_value)
                    #sampling_params = kwargs.get('sampling_params', None)

                    span.measure_event_as_counter('vllm.latency.e2e')
                    if ret and isinstance(ret, list):
                        for item in ret:
                            if span.get_tag('vllm.request.id') is None and hasattr(item, 'request_id') and item.request_id:
                                span.set_tag('vllm.request.id', item.request_id)
                            if hasattr(item, 'prompt_token_ids') and item.prompt_token_ids and len(item.prompt_token_ids) > 0:
                                span.set_counter('vllm.usage.prompt_tokens', len(item.prompt_token_ids))
                            if hasattr(item, 'num_cached_tokens') and item.num_cached_tokens > 0:
                                span.set_counter('vllm.usage.cached_tokens', item.num_cached_tokens)
                            for output in item.outputs:
                                if hasattr(output, 'token_ids') and output.token_ids and len(output.token_ids) > 0:
                                    span.inc_counter('vllm.usage.completion_tokens', len(output.token_ids))

                trace_method(llm_obj, 'generate', 'vllm.llm.generate', trace_func=trace_generate)

            patch_method(vllm.LLM, '__init__', before_func=before_llm_init, after_func=after_llm_init)

            # AsyncLLM
            def patch_generate(async_llm_engine, tags, params):
                def trace_async_generate(span, args, kwargs, ret, exc):
                    for tag_name, tag_value in tags.items():
                        span.set_tag(tag_name, tag_value)
                    for param_name, param_value in params.items():
                        span.set_attribute(param_name, param_value)

                def trace_async_generate_data(span, item, exc, stopped):
                    if not stopped:
                        if not span.get_counter('vllm.latency.time_to_first_token'):
                            span.measure_event_as_counter('vllm.latency.time_to_first_token')
                            if hasattr(item, 'request_id') and item.request_id:
                                span.set_tag('vllm.request.id', item.request_id)
                            if hasattr(item, 'prompt_token_ids') and item.prompt_token_ids and len(item.prompt_token_ids) > 0:
                                span.set_counter('vllm.usage.prompt_tokens', len(item.prompt_token_ids))
                            if hasattr(item, 'num_cached_tokens') and item.num_cached_tokens > 0:
                                span.set_counter('vllm.usage.cached_tokens', item.num_cached_tokens)
                        for output in item.outputs:
                            if hasattr(output, 'token_ids') and output.token_ids and len(output.token_ids) > 0:
                                span.inc_counter('vllm.usage.completion_tokens', len(output.token_ids))
                    else:
                        span.measure_event_as_counter('vllm.latency.e2e')

                        e2e = span.get_counter('vllm.latency.e2e')
                        ttft = span.get_counter('vllm.latency.time_to_first_token')
                        completion_tokens = span.get_counter('vllm.usage.completion_tokens')
                        if e2e and e2e > 0 and ttft and ttft > 0 and completion_tokens and completion_tokens > 0:
                            tpot = (e2e - ttft) / completion_tokens
                            span.set_counter('vllm.latency.time_per_output_token', tpot)

                trace_method(async_llm_engine, 'generate', 'vllm.asyncllm.generate', trace_func=trace_async_generate, data_func=trace_async_generate_data)

            def before_from_engine_args(args, kwargs):
                if len(args) > 0:
                    engine_args = args[0]
                else:
                    engine_args = kwargs['engine_args']

                if engine_args.disable_log_stats:
                    engine_args.disable_log_stats = False

            def after_from_engine_args(args, kwargs, ret, exc, context):
                if len(args) > 0:
                    engine_args = args[0]
                else:
                    engine_args = kwargs['engine_args']
                async_llm = ret

                tags = {}
                params = {}

                if engine_args.model:
                    tags['vllm.model.name'] = engine_args.model
                if engine_args.tokenizer:
                    tags['vllm.tokenizer.name'] = engine_args.tokenizer
                if engine_args.dtype:
                    tags['vllm.dtype'] = engine_args.dtype
                if engine_args.kv_cache_dtype:
                    tags['vllm.kv_cache_dtype'] = engine_args.kv_cache_dtype
                if engine_args.quantization:
                    tags['vllm.quantization'] = engine_args.quantization

                if (engine_args.tensor_parallel_size > 1 or
                    engine_args.pipeline_parallel_size > 1 or
                    engine_args.data_parallel_size > 1):

                    if engine_args.pipeline_parallel_size:
                        tags['vllm.pipeline_parallel_size'] = engine_args.pipeline_parallel_size
                    if engine_args.tensor_parallel_size:
                        tags['vllm.tensor_parallel_size'] = engine_args.tensor_parallel_size
                    if engine_args.data_parallel_size:
                        tags['vllm.data_parallel_size'] = engine_args.data_parallel_size
                    if engine_args.data_parallel_rank:
                        tags['vllm.data_parallel_rank'] = engine_args.data_parallel_rank
                    if engine_args.data_parallel_rank_local:
                        tags['vllm.data_parallel_rank_local'] = engine_args.data_parallel_rank_local
                    if engine_args.data_parallel_master_ip:
                        tags['vllm.data_parallel_master_ip'] = engine_args.data_parallel_master_ip
                    if engine_args.data_parallel_master_port:
                        tags['vllm.data_parallel_master_port'] = engine_args.data_parallel_master_port
                    if engine_args.data_parallel_rpc_port:
                        tags['vllm.data_parallel_rpc_port'] = engine_args.data_parallel_rpc_port
                    if engine_args.data_parallel_backend:
                        tags['vllm.data_parallel_backend'] = engine_args.data_parallel_backend

                if engine_args.enforce_eager:
                    tags['vllm.enforce_eager'] = engine_args.enforce_eager

                patch_generate(async_llm, tags, params)

            def before_from_vllm_config(args, kwargs):
                kwargs['disable_log_stats'] = False

            def after_from_vllm_config(args, kwargs, ret, exc, context):
                if len(args) > 0:
                    vllm_config = args[0]
                else:
                    vllm_config = kwargs['vllm_config']
                async_llm = ret

                tags = {}
                params = {}

                if vllm_config.model_config.model:
                    tags['vllm.model.name'] = vllm_config.model_config.model
                if vllm_config.model_config.tokenizer:
                    tags['vllm.tokenizer.name'] = vllm_config.model_config.tokenizer
                if vllm_config.model_config.dtype:
                    tags['vllm.dtype'] = vllm_config.model_config.dtype
                if vllm_config.cache_config.cache_dtype:
                    tags['vllm.kv_cache_dtype'] = vllm_config.cache_config.cache_dtype
                if vllm_config.model_config.quantization:
                    tags['vllm.quantization'] = vllm_config.model_config.quantization
                
                if (vllm_config.parallel_config.tensor_parallel_size > 1 or
                    vllm_config.parallel_config.pipeline_parallel_size > 1 or
                    vllm_config.parallel_config.data_parallel_size > 1):
                    
                    if vllm_config.parallel_config.pipeline_parallel_size:
                        tags['vllm.pipeline_parallel_size'] = vllm_config.parallel_config.pipeline_parallel_size
                    if vllm_config.parallel_config.tensor_parallel_size:
                        tags['vllm.tensor_parallel_size'] = vllm_config.parallel_config.tensor_parallel_size
                    if vllm_config.parallel_config.data_parallel_size:
                        tags['vllm.data_parallel_size'] = vllm_config.parallel_config.data_parallel_size
                    if vllm_config.parallel_config.data_parallel_rank:
                        tags['vllm.data_parallel_rank'] = vllm_config.parallel_config.data_parallel_rank
                    if vllm_config.parallel_config.data_parallel_rank_local:
                        tags['vllm.data_parallel_rank_local'] = vllm_config.parallel_config.data_parallel_rank_local
                    if vllm_config.parallel_config.data_parallel_master_ip:
                        tags['vllm.data_parallel_master_ip'] = vllm_config.parallel_config.data_parallel_master_ip
                    if vllm_config.parallel_config.data_parallel_master_port:
                        tags['vllm.data_parallel_master_port'] = vllm_config.parallel_config.data_parallel_master_port
                    if vllm_config.parallel_config.data_parallel_rpc_port:
                        tags['vllm.data_parallel_rpc_port'] = vllm_config.parallel_config.data_parallel_rpc_port
                    if vllm_config.parallel_config.data_parallel_backend:
                        tags['vllm.data_parallel_backend'] = vllm_config.parallel_config.data_parallel_backend
                
                if vllm_config.model_config.enforce_eager:
                    tags['vllm.enforce_eager'] = vllm_config.model_config.enforce_eager

                patch_generate(async_llm, tags, params)

            from vllm.v1.engine.async_llm import AsyncLLM
            patch_method(AsyncLLM, 'from_engine_args', before_func=before_from_engine_args, after_func=after_from_engine_args)
            patch_method(AsyncLLM, 'from_vllm_config', before_func=before_from_vllm_config, after_func=after_from_vllm_config)
        else:
            logger.debug('VLLM tracing is only supported for >= 0.8.0.')
            return
        
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

    def on_metric_update(self):
        if self._prometheus_adapter:
            self._prometheus_adapter.collect()

    def _convert_otel_span(self, otel_span):
        if not graphsignal._tracer.should_sample('vllm.' + otel_span.name):
            return

        if not otel_span.name:
            logger.error(f'Invalid Open Telemetry span: name={otel_span.name}')
            return

        if not (otel_span.start_time > 0 and otel_span.end_time > 0):
            logger.error(f'Invalid Open Telemetry span: start_time={otel_span.start_time}, end_time={otel_span.end_time}')
            return

        span = client.Span(
            span_id=uuid_sha1(size=12),
            trace_id=uuid_sha1(size=12),
            start_ns=otel_span.start_time,
            end_ns=otel_span.end_time,
            name=f'vllm.{otel_span.name}',
            tags=[],
            params=[],
            counters=[],
            attributes=[]
        )

        # set process tags
        if graphsignal._tracer.tags:
            for tag_key, tag_value in graphsignal._tracer.tags.items():
                _add_tag(span, tag_key, tag_value)

        attributes = otel_span.attributes
        _add_attribute(span, 'sampling.reason', 'vllm.otel')
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
            _add_counter(span, 'vllm.latency.time_to_first_token', attributes['gen_ai.latency.time_to_first_token'],  sec_to_ns=True)
        if 'gen_ai.latency.e2e' in attributes:
            _add_counter(span, 'vllm.latency.e2e', attributes['gen_ai.latency.e2e'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_scheduler' in attributes:
            _add_counter(span, 'vllm.latency.time_in_scheduler', attributes['gen_ai.latency.time_in_scheduler'], sec_to_ns=True)      
        if 'gen_ai.latency.time_in_model_forward' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_forward', attributes['gen_ai.latency.time_in_model_forward'], sec_to_ns=True)
        if 'gen_ai.latency.time_in_model_execute' in attributes:
            _add_counter(span, 'vllm.latency.time_in_model_execute', attributes['gen_ai.latency.time_in_model_execute'], sec_to_ns=True)

        graphsignal._tracer.uploader().upload_span(span)


    def shutdown(self):
        if self._otel_adapter:
            self._otel_adapter.shutdown()

def _add_tag(span, key, value):
    span.tags.append(client.Tag(
        key=sanitize_str(key, max_len=50), 
        value=sanitize_str(value, max_len=250)))

def _add_attribute(span, name, value):
    span.attributes.append(client.Attribute(
        name=sanitize_str(name, max_len=50), 
        value=sanitize_str(value, max_len=2500)))

def _add_counter(span, name, value, sec_to_ns: bool = False):
    if sec_to_ns:
        value = int(value * 1e9)
    span.counters.append(client.Counter(name=name, value=float(value)))
