import logging
import time
import vllm

import graphsignal
from graphsignal import client
from graphsignal.utils import uuid_sha1
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
            def read_kwarg(store, kwargs, key, new_key=None, default=None):
                if new_key is None:
                    new_key = key
                if key in kwargs:
                    store[new_key] = str(kwargs[key])
                elif default is not None:
                    store[new_key] = str(default)

            def after_llm_init(args, kwargs, ret, exc, context):
                llm_obj = args[0]
                llm_tags = {}
                llm_params = {}

                if self._otel_adapter is None:
                    # Define the export callback function
                    def otel_export_callback(otel_spans):
                        for otel_span in otel_spans:
                            self._convert_otel_span(otel_span)
                        graphsignal._tracer.tick()

                    self._otel_adapter = OTELAdapter(export_callback=otel_export_callback)
                    self._otel_adapter.setup()

                model = None
                if len(args) > 1 and args[1] is not None:
                    model = args[1]
                read_kwarg(llm_tags, kwargs, 'model', 'vllm.model.name', default=model)
                read_kwarg(llm_params, kwargs, 'model', 'vllm.model.name', default=model)
                read_kwarg(llm_params, kwargs, 'tokenizer', 'vllm.tokenizer.name')
                read_kwarg(llm_params, kwargs, 'tensor_parallel_size', 'vllm.tensor_parallel_size')
                read_kwarg(llm_params, kwargs, 'dtype', 'vllm.dtype')
                read_kwarg(llm_params, kwargs, 'quantization', 'vllm.quantization')
                read_kwarg(llm_params, kwargs, 'gpu_memory_utilization', 'vllm.gpu_memory_utilization')
                read_kwarg(llm_params, kwargs, 'swap_space', 'vllm.swap_space')
                read_kwarg(llm_params, kwargs, 'cpu_offload_gb', 'vllm.cpu_offload_gb')
                read_kwarg(llm_params, kwargs, 'enforce_eager', 'vllm.enforce_eager')
                read_kwarg(llm_params, kwargs, 'max_seq_len_to_capture', 'vllm.max_seq_len_to_capture')
                read_kwarg(llm_params, kwargs, 'disable_custom_all_reduce', 'vllm.disable_custom_all_reduce')
                read_kwarg(llm_params, kwargs, 'disable_async_output_proc', 'vllm.disable_async_output_proc')

                def trace_generate(span, args, kwargs, ret, exc):
                    for param_name, param_value in llm_tags.items():
                        span.set_tag(param_name, param_value)
                    for param_name, param_value in llm_params.items():
                        span.set_param(param_name, param_value)
                    #sampling_params = kwargs.get('sampling_params', None)

                trace_method(llm_obj, 'generate', 'llm.generate', trace_func=trace_generate)

            patch_method(vllm.LLM, '__init__', after_func=after_llm_init)

            '''def after_llm_engine_init(args, kwargs, ret, exc, context):
                llm_engine = args[0]
                trace_method(llm_engine, 'generate', 'LLMEngine.generate', trace_func=trace_generate)

            patch_method(vllm.llm_engine.LLMEngine, '__init__', after_func=after_llm_engine_init)'''

            '''def after_async_llm_init(args, kwargs, ret, exc, context):
                llm_engine = args[0]

                trace_method(llm_engine, 'generate', 'AsyncLLMEngine.generate', trace_func=trace_generate)

            patch_method(vllm.engine.async_llm_engine.AsyncLLMEngine, '__init__', after_func=after_async_llm_init)'''
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
        if not graphsignal._tracer.should_sample('vllm'):
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
            counters=[]
        )

        attributes = otel_span.attributes
        if 'gen_ai.request.id' in attributes:
            span.tags.append(client.Tag(key='vllm.request.id', value=attributes['gen_ai.request.id']))
        if 'gen_ai.response.model' in attributes:
            span.tags.append(client.Tag(key='vllm.response.model', value=attributes['gen_ai.response.model']))
            span.params.append(client.Param(name='vllm.response.model', value=str(attributes['gen_ai.response.model'])))
        if 'gen_ai.request.temperature' in attributes:
            span.params.append(client.Param(name='vllm.request.temperature', value=str(attributes['gen_ai.request.temperature'])))
        if 'gen_ai.request.top_p' in attributes:
            span.params.append(client.Param(name='vllm.request.top_p', value=str(attributes['gen_ai.request.top_p'])))
        if 'gen_ai.request.max_tokens' in attributes:
            span.params.append(client.Param(name='vllm.request.max_tokens', value=str(attributes['gen_ai.request.max_tokens'])))
        if 'gen_ai.request.n' in attributes:
            span.params.append(client.Param(name='vllm.request.n', value=str(attributes['gen_ai.request.n'])))
        if 'gen_ai.usage.num_sequences' in attributes:
            span.counters.append(client.Counter(name='vllm.usage.num_sequences', value=float(attributes['gen_ai.usage.num_sequences'])))
        if 'gen_ai.usage.prompt_tokens' in attributes:
            span.counters.append(client.Counter(name='vllm.usage.prompt_tokens', value=float(attributes['gen_ai.usage.prompt_tokens'])))
        if 'gen_ai.usage.completion_tokens' in attributes:
            span.counters.append(client.Counter(name='vllm.usage.completion_tokens', value=float(attributes['gen_ai.usage.completion_tokens'])))
        if 'gen_ai.latency.time_in_queue' in attributes:
            span.counters.append(client.Counter(name='vllm.latency.time_in_queue', value=float(attributes['gen_ai.latency.time_in_queue'])))
        if 'gen_ai.latency.time_to_first_token' in attributes:
            span.counters.append(client.Counter(name='vllm.latency.time_to_first_token', value=float(attributes['gen_ai.latency.time_to_first_token'])))
        if 'gen_ai.latency.e2e' in attributes:
            span.counters.append(client.Counter(name='vllm.latency.e2e', value=float(attributes['gen_ai.latency.e2e'])))
        if 'gen_ai.latency.time_in_scheduler' in attributes:
            span.counters.append(client.Counter(name='vllm.latency.time_in_scheduler', value=float(attributes['gen_ai.latency.time_in_scheduler'])))        

        graphsignal._tracer.uploader().upload_span(span)


    def shutdown(self):
        if self._otel_adapter:
            self._otel_adapter.shutdown()
