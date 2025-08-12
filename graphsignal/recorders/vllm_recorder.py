import logging
import vllm

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import trace_method, patch_method, parse_semver, compare_semver
from graphsignal.recorders.prometheus_adapter import PrometheusAdapter

logger = logging.getLogger('graphsignal')

class VLLMRecorder(BaseRecorder):
    def __init__(self):
        self._active_profile = None
        self._profiling = False
        self._otel_processor = None

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

                model = None
                if len(args) > 1 and args[1] is not None:
                    model = args[1]
                read_kwarg(llm_tags, kwargs, 'model', 'model.name', default=model)
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
        
        self._prometheus_adapter = PrometheusAdapter()
        self._prometheus_adapter.setup(dict(
           # map here 
        ))

    def on_metric_update(self):
        if self._prometheus_adapter:
            self._prometheus_adapter.collect()

    def shutdown(self):
        if self._otel_processor:
            self._otel_processor.shutdown()
