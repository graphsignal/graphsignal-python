import logging
import os
import copy
import importlib
import openai

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import trace_method, profile_method, patch_method, parse_semver, compare_semver
from graphsignal.profiles import EventAverages

logger = logging.getLogger('graphsignal')

class VLLMRecorder(BaseRecorder):
    def __init__(self):
        self._active_profile = None
        self._profiling = False

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        version = vllm.__version__
        self._library_version = version
        parsed_version = parse_semver(version)

        if compare_semver(parsed_version, (0, 8, 0)) >= 0:
            def trace_generate(span, args, kwargs, ret, exc):
                llm = args[0]

                # read and set params to span

            def after_llm_init(args, kwargs, ret, exc, context):
                llm = args[0]
                trace_method(llm, 'generate', 'LLM.generate', trace_func=trace_generate)

            patch_method(vllm.LLM, '__init__', after_func=after_llm_init)


            def after_llm_engine_init(args, kwargs, ret, exc, context):
                llm_engine = args[0]
                trace_method(llm_engine, 'generate', 'LLMEngine.generate', trace_func=trace_generate)

            patch_method(vllm.llm_engine.LLMEngine, '__init__', after_func=after_llm_engine_init)


            '''def after_async_llm_init(args, kwargs, ret, exc, context):
                llm_engine = args[0]

                trace_method(llm_engine, 'generate', 'AsyncLLMEngine.generate', trace_func=trace_generate)

            patch_method(vllm.engine.async_llm_engine.AsyncLLMEngine, '__init__', after_func=after_async_llm_init)'''
        else:
            logger.debug('VLLM tracing is only supported for >= 0.8.0.')
            return


    def on_span_start(self, span, context):
        if not span.profiled():
            return

        self._active_profile = EventAverages()
        self._profiling = True

    def on_span_stop(self, span, context):
        if not span.profiled():
            return

        self._profiling = False

    def on_span_read(self, span, context):
        if not span.profiled():
            return

        if self._active_profile and not self._active_profile.is_empty():
            span.set_profile(
                name='vllm-profile', 
                format='event-averages', 
                content=self._active_profile.dumps())
        
        self._active_profile = None

    def shutdown(self):
        pass
