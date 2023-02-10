import logging
import sys
import time
import openai

import graphsignal
from graphsignal.endpoint_trace import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.recorder_utils import patch_method, unpatch_method
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class OpenAIRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None
        self._is_sampling = False

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.type = signals_pb2.FrameworkInfo.FrameworkType.OPENAI_FRAMEWORK
        parse_semver(self._framework.version, openai.version.VERSION)

        if compare_semver(self._framework.version, (0, 26, 0)) < 1:
            logger.debug('OpenAI tracing is only supported for >= 0.26.0.')
            return

        self._instrument(openai.Completion, 'create', 'openai.Completion.create', self.trace_completion)
        self._instrument(openai.Completion, 'acreate', 'openai.Completion.acreate', self.trace_completion)
        self._instrument(openai.Edit, 'create', 'openai.Edit.create', self.trace_edits)
        self._instrument(openai.Edit, 'acreate', 'openai.Edit.acreate', self.trace_edits)
        self._instrument(openai.Embedding, 'create', 'openai.Embedding.create', self.trace_embedding)
        self._instrument(openai.Embedding, 'acreate', 'openai.Embedding.acreate', self.trace_embedding)
        self._instrument(openai.Image, 'create', 'openai.Image.create', self.trace_image)
        self._instrument(openai.Image, 'acreate', 'openai.Image.acreate', self.trace_image)
        self._instrument(openai.Image, 'create_variation', 'openai.Image.create_variation', self.trace_image)
        self._instrument(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation', self.trace_image)
        self._instrument(openai.Image, 'create_edit', 'openai.Image.create_edit', self.trace_image)
        self._instrument(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit', self.trace_image)
        self._instrument(openai.Moderation, 'create', 'openai.Moderation.create', self.trace_moderation)
        self._instrument(openai.Moderation, 'acreate', 'openai.Moderation.acreate', self.trace_moderation)

    def shutdown(self):
        self._uninstrument(openai.Completion, 'create', 'openai.Completion.create')
        self._uninstrument(openai.Completion, 'acreate', 'openai.Completion.acreate')
        self._uninstrument(openai.Edit, 'create', 'openai.Edit.create')
        self._uninstrument(openai.Edit, 'acreate', 'openai.Edit.acreate')
        self._uninstrument(openai.Embedding, 'create', 'openai.Embedding.create')
        self._uninstrument(openai.Embedding, 'acreate', 'openai.Embedding.acreate')
        self._uninstrument(openai.Image, 'create', 'openai.Image.create')
        self._uninstrument(openai.Image, 'acreate', 'openai.Image.acreate')
        self._uninstrument(openai.Image, 'create_variation', 'openai.Image.create_variation')
        self._uninstrument(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation')
        self._uninstrument(openai.Image, 'create_edit', 'openai.Image.create_edit')
        self._uninstrument(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit')
        self._uninstrument(openai.Moderation, 'create', 'openai.Moderation.create')
        self._uninstrument(openai.Moderation, 'acreate', 'openai.Moderation.acreate')

    def _instrument(self, obj, func_name, endpoint, trace_func):
        def before_func(args, kwargs):
            return graphsignal.start_trace(endpoint=endpoint)

        def after_func(args, kwargs, ret, exc, trace):
            trace.measure()
            try:
                if exc is not None:
                    trace.set_exception(exc)

                trace_func(trace, kwargs, ret, exc)
            except Exception as e:
                logger.debug('Error tracing %s', func_name, exc_info=True)

            trace.stop()

        if not patch_method(obj, func_name, before_func=before_func, after_func=after_func):
            logger.debug('Cannot instrument %s.', endpoint)

    def _uninstrument(self, obj, func_name, endpoint):
        if not unpatch_method(obj, func_name):
            logger.debug('Cannot uninstrument %s.', endpoint)

    def trace_completion(self, trace, kwargs, ret, exc):
        if self._is_sampling:
            param_args = [
                'model',
                'max_tokens',
                'temperature',
                'top_p',
                'n',
                'stream',
                'logprobs',
                'echo',
                'stop',
                'presence_penalty',
                'frequency_penalty',
                'best_of'
            ]
            for param_arg in param_args:
                if param_arg in kwargs:
                    trace.set_param(param_arg, kwargs[param_arg])
        if 'stream' in kwargs and kwargs['stream']:
            if 'prompt' in kwargs:
                trace.set_data('prompt', kwargs['prompt'])
            return

        prompt_usage = {}
        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0
        }
        if ret and 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'prompt' in kwargs:
            trace.set_data('prompt', kwargs['prompt'], extra_counts=prompt_usage)

        if ret and 'choices' in ret:
            completion = []
            for choice in ret['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                if 'text' in choice:
                    completion.append(choice['text'])
            trace.set_data('completion', completion, extra_counts=completion_usage)

    def trace_edits(self, trace, kwargs, ret, exc):
        if self._is_sampling:
            param_args = [
                'model',
                'temperature',
                'top_p',
                'n'
            ]
            for param_arg in param_args:
                if param_arg in kwargs:
                    trace.set_param(param_arg, kwargs[param_arg])

        prompt_usage = {}
        completion_usage = {}
        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'input' in kwargs:
            trace.set_data('input', kwargs['input'])

        if 'instruction' in kwargs:
            trace.set_data('instruction', kwargs['instruction'], extra_counts=prompt_usage)

        if ret and 'choices' in ret:
            edits = []
            for choice in ret['choices']:
                if 'text' in choice:
                    edits.append(choice['text'])
            trace.set_data('edits', edits, extra_counts=completion_usage)

    def trace_embedding(self, trace, kwargs, ret, exc):
        if self._is_sampling:
            param_args = [
                'engine'
            ]
            for param_arg in param_args:
                if param_arg in kwargs:
                    trace.set_param(param_arg, kwargs[param_arg])

        prompt_usage = {}
        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']

        if 'input' in kwargs:
            trace.set_data('input', kwargs['input'], extra_counts=prompt_usage)

        if ret and 'data' in ret:
            embedding = []
            for choice in ret['data']:
                if 'embedding' in choice:
                    embedding.append(choice['embedding'])
            trace.set_data('embedding', embedding)

    def trace_image(self, trace, kwargs, ret, exc):
        if self._is_sampling:
            param_args = [
                'n',
                'size',
                'response_format'
            ]
            for param_arg in param_args:
                if param_arg in kwargs:
                    trace.set_param(param_arg, kwargs[param_arg])

        if 'prompt' in kwargs:
            trace.set_data('prompt', kwargs['prompt'])

        if ret and 'data' in ret:
            image_data = []
            for image in ret['data']:
                if 'url' in image:
                    image_data.append(image['url'])
                elif 'b64_json' in image:
                    image_data.append(image['b64_json'])
            trace.set_data('image', image_data)

    def trace_moderation(self, trace, kwargs, ret, exc):
        if self._is_sampling:
            param_args = [
                'model'
            ]
            for param_arg in param_args:
                if param_arg in kwargs:
                    trace.set_param(param_arg, kwargs[param_arg])

        if 'input' in kwargs:
            trace.set_data('input', kwargs['input'])

    def on_trace_start(self, signal, context, options):
        self._is_sampling = True

    def on_trace_stop(self, signal, context, options):
        self._is_sampling = False

    def on_trace_read(self, signal, context, options):
        if self._framework:
            signal.frameworks.append(self._framework)
