import logging
import sys
import time
import types
import openai

import graphsignal
from graphsignal.endpoint_trace import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method
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
        self._framework.name = 'OpenAI Python Library'
        parse_semver(self._framework.version, openai.version.VERSION)

        if compare_semver(self._framework.version, (0, 26, 0)) < 1:
            logger.debug('OpenAI tracing is only supported for >= 0.26.0.')
            return

        instrument_method(openai.Completion, 'create', 'openai.Completion.create', self.trace_completion)
        instrument_method(openai.Completion, 'acreate', 'openai.Completion.acreate', self.trace_completion)
        instrument_method(openai.Edit, 'create', 'openai.Edit.create', self.trace_edits)
        instrument_method(openai.Edit, 'acreate', 'openai.Edit.acreate', self.trace_edits)
        instrument_method(openai.Embedding, 'create', 'openai.Embedding.create', self.trace_embedding)
        instrument_method(openai.Embedding, 'acreate', 'openai.Embedding.acreate', self.trace_embedding)
        instrument_method(openai.Image, 'create', 'openai.Image.create', self.trace_image)
        instrument_method(openai.Image, 'acreate', 'openai.Image.acreate', self.trace_image)
        instrument_method(openai.Image, 'create_variation', 'openai.Image.create_variation', self.trace_image)
        instrument_method(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation', self.trace_image)
        instrument_method(openai.Image, 'create_edit', 'openai.Image.create_edit', self.trace_image)
        instrument_method(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit', self.trace_image)
        instrument_method(openai.Moderation, 'create', 'openai.Moderation.create', self.trace_moderation)
        instrument_method(openai.Moderation, 'acreate', 'openai.Moderation.acreate', self.trace_moderation)

    def shutdown(self):
        uninstrument_method(openai.Completion, 'create', 'openai.Completion.create')
        uninstrument_method(openai.Completion, 'acreate', 'openai.Completion.acreate')
        uninstrument_method(openai.Edit, 'create', 'openai.Edit.create')
        uninstrument_method(openai.Edit, 'acreate', 'openai.Edit.acreate')
        uninstrument_method(openai.Embedding, 'create', 'openai.Embedding.create')
        uninstrument_method(openai.Embedding, 'acreate', 'openai.Embedding.acreate')
        uninstrument_method(openai.Image, 'create', 'openai.Image.create')
        uninstrument_method(openai.Image, 'acreate', 'openai.Image.acreate')
        uninstrument_method(openai.Image, 'create_variation', 'openai.Image.create_variation')
        uninstrument_method(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation')
        uninstrument_method(openai.Image, 'create_edit', 'openai.Image.create_edit')
        uninstrument_method(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit')
        uninstrument_method(openai.Moderation, 'create', 'openai.Moderation.create')
        uninstrument_method(openai.Moderation, 'acreate', 'openai.Moderation.acreate')

    def trace_completion(self, trace, args, kwargs, ret, exc):
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

    def trace_edits(self, trace, args, kwargs, ret, exc):
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

    def trace_embedding(self, trace, args, kwargs, ret, exc):
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

    def trace_image(self, trace, args, kwargs, ret, exc):
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

    def trace_moderation(self, trace, args, kwargs, ret, exc):
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
