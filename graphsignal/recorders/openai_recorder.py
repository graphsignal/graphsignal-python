import logging
import sys
import os
import time
import types
import openai

import graphsignal
from graphsignal.traces import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2
from graphsignal.proto_utils import add_framework_param, add_driver

logger = logging.getLogger('graphsignal')

class OpenAIRecorder(BaseRecorder):
    def __init__(self):
        self._framework = None

    def setup(self):
        if not graphsignal._agent.auto_instrument:
            return

        self._framework = signals_pb2.FrameworkInfo()
        self._framework.name = 'OpenAI Python Library'
        parse_semver(self._framework.version, openai.version.VERSION)

        if compare_semver(self._framework.version, (0, 26, 0)) < 1:
            logger.debug('OpenAI tracing is only supported for >= 0.26.0.')
            return

        self._api_base = openai.api_base

        instrument_method(openai.Completion, 'create', 'openai.Completion.create', self.trace_completion, self.trace_completion_data)
        instrument_method(openai.Completion, 'acreate', 'openai.Completion.acreate', self.trace_completion, self.trace_completion_data)
        instrument_method(openai.ChatCompletion, 'create', 'openai.ChatCompletion.create', self.trace_chat_completion, self.trace_chat_completion_data)
        instrument_method(openai.ChatCompletion, 'acreate', 'openai.ChatCompletion.acreate', self.trace_chat_completion, self.trace_chat_completion_data)
        instrument_method(openai.Edit, 'create', 'openai.Edit.create', self.trace_edits)
        instrument_method(openai.Edit, 'acreate', 'openai.Edit.acreate', self.trace_edits)
        instrument_method(openai.Embedding, 'create', 'openai.Embedding.create', self.trace_embedding)
        instrument_method(openai.Embedding, 'acreate', 'openai.Embedding.acreate', self.trace_embedding)
        instrument_method(openai.Image, 'create', 'openai.Image.create', self.trace_image_generation)
        instrument_method(openai.Image, 'acreate', 'openai.Image.acreate', self.trace_image_generation)
        instrument_method(openai.Image, 'create_variation', 'openai.Image.create_variation', self.trace_image_variation)
        instrument_method(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation', self.trace_image_variation)
        instrument_method(openai.Image, 'create_edit', 'openai.Image.create_edit', self.trace_image_edit)
        instrument_method(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit', self.trace_image_edit)
        instrument_method(openai.Audio, 'transcribe', 'openai.Audio.transcribe', self.trace_audio_transcription)
        instrument_method(openai.Audio, 'atranscribe', 'openai.Audio.atranscribe', self.trace_audio_transcription)
        instrument_method(openai.Audio, 'translate', 'openai.Audio.translate', self.trace_audio_translation)
        instrument_method(openai.Audio, 'atranslate', 'openai.Audio.atranslate', self.trace_audio_translation)
        instrument_method(openai.Moderation, 'create', 'openai.Moderation.create', self.trace_moderation)
        instrument_method(openai.Moderation, 'acreate', 'openai.Moderation.acreate', self.trace_moderation)

    def shutdown(self):
        uninstrument_method(openai.Completion, 'create', 'openai.Completion.create')
        uninstrument_method(openai.Completion, 'acreate', 'openai.Completion.acreate')
        uninstrument_method(openai.ChatCompletion, 'create', 'openai.ChatCompletion.create')
        uninstrument_method(openai.ChatCompletion, 'acreate', 'openai.ChatCompletion.acreate')
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
        uninstrument_method(openai.Audio, 'transcribe', 'openai.Audio.transcribe')
        uninstrument_method(openai.Audio, 'atranscribe', 'openai.Audio.atranscribe')
        uninstrument_method(openai.Audio, 'translate', 'openai.Audio.translate')
        uninstrument_method(openai.Audio, 'atranslate', 'openai.Audio.atranslate')
        uninstrument_method(openai.Moderation, 'create', 'openai.Moderation.create')
        uninstrument_method(openai.Moderation, 'acreate', 'openai.Moderation.acreate')

    def trace_completion(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        trace.set_tag('component', 'LLM')
        trace.set_tag('endpoint', f'{self._api_base}/completions')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
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
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'stream' in params and params['stream']:
            if 'prompt' in params:
                trace.set_data('prompt', params['prompt'])
            return

        if ret and 'model' in ret:
            trace.set_tag('model', ret['model'])

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

        if 'prompt' in params:
            trace.set_data('prompt', params['prompt'], counts=prompt_usage)

        if ret:
            if 'choices' in ret:
                for choice in ret['choices']:
                    if 'finish_reason' in choice:
                        if choice['finish_reason'] == 'stop':
                            completion_usage['finish_reason_stop'] += 1
                        elif choice['finish_reason'] == 'length':
                            completion_usage['finish_reason_length'] += 1
            trace.set_data('completion', ret, counts=completion_usage)

    def trace_completion_data(self, trace, item):
        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0
        }
        if item and 'choices' in item:
            completion = []
            for choice in item['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                if 'text' in choice:
                    completion.append(choice['text'])

            trace.append_data('completion', completion, counts=completion_usage)

    def trace_chat_completion(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        trace.set_tag('component', 'LLM')
        trace.set_tag('endpoint', f'{self._api_base}/chat/completions')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
            'model',
            'max_tokens',
            'temperature',
            'top_p',
            'n',
            'stream',
            'logprobs',
            'stop',
            'presence_penalty',
            'frequency_penalty',
            'best_of'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'stream' in params and params['stream']:
            if 'messages' in params:
                trace.set_data('messages', params['messages'])
            return

        if ret and 'model' in ret:
            trace.set_tag('model', ret['model'])

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

        if 'messages' in params:
            trace.set_data('messages', params['messages'], counts=prompt_usage)

        if ret:
            if 'choices' in ret:
                for choice in ret['choices']:
                    if 'finish_reason' in choice:
                        if choice['finish_reason'] == 'stop':
                            completion_usage['finish_reason_stop'] += 1
                        elif choice['finish_reason'] == 'length':
                            completion_usage['finish_reason_length'] += 1
            trace.set_data('completion', ret, counts=completion_usage)

    def trace_chat_completion_data(self, trace, item):
        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0
        }
        if item and 'choices' in item:
            completion = []
            for choice in item['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                if 'delta' in choice and 'content' in choice['delta']:
                    completion.append(choice['delta']['content'])

            trace.append_data('completion', completion, counts=completion_usage)

    def trace_edits(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        trace.set_tag('component', 'LLM')
        trace.set_tag('endpoint', f'{self._api_base}/edits')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'top_p',
            'n'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        prompt_usage = {}
        completion_usage = {}
        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'input' in params:
            trace.set_data('input', params['input'])

        if 'instruction' in params:
            trace.set_data('instruction', params['instruction'], counts=prompt_usage)

        if ret:
            trace.set_data('edits', ret, counts=completion_usage)

    def trace_embedding(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        trace.set_tag('component', 'LLM')
        trace.set_tag('endpoint', f'{self._api_base}/embeddings')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
            'model'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        prompt_usage = {}
        if 'model' in ret:
            trace.set_tag('model', ret['model'])

        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']

        if 'input' in params:
            trace.set_data('input', params['input'], counts=prompt_usage)

        if ret:
            trace.set_data('embedding', ret)

    def trace_image_generation(self, trace, args, kwargs, ret, exc):
        trace.set_tag('endpoint', f'{self._api_base}/images/generations')
        self.trace_image_endpoint(trace, args, kwargs, ret, exc)

    def trace_image_variation(self, trace, args, kwargs, ret, exc):
        trace.set_tag('endpoint', f'{self._api_base}/images/variations')
        self.trace_image_endpoint(trace, args, kwargs, ret, exc)

    def trace_image_edit(self, trace, args, kwargs, ret, exc):
        trace.set_tag('endpoint', f'{self._api_base}/images/edits')
        self.trace_image_endpoint(trace, args, kwargs, ret, exc)

    def trace_image_endpoint(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        trace.set_tag('component', 'Model')

        param_names = [
            'n',
            'size',
            'response_format'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'prompt' in params:
            trace.set_data('prompt', params['prompt'])

        if ret and 'data' in ret:
            image_data = []
            for image in ret['data']:
                if 'url' in image:
                    image_data.append(image['url'])
                elif 'b64_json' in image:
                    image_data.append(image['b64_json'])
            trace.set_data('image', image_data)

    def trace_audio_transcription(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['model', 'file', 'prompt', 'response_format', 'temperature', 'language'])

        trace.set_tag('component', 'Model')
        trace.set_tag('endpoint', f'{self._api_base}/audio/transcriptions')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'response_format',
            'language'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'file' in params and hasattr(params['file'], 'name'):
            try:
                file_size = os.path.getsize(params['file'].name)
                trace.set_data('file', params['file'], counts={'byte_count': file_size})
            except OSError:
                pass

        if 'prompt' in params:
            trace.set_data('prompt', params['prompt'])

        if ret and 'text' in ret:
            trace.set_data('text', ret['text'])

    def trace_audio_translation(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['model', 'file', 'prompt', 'response_format', 'temperature'])

        trace.set_tag('component', 'Model')
        trace.set_tag('endpoint', f'{self._api_base}/audio/translations')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'response_format'
        ]
        for param_name in param_names:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'file' in params and hasattr(params['file'], 'name'):
            try:
                file_size = os.path.getsize(params['file'].name)
                trace.set_data('file', params['file'], counts={'byte_count': file_size})
            except OSError:
                pass

        if 'prompt' in params:
            trace.set_data('prompt', params['prompt'])

        if ret and 'text' in ret:
            trace.set_data('text', ret['text'])

    def trace_moderation(self, trace, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['input', 'model'])

        trace.set_tag('component', 'Model')
        trace.set_tag('endpoint', f'{self._api_base}/moderations')

        if 'model' in params:
            trace.set_tag('model', params['model'])

        for param_name in params:
            if param_name in params:
                trace.set_param(param_name, params[param_name])

        if 'input' in params:
            trace.set_data('input', params['input'])

    def on_trace_start(self, proto, context, options):
        pass

    def on_trace_stop(self, proto, context, options):
        pass

    def on_trace_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)

    def on_metric_update(self):
        pass
