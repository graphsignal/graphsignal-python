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

        self._api_base = openai.api_base

        instrument_method(openai.Completion, 'create', f'{self._api_base}/completions', self.trace_completion, self.trace_completion_data)
        instrument_method(openai.Completion, 'acreate', f'{self._api_base}/completions', self.trace_completion, self.trace_completion_data)
        instrument_method(openai.ChatCompletion, 'create', f'{self._api_base}/chat/completions', self.trace_chat_completion, self.trace_chat_completion_data)
        instrument_method(openai.ChatCompletion, 'acreate', f'{self._api_base}/chat/completions', self.trace_chat_completion, self.trace_chat_completion_data)
        instrument_method(openai.Edit, 'create', f'{self._api_base}/edits', self.trace_edits)
        instrument_method(openai.Edit, 'acreate', f'{self._api_base}/edits', self.trace_edits)
        instrument_method(openai.Embedding, 'create', f'{self._api_base}/embeddings', self.trace_embedding)
        instrument_method(openai.Embedding, 'acreate', f'{self._api_base}/embeddings', self.trace_embedding)
        instrument_method(openai.Image, 'create', f'{self._api_base}/images/generations', self.trace_image_generation)
        instrument_method(openai.Image, 'acreate', f'{self._api_base}/images/generations', self.trace_image_generation)
        instrument_method(openai.Image, 'create_variation', f'{self._api_base}/images/variations', self.trace_image_generation)
        instrument_method(openai.Image, 'acreate_variation', f'{self._api_base}/images/variations', self.trace_image_generation)
        instrument_method(openai.Image, 'create_edit', f'{self._api_base}/images/edits', self.trace_image_generation)
        instrument_method(openai.Image, 'acreate_edit', f'{self._api_base}/images/edits', self.trace_image_generation)
        instrument_method(openai.Audio, 'transcribe', f'{self._api_base}/audio/transcriptions', self.trace_audio_transcription)
        instrument_method(openai.Audio, 'atranscribe', f'{self._api_base}/audio/transcriptions', self.trace_audio_transcription)
        instrument_method(openai.Audio, 'translate', f'{self._api_base}/audio/translations', self.trace_audio_translation)
        instrument_method(openai.Audio, 'atranslate', f'{self._api_base}/audio/translations', self.trace_audio_translation)
        instrument_method(openai.Moderation, 'create', f'{self._api_base}/moderations', self.trace_moderation)
        instrument_method(openai.Moderation, 'acreate', f'{self._api_base}/moderations', self.trace_moderation)

    def shutdown(self):
        uninstrument_method(openai.Completion, 'create', f'{self._api_base}/completions')
        uninstrument_method(openai.Completion, 'acreate', f'{self._api_base}/completions')
        uninstrument_method(openai.ChatCompletion, 'create', f'{self._api_base}/chat/completions')
        uninstrument_method(openai.ChatCompletion, 'acreate', 'openai.ChatCompletion.acreate')
        uninstrument_method(openai.Edit, 'create', f'{self._api_base}/edits')
        uninstrument_method(openai.Edit, 'acreate', f'{self._api_base}/edits')
        uninstrument_method(openai.Embedding, 'create', f'{self._api_base}/embeddings')
        uninstrument_method(openai.Embedding, 'acreate', f'{self._api_base}/embeddings')
        uninstrument_method(openai.Image, 'create', f'{self._api_base}/images/generations')
        uninstrument_method(openai.Image, 'acreate', f'{self._api_base}/images/generations')
        uninstrument_method(openai.Image, 'create_variation', f'{self._api_base}/images/variations')
        uninstrument_method(openai.Image, 'acreate_variation', f'{self._api_base}/images/variations')
        uninstrument_method(openai.Image, 'create_edit', f'{self._api_base}/images/edits')
        uninstrument_method(openai.Image, 'acreate_edit', f'{self._api_base}/images/edits')
        uninstrument_method(openai.Audio, 'transcribe', f'{self._api_base}/audio/transcriptions')
        uninstrument_method(openai.Audio, 'atranscribe', f'{self._api_base}/audio/transcriptions')
        uninstrument_method(openai.Audio, 'translate', f'{self._api_base}/audio/translations')
        uninstrument_method(openai.Audio, 'atranslate', f'{self._api_base}/audio/translations')
        uninstrument_method(openai.Moderation, 'create', f'{self._api_base}/moderations')
        uninstrument_method(openai.Moderation, 'acreate', f'{self._api_base}/moderations')

    def trace_completion(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
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
            trace.set_data('completion', completion, counts=completion_usage)

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

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
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

        if ret and 'choices' in ret:
            completion = []
            for choice in ret['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                if 'message' in choice and 'content' in choice['message']:
                    completion.append(choice['message']['content'])
            trace.set_data('completion', completion, counts=completion_usage)

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

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
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

        if ret and 'choices' in ret:
            edits = []
            for choice in ret['choices']:
                if 'text' in choice:
                    edits.append(choice['text'])
            trace.set_data('edits', edits, counts=completion_usage)

    def trace_embedding(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        if 'engine' in params:
            trace.set_tag('model', params['engine'])

        if self._is_sampling:
            param_names = [
                'engine'
            ]
            for param_name in param_names:
                if param_name in params:
                    trace.set_param(param_name, params[param_name])

        prompt_usage = {}
        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']

        if 'input' in params:
            trace.set_data('input', params['input'], counts=prompt_usage)

        if ret and 'data' in ret:
            embedding = []
            for choice in ret['data']:
                if 'embedding' in choice:
                    embedding.append(choice['embedding'])
            trace.set_data('embedding', embedding)

    def trace_image_generation(self, trace, args, kwargs, ret, exc):
        params = kwargs # no positional args

        if self._is_sampling:
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

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
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

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
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

        if 'model' in params:
            trace.set_tag('model', params['model'])

        if self._is_sampling:
            for param_name in params:
                if param_name in params:
                    trace.set_param(param_name, params[param_name])

        if 'input' in params:
            trace.set_data('input', params['input'])

    def on_trace_start(self, proto, context, options):
        self._is_sampling = True

    def on_trace_stop(self, proto, context, options):
        self._is_sampling = False

    def on_trace_read(self, proto, context, options):
        if self._framework:
            proto.frameworks.append(self._framework)

    def on_metric_update(self):
        pass
