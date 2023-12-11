import logging
import os
import copy
import importlib
import openai

import graphsignal
from graphsignal.spans import TraceOptions
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, uninstrument_method, read_args, patch_method
from graphsignal.proto_utils import parse_semver, compare_semver
from graphsignal.proto import signals_pb2

logger = logging.getLogger('graphsignal')

class OpenAIRecorder(BaseRecorder):
    def __init__(self):
        self._library = None
        self._base_url = None
        self._openai_version_0_26_0 = False
        self._openai_version_1_0_0 = False
        self._cached_encodings = {}
        self._extra_content_tokens = {
            'gpt-3.5-turbo': 3,
            'gpt-3.5-turbo-0301': 4,
            'gpt-4': 3,
            'gpt-4-0314': 3,
            'gpt-4-32k-0314': 3,
            'gpt-3.5-turbo-0613': 3,
            'gpt-3.5-turbo-16k-0613': 3,
            'gpt-4-0613': 3,
            'gpt-4-32k-0613': 3
        }
        self._extra_name_tokens = {
            'gpt-3.5-turbo': 1,
            'gpt-3.5-turbo-0301': -1,
            'gpt-4': 1,
            'gpt-4-0314': 1,
            'gpt-4-32k-0314': 1,
            'gpt-3.5-turbo-0613': 1,
            'gpt-3.5-turbo-16k-0613': 1,
            'gpt-4-0613': 1,
            'gpt-4-32k-0613': 1
        }
        self._extra_reply_tokens = {
            'gpt-3.5-turbo': 3,
            'gpt-3.5-turbo-0301': 3,
            'gpt-4': 3,
            'gpt-4-0314': 3,
            'gpt-4-32k-0314': 3,
            'gpt-3.5-turbo-0613': 3,
            'gpt-3.5-turbo-16k-0613': 3,
            'gpt-4-0613': 3,
            'gpt-4-32k-0613': 3
        }

    def setup(self):
        if not graphsignal._tracer.auto_instrument:
            return

        self._library = signals_pb2.LibraryInfo()
        self._library.name = 'OpenAI Python Library'
        parse_semver(self._library.version, openai.version.VERSION)

        if compare_semver(self._library.version, (1, 0, 0)) >= 0:
            self._openai_version_1_0_0 = True

            def after_init(args, kwargs, ret, exc, context):
                client = args[0]
                self._base_url = client.base_url

                instrument_method(client.completions, 'create', 'openai.completions.create', trace_func=self.trace_completion, data_func=self.trace_completion_data)
                instrument_method(client.chat.completions, 'create', 'openai.chat.completions.create', trace_func=self.trace_chat_completion, data_func=self.trace_chat_completion_data)
                instrument_method(client.edits, 'create', 'openai.edits.create', trace_func=self.trace_edits)
                instrument_method(client.embeddings, 'create', 'openai.embeddings.create', trace_func=self.trace_embedding)
                instrument_method(client.images, 'create', 'openai.images.create', trace_func=self.trace_image_generation)
                instrument_method(client.images, 'create_variation', 'openai.images.create_variation', trace_func=self.trace_image_variation)
                instrument_method(client.images, 'edit', 'openai.images.edit', trace_func=self.trace_image_edit)
                instrument_method(client.audio.transcriptions, 'create', 'openai.audio.transcriptions.create', trace_func=self.trace_audio_transcription)
                instrument_method(client.audio.translations, 'create', 'openai.audio.translations.create', trace_func=self.trace_audio_translation)
                instrument_method(client.moderations, 'create', 'openai.moderations.create', trace_func=self.trace_moderation)

            patch_method(openai.OpenAI, '__init__', after_func=after_init)
            patch_method(openai.AsyncOpenAI, '__init__', after_func=after_init)
            patch_method(openai.AzureOpenAI, '__init__', after_func=after_init)
            patch_method(openai.AsyncAzureOpenAI, '__init__', after_func=after_init)

        elif compare_semver(self._library.version, (0, 26, 0)) >= 0:
            self._openai_version_0_26_0 = True
            self._base_url = openai.api_base + '/'

            instrument_method(openai.Completion, 'create', 'openai.Completion.create', trace_func=self.trace_completion, data_func=self.trace_completion_data)
            instrument_method(openai.Completion, 'acreate', 'openai.Completion.acreate', trace_func=self.trace_completion, data_func=self.trace_completion_data)
            instrument_method(openai.ChatCompletion, 'create', 'openai.ChatCompletion.create', trace_func=self.trace_chat_completion, data_func=self.trace_chat_completion_data)
            instrument_method(openai.ChatCompletion, 'acreate', 'openai.ChatCompletion.acreate', trace_func=self.trace_chat_completion, data_func=self.trace_chat_completion_data)
            instrument_method(openai.Edit, 'create', 'openai.Edit.create', trace_func=self.trace_edits)
            instrument_method(openai.Edit, 'acreate', 'openai.Edit.acreate', trace_func=self.trace_edits)
            instrument_method(openai.Embedding, 'create', 'openai.Embedding.create', trace_func=self.trace_embedding)
            instrument_method(openai.Embedding, 'acreate', 'openai.Embedding.acreate', trace_func=self.trace_embedding)
            instrument_method(openai.Image, 'create', 'openai.Image.create', trace_func=self.trace_image_generation)
            instrument_method(openai.Image, 'acreate', 'openai.Image.acreate', trace_func=self.trace_image_generation)
            instrument_method(openai.Image, 'create_variation', 'openai.Image.create_variation', trace_func=self.trace_image_variation)
            instrument_method(openai.Image, 'acreate_variation', 'openai.Image.acreate_variation', trace_func=self.trace_image_variation)
            instrument_method(openai.Image, 'create_edit', 'openai.Image.create_edit', trace_func=self.trace_image_edit)
            instrument_method(openai.Image, 'acreate_edit', 'openai.Image.acreate_edit', trace_func=self.trace_image_edit)
            instrument_method(openai.Audio, 'transcribe', 'openai.Audio.transcribe', trace_func=self.trace_audio_transcription)
            instrument_method(openai.Audio, 'atranscribe', 'openai.Audio.atranscribe', trace_func=self.trace_audio_transcription)
            instrument_method(openai.Audio, 'translate', 'openai.Audio.translate', trace_func=self.trace_audio_translation)
            instrument_method(openai.Audio, 'atranslate', 'openai.Audio.atranslate', trace_func=self.trace_audio_translation)
            instrument_method(openai.Moderation, 'create', 'openai.Moderation.create', trace_func=self.trace_moderation)
            instrument_method(openai.Moderation, 'acreate', 'openai.Moderation.acreate', trace_func=self.trace_moderation)

        else:
            logger.debug('OpenAI tracing is only supported for >= 0.26.0.')
            return


    def shutdown(self):
        if compare_semver(self._library.version, (1, 0, 0)) >= 0:
            pass
        elif compare_semver(self._library.version, (0, 26, 0)) >= 0:
            uninstrument_method(openai.Completion, 'create')
            uninstrument_method(openai.Completion, 'acreate')
            uninstrument_method(openai.ChatCompletion, 'create')
            uninstrument_method(openai.ChatCompletion, 'acreate')
            uninstrument_method(openai.Edit, 'create')
            uninstrument_method(openai.Edit, 'acreate')
            uninstrument_method(openai.Embedding, 'create')
            uninstrument_method(openai.Embedding, 'acreate')
            uninstrument_method(openai.Image, 'create')
            uninstrument_method(openai.Image, 'acreate')
            uninstrument_method(openai.Image, 'create_variation')
            uninstrument_method(openai.Image, 'acreate_variation')
            uninstrument_method(openai.Image, 'create_edit')
            uninstrument_method(openai.Image, 'acreate_edit')
            uninstrument_method(openai.Audio, 'transcribe')
            uninstrument_method(openai.Audio, 'atranscribe')
            uninstrument_method(openai.Audio, 'translate')
            uninstrument_method(openai.Audio, 'atranslate')
            uninstrument_method(openai.Moderation, 'create')
            uninstrument_method(openai.Moderation, 'acreate')

    def count_tokens(self, model, text):
        if model not in self._cached_encodings:
            try:
                tiktoken = importlib.import_module('tiktoken')
                encoding = tiktoken.encoding_for_model(model)
                self._cached_encodings[model] = encoding
                if encoding:
                    logger.debug('Cached encoding for model %s', model)
                else:
                    logger.debug('No encoding returned for model %s', model)
            except ModuleNotFoundError:
                self._cached_encodings[model] = None
                logger.debug('tiktoken not installed, will not count OpenAI stream tokens.')
            except Exception:
                self._cached_encodings[model] = None
                logger.error('Error using tiktoken for model %s', model, exc_info=True)

        encoding = self._cached_encodings.get(model, None)
        if encoding:
            return len(encoding.encode(text))
        return None

    def trace_completion(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('component', 'LLM')
        span.set_tag('endpoint', f'{self._base_url}completions')

        if 'model' in params:
            span.set_tag('model', params['model'])

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
            'best_of',
            'seed'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'stream' in params and params['stream']:
            if 'model' in params and 'prompt' in params:
                prompt_usage = {
                    'token_count': 0
                }
                if not exc:
                    if isinstance(params['prompt'], str):
                        prompt_tokens = self.count_tokens(params['model'], params['prompt'])
                        if prompt_tokens:
                            prompt_usage['token_count'] = prompt_tokens
                    elif isinstance(params['prompt'], list):
                        for prompt in params['prompt']:
                            prompt_tokens = self.count_tokens(params['model'], prompt)
                            if prompt_tokens:
                                prompt_usage['token_count'] += prompt_tokens
                span.set_data('prompt', params['prompt'], counts=prompt_usage)
            return

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        prompt_usage = {}
        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0
        }
        if ret and 'usage' in ret and not exc:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'prompt' in params:
            span.set_data('prompt', params['prompt'], counts=prompt_usage)

        if ret:
            if 'choices' in ret:
                for choice in ret['choices']:
                    if 'finish_reason' in choice:
                        if choice['finish_reason'] == 'stop':
                            completion_usage['finish_reason_stop'] += 1
                        elif choice['finish_reason'] == 'length':
                            completion_usage['finish_reason_length'] += 1
            span.set_data('completion', ret, counts=completion_usage)

    def trace_completion_data(self, span, item):
        if self._openai_version_1_0_0:
            item = item.model_dump()

        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0,
            'token_count': 0
        }
        if item and 'choices' in item:
            for choice in item['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                completion_usage['token_count'] += 1

            span.append_data('completion', [item], counts=completion_usage)

    def trace_chat_completion(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('component', 'LLM')
        span.set_tag('endpoint', f'{self._base_url}chat/completions')

        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'model',
            'function_call',
            'max_tokens',
            'temperature',
            'top_p',
            'n',
            'stream',
            'logprobs',
            'stop',
            'presence_penalty',
            'frequency_penalty',
            'best_of',
            'response_format',
            'tool_choice',
            'seed'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'stream' in params and params['stream']:
            if 'model' in params and 'messages' in params:
                prompt_usage = {
                    'token_count': 0
                }
                if not exc:
                    # Based on token counting example
                    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
                    model = params['model']
                    for message in params['messages']:
                        if 'content' in message:
                            prompt_tokens = self.count_tokens(model, message['content'])
                            if prompt_tokens:
                                prompt_usage['token_count'] += prompt_tokens
                                prompt_usage['token_count'] += self._extra_content_tokens.get(model, 0)
                        if 'role' in message:
                            prompt_tokens = self.count_tokens(model, message['role'])
                            if prompt_tokens:
                                prompt_usage['token_count'] += prompt_tokens
                        if 'name' in message:
                            prompt_tokens = self.count_tokens(model, message['name'])
                            if prompt_tokens:
                                prompt_usage['token_count'] += prompt_tokens
                                prompt_usage['token_count'] += self._extra_name_tokens.get(model, 0)
                    if prompt_usage['token_count'] > 0:
                        prompt_usage['token_count'] += self._extra_reply_tokens.get(model, 0)
                span.set_data('messages', params['messages'], counts=prompt_usage)
            return

        if ret and 'model' in ret:
            span.set_tag('model', ret['model'])

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        prompt_usage = {}
        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0,
            'finish_reason_function_call': 0
        }
        if ret and 'usage' in ret and not exc:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'messages' in params:
            span.set_data('messages', params['messages'], counts=prompt_usage)

        if 'functions' in params:
            span.set_data('functions', params['functions'])

        if 'tools' in params:
            span.set_data('tools', params['tools'])

        if ret:
            if 'choices' in ret:
                for choice in ret['choices']:
                    if 'finish_reason' in choice:
                        if choice['finish_reason'] == 'stop':
                            completion_usage['finish_reason_stop'] += 1
                        elif choice['finish_reason'] == 'length':
                            completion_usage['finish_reason_length'] += 1
                        elif choice['finish_reason'] == 'function_call':
                            completion_usage['finish_reason_function_call'] += 1
            span.set_data('completion', ret, counts=completion_usage)

    def trace_chat_completion_data(self, span, item):
        if self._openai_version_1_0_0:
            item = item.model_dump()

        completion_usage = {
            'finish_reason_stop': 0,
            'finish_reason_length': 0,
            'token_count': 0
        }
        if item and 'choices' in item:
            for choice in item['choices']:
                if 'finish_reason' in choice:
                    if choice['finish_reason'] == 'stop':
                        completion_usage['finish_reason_stop'] += 1
                    elif choice['finish_reason'] == 'length':
                        completion_usage['finish_reason_length'] += 1
                completion_usage['token_count'] += 1

            span.append_data('completion', [item], counts=completion_usage)

    def trace_edits(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('component', 'LLM')
        span.set_tag('endpoint', f'{self._base_url}edits')

        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'top_p',
            'n'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        prompt_usage = {}
        completion_usage = {}
        if 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']
            if 'completion_tokens' in ret['usage']:
                completion_usage['token_count'] = ret['usage']['completion_tokens']

        if 'input' in params:
            span.set_data('input', params['input'])

        if 'instruction' in params:
            span.set_data('instruction', params['instruction'], counts=prompt_usage)

        if ret:
            span.set_data('edits', ret, counts=completion_usage)

    def trace_embedding(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('component', 'LLM')
        span.set_tag('endpoint', f'{self._base_url}embeddings')

        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'model'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        prompt_usage = {}
        if ret and 'model' in ret:
            span.set_tag('model', ret['model'])

        if ret and 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                prompt_usage['token_count'] = ret['usage']['prompt_tokens']

        if 'input' in params:
            span.set_data('input', params['input'], counts=prompt_usage)

        if ret:
            if 'data' in ret and isinstance(ret['data'], list):
                ret = copy.deepcopy(ret)
                for item in ret['data']:
                    if 'embedding' in item and isinstance(item['embedding'], list):
                        size = len(item['embedding'])
                        item['embedding'] = f'({size} floats)'
                span.set_data('embeddings', ret)

    def trace_image_generation(self, span, args, kwargs, ret, exc):
        span.set_tag('endpoint', f'{self._base_url}images/generations')
        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_variation(self, span, args, kwargs, ret, exc):
        span.set_tag('endpoint', f'{self._base_url}images/variations')
        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_edit(self, span, args, kwargs, ret, exc):
        span.set_tag('endpoint', f'{self._base_url}images/edits')
        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_endpoint(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('component', 'Model')

        param_names = [
            'n',
            'size',
            'response_format'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'prompt' in params:
            span.set_data('prompt', params['prompt'])

        if ret and 'data' in ret:
            image_data = []
            for image in ret['data']:
                if 'url' in image:
                    image_data.append(image['url'])
                elif 'b64_json' in image:
                    image_data.append(image['b64_json'])
            span.set_data('image', image_data)

    def trace_audio_transcription(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['model', 'file', 'prompt', 'response_format', 'temperature', 'language'])

        span.set_tag('component', 'Model')
        span.set_tag('endpoint', f'{self._base_url}audio/transcriptions')

        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'response_format',
            'language'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'file' in params and hasattr(params['file'], 'name'):
            try:
                file_size = os.path.getsize(params['file'].name)
                span.set_data('file', params['file'], counts={'byte_count': file_size})
            except OSError:
                pass

        if 'prompt' in params:
            span.set_data('prompt', params['prompt'])

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        if ret and 'text' in ret:
            span.set_data('text', ret['text'])

    def trace_audio_translation(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['model', 'file', 'prompt', 'response_format', 'temperature'])

        span.set_tag('component', 'Model')
        span.set_tag('endpoint', f'{self._base_url}audio/translations')

        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'model',
            'temperature',
            'response_format'
        ]
        for param_name in param_names:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'file' in params and hasattr(params['file'], 'name'):
            try:
                file_size = os.path.getsize(params['file'].name)
                span.set_data('file', params['file'], counts={'byte_count': file_size})
            except OSError:
                pass

        if 'prompt' in params:
            span.set_data('prompt', params['prompt'])

        if self._openai_version_1_0_0:
            ret = ret.model_dump()

        if ret and 'text' in ret:
            span.set_data('text', ret['text'])

    def trace_moderation(self, span, args, kwargs, ret, exc):
        params = read_args(args, kwargs, ['input', 'model'])

        span.set_tag('component', 'Model')
        span.set_tag('endpoint', f'{self._base_url}moderations')

        if 'model' in params:
            span.set_tag('model', params['model'])

        for param_name in params:
            if param_name in params:
                span.set_param(param_name, params[param_name])

        if 'input' in params:
            span.set_data('input', params['input'])

    def on_span_read(self, proto, context, options):
        if self._library:
            proto.libraries.append(self._library)
