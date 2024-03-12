import logging
import os
import copy
import importlib
import openai

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, patch_method, parse_semver, compare_semver

logger = logging.getLogger('graphsignal')

class OpenAIRecorder(BaseRecorder):
    def __init__(self):
        self._library_version = None
        self._base_url = None
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

        version = openai.version.VERSION
        self._library_version = version
        parsed_version = parse_semver(version)

        if compare_semver(parsed_version, (1, 0, 0)) >= 0:
            def after_init(args, kwargs, ret, exc, context):
                client = args[0]
                self._base_url = client.base_url

                instrument_method(client.chat.completions, 'create', 'openai.chat.completions.create', trace_func=self.trace_chat_completion, data_func=self.trace_chat_completion_data)
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
        else:
            logger.debug('OpenAI tracing is only supported for >= 1.0.0.')
            return


    def shutdown(self):
        pass

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

    def set_common_tags(self, span, endpoint):
        span.set_tag('library', 'openai')
        if 'openai.com' in str(self._base_url):
            span.set_tag('api_provider', 'openai')
        if 'azure.com' in str(self._base_url):
            span.set_tag('api_provider', 'azure')
        span.set_tag('endpoint', f'{self._base_url}{endpoint}')

    def trace_chat_completion(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        span.set_tag('model_type', 'chat')
        self.set_common_tags(span, 'chat/completions')
        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'messages',
            'model',
            'function_call',
            'functions',
            'tools',
            'max_tokens',
            'temperature',
            'top_logprobs',
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
            'seed',
            'user',
            'extra_headers',
            'extra_query',
            'extra_body',
            'timeout'
        ]

        input_data = {}
        input_usage = {
            'token_count': 0
        }
        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]
        if 'stream' in params and params['stream']:
            if 'messages' in params:
                if 'model' in params and not exc:
                    # Based on token counting example
                    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
                    model = params['model']
                    for message in params['messages']:
                        if 'content' in message:
                            prompt_tokens = self.count_tokens(model, message['content'])
                            if prompt_tokens:
                                input_usage['token_count'] += prompt_tokens
                                input_usage['token_count'] += self._extra_content_tokens.get(model, 0)
                        if 'role' in message:
                            prompt_tokens = self.count_tokens(model, message['role'])
                            if prompt_tokens:
                                input_usage['token_count'] += prompt_tokens
                        if 'name' in message:
                            prompt_tokens = self.count_tokens(model, message['name'])
                            if prompt_tokens:
                                input_usage['token_count'] += prompt_tokens
                                input_usage['token_count'] += self._extra_name_tokens.get(model, 0)
                    if input_usage['token_count'] > 0:
                        input_usage['token_count'] += self._extra_reply_tokens.get(model, 0)

            span.set_payload('input', input_data, usage=input_usage)
        else:
            if ret and 'model' in ret:
                span.set_tag('model', ret['model'])

            ret = ret.model_dump()

            input_data = {}
            input_usage = {}
            output_usage = {
                'token_count': 0
            }
            if ret and 'usage' in ret and not exc:
                if 'prompt_tokens' in ret['usage']:
                    input_usage['token_count'] = ret['usage']['prompt_tokens']
                if 'completion_tokens' in ret['usage']:
                    output_usage['token_count'] = ret['usage']['completion_tokens']

            span.set_payload('input', input_data, usage=input_usage)

            if ret:
                span.set_payload('output', ret, usage=output_usage)

    def trace_chat_completion_data(self, span, item):
        item = item.model_dump()

        output_usage = {
            'token_count': 0
        }
        if item and 'choices' in item:
            for _ in item['choices']:
                output_usage['token_count'] += 1

            span.append_payload('output', [item], usage=output_usage)

    def trace_embedding(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        self.set_common_tags(span, 'embeddings')
        if 'model' in params:
            span.set_tag('model', params['model'])

        input_data = {}
        input_usage = {}

        param_names = [
            'model',
            'input',
            'encoding_format',
            'dimensions',
            'user',
            'extra_headers',
            'extra_query',
            'extra_body'
        ]
        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]

        ret = ret.model_dump()

        if ret and 'model' in ret:
            span.set_tag('model', ret['model'])

        if ret and 'usage' in ret:
            if 'prompt_tokens' in ret['usage']:
                input_usage['token_count'] = ret['usage']['prompt_tokens']

        span.set_payload('input', input_data, usage=input_usage)

    def trace_image_generation(self, span, args, kwargs, ret, exc):
        self.set_common_tags(span, 'images/generations')

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_variation(self, span, args, kwargs, ret, exc):
        self.set_common_tags(span, 'images/variations')

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_edit(self, span, args, kwargs, ret, exc):
        self.set_common_tags(span, 'images/edits')

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_endpoint(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        param_names = [
            'prompt',
            'n',
            'size',
            'response_format',
            'extra_headers',
            'extra_query',
            'extra_body'
        ]

        input_data = {}

        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]

        span.set_payload('input', input_data)

        if ret and 'data' in ret:
            image_data = []
            for image in ret['data']:
                if 'url' in image:
                    image_data.append(image['url'])
                elif 'b64_json' in image:
                    image_data.append(image['b64_json'])
            span.set_payload('output', image_data)

    def trace_audio_transcription(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        self.set_common_tags(span, 'audio/transcriptions')
        if 'model' in params:
            span.set_tag('model', params['model'])

        param_names = [
            'prompt',
            'file',
            'model',
            'temperature',
            'response_format',
            'language',
            'extra_headers',
            'extra_query',
            'extra_body'
        ]

        input_data = {}

        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]

        span.set_payload('input', input_data)

        ret = ret.model_dump()

        if ret and 'text' in ret:
            span.set_payload('output', ret['text'])

    def trace_audio_translation(self, span, args, kwargs, ret, exc):
        params = kwargs # no positional args

        self.set_common_tags(span, 'audio/translations')
        if 'model' in params:
            span.set_tag('model', params['model'])

        input_data = {}

        param_names = [
            'prompt',
            'file',
            'model',
            'temperature',
            'response_format',
            'extra_headers',
            'extra_query',
            'extra_body'
        ]
        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]

        span.set_payload('input', input_data)

        ret = ret.model_dump()

        if ret and 'text' in ret:
            span.set_payload('input', ret['text'])

    def trace_moderation(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'moderations')
        if 'model' in params:
            span.set_tag('model', params['model'])

        input_data = {}

        param_names = [
            'input',
            'model',
            'extra_headers',
            'extra_query',
            'extra_body'
        ]
        for param_name in param_names:
            if param_name in params:
                input_data[param_name] = params[param_name]

        span.set_payload('input', input_data)

    def on_span_read(self, span, context):
        if self._library_version:
            entry = span.config.add()
            entry.key = 'openai.library.version'
            entry.value = self._library_version