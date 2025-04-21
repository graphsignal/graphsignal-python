import logging
import os
import copy
import importlib
import openai

import graphsignal
from graphsignal.recorders.base_recorder import BaseRecorder
from graphsignal.recorders.instrumentation import instrument_method, patch_method, parse_semver, compare_semver
from graphsignal import client

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

    def set_common_tags(self, span, endpoint, params):
        span.set_param('openai_version', self._library_version)
        if 'openai.com' in str(self._base_url):
            span.set_tag('api_provider', 'openai')
        if 'azure.com' in str(self._base_url):
            span.set_tag('api_provider', 'azure')
        span.set_tag('endpoint', f'{self._base_url}{endpoint}')

        # header tags
        try:
            tags_header_value = None
            if 'extra_headers' in params and isinstance(params['extra_headers'], dict):
                for header_key, header_value in params['extra_headers'].items():
                    if header_key.lower() == 'graphsignal-tags':
                        tags_header_value = header_value
                        del params['extra_headers'][header_key]
                        break

            if isinstance(tags_header_value, str):
                header_tags = dict([el.strip(' ') for el in kv.split('=')] for kv in tags_header_value.split(','))
                for tag_key, tag_value in header_tags.items():
                    span.set_tag(tag_key, tag_value)
        except Exception:
            logger.error('Error parsing Graphsignal-Tags extra header', exc_info=True)

        # user tag
        if 'user' in params:
            if not graphsignal.get_tag('user_id') and not graphsignal.get_context_tag('user_id') and not span.get_tag('user_id'):
                span.set_tag('user_id', params['user'])

    def read_usage(self, span, usage):
        if not usage:
            return

        def set_usage(span, obj, key, new_key=None):
            if not isinstance(obj, dict):
                return 0

            if key in obj:
                value = obj[key]
                if isinstance(value, int):
                    if new_key:
                        span.set_counter(new_key, value)
                        span.inc_counter_metric('usage', new_key, value)
                    else:
                        span.set_counter(key, value)
                        span.inc_counter_metric('usage', key, value)
                    return value
            return None

        set_usage(span, usage, 'total_tokens')
        prompt_tokens = set_usage(span, usage, 'prompt_tokens')
        cached_prompt_tokens = None
        set_usage(span, usage, 'completion_tokens')
        if 'completion_tokens' in usage:
            span.set_counter('output_tokens', usage['completion_tokens'])
        if 'prompt_tokens_details' in usage:
            prompt_tokens_details = usage['prompt_tokens_details']
            set_usage(span, prompt_tokens_details, 'audio_tokens', 'prompt_audio_tokens')
            cached_prompt_tokens = set_usage(span, prompt_tokens_details, 'cached_tokens', 'cached_prompt_tokens')
        if 'completion_tokens_details' in usage:
            completion_tokens_details = usage['completion_tokens_details']
            set_usage(span, completion_tokens_details, 'audio_tokens', 'completion_audio_tokens')
            set_usage(span, completion_tokens_details, 'reasoning_tokens')
            set_usage(span, completion_tokens_details, 'accepted_prediction_tokens')
            set_usage(span, completion_tokens_details, 'rejected_prediction_tokens')

        if prompt_tokens:
            if cached_prompt_tokens and cached_prompt_tokens > 0:
                uncached_prompt_tokens = prompt_tokens - cached_prompt_tokens
                span.set_counter('uncached_prompt_tokens', uncached_prompt_tokens)
                span.inc_counter_metric('usage', 'uncached_prompt_tokens', uncached_prompt_tokens)
            else:
                span.set_counter('uncached_prompt_tokens', prompt_tokens)
                span.inc_counter_metric('usage', 'uncached_prompt_tokens', prompt_tokens)

    def trace_chat_completion(self, span, args, kwargs, ret, exc):
        params = kwargs

        span.set_tag('model_type', 'chat')
        self.set_common_tags(span, 'chat/completions', params)
        if 'model' in params:
            span.set_tag('model', params['model'])
            span.set_param('model', params['model'])
        if 'reasoning_effort' in params:
            span.set_param('reasoning_effort', params['reasoning_effort'])
        if 'max_completion_tokens' in params:
            span.set_param('max_completion_tokens', params['max_completion_tokens'])

        if 'stream' in params and params['stream']:
            if 'messages' in params:
                if 'model' in params and not exc:
                    # Based on token counting example
                    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
                    model = params['model']
                    total_prompt_tokens = 0
                    for message in params['messages']:
                        if 'content' in message:
                            prompt_tokens = self.count_tokens(model, message['content'])
                            if prompt_tokens:
                                total_prompt_tokens += prompt_tokens
                                total_prompt_tokens += self._extra_content_tokens.get(model, 0)
                        if 'role' in message:
                            prompt_tokens = self.count_tokens(model, message['role'])
                            if prompt_tokens:
                                total_prompt_tokens += prompt_tokens
                        if 'name' in message:
                            prompt_tokens = self.count_tokens(model, message['name'])
                            if prompt_tokens:
                                total_prompt_tokens += prompt_tokens
                                total_prompt_tokens += self._extra_name_tokens.get(model, 0)
                    if total_prompt_tokens > 0:
                        total_prompt_tokens += self._extra_reply_tokens.get(model, 0)
                    span.set_counter('prompt_tokens', total_prompt_tokens)
                    span.set_counter('uncached_prompt_tokens', total_prompt_tokens)
                    span.inc_counter_metric('usage', 'prompt_tokens', total_prompt_tokens)
                    span.inc_counter_metric('usage', 'uncached_prompt_tokens', total_prompt_tokens)
        else:
            ret = ret.model_dump()

            if ret and 'model' in ret:
                span.set_tag('effective_model', ret['model'])

            if ret and 'usage' in ret and not exc:
                self.read_usage(span, ret['usage'])

    def trace_chat_completion_data(self, span, item):
        item = item.model_dump()

        span.set_perf_counter('first_token_ns')
        if item and 'choices' in item:
            for _ in item['choices']:
                span.inc_counter('completion_tokens', 1)
                span.inc_counter_metric('usage', 'completion_tokens', 1)
                span.inc_counter('output_tokens', 1)

        if item and 'usage' in item:
            self.read_usage(span, item['usage'])

    def trace_embedding(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'embeddings', params)
        if 'model' in params:
            span.set_tag('model', params['model'])
            span.set_param('model', params['model'])

        ret = ret.model_dump()

        if ret and 'model' in ret:
            span.set_tag('effective_model', ret['model'])

        if ret and 'usage' in ret:
            usage = ret['usage']
            if 'total_tokens' in usage:
                span.set_counter('total_tokens', usage['total_tokens'])
                span.inc_counter_metric('usage', 'total_tokens', usage['total_tokens'])
            if 'prompt_tokens' in usage:
                span.set_counter('prompt_tokens', usage['prompt_tokens'])
                span.inc_counter_metric('usage', 'prompt_tokens', usage['prompt_tokens'])

    def trace_image_generation(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'images/generations', params)

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_variation(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'images/variations', params)

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_edit(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'images/edits', params)

        self.trace_image_endpoint(span, args, kwargs, ret, exc)

    def trace_image_endpoint(self, span, args, kwargs, ret, exc):
        params = kwargs

        if ret and 'data' in ret:
            image_data = []
            for image in ret['data']:
                if 'url' in image:
                    image_data.append(image['url'])
                elif 'b64_json' in image:
                    image_data.append(image['b64_json'])

    def trace_audio_transcription(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'audio/transcriptions', params)
        if 'model' in params:
            span.set_tag('model', params['model'])
            span.set_param('model', params['model'])

    def trace_audio_translation(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'audio/translations', params)
        if 'model' in params:
            span.set_tag('model', params['model'])
            span.set_param('model', params['model'])

    def trace_moderation(self, span, args, kwargs, ret, exc):
        params = kwargs

        self.set_common_tags(span, 'moderations', params)
        if 'model' in params:
            span.set_tag('model', params['model'])
            span.set_param('model', params['model'])
