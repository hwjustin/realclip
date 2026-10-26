"""OpenAI-compatible local Chat Completions adapter for self-hosted VLMs.

Tool images are delivered as associated user image messages because ordinary
Chat Completions tool messages cannot contain images. Token counting uses the
same SDK conversion as generation and the server's /tokenize endpoint.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from agents import OpenAIChatCompletionsModel
from agents.models.chatcmpl_converter import Converter

from .storage import write_json


def prepare_local_input(items):
    if isinstance(items, str):
        return items
    prepared, images = [], []
    def flush():
        if images:
            prepared.append({'role': 'user', 'content': list(images)})
            images.clear()
    for original in items:
        item = deepcopy(original)
        if item.get('type') != 'function_call_output':
            flush()
            # Open-source chat templates commonly expose system/user/assistant/
            # tool roles. Preserve developer content as a system message.
            if item.get('role') == 'developer':
                item['role'] = 'system'
            prepared.append(item)
            continue
        content = item.get('output')
        if not isinstance(content, list):
            prepared.append(item)
            continue
        text = []
        for part in content:
            kind = part.get('type')
            if kind in {'input_text', 'output_text', 'text'}:
                text.append(part['text'])
            elif kind == 'input_image':
                url = part.get('image_url')
                if not isinstance(url, str) or not url.startswith('data:image/'):
                    raise ValueError('Local tool images require inline image data; remote OpenAI file IDs are unsupported')
                images.extend([{'type': 'input_text', 'text': f"Image returned by tool call {item['call_id']}; retain its tool metadata and original image scope."}, part])
            else:
                raise ValueError(f'Unsupported local tool output content: {kind}')
        item['output'] = '\n'.join(text)
        prepared.append(item)
    flush()
    return prepared


class LocalChatModel(OpenAIChatCompletionsModel):
    def __init__(self, *args, settings, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_settings = settings

    def prepared_arguments(self, args, kwargs):
        bound = inspect.signature(OpenAIChatCompletionsModel.get_response).bind(self, *args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        values.pop('self')
        values['input'] = prepare_local_input(values['input'])
        return values

    async def count_input(self, args, kwargs, receipt_path: Path) -> int:
        values = self.prepared_arguments(args, kwargs)
        messages = Converter.items_to_messages(values['input'], model=self.model,
            base_url=str(self._client.base_url),
            should_replay_reasoning_content=self.should_replay_reasoning_content)
        if values['system_instructions']:
            messages.insert(0, {'role': 'system', 'content': values['system_instructions']})
        tools = [Converter.tool_to_openai(t) for t in values['tools']]
        tools.extend(Converter.convert_handoff_tool(t) for t in values['handoffs'])
        payload = {'model': self.model, 'messages': messages, 'add_generation_prompt': True}
        if tools:
            payload['tools'] = tools
        # These change chat-template rendering and therefore the token count.
        # Sampling/decoding-only options remain generation parameters.
        for key in ('chat_template', 'chat_template_kwargs', 'mm_processor_kwargs'):
            if key in self.local_settings.local_extra_body:
                payload[key] = self.local_settings.local_extra_body[key]
        choice = Converter.convert_tool_choice(values['model_settings'].tool_choice)
        if isinstance(choice, (str, dict)):
            payload['tool_choice'] = choice
        encoded = json.dumps(payload, ensure_ascii=False).encode()
        record = {'provider': 'openai_compatible', 'count_payload_sha256': hashlib.sha256(encoded).hexdigest(),
                  'encoded_bytes': len(encoded), 'phase': 'before_generation',
                  'safety_margin_tokens': self.local_settings.local_context_margin_tokens,
                  'includes_tools': bool(tools), 'image_transport': 'inline_data_urls'}
        try:
            if len(encoded) > self.local_settings.local_max_request_bytes:
                raise ValueError('Local request exceeds local_max_request_bytes')
            base = urlsplit(self.local_settings.base_url)
            path = base.path.rstrip('/')
            if path.endswith('/v1'):
                path = path[:-3]
            endpoint = urlunsplit((base.scheme, base.netloc, path + '/tokenize', '', ''))
            # Use the same local client and credential. No OpenAI token-count
            # endpoint, hosted Files API, or fallback to a cloud model is used.
            response = await self._client.post(endpoint, cast_to=dict, body=payload)
            count = response.get('count')
            if type(count) is not int or count < 0:
                raise ValueError('Local /tokenize must return a nonnegative integer count')
            limit = count + self.local_settings.local_context_margin_tokens
            record.update(status='counted', input_tokens=count, input_token_bound=limit)
            write_json(receipt_path, record)
            return limit
        except BaseException as error:
            record.update(status='failed', error=f'{type(error).__name__}: {error}')
            write_json(receipt_path, record)
            raise

    async def get_response(self, *args, **kwargs):
        response = await super().get_response(**self.prepared_arguments(args, kwargs))
        raw = response.raw_usage
        if raw is not None:
            # The application ledger consumes Responses-style usage fields;
            # retain the unmodified provider usage alongside normalized counts.
            raw = raw.model_dump() if hasattr(raw, 'model_dump') else raw
            if 'prompt_tokens' not in raw or 'completion_tokens' not in raw:
                raise ValueError('Local provider omitted required prompt/completion usage')
            response = replace(response, raw_usage={
                'input_tokens': raw['prompt_tokens'], 'output_tokens': raw['completion_tokens'],
                'total_tokens': raw.get('total_tokens', raw['prompt_tokens'] + raw['completion_tokens']),
                'input_tokens_details': raw.get('prompt_tokens_details') or {},
                'output_tokens_details': raw.get('completion_tokens_details') or {},
                'provider_usage': raw})
        return response
