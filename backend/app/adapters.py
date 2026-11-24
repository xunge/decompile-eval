"""Inference adapters.

Two interchangeable generation backends:

* ``OnlineAdapter`` — vLLM OpenAI-compatible HTTP API (``/v1/completions`` or
  ``/v1/chat/completions``) with a shared ``httpx.AsyncClient`` connection pool,
  bounded retries and real tokenizer checks through ``POST /tokenize``.
* ``OfflineBatchAdapter`` — optional vLLM Python API ``LLM.generate(prompts,
  sampling_params)`` for pure offline generation.  Prompts stay independent;
  they are never concatenated into one function.

The rest of the pipeline only depends on ``Generation`` / ``InputTooLong`` /
``InferenceError``, so both adapters are interchangeable.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from .config import (
    INFERENCE_RETRIES, INFERENCE_TIMEOUT, MAX_MODEL_LEN,
    OFFLINE_BATCH_MODEL_PATH, OFFLINE_BATCH_SIZE,
    TOKENIZER_CACHE_SIZE, TOKENIZER_TIMEOUT, VLLM_METRICS_TIMEOUT,
)


@dataclass
class Generation:
    text: str
    inference_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt: str = ''
    params: dict = field(default_factory=dict)
    tokenizer: str = ''


class InferenceError(RuntimeError):
    pass


class TokenizerError(RuntimeError):
    pass


class InputTooLong(InferenceError):
    """The prompt plus one generated token does not fit in the model window."""

    def __init__(self, input_tokens: int, max_model_len: int, prompt: str = '',
                 params: dict | None = None):
        self.input_tokens = int(input_tokens)
        self.max_model_len = int(max_model_len)
        self.prompt = prompt
        self.params = dict(params or {})
        super().__init__(
            f'Input prompt has {input_tokens} tokens, which exceeds the '
            f'max_model_len={max_model_len} window; refusing to truncate assembly.')


def estimate_tokens(text: str) -> int:
    """Conservative fallback when /tokenize is unavailable (never truncates)."""
    return max(1, (len(text) + 2) // 3)


def _root_url(base_url: str) -> str:
    base = (base_url or '').rstrip('/')
    if base.endswith('/v1'):
        base = base[:-3]
    return base


class VllmTokenizer:
    """Real model tokenizer exposed by vLLM at ``POST /tokenize``."""

    def __init__(self, client: httpx.AsyncClient, base_url: str, model_id: str,
                 timeout: float = TOKENIZER_TIMEOUT, cache_size: int = TOKENIZER_CACHE_SIZE):
        self.client = client
        self.base_url = base_url
        self.model_id = model_id
        self.timeout = timeout
        self.cache_size = cache_size
        self.cache: OrderedDict[str, int] = OrderedDict()

    def _urls(self) -> list[str]:
        root = _root_url(self.base_url)
        return [f'{root}/tokenize', f'{self.base_url.rstrip("/")}/tokenize']

    async def count(self, text: str) -> int:
        key = hashlib.sha1(text.encode('utf-8', 'replace')).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached
        last: Exception | None = None
        for url in dict.fromkeys(self._urls()):
            try:
                response = await asyncio.wait_for(
                    self.client.post(url, json={'model': self.model_id, 'prompt': text}),
                    timeout=self.timeout)
            except (asyncio.TimeoutError, httpx.TransportError) as exc:
                last = exc
                continue
            if response.status_code == 200:
                body = response.json()
                count = body.get('count')
                if count is None:
                    count = len(body.get('tokens') or [])
                count = int(count)
                self.cache[key] = count
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
                return count
            last = TokenizerError(f'HTTP {response.status_code} from {url}')
            if response.status_code not in (404, 405):
                break
        raise TokenizerError(f'vLLM tokenizer unavailable: {last}')


async def post_with_retries(client: httpx.AsyncClient, url: str, headers: dict,
                            payload: dict, timeout: float, retries: int,
                            backoff_base: float = 1.0) -> dict:
    """Bounded exponential backoff for 429 / 5xx / transport errors.

    Deterministic 4xx responses (other than 429) are raised immediately:
    retrying them only wastes GPU time and distorts latency statistics.
    """
    last_error: Exception | None = None
    attempts = max(0, retries) + 1
    for attempt in range(attempts):
        try:
            response = await asyncio.wait_for(
                client.post(url, headers=headers, json=payload), timeout=timeout)
        except asyncio.TimeoutError:
            last_error = InferenceError(f'Inference timed out after {timeout}s')
        except httpx.TransportError as exc:
            last_error = InferenceError(f'{type(exc).__name__}: {exc}')
        else:
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_error = InferenceError(
                    f'HTTP {response.status_code}: {response.text[:200]}')
            elif response.status_code >= 400:
                raise InferenceError(
                    f'HTTP {response.status_code}: {response.text[:300]}')
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise InferenceError(f'Invalid JSON response: {exc}') from exc
        if attempt + 1 < attempts:
            delay = min(30.0, backoff_base * (2 ** attempt)) + random.uniform(0, 0.25)
            await asyncio.sleep(delay)
    raise InferenceError(f'Inference failed after {attempts} attempt(s): {last_error}')


def _extract_content(body: dict, api_mode: str) -> str:
    choice = body['choices'][0]
    content = choice.get('text', '') if api_mode == 'completion' else choice['message']['content']
    if isinstance(content, list):
        return '\n'.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
    return content or ''


class OnlineAdapter:
    """Realtime HTTP adapter used by the evaluation pipeline."""

    def __init__(self, model: dict, cfg: dict, client: httpx.AsyncClient | None = None,
                 tokenizer: VllmTokenizer | None = None):
        self.model = model
        self.cfg = cfg
        self.client = client
        self.max_model_len = int(cfg.get('max_model_len') or MAX_MODEL_LEN)
        self.api_mode = model.get('api_mode') if isinstance(model, dict) else model['api_mode']
        self.api_mode = self.api_mode or 'chat'
        self.tokenizer = tokenizer
        if (self.tokenizer is None and model['kind'] == 'openai'
                and client is not None):
            self.tokenizer = VllmTokenizer(client, model['base_url'], model['api_model'])

    def _prompt(self, sample: dict) -> str:
        return str(self.cfg['prompt_template']).replace('{assembly}', sample['assembly'])

    def _count_text(self, prompt: str) -> str:
        if self.api_mode == 'chat':
            system = self.model.get('system_prompt') or ''
            return (system + '\n' + prompt) if system else prompt
        return prompt

    async def _input_tokens(self, prompt: str) -> tuple[int, str]:
        if self.tokenizer is None:
            return estimate_tokens(self._count_text(prompt)), 'estimated'
        try:
            return await self.tokenizer.count(self._count_text(prompt)), 'vllm'
        except (TokenizerError, asyncio.TimeoutError, httpx.HTTPError):
            # Never fail the whole run because /tokenize is temporarily down;
            # the estimate is conservative and marked in generation params.
            return estimate_tokens(self._count_text(prompt)), 'estimated'

    def _payload(self, prompt: str, max_tokens: int) -> dict:
        payload = {'model': self.model['api_model'],
                   'temperature': self.cfg['temperature'],
                   'max_tokens': max_tokens, 'stream': False}
        if self.api_mode == 'completion':
            payload['prompt'] = prompt
        else:
            payload['messages'] = [
                {'role': 'system', 'content': self.model.get('system_prompt') or ''},
                {'role': 'user', 'content': prompt},
            ]
        return payload

    def _url_headers(self) -> tuple[str, dict]:
        mode = self.api_mode
        url = self.model['base_url'].rstrip('/') + (
            '/completions' if mode == 'completion' else '/chat/completions')
        headers = {'Content-Type': 'application/json'}
        key_env = self.model.get('api_key_env') or ''
        api_key = os.getenv(key_env, '') if key_env else ''
        if key_env and not api_key:
            raise InferenceError(f'Missing environment variable {key_env}')
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        return url, headers

    async def predict(self, sample: dict) -> Generation:
        kind = self.model['kind']
        if kind == 'fixture':
            if not sample['demo_prediction']:
                raise ValueError('Fixture model can only be used on samples with demo_prediction.')
            return Generation(sample['demo_prediction'], params={'model_kind': 'fixture'})
        if kind == 'offline':
            lookup = self.cfg.get('_predictions', {})
            if sample['sample_key'] not in lookup:
                raise ValueError(f'Missing offline prediction for {sample["sample_key"]}')
            return Generation(lookup[sample['sample_key']], params={'model_kind': 'offline'})
        if kind == 'reference':
            return Generation(sample['reference_source'], params={'model_kind': 'reference'})
        if kind != 'openai':
            raise ValueError(f'Unsupported model kind: {kind}')
        if self.client is None:
            raise InferenceError('HTTP client is not configured for an openai model.')

        prompt = self._prompt(sample)
        requested = int(self.cfg['max_tokens'])
        input_tokens, source = await self._input_tokens(prompt)
        available = self.max_model_len - input_tokens
        params = {
            'model': self.model['api_model'],
            'api_mode': self.api_mode,
            'temperature': self.cfg['temperature'],
            'max_tokens_requested': requested,
            'max_tokens_effective': max(0, min(requested, available)),
            'max_model_len': self.max_model_len,
            'input_tokens': input_tokens,
            'tokenizer': source,
        }
        if available <= 0:
            raise InputTooLong(input_tokens, self.max_model_len, prompt=prompt, params=params)
        effective = min(requested, available)
        params['max_tokens_effective'] = effective
        if effective != requested:
            params['budget_reason'] = 'prompt leaves less room than requested max_tokens'
        url, headers = self._url_headers()
        started = time.perf_counter()
        body = await post_with_retries(
            self.client, url, headers, self._payload(prompt, effective),
            float(self.cfg.get('inference_timeout') or INFERENCE_TIMEOUT),
            int(self.cfg.get('inference_retries', INFERENCE_RETRIES)))
        inference_ms = round((time.perf_counter() - started) * 1000)
        usage = body.get('usage') or {}
        prompt_tokens = int(usage.get('prompt_tokens') or input_tokens)
        completion_tokens = int(usage.get('completion_tokens') or 0)
        return Generation(
            text=_extract_content(body, self.api_mode),
            inference_ms=inference_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt=prompt,
            params=params,
            tokenizer=source,
        )


class BatchUnavailable(RuntimeError):
    pass


class OfflineBatchAdapter:
    """Optional offline generation through ``vllm.LLM.generate``.

    Only used for pure offline generation.  Prompts are passed as a list of
    independent strings; outputs are mapped back by list position, which the
    caller associates with ``sample_id``.
    """

    def __init__(self, cfg: dict, llm_factory: Callable | None = None,
                 sampling_params_factory: Callable | None = None):
        self.cfg = cfg
        self.model_path = (cfg.get('model_path') or OFFLINE_BATCH_MODEL_PATH or '').strip()
        self.max_model_len = int(cfg.get('max_model_len') or MAX_MODEL_LEN)
        self.batch_size = int(cfg.get('batch_size') or OFFLINE_BATCH_SIZE)
        self.llm_factory = llm_factory
        self.sampling_params_factory = sampling_params_factory
        self._llm = None
        self._tokenizer = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec('vllm') is not None

    def _ensure(self):
        if self._llm is not None:
            return
        if not self.model_path:
            raise BatchUnavailable(
                'No local model path configured. Set DEC_EVAL_BATCH_MODEL_PATH '
                'or pass model_path in the batch request.')
        if self.llm_factory is not None:
            self._llm = self.llm_factory(self.model_path, self.max_model_len)
        else:
            try:
                from vllm import LLM
            except ImportError as exc:
                raise BatchUnavailable(
                    'The vLLM Python package is not installed in the backend '
                    'environment; install vllm or use the online adapter.') from exc
            self._llm = LLM(model=self.model_path, max_model_len=self.max_model_len)
        self._tokenizer = self._llm.get_tokenizer()

    def _sampling_params(self, max_tokens: int):
        if self.sampling_params_factory is not None:
            return self.sampling_params_factory(self.cfg['temperature'], max_tokens)
        from vllm import SamplingParams
        return SamplingParams(temperature=self.cfg['temperature'], max_tokens=max_tokens)

    def _prompt(self, sample: dict) -> str:
        return str(self.cfg['prompt_template']).replace('{assembly}', sample['assembly'])

    def generate(self, samples: list[dict]) -> list[Generation | BaseException]:
        self._ensure()
        prompts = [self._prompt(sample) for sample in samples]
        counts = [len(self._tokenizer.encode(prompt)) for prompt in prompts]
        requested = int(self.cfg['max_tokens'])
        results: list[Generation | BaseException] = [None] * len(samples)  # type: ignore[list-item]
        groups: dict[int, list[int]] = defaultdict(list)
        for index, (prompt, count) in enumerate(zip(prompts, counts)):
            available = self.max_model_len - count
            params = {
                'model': self.model_path,
                'api_mode': 'offline_batch',
                'temperature': self.cfg['temperature'],
                'max_tokens_requested': requested,
                'max_tokens_effective': max(0, min(requested, available)),
                'max_model_len': self.max_model_len,
                'input_tokens': count,
                'tokenizer': 'vllm-local',
            }
            if available <= 0:
                results[index] = InputTooLong(count, self.max_model_len,
                                              prompt=prompt, params=params)
                continue
            effective = min(requested, available)
            params['max_tokens_effective'] = effective
            groups[effective].append(index)
            # Stash per-index metadata for the generation phase.
            samples[index]['_batch_prompt'] = prompt
            samples[index]['_batch_params'] = params

        for effective, indexes in sorted(groups.items(), reverse=True):
            started = time.perf_counter()
            try:
                outputs = self._llm.generate(
                    [samples[i]['_batch_prompt'] for i in indexes],
                    self._sampling_params(effective), use_tqdm=False)
            except Exception as exc:  # one bad batch must not kill the run
                for index in indexes:
                    results[index] = InferenceError(f'Batch generation failed: {type(exc).__name__}: {exc}')
                continue
            batch_ms = round((time.perf_counter() - started) * 1000)
            if len(outputs) != len(indexes):
                for index in indexes:
                    results[index] = InferenceError(
                        f'vLLM returned {len(outputs)} outputs for {len(indexes)} prompts')
                continue
            for index, output in zip(indexes, outputs):
                try:
                    completion = output.outputs[0]
                    text = completion.text or ''
                    token_ids = getattr(completion, 'token_ids', None)
                    completion_tokens = len(token_ids) if token_ids is not None else 0
                except (AttributeError, IndexError) as exc:
                    results[index] = InferenceError(f'Malformed vLLM output: {exc}')
                    continue
                params = dict(samples[index]['_batch_params'])
                params['batch_ms'] = batch_ms
                results[index] = Generation(
                    text=text,
                    inference_ms=round(batch_ms / max(1, len(indexes))),
                    prompt_tokens=len(self._tokenizer.encode(samples[index]['_batch_prompt'])),
                    completion_tokens=completion_tokens,
                    prompt=samples[index]['_batch_prompt'],
                    params=params,
                    tokenizer='vllm-local',
                )
        return results


_METRIC_NAMES = {
    'num_requests_running', 'num_requests_waiting', 'gpu_cache_usage_perc',
    'kv_cache_usage_perc', 'prompt_tokens_total', 'generation_tokens_total',
    'request_success_total',
}


def parse_vllm_metrics(text: str) -> dict[str, float]:
    """Parse the subset of vLLM Prometheus metrics we surface in the UI."""
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        try:
            head, value = line.rsplit(' ', 1)
        except ValueError:
            continue
        name = head.split('{', 1)[0].strip()
        if not name.startswith('vllm:'):
            continue
        short = name.split(':', 1)[1]
        if short not in _METRIC_NAMES:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if short in ('num_requests_running', 'num_requests_waiting'):
            values[short] = values.get(short, 0.0) + number
        else:
            values[short] = number
    if 'kv_cache_usage_perc' in values:
        values.setdefault('gpu_cache_usage_perc', values['kv_cache_usage_perc'])
    return values


def fetch_vllm_metrics(base_url: str, timeout: float = VLLM_METRICS_TIMEOUT) -> dict[str, Any] | None:
    """Best-effort synchronous scrape used by the progress API."""
    if not base_url:
        return None
    url = _root_url(base_url) + '/metrics'
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            response = client.get(url)
            if response.status_code != 200:
                return None
            metrics = parse_vllm_metrics(response.text)
    except (httpx.HTTPError, OSError, ValueError):
        return None
    if not metrics:
        return None
    metrics['url'] = url
    return metrics
