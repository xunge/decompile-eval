"""Adapter tests: tokenizer budgeting, metrics parsing and offline batching."""
import asyncio

import pytest

from app.adapters import (
    BatchUnavailable, Generation, InputTooLong, OfflineBatchAdapter, OnlineAdapter,
    TokenizerError, estimate_tokens, fetch_vllm_metrics, parse_vllm_metrics,
)


class FakeResponse:
    def __init__(self, status=200, payload=None, text=''):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def post(self, url, headers=None, json=None):
        self.requests.append((url, json))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeTokenizer:
    def __init__(self, counts):
        self.counts = list(counts)
        self.calls = []

    async def count(self, text):
        self.calls.append(text)
        value = self.counts.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def model():
    return {'kind': 'openai', 'base_url': 'http://127.0.0.1:8001/v1',
            'api_model': 'm', 'api_mode': 'completion',
            'api_key_env': '', 'system_prompt': ''}


def cfg(**overrides):
    base = {
        'prompt_template': '# This is the assembly code:\n{assembly}\n# What is the source code?\n',
        'temperature': 0, 'max_tokens': 1024,
    }
    base.update(overrides)
    return base


def test_online_adapter_uses_effective_token_budget():
    client = FakeClient([FakeResponse(200, {
        'choices': [{'text': 'int f(void){return 0;}'}],
        'usage': {'prompt_tokens': 4000, 'completion_tokens': 96},
    })])
    tokenizer = FakeTokenizer([4000])
    adapter = OnlineAdapter(model(), cfg(), client, tokenizer=tokenizer)
    generation = asyncio.run(adapter.predict({'assembly': 'ret', 'sample_key': 's'}))
    sent = client.requests[0][1]
    assert sent['max_tokens'] == 96, 'only context-required budget reduction is allowed'
    assert sent['prompt'].startswith('# This is the assembly code:')
    assert generation.params['max_tokens_requested'] == 1024
    assert generation.params['max_tokens_effective'] == 96
    assert generation.params['input_tokens'] == 4000
    assert generation.prompt_tokens == 4000
    assert generation.completion_tokens == 96
    assert 'ret' in generation.prompt


def test_online_adapter_input_too_long_is_explicit():
    client = FakeClient([])
    adapter = OnlineAdapter(model(), cfg(), client, tokenizer=FakeTokenizer([4096]))
    with pytest.raises(InputTooLong) as info:
        asyncio.run(adapter.predict({'assembly': 'ret', 'sample_key': 's'}))
    assert info.value.input_tokens == 4096
    assert info.value.max_model_len == 4096
    assert 'ret' in info.value.prompt
    assert client.requests == [], 'no request may be sent for an over-long prompt'


def test_online_adapter_estimates_when_tokenizer_fails():
    client = FakeClient([FakeResponse(200, {
        'choices': [{'text': 'int f(void){return 0;}'}],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 4},
    })])
    adapter = OnlineAdapter(model(), cfg(), client,
                            tokenizer=FakeTokenizer([TokenizerError('down')]))
    generation = asyncio.run(adapter.predict({'assembly': 'ret', 'sample_key': 's'}))
    assert generation.tokenizer == 'estimated'
    assert generation.params['tokenizer'] == 'estimated'
    assert generation.params['max_tokens_effective'] == 1024


def test_estimate_tokens_is_conservative():
    assert estimate_tokens('') == 1
    assert estimate_tokens('abcd') >= 1
    assert estimate_tokens('a' * 300) >= 100


def test_parse_vllm_metrics_subset():
    text = '''
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="m"} 3.0
vllm:num_requests_waiting{engine="0",model_name="m"} 7.0
vllm:gpu_cache_usage_perc{engine="0",model_name="m"} 0.25
vllm:prompt_tokens_total{engine="0",model_name="m"} 1000.0
vllm:generation_tokens_total{engine="0",model_name="m"} 500.0
unrelated_metric 99
'''
    metrics = parse_vllm_metrics(text)
    assert metrics['num_requests_running'] == 3.0
    assert metrics['num_requests_waiting'] == 7.0
    assert metrics['gpu_cache_usage_perc'] == 0.25
    assert metrics['prompt_tokens_total'] == 1000.0
    assert metrics['generation_tokens_total'] == 500.0
    assert 'unrelated_metric' not in metrics


def test_fetch_vllm_metrics_unreachable_is_none():
    # Port 9 is never a vLLM server; the helper must fail soft and fast.
    assert fetch_vllm_metrics('http://127.0.0.1:9/v1', timeout=0.2) is None


class FakeHFTokenizer:
    def encode(self, text):
        return list(range(1 + len(text) // 10))


class FakeVllmTokenizer(FakeHFTokenizer):
    def encode(self, text):
        return [0] * (4096 if 'LONG' in text else 1 + len(text) // 10)


class FakeCompletion:
    def __init__(self, text='int f(void){return 0;}'):
        self.text = text
        self.token_ids = [1, 2, 3, 4, 5]


class FakeOutput:
    def __init__(self, text='int f(void){return 0;}'):
        self.outputs = [FakeCompletion(text)]


class FakeLLM:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.calls = []

    def get_tokenizer(self):
        return self._tokenizer

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((list(prompts), getattr(sampling_params, 'max_tokens', None)
                           if not isinstance(sampling_params, dict) else sampling_params['max_tokens']))
        return [FakeOutput(f'int f(void){{return {len(prompt)};}}') for prompt in prompts]


def batch_cfg(**overrides):
    base = {'prompt_template': 'ASM:{assembly}', 'temperature': 0,
            'max_tokens': 64, 'max_model_len': 4096, 'batch_size': 2,
            'model_path': 'fake-model'}
    base.update(overrides)
    return base


class FakeSamplingParams:
    def __init__(self, temperature, max_tokens):
        self.temperature = temperature
        self.max_tokens = max_tokens


def test_offline_batch_keeps_prompts_independent():
    llm = FakeLLM(FakeHFTokenizer())
    adapter = OfflineBatchAdapter(batch_cfg(), llm_factory=lambda path, ml: llm,
                                  sampling_params_factory=FakeSamplingParams)
    samples = [
        {'sample_key': 'a', 'assembly': 'AAA'},
        {'sample_key': 'b', 'assembly': 'BBB'},
    ]
    results = adapter.generate(samples)
    assert len(results) == 2
    assert all(isinstance(item, Generation) for item in results)
    assert len(llm.calls) == 1
    prompts = llm.calls[0][0]
    assert prompts == ['ASM:AAA', 'ASM:BBB'], 'prompts must not be concatenated'
    assert results[0].prompt == 'ASM:AAA' and results[1].prompt == 'ASM:BBB'
    assert results[0].completion_tokens == 5
    assert results[0].params['max_tokens_effective'] == 64


def test_offline_batch_reports_input_too_long_without_truncation():
    llm = FakeLLM(FakeVllmTokenizer())
    adapter = OfflineBatchAdapter(batch_cfg(max_tokens=64),
                                  llm_factory=lambda path, ml: llm,
                                  sampling_params_factory=FakeSamplingParams)
    samples = [
        {'sample_key': 'ok', 'assembly': 'fine'},
        {'sample_key': 'long', 'assembly': 'LONG'},
    ]
    results = adapter.generate(samples)
    assert isinstance(results[0], Generation)
    assert isinstance(results[1], InputTooLong)
    assert results[1].input_tokens == 4096
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == ['ASM:fine'], 'over-long prompt must not be sent/truncated'


def test_offline_batch_requires_model_path():
    with pytest.raises(BatchUnavailable):
        OfflineBatchAdapter(batch_cfg(model_path='')).generate([{'sample_key': 'a', 'assembly': 'x'}])


def test_online_adapter_preserves_requested_budget_when_it_fits():
    client = FakeClient([FakeResponse(200, {
        'choices': [{'text': 'int f(void){return 0;}'}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 1024},
    })])
    adapter = OnlineAdapter(model(), cfg(), client, tokenizer=FakeTokenizer([100]))
    generation = asyncio.run(adapter.predict({'assembly': 'ret', 'sample_key': 's'}))
    assert client.requests[0][1]['max_tokens'] == 1024
    assert generation.params['max_tokens_effective'] == 1024
    assert 'budget_reason' not in generation.params


def test_tokenizer_caches_repeated_prompts():
    from app.adapters import VllmTokenizer
    client = FakeClient([FakeResponse(200, {'count': 17})])
    tokenizer = VllmTokenizer(client, 'http://127.0.0.1:8001/v1', 'm')
    first = asyncio.run(tokenizer.count('same prompt'))
    second = asyncio.run(tokenizer.count('same prompt'))
    assert first == second == 17
    assert len(client.requests) == 1, '/tokenize must be cached per prompt'


def test_parse_vllm_metrics_supports_new_kv_cache_name():
    text = 'vllm:kv_cache_usage_perc{engine="0"} 0.42\n'
    metrics = parse_vllm_metrics(text)
    assert metrics['gpu_cache_usage_perc'] == 0.42
