#!/usr/bin/env python3
"""Measure real vLLM HTTP inference throughput at several concurrency levels.

The same dataset slice, model, prompt template, temperature and max_tokens are
used for every concurrency setting.  Prompt tokenization is warmed through the
real /tokenize endpoint before timing, so the numbers isolate generation
throughput (the online evaluation pipeline still tokenizes each prompt once).

Example:
    source backend/.venv/bin/activate
    python scripts/bench_inference.py \
        --jsonl data/import/humaneval.jsonl --opt O0 --limit 128 \
        --base-url http://127.0.0.1:8001/v1 \
        --model-id llm4decompile-1.3b-v1.5 \
        --concurrency 16,32,64,128 --max-tokens 512

Outputs a JSON report under data/reports/ unless --out is given.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / 'backend'
sys.path.insert(0, str(BACKEND))

import httpx  # noqa: E402

from app.adapters import (  # noqa: E402
    Generation, InputTooLong, OnlineAdapter, VllmTokenizer, TokenizerError,
)
from app.config import MAX_MODEL_LEN  # noqa: E402

PROMPT = '# This is the assembly code:\n{assembly}\n# What is the source code?\n'


def load_samples(path: Path, opt: str, limit: int):
    rows = []
    with path.open(encoding='utf-8') as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if opt and str(row.get('optimization')) != opt:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise SystemExit(f'No samples in {path} for optimization={opt}')
    return rows


async def run_config(samples, args, concurrency, tokenizer):
    model = {
        'kind': 'openai', 'base_url': args.base_url, 'api_model': args.model_id,
        'api_mode': 'completion', 'api_key_env': '', 'system_prompt': '',
    }
    cfg = {
        'prompt_template': PROMPT, 'temperature': args.temperature,
        'max_tokens': args.max_tokens, 'max_model_len': args.max_model_len,
        'inference_timeout': args.timeout, 'inference_retries': 0,
    }
    limits = httpx.Limits(max_connections=concurrency,
                          max_keepalive_connections=concurrency)
    sem = asyncio.Semaphore(concurrency)
    state = {'active': 0, 'max_active': 0}
    latencies = []
    results = []

    async with httpx.AsyncClient(timeout=None, limits=limits,
                                 follow_redirects=False) as client:
        adapter = OnlineAdapter(model, cfg, client, tokenizer=tokenizer)

        async def one(sample):
            async with sem:
                state['active'] += 1
                state['max_active'] = max(state['max_active'], state['active'])
                started = time.perf_counter()
                try:
                    generation = await adapter.predict(sample)
                    results.append(('ok', generation))
                except InputTooLong as exc:
                    results.append(('input_too_long', exc))
                except Exception as exc:  # keep other requests running
                    results.append(('error', exc))
                finally:
                    latencies.append(round((time.perf_counter() - started) * 1000))
                    state['active'] -= 1

        started = time.perf_counter()
        await asyncio.gather(*(one(sample) for sample in samples))
        wall = time.perf_counter() - started

    ok = [item for status, item in results if status == 'ok']
    counts = Counter(status for status, _ in results)
    prompt_tokens = sum(g.prompt_tokens for g in ok)
    completion_tokens = sum(g.completion_tokens for g in ok)
    latencies_sorted = sorted(latencies)
    def quantile(q):
        if not latencies_sorted:
            return 0
        index = min(len(latencies_sorted) - 1, int(q * (len(latencies_sorted) - 1)))
        return latencies_sorted[index]
    return {
        'concurrency': concurrency,
        'requests': len(samples),
        'ok': counts.get('ok', 0),
        'failures': counts.get('error', 0),
        'input_too_long': counts.get('input_too_long', 0),
        'wall_seconds': round(wall, 3),
        'samples_per_second': round(len(ok) / wall, 3) if wall else 0,
        'completion_tokens_per_second': round(completion_tokens / wall, 1) if wall else 0,
        'total_tokens_per_second': round((prompt_tokens + completion_tokens) / wall, 1) if wall else 0,
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'avg_latency_ms': round(statistics.mean(latencies), 1) if latencies else 0,
        'p50_latency_ms': quantile(0.50),
        'p95_latency_ms': quantile(0.95),
        'max_in_flight_observed': state['max_active'],
        'errors': [str(item)[:200] for status, item in results if status == 'error'][:5],
    }


async def main_async(args):
    samples = load_samples(Path(args.jsonl), args.opt, args.limit)
    print(f'Loaded {len(samples)} samples from {args.jsonl} (opt={args.opt})')
    # Tokenize once outside the timed window through the real vLLM tokenizer,
    # then serve cached counts to every concurrency configuration so all runs
    # see identical prompt budgets.
    class CachedTokenizer:
        def __init__(self, counts):
            self.counts = counts
            self.calls = 0

        async def count(self, text):
            self.calls += 1
            return self.counts.get(text, max(1, (len(text) + 2) // 3))

    counts = {}
    client = httpx.AsyncClient(timeout=None)
    real = VllmTokenizer(client, args.base_url, args.model_id)
    try:
        async def fill(sample):
            prompt = PROMPT.replace('{assembly}', sample['assembly'])
            counts[prompt] = await real.count(prompt)
        await asyncio.gather(*(fill(sample) for sample in samples))
    except TokenizerError:
        pass
    finally:
        await client.aclose()
    cached = CachedTokenizer(counts)

    report = []
    for concurrency in args.concurrency:
        result = await run_config(samples, args, concurrency, cached)
        report.append(result)
        print(json.dumps(result, ensure_ascii=False))
    out = Path(args.out) if args.out else (
        Path('data/reports') / f'bench_inference_{time.strftime("%Y%m%d_%H%M%S")}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'args': vars(args), 'results': report},
                              ensure_ascii=False, indent=2), encoding='utf-8')
    print('Report:', out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jsonl', default='data/import/humaneval.jsonl')
    p.add_argument('--opt', default='O0')
    p.add_argument('--limit', type=int, default=128)
    p.add_argument('--base-url', default='http://127.0.0.1:8001/v1')
    p.add_argument('--model-id', default='llm4decompile-1.3b-v1.5')
    p.add_argument('--concurrency', default='16,32,64,128')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--max-model-len', type=int, default=MAX_MODEL_LEN)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--out', default='')
    args = p.parse_args()
    args.concurrency = [int(x) for x in str(args.concurrency).split(',') if x.strip()]
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
