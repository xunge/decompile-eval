"""Concurrent producer/consumer evaluation pipeline.

Model inference and Docker compile/test are fully decoupled:

    samples -> bounded inference workers (asyncio + httpx.AsyncClient)
            -> bounded compile queue (asyncio.Queue + backpressure)
            -> compile/test workers (asyncio subprocess + Docker)
            -> immediate SQLite persistence per sample

Runs execute one at a time (single-host MVP), but inside a run inference keeps
flowing while compilation is in progress.  Progress counters are held in memory
and exposed through :func:`get_progress`; durable per-sample state lives in the
``results`` table, which makes cancellation and resume safe.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx

from .config import (
    COMPILE_CONCURRENCY, COMPILE_TIMEOUT, INFERENCE_CONCURRENCY,
    INFERENCE_RETRIES, INFERENCE_TIMEOUT, LOG_LIMIT_BYTES,
    MAX_COMPILE_CONCURRENCY, MAX_INFERENCE_CONCURRENCY, MAX_MODEL_LEN,
    MAX_PENDING_COMPILE, OFFLINE_BATCH_SIZE, TEST_TIMEOUT,
)
from .db import connect, now
from .adapters import (
    BatchUnavailable, Generation, InferenceError, InputTooLong, OfflineBatchAdapter,
    OnlineAdapter, post_with_retries as _post_with_retries,
)
from .sandbox import EvalOutcome, evaluate_code_async

EXCLUDED = {'INFRA_ERROR', 'INFERENCE_ERROR'}
TRANSIENT = {'GENERATED', 'COMPILING'}
TERMINAL_EVAL = {'TEST_PASS', 'WRONG_ANSWER', 'RUNTIME_ERROR', 'TIMEOUT',
                 'COMPILE_ERROR', 'COMPILE_TIMEOUT', 'MODEL_EMPTY', 'INPUT_TOO_LONG'}
REFERENCE_MODEL_NAME = '参考源码基线（内部）'


# ---------------------------------------------------------------------------
# SQLite writes: one lock + short transactions, WAL and busy_timeout make
# concurrent writes safe without risking "database is locked".
# ---------------------------------------------------------------------------
_WRITE_LOCK = threading.RLock()


def _db_write(fn: Callable[[Any], Any]):
    with _WRITE_LOCK:
        with connect() as db:
            return fn(db)


async def _db_write_async(fn: Callable[[Any], Any]):
    return await asyncio.to_thread(_db_write, fn)


# ---------------------------------------------------------------------------
# In-memory progress registry (read by the API thread, written by the loop).
# ---------------------------------------------------------------------------
_PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict[int, dict[str, Any]] = {}


def _progress_init(run_id: int, data: dict[str, Any]) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS[run_id] = dict(data)


def _progress_patch(run_id: int, **deltas) -> None:
    with _PROGRESS_LOCK:
        state = _PROGRESS.setdefault(run_id, {})
        for key, value in deltas.items():
            state[key] = state.get(key, 0) + value


def _progress_set(run_id: int, **values) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS.setdefault(run_id, {}).update(values)


def _progress_get(run_id: int) -> dict[str, Any] | None:
    with _PROGRESS_LOCK:
        state = _PROGRESS.get(run_id)
        return dict(state) if state is not None else None


def _summarize_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    snap = dict(snap)
    if snap.get('finished_monotonic'):
        elapsed = snap['finished_monotonic'] - snap['started_monotonic']
    else:
        elapsed = time.monotonic() - snap['started_monotonic']
    elapsed = max(0.0, elapsed)
    snap['elapsed_seconds'] = round(elapsed, 2)
    snap['inference_rate'] = round(snap.get('inference_done', 0) / elapsed, 3) if elapsed else 0.0
    snap['eval_rate'] = round(snap.get('evaluated', 0) / elapsed, 3) if elapsed else 0.0
    completion = snap.get('completion_tokens', 0)
    prompt_total = snap.get('prompt_tokens', 0)
    snap['token_rate'] = round(completion / elapsed, 1) if elapsed else 0.0
    snap['total_token_rate'] = round((completion + prompt_total) / elapsed, 1) if elapsed else 0.0
    done = snap.get('inference_done', 0)
    snap['average_inference_ms'] = (round(snap.get('inference_ms_sum', 0) / done, 1)
                                    if done else 0.0)
    snap['in_flight'] = snap.get('inferring', 0)
    snap['inference_requests'] = (snap.get('inference_done', 0)
                                  + snap.get('inference_failed', 0))
    snap['waiting_inference'] = max(
        0, snap.get('total', 0) - done
        - snap.get('inference_failed', 0) - snap.get('input_too_long', 0)
        - snap.get('inferring', 0))
    # Counter is mutated by the pipeline thread; copy it before serialisation.
    snap['status_counts'] = dict(snap.get('status_counts') or {})
    snap.pop('started_monotonic', None)
    snap.pop('finished_monotonic', None)
    return snap



# ---------------------------------------------------------------------------
# Prediction helpers.  The synchronous variant stays for /api/models/{id}/test
# and for backward-compatible callers/tests.
# ---------------------------------------------------------------------------
def _request_config(model, sample, cfg: dict) -> tuple[str, dict, dict]:
    api_mode = model['api_mode'] if 'api_mode' in model.keys() else 'chat'
    url = model['base_url'].rstrip('/') + ('/completions' if api_mode == 'completion' else '/chat/completions')
    prompt = cfg['prompt_template'].replace('{assembly}', sample['assembly'])
    headers = {'Content-Type': 'application/json'}
    key_env = model['api_key_env']
    api_key = os.getenv(key_env, '') if key_env else ''
    if key_env and not api_key:
        raise InferenceError(f'Missing environment variable {key_env}')
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {'model': model['api_model'], 'temperature': cfg['temperature'],
               'max_tokens': cfg['max_tokens'], 'stream': False}
    if api_mode == 'completion':
        payload['prompt'] = prompt
    else:
        payload['messages'] = [
            {'role': 'system', 'content': model['system_prompt']},
            {'role': 'user', 'content': prompt},
        ]
    return url, headers, payload


def _extract_content(body: dict, api_mode: str) -> str:
    choice = body['choices'][0]
    content = choice.get('text', '') if api_mode == 'completion' else choice['message']['content']
    if isinstance(content, list):
        return '\n'.join(str(part.get('text', '')) for part in content if isinstance(part, dict))
    return content or ''


def generate_prediction(model, sample, cfg: dict) -> str:
    """Synchronous prediction used by the model-check endpoint and old callers."""
    if model['kind'] == 'fixture':
        if not sample['demo_prediction']:
            raise ValueError('Fixture model can only be used on samples with demo_prediction.')
        return sample['demo_prediction']
    if model['kind'] == 'offline':
        lookup = cfg.get('_predictions', {})
        if sample['sample_key'] not in lookup:
            raise ValueError(f'Missing offline prediction for {sample["sample_key"]}')
        return lookup[sample['sample_key']]
    if model['kind'] == 'reference':
        return sample['reference_source']
    if model['kind'] != 'openai':
        raise ValueError(f'Unsupported model kind: {model["kind"]}')
    api_mode = model['api_mode'] if 'api_mode' in model.keys() else 'chat'
    url, headers, payload = _request_config(model, sample, cfg)
    with httpx.Client(timeout=cfg.get('inference_timeout', 300), follow_redirects=False) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    return _extract_content(body, api_mode)


async def generate_prediction_async(model, sample, cfg: dict,
                                    client: httpx.AsyncClient | None) -> Generation:
    """Delegate to the per-run OnlineAdapter (kept patchable for tests)."""
    adapter = cfg.get('_online_adapter')
    if adapter is None:
        adapter = OnlineAdapter(model, cfg, client)
        cfg['_online_adapter'] = adapter
    return await adapter.predict(sample)


# ---------------------------------------------------------------------------
# Run lifecycle helpers.
# ---------------------------------------------------------------------------
def _get_samples(db, run_id: int, cfg: dict):
    args = [cfg['dataset_id'], *cfg['optimizations']]
    markers = ','.join('?' for _ in cfg['optimizations'])
    query = (f'SELECT * FROM samples WHERE dataset_id=? '
             f'AND optimization IN ({markers}) ORDER BY id')
    if cfg['sample_limit']:
        query += ' LIMIT ?'
        args.append(cfg['sample_limit'])
    return db.execute(query, args).fetchall()


def _load_run(run_id: int):
    with connect() as db:
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if run is None:
            return None
        model = db.execute('SELECT * FROM models WHERE id=?', (run['model_id'],)).fetchone()
        cfg = json.loads(run['config_json'])
        samples = _get_samples(db, run_id, cfg)
        existing = {row['sample_id']: dict(row) for row in
                    db.execute('SELECT * FROM results WHERE run_id=?', (run_id,))}
        if cfg.get('offline_file'):
            try:
                with open(cfg['offline_file'], encoding='utf-8') as fp:
                    cfg['_predictions'] = json.load(fp)
            except OSError as exc:
                raise InferenceError(f'Offline prediction file unreadable: {exc}') from exc
        return dict(run), dict(model), cfg, [dict(s) for s in samples], existing


def _resolve_concurrency(cfg: dict) -> tuple[int, int]:
    inference = cfg.get('inference_concurrency') or INFERENCE_CONCURRENCY
    compile_workers = cfg.get('compile_concurrency') or COMPILE_CONCURRENCY
    inference = max(1, min(int(inference), MAX_INFERENCE_CONCURRENCY))
    compile_workers = max(1, min(int(compile_workers), MAX_COMPILE_CONCURRENCY))
    return inference, compile_workers


def _mark_running(run_id: int) -> bool:
    def write(db):
        cur = db.execute(
            "UPDATE runs SET status='running', started_at=COALESCE(started_at,?), error='' "
            "WHERE id=? AND status='queued'", (now(), run_id))
        return cur.rowcount
    return bool(_db_write(write))


def _read_run_status(run_id: int) -> str | None:
    with connect() as db:
        row = db.execute('SELECT status FROM runs WHERE id=?', (run_id,)).fetchone()
        return row['status'] if row is not None else None


def _insert_generated(db, run_id: int, sample: dict, gen: Generation) -> None:
    db.execute('''
            INSERT INTO results(run_id,sample_id,status,compile_success,tests_passed,tests_total,
                                model_output,inference_ms,latency_ms,prompt_tokens,completion_tokens,
                                prompt_text,generation_params_json,created_at)
            VALUES(?,?,'GENERATED',NULL,0,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,sample_id) DO UPDATE SET
                status='GENERATED', compile_success=NULL, tests_passed=0,
                tests_total=excluded.tests_total, model_output=excluded.model_output,
                inference_ms=excluded.inference_ms, latency_ms=excluded.latency_ms,
                prompt_tokens=excluded.prompt_tokens, completion_tokens=excluded.completion_tokens,
                prompt_text=excluded.prompt_text,
                generation_params_json=excluded.generation_params_json,
                compile_log='', run_log='', compile_ms=0, test_ms=0,
                error='', detail_json='', created_at=excluded.created_at
            WHERE results.status IN ('GENERATED','INFERENCE_ERROR','INFRA_ERROR')
        ''', (run_id, sample['id'], len(json.loads(sample['tests_json'])),
              gen.text[:150000], gen.inference_ms, gen.inference_ms,
              gen.prompt_tokens, gen.completion_tokens, gen.prompt[:150000],
              json.dumps(gen.params, ensure_ascii=False), now()))


def _insert_input_too_long(db, run_id: int, sample: dict, exc: InputTooLong,
                           inference_ms: int) -> None:
    tests = json.loads(sample['tests_json'])
    detail = json.dumps({
        'failure_reason': str(exc),
        'input_tokens': exc.input_tokens,
        'max_model_len': exc.max_model_len,
        'harness_mode': 'program' if len(tests) == 1 and 'program' in tests[0] else 'cases',
    }, ensure_ascii=False)
    db.execute('''
            INSERT INTO results(run_id,sample_id,status,compile_success,tests_passed,tests_total,
                                model_output,inference_ms,latency_ms,prompt_tokens,
                                prompt_text,generation_params_json,error,detail_json,created_at)
            VALUES(?,?,'INPUT_TOO_LONG',0,0,?,'',?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,sample_id) DO UPDATE SET
                status='INPUT_TOO_LONG', compile_success=0, tests_passed=0,
                tests_total=excluded.tests_total, model_output='',
                inference_ms=excluded.inference_ms, latency_ms=excluded.latency_ms,
                prompt_tokens=excluded.prompt_tokens, prompt_text=excluded.prompt_text,
                generation_params_json=excluded.generation_params_json,
                error=excluded.error, detail_json=excluded.detail_json,
                compile_log='', run_log='', created_at=excluded.created_at
            WHERE results.status IN ('GENERATED','INFERENCE_ERROR','INFRA_ERROR')
        ''', (run_id, sample['id'], len(json.loads(sample['tests_json'])),
              inference_ms, inference_ms, exc.input_tokens,
              exc.prompt[:150000], json.dumps(exc.params, ensure_ascii=False),
              str(exc)[:1000], detail, now()))


async def _persist_generation(run_id: int, sample: dict, gen: Generation,
                              compile_queue: asyncio.Queue) -> None:
    await _db_write_async(lambda db: _insert_generated(db, run_id, sample, gen))
    _progress_patch(run_id, inference_done=1, waiting_compile=1,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    inference_ms_sum=gen.inference_ms)
    await compile_queue.put((sample, gen.text, gen.inference_ms))


async def _persist_inference_error(run_id: int, sample: dict,
                                   exc: BaseException, inference_ms: int) -> None:
    error = f'{type(exc).__name__}: {str(exc)[:600]}'
    await _db_write_async(lambda db: _insert_inference_error(
        db, run_id, sample, error, inference_ms))
    _progress_patch(run_id, inference_failed=1, excluded=1,
                    inference_ms_sum=inference_ms)
    with _PROGRESS_LOCK:
        counts = _PROGRESS.setdefault(run_id, {}).setdefault('status_counts', Counter())
        counts['INFERENCE_ERROR'] += 1


async def _persist_input_too_long(run_id: int, sample: dict, exc: InputTooLong,
                                  inference_ms: int) -> None:
    await _db_write_async(lambda db: _insert_input_too_long(
        db, run_id, sample, exc, inference_ms))
    _progress_patch(run_id, evaluated=1, failed=1, input_too_long=1,
                    prompt_tokens=exc.input_tokens,
                    inference_ms_sum=inference_ms)
    with _PROGRESS_LOCK:
        counts = _PROGRESS.setdefault(run_id, {}).setdefault('status_counts', Counter())
        counts['INPUT_TOO_LONG'] += 1


def _insert_inference_error(db, run_id: int, sample: dict, error: str, inference_ms: int) -> None:
    db.execute('''
            INSERT INTO results(run_id,sample_id,status,compile_success,tests_passed,tests_total,
                                model_output,inference_ms,latency_ms,error,created_at)
            VALUES(?,?,'INFERENCE_ERROR',NULL,0,?,'',?,?,?,?)
            ON CONFLICT(run_id,sample_id) DO UPDATE SET
                status='INFERENCE_ERROR', compile_success=NULL, tests_passed=0,
                model_output='', inference_ms=excluded.inference_ms,
                latency_ms=excluded.latency_ms, error=excluded.error,
                compile_log='', run_log='', detail_json='', created_at=excluded.created_at
            WHERE results.status IN ('GENERATED','INFERENCE_ERROR','INFRA_ERROR')
        ''', (run_id, sample['id'], len(json.loads(sample['tests_json'])),
              inference_ms, inference_ms, error[:1000], now()))


def _update_compiled(db, run_id: int, sample: dict, outcome: EvalOutcome,
                     inference_ms: int, sandbox_ms: int) -> bool:
    latency = inference_ms + sandbox_ms
    cur = db.execute('''
            UPDATE results SET status=?, compile_success=?, tests_passed=?, tests_total=?,
                compile_log=?, run_log=?, error=?, compile_ms=?, test_ms=?, sandbox_ms=?,
                latency_ms=?, detail_json=?, created_at=?
            WHERE run_id=? AND sample_id=? AND status='GENERATED'
        ''', (outcome.status,
              None if outcome.compile_success is None else int(outcome.compile_success),
              outcome.tests_passed, outcome.tests_total,
              outcome.compile_log, outcome.run_log, outcome.error,
              outcome.compile_ms, outcome.test_ms, sandbox_ms, latency,
              outcome.detail(), now(), run_id, sample['id']))
    return bool(cur.rowcount)


def _finalize(run_id: int, status: str) -> None:
    def write(db):
        excluded = db.execute(
            "SELECT COUNT(*) FROM results WHERE run_id=? AND status IN ('INFRA_ERROR','INFERENCE_ERROR')",
            (run_id,)).fetchone()[0]
        completed = db.execute(
            "SELECT COUNT(*) FROM results WHERE run_id=? AND status NOT IN ('GENERATED','COMPILING')",
            (run_id,)).fetchone()[0]
        pending = db.execute(
            "SELECT COUNT(*) FROM results WHERE run_id=? AND status IN ('GENERATED','COMPILING')",
            (run_id,)).fetchone()[0]
        final = status
        error = ''
        if status == 'completed' and pending:
            # A worker crash / DB error must not masquerade as a completed run.
            final = 'interrupted'
            error = f'{pending} sample(s) have no final result; select Retry to resume.'
        elif status == 'completed' and excluded:
            final = 'completed_with_errors'
        db.execute('UPDATE runs SET status=?, completed=?, finished_at=?, error=? WHERE id=?',
                   (final, completed, now(), error, run_id))
        return final
    return _db_write(write)


def _store_progress_final(run_id: int, status: str) -> None:
    _progress_set(run_id, status=status, phase='finished',
                  finished_monotonic=time.monotonic(), waiting_compile=0,
                  compiling=0, inferring=0,
                  finished_at=now())


# ---------------------------------------------------------------------------
# Compile worker.
# ---------------------------------------------------------------------------
async def _compile_one(run_id: int, sample: dict, prediction: str, inference_ms: int) -> None:
    tests = json.loads(sample['tests_json'])
    started = time.perf_counter()
    try:
        outcome = await evaluate_code_async(
            prediction, sample['dependencies'], tests, language=sample['language'],
            compile_timeout=COMPILE_TIMEOUT, test_timeout=TEST_TIMEOUT,
            log_limit=LOG_LIMIT_BYTES)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # never let one sample kill the pipeline
        outcome = EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                              error=f'{type(exc).__name__}: {str(exc)[:400]}')
    if outcome.status == 'INFRA_ERROR' and outcome.error:
        outcome.error = outcome.error[:1000]
    sandbox_ms = round((time.perf_counter() - started) * 1000)
    await _db_write_async(lambda db: _update_compiled(
        db, run_id, sample, outcome, inference_ms, sandbox_ms))
    deltas = dict(compiling=-1, evaluated=1,
                  compile_ms_sum=outcome.compile_ms, test_ms_sum=outcome.test_ms,
                  sandbox_ms_sum=sandbox_ms)
    if outcome.status != 'TEST_PASS':
        deltas['failed'] = 1
    if outcome.status == 'INFRA_ERROR':
        deltas['infra_errors'] = 1
        deltas['excluded'] = 1
    _progress_patch(run_id, **deltas)
    with _PROGRESS_LOCK:
        state = _PROGRESS.setdefault(run_id, {})
        counts = state.setdefault('status_counts', Counter())
        counts[outcome.status] += 1


async def _compile_worker(run_id: int, queue: asyncio.Queue,
                          cancel_event: asyncio.Event) -> None:
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            if cancel_event.is_set():
                return
            sample, prediction, inference_ms = item
            _progress_patch(run_id, waiting_compile=-1, compiling=1)
            await _compile_one(run_id, sample, prediction, inference_ms)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Run main loop.
# ---------------------------------------------------------------------------
async def _infer_one(run_id: int, model: dict, sample: dict, cfg: dict,
                     client: httpx.AsyncClient | None, sem: asyncio.Semaphore,
                     compile_queue: asyncio.Queue, cancel_event: asyncio.Event) -> None:
    _progress_patch(run_id, inferring=1)
    try:
        if cancel_event.is_set():
            return
        started = time.perf_counter()
        try:
            gen = await generate_prediction_async(model, sample, cfg, client)
        except InputTooLong as exc:
            await _persist_input_too_long(
                run_id, sample, exc, round((time.perf_counter() - started) * 1000))
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await _persist_inference_error(
                run_id, sample, exc, round((time.perf_counter() - started) * 1000))
            return
        await _persist_generation(run_id, sample, gen, compile_queue)
    finally:
        _progress_patch(run_id, inferring=-1)
        sem.release()


async def _watch_cancel(run_id: int, cancel_event: asyncio.Event,
                        active_tasks: set[asyncio.Task]) -> None:
    while not cancel_event.is_set():
        await asyncio.sleep(0.25)
        status = await asyncio.to_thread(_read_run_status, run_id)
        if status in ('cancel_requested', 'cancelled'):
            cancel_event.set()
            for task in list(active_tasks):
                task.cancel()
            return


async def _batch_producer(run_id: int, cfg: dict, samples: list[dict],
                          compile_queue: asyncio.Queue,
                          cancel_event: asyncio.Event) -> None:
    adapter = cfg.get('_batch_adapter')
    if adapter is None:
        adapter = OfflineBatchAdapter(cfg)
        cfg['_batch_adapter'] = adapter
    batch_size = max(1, int(cfg.get('batch_size') or OFFLINE_BATCH_SIZE))
    for offset in range(0, len(samples), batch_size):
        if cancel_event.is_set():
            return
        chunk = samples[offset:offset + batch_size]
        started = time.perf_counter()
        _progress_patch(run_id, inferring=len(chunk))
        try:
            try:
                outputs = await asyncio.to_thread(adapter.generate, chunk)
            except BatchUnavailable:
                raise
            except Exception as exc:
                outputs = [InferenceError(f'{type(exc).__name__}: {exc}') for _ in chunk]
        finally:
            _progress_patch(run_id, inferring=-len(chunk))
        elapsed = round((time.perf_counter() - started) * 1000)
        for sample, item in zip(chunk, outputs):
            if isinstance(item, InputTooLong):
                await _persist_input_too_long(run_id, sample, item, elapsed)
            elif isinstance(item, BaseException):
                await _persist_inference_error(run_id, sample, item, elapsed)
            else:
                await _persist_generation(run_id, sample, item, compile_queue)


async def _evaluate_run(run_id: int) -> None:
    loaded = await asyncio.to_thread(_load_run, run_id)
    if loaded is None:
        return
    run, model, cfg, samples, existing = loaded
    if run['status'] != 'queued':
        return
    if not await asyncio.to_thread(_mark_running, run_id):
        return

    inference_conc, compile_conc = _resolve_concurrency(cfg)
    batch_mode = cfg.get('generation_mode') == 'offline_batch'
    batch_size = max(1, int(cfg.get('batch_size') or OFFLINE_BATCH_SIZE))
    max_pending = int(cfg.get('max_pending_compile') or MAX_PENDING_COMPILE)
    compile_queue: asyncio.Queue = asyncio.Queue(maxsize=max_pending)
    cancel_event = asyncio.Event()

    to_infer: list[dict] = []
    to_compile: list[tuple[dict, str, int]] = []
    for sample in samples:
        row = existing.get(sample['id'])
        if row is None:
            to_infer.append(sample)
        elif row['status'] in ('GENERATED', 'COMPILING'):
            to_compile.append((sample, row['model_output'], int(row['inference_ms'] or 0)))
        # INFERENCE_ERROR and terminal evaluation rows are kept: the retry
        # endpoint is responsible for explicitly requesting them again.

    inference_failed = sum(1 for r in existing.values() if r['status'] == 'INFERENCE_ERROR')
    input_too_long = sum(1 for r in existing.values() if r['status'] == 'INPUT_TOO_LONG')
    terminal_rows = [r for r in existing.values() if r['status'] in TERMINAL_EVAL]
    infra = sum(1 for r in terminal_rows if r['status'] == 'INFRA_ERROR')
    status_counts = Counter(r['status'] for r in existing.values())
    # Every existing non-failed row implies a successful generation, including
    # rows that still wait for compilation.
    inference_done = len(existing) - inference_failed - input_too_long
    prompt_tokens = sum(int(r.get('prompt_tokens') or 0) for r in existing.values())
    completion_tokens = sum(int(r.get('completion_tokens') or 0) for r in existing.values())
    inference_ms_sum = sum(int(r.get('inference_ms') or 0) for r in existing.values())
    compile_ms_sum = sum(int(r.get('compile_ms') or 0) for r in existing.values())
    test_ms_sum = sum(int(r.get('test_ms') or 0) for r in existing.values())
    sandbox_ms_sum = sum(int(r.get('sandbox_ms') or 0) for r in existing.values())

    _progress_init(run_id, {
        'run_id': run_id,
        'status': 'running',
        'phase': 'running',
        'total': len(samples),
        'started_at': run['started_at'] or now(),
        'finished_at': None,
        'started_monotonic': time.monotonic(),
        'finished_monotonic': None,
        'inferring': 0,
        'in_flight': 0,
        'inference_done': inference_done,
        'inference_failed': inference_failed,
        'inference_requests': inference_done + inference_failed,
        'input_too_long': input_too_long,
        'waiting_compile': len(to_compile),
        'compiling': 0,
        'evaluated': len(terminal_rows),
        'failed': sum(1 for r in terminal_rows if r['status'] != 'TEST_PASS'),
        'infra_errors': infra,
        'excluded': inference_failed + infra,
        'inference_concurrency': inference_conc,
        'compile_concurrency': compile_conc,
        'max_pending_compile': max_pending,
        'generation_mode': 'offline_batch' if batch_mode else 'online',
        'batch_size': batch_size if batch_mode else 0,
        'max_model_len': int(cfg.get('max_model_len') or MAX_MODEL_LEN),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'inference_ms_sum': inference_ms_sum,
        'compile_ms_sum': compile_ms_sum,
        'test_ms_sum': test_ms_sum,
        'sandbox_ms_sum': sandbox_ms_sum,
        'status_counts': status_counts,
    })

    sem = asyncio.Semaphore(inference_conc)
    active_inference: set[asyncio.Task] = set()
    all_inference: list[asyncio.Task] = []
    compile_tasks: list[asyncio.Task] = []

    async def inference_producer() -> None:
        # Each iteration waits for a free slot, so exactly
        # inference_concurrency requests are in flight and a finished request
        # immediately frees one slot for the next sample.  No other request is
        # awaited, and compilation never blocks this loop.
        for sample in to_infer:
            if cancel_event.is_set():
                return
            await sem.acquire()
            if cancel_event.is_set():
                sem.release()
                return
            task = asyncio.create_task(_infer_one(
                run_id, model, sample, cfg, client, sem, compile_queue, cancel_event))
            all_inference.append(task)
            active_inference.add(task)
            task.add_done_callback(active_inference.discard)

    for _ in range(compile_conc):
        compile_tasks.append(asyncio.create_task(
            _compile_worker(run_id, compile_queue, cancel_event)))
    watchdog = asyncio.create_task(_watch_cancel(run_id, cancel_event, active_inference))
    client: httpx.AsyncClient | None = None
    if not batch_mode and model['kind'] == 'openai':
        client = httpx.AsyncClient(follow_redirects=False, timeout=None)
    if batch_mode:
        cfg['_batch_adapter'] = OfflineBatchAdapter(cfg)
    else:
        cfg['_online_adapter'] = OnlineAdapter(model, cfg, client)
    try:
        for item in to_compile:
            if cancel_event.is_set():
                break
            # Init already counted these rows as waiting; workers decrement.
            await compile_queue.put(item)
        if batch_mode:
            await _batch_producer(run_id, cfg, to_infer, compile_queue, cancel_event)
            await asyncio.gather(*all_inference, return_exceptions=True)
        else:
            await inference_producer()
            # Let every inference coroutine finish (and enqueue its prediction)
            # before telling compile workers that the stream is over.
            await asyncio.gather(*all_inference, return_exceptions=True)
        if cancel_event.is_set():
            for task in compile_tasks:
                task.cancel()
        else:
            for _ in range(compile_conc):
                await compile_queue.put(None)
        results = await asyncio.gather(*compile_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                    result, (asyncio.CancelledError,)):
                # Individual failures are handled inside the tasks; anything
                # reaching here is a programming error worth surfacing.
                _progress_patch(run_id, pipeline_errors=1)
    finally:
        if client is not None:
            await client.aclose()
        watchdog.cancel()
        try:
            await watchdog
        except (asyncio.CancelledError, Exception):
            pass
        for task in all_inference + compile_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_inference, *compile_tasks, return_exceptions=True)

    status = await asyncio.to_thread(_read_run_status, run_id)
    final = 'cancelled' if cancel_event.is_set() or status in ('cancelled', 'cancel_requested') \
        else 'completed'
    final = await asyncio.to_thread(_finalize, run_id, final)
    _store_progress_final(run_id, final)


async def _process_run(run_id: int) -> None:
    """One run at a time; queued runs wait for the global lock."""
    acquired = await asyncio.to_thread(_RUN_LOCK.acquire)
    try:
        if not acquired:
            return
        try:
            await _evaluate_run(run_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await asyncio.to_thread(_mark_interrupted, run_id, exc)
    finally:
        _RUN_LOCK.release()


_RUN_LOCK = threading.Lock()


def _mark_interrupted(run_id: int, exc: Exception) -> None:
    def write(db):
        db.execute("UPDATE runs SET status='interrupted', error=?, finished_at=? WHERE id=?",
                   (f'{type(exc).__name__}: {str(exc)[:500]}', now(), run_id))
    _db_write(write)
    _progress_set(run_id, status='interrupted', phase='finished',
                  finished_monotonic=time.monotonic())
    _progress_patch(run_id, compiling=0, inferring=0)


def process_run(run_id: int) -> None:
    """Synchronous compatibility wrapper (tests / manual retry)."""
    asyncio.run(_process_run(run_id))


# ---------------------------------------------------------------------------
# Thread dispatch.
# ---------------------------------------------------------------------------
_THREADS: dict[int, threading.Thread] = {}
_THREADS_LOCK = threading.Lock()


def _thread_main(run_id: int) -> None:
    try:
        asyncio.run(_process_run(run_id))
    except Exception as exc:
        _mark_interrupted(run_id, exc)
    finally:
        with _THREADS_LOCK:
            _THREADS.pop(run_id, None)


def submit_run(run_id: int) -> threading.Thread:
    with _THREADS_LOCK:
        thread = _THREADS.get(run_id)
        if thread is not None and thread.is_alive():
            return thread
        thread = threading.Thread(target=_thread_main, args=(run_id,),
                                  name=f'eval-run-{run_id}', daemon=True)
        _THREADS[run_id] = thread
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# Progress API used by main.py.
# ---------------------------------------------------------------------------
def _progress_from_db(run_id: int) -> dict[str, Any]:
    with connect() as db:
        run = db.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone()
        if run is None:
            return {}
        rows = [dict(r) for r in db.execute(
            'SELECT status,prompt_tokens,completion_tokens,inference_ms,compile_ms,test_ms,sandbox_ms '
            'FROM results WHERE run_id=?', (run_id,))]
    counts = Counter(r['status'] for r in rows)
    evaluated = sum(counts[s] for s in TERMINAL_EVAL)
    failed = sum(n for s, n in counts.items() if s in TERMINAL_EVAL and s != 'TEST_PASS')
    inference_failed = counts.get('INFERENCE_ERROR', 0)
    input_too_long = counts.get('INPUT_TOO_LONG', 0)
    generated = counts.get('GENERATED', 0) + counts.get('COMPILING', 0)
    prompt_tokens = sum(int(r.get('prompt_tokens') or 0) for r in rows)
    completion_tokens = sum(int(r.get('completion_tokens') or 0) for r in rows)
    inference_ms_sum = sum(int(r.get('inference_ms') or 0) for r in rows)
    compile_ms_sum = sum(int(r.get('compile_ms') or 0) for r in rows)
    test_ms_sum = sum(int(r.get('test_ms') or 0) for r in rows)
    sandbox_ms_sum = sum(int(r.get('sandbox_ms') or 0) for r in rows)
    inference_done = len(rows) - inference_failed - input_too_long
    try:
        cfg = json.loads(run['config_json'] or '{}')
    except (TypeError, ValueError):
        cfg = {}
    active = run['status'] in ('queued', 'running', 'cancel_requested')
    started_at = run['started_at'] or run['created_at']
    finished_at = run['finished_at']
    elapsed = 0.0
    try:
        from datetime import datetime
        fmt = lambda value: datetime.fromisoformat(value) if value else None
        end = fmt(finished_at)
        start = fmt(started_at)
        if start and end:
            elapsed = max(0.0, (end - start).total_seconds())
    except (TypeError, ValueError):
        elapsed = 0.0
    return {
        'run_id': run_id,
        'status': run['status'],
        'phase': 'finished' if not active else 'running',
        'total': run['total'],
        'started_at': started_at,
        'finished_at': finished_at,
        'elapsed_seconds': round(elapsed, 2),
        'waiting_inference': max(0, run['total'] - len(rows)) if active else 0,
        'inferring': 0,
        'in_flight': 0,
        'inference_done': inference_done,
        'inference_failed': inference_failed,
        'inference_requests': inference_done + inference_failed,
        'input_too_long': input_too_long,
        'waiting_compile': generated,
        'compiling': 0,
        'evaluated': evaluated,
        'failed': failed,
        'infra_errors': counts.get('INFRA_ERROR', 0),
        'excluded': inference_failed + counts.get('INFRA_ERROR', 0),
        'inference_concurrency': int(cfg.get('inference_concurrency') or 0) if active else 0,
        'compile_concurrency': int(cfg.get('compile_concurrency') or 0) if active else 0,
        'max_pending_compile': int(cfg.get('max_pending_compile') or MAX_PENDING_COMPILE),
        'generation_mode': cfg.get('generation_mode') or 'online',
        'batch_size': int(cfg.get('batch_size') or 0),
        'max_model_len': int(cfg.get('max_model_len') or MAX_MODEL_LEN),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'inference_ms_sum': inference_ms_sum,
        'compile_ms_sum': compile_ms_sum,
        'test_ms_sum': test_ms_sum,
        'sandbox_ms_sum': sandbox_ms_sum,
        'inference_ms_avg': round(inference_ms_sum / inference_done, 1) if inference_done else 0.0,
        'inference_rate': round(inference_done / elapsed, 3) if elapsed else 0.0,
        'eval_rate': round(evaluated / elapsed, 3) if elapsed else 0.0,
        'token_rate': round(completion_tokens / elapsed, 1) if elapsed else 0.0,
        'total_token_rate': round((prompt_tokens + completion_tokens) / elapsed, 1) if elapsed else 0.0,
        'status_counts': dict(counts),
    }


def get_progress(run_id: int) -> dict[str, Any]:
    snap = _progress_get(run_id)
    if snap:
        return _summarize_snapshot(snap)
    return _progress_from_db(run_id)


def clear_progress(run_id: int) -> None:
    """Drop stale in-memory counters (delete/retry)."""
    with _PROGRESS_LOCK:
        _PROGRESS.pop(run_id, None)
