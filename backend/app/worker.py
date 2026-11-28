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
import hashlib
import json
import os
import random
import re
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
    DIAGNOSTICS_ENABLED, MAX_DIAGNOSTIC_CASES,
    AUTO_INITIAL_INFLIGHT, AUTO_MAX_INFLIGHT, AUTO_STEP,
    COMPILE_MEMORY_FRACTION, COMPILE_MEMORY_PER_WORKER_MB,
)
from .case_extract import DIAG_HEADER, METRIC_VERSION, PARSER_VERSION, extract_manifest
from .db import connect, now
from .adapters import (
    VllmTokenizer,
    BatchUnavailable, Generation, InferenceError, InputTooLong, OfflineBatchAdapter,
    OnlineAdapter, post_with_retries as _post_with_retries,
)
from .sandbox import EvalOutcome, evaluate_code_async, extract_c
from .host_sandbox import evaluate_code_host_async
from .exebench import evaluate_exebench_async, diff_io
from .edit_sim import edit_similarity

EXCLUDED = {'INFRA_ERROR', 'INFERENCE_ERROR', 'ORACLE_ERROR', 'INVALID_BENCHMARK'}
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


def _cached_manifest(sample: dict, harness: str, language: str) -> dict:
    digest = hashlib.sha256((language + '\0' + harness).encode('utf-8')).hexdigest()
    with connect() as db:
        row = db.execute(
            'SELECT manifest_json FROM case_manifests WHERE sample_id=? AND harness_hash=? AND parser_version=?',
            (sample['id'], digest, PARSER_VERSION)).fetchone()
        if row is not None:
            try:
                return json.loads(row['manifest_json'])
            except ValueError:
                pass
    manifest = extract_manifest(harness, language)
    data = manifest.as_dict()
    with _WRITE_LOCK:
        with connect() as db:
            db.execute('''INSERT OR REPLACE INTO case_manifests
                (sample_id,harness_hash,parser_version,status,manifest_json,created_at)
                VALUES(?,?,?,?,?,?)''',
                (sample['id'], digest, PARSER_VERSION, data['status'],
                 json.dumps(data, ensure_ascii=False), now()))
    return data


def _build_diagnostics(samples: list[dict]) -> dict[int, dict]:
    """Load/cache a per-sample test-case manifest (tree-sitter extraction)."""
    result: dict[int, dict] = {}
    for sample in samples:
        try:
            tests = json.loads(sample['tests_json'])
        except (TypeError, ValueError):
            result[sample['id']] = {'status': 'INVALID_HARNESS', 'diagnostic': None, 'manifest': {}}
            continue
        if not DIAGNOSTICS_ENABLED:
            result[sample['id']] = {'status': 'DISABLED', 'diagnostic': None, 'manifest': {}}
            continue
        if not (len(tests) == 1 and 'program' in tests[0]):
            result[sample['id']] = {
                'status': 'DISCRETE', 'diagnostic': None,
                'manifest': {'mode': 'discrete', 'static_case_count': len(tests)},
            }
            continue
        language = 'cpp' if sample['language'] == 'cpp' else 'c'
        try:
            data = _cached_manifest(sample, tests[0]['program'], language)
        except Exception as exc:  # never break a run because diagnostics failed
            data = {'status': 'EXTRACTION_ERROR',
                    'reason': f'{type(exc).__name__}: {str(exc)[:300]}',
                    'cases': [], 'rewritten_harness': ''}
        diagnostic = None
        if data.get('status') == 'OK' and data.get('rewritten_harness'):
            diagnostic = {
                'harness': data['rewritten_harness'],
                'header': DIAG_HEADER,
                'max_cases': MAX_DIAGNOSTIC_CASES,
            }
        result[sample['id']] = {
            'status': data.get('status', 'UNSUPPORTED_CASE_EXTRACTION'),
            'diagnostic': diagnostic,
            'manifest': data,
            'language': language,
        }
    return result


def _value_json(text, supported: bool) -> str:
    if not supported:
        return ''
    value = '' if text is None else str(text)
    if value == '':
        return ''
    if re.fullmatch(r'-?\d+', value):
        return json.dumps(int(value))
    try:
        return json.dumps(float(value))
    except ValueError:
        return json.dumps(value)


def _case_error_type(event: dict) -> str:
    if event.get('passed'):
        return 'NONE'
    if event.get('timed_out'):
        return 'TIMEOUT'
    if event.get('signal') is not None:
        return 'RUNTIME'
    return 'ASSERTION'


def _case_source_span(manifest_case: dict) -> str:
    if not manifest_case:
        return ''
    return json.dumps({
        'start_line': manifest_case.get('line'),
        'start_column': manifest_case.get('column'),
        'end_line': manifest_case.get('end_line'),
        'end_column': manifest_case.get('end_column'),
        'span': manifest_case.get('span'),
    }, ensure_ascii=False)


def _case_row(**overrides) -> dict:
    row = {
        'case_id': '', 'case_index': 0, 'static_id': -1,
        'case_source_span': '', 'input_text': '', 'input_values_json': '',
        'expected_text': '', 'expected_value_json': '',
        'actual_text': '', 'actual_value_json': '',
        'compare_expression': '', 'comparison_rule': '',
        'passed': 0, 'status': 'NOT_EXECUTED',
        'exit_code': None, 'signal': None,
        'stdout': '', 'stderr': '',
        'stdout_truncated': 0, 'stderr_truncated': 0,
        'duration_ms': 0, 'oracle_source': '', 'diagnostic_status': '',
        'error_type': '',
    }
    row.update(overrides)
    return row


def _discrete_cases(outcome, tests: list[dict], compile_failed: bool):
    rows = []
    events = {int(ev.get('index', 0)): ev for ev in outcome.tests}
    for index, test in enumerate(tests):
        code = test.get('code') or test.get('program') or ''
        event = events.get(index)
        base = dict(case_id=f't{index}', case_index=index, static_id=index,
                    input_text=code[:150000],
                    compare_expression='', comparison_rule='discrete_test',
                    oracle_source='discrete_harness', diagnostic_status='DISCRETE')
        if compile_failed:
            rows.append(_case_row(**{**base, 'status': 'NOT_RUN_COMPILE_ERROR',
                                     'error_type': 'COMPILE_ERROR'}))
            continue
        if event is None:
            rows.append(_case_row(**{**base, 'status': 'NOT_EXECUTED',
                                     'diagnostic_status': 'DISCRETE;NOT_EXECUTED'}))
            continue
        passed = event.get('status') == 'PASS'
        rows.append(_case_row(
            **base,
            passed=int(passed),
            status='PASS' if passed else 'FAIL',
            exit_code=event.get('exit_code'),
            signal=event.get('signal'),
            stdout=str(event.get('stdout') or '')[:150000],
            stderr=str(event.get('stderr') or '')[:150000],
            stdout_truncated=int(bool(event.get('stdout_truncated'))),
            stderr_truncated=int(bool(event.get('stderr_truncated'))),
            duration_ms=int(event.get('duration_ms') or 0),
            error_type=_case_error_type(event),
        ))
    executed = [row for row in rows if row['status'] in ('PASS', 'FAIL')]
    status = 'NOT_RUN_COMPILE_ERROR' if compile_failed else 'OK'
    return rows, len(tests), len(executed), sum(r['passed'] for r in executed), status


def _program_cases(outcome, info: dict, compile_failed: bool):
    manifest = info.get('manifest') or {}
    manifest_cases = {int(c['static_id']): c for c in manifest.get('cases', [])
                      if isinstance(c, dict) and 'static_id' in c}
    declared = len(manifest_cases)
    if compile_failed:
        rows = []
        for sid, case in sorted(manifest_cases.items()):
            rows.append(_case_row(
                case_id=f'a{sid}', case_index=sid + 1, static_id=sid,
                case_source_span=_case_source_span(case),
                input_text=(case.get('lhs') or case.get('expression') or '')[:150000],
                expected_text=(case.get('rhs') or '1')[:150000],
                compare_expression=(case.get('expression') or '')[:150000],
                comparison_rule=case.get('comparison_rule') or '',
                status='NOT_RUN_COMPILE_ERROR', oracle_source='assertion',
                diagnostic_status='NOT_RUN_COMPILE_ERROR', error_type='COMPILE_ERROR'))
        return rows, declared, 0, 0, 'NOT_RUN_COMPILE_ERROR'

    if info.get('status') != 'OK':
        rows = []
        for sid, case in sorted(manifest_cases.items()):
            rows.append(_case_row(
                case_id=f'a{sid}', case_index=sid + 1, static_id=sid,
                case_source_span=_case_source_span(case),
                input_text=(case.get('lhs') or case.get('expression') or '')[:150000],
                expected_text=(case.get('rhs') or '1')[:150000],
                compare_expression=(case.get('expression') or '')[:150000],
                comparison_rule=case.get('comparison_rule') or '',
                status='NOT_RUN_DIAG_ERROR', oracle_source='assertion',
                diagnostic_status=info.get('status') or 'UNSUPPORTED_CASE_EXTRACTION',
                error_type=info.get('status') or 'UNSUPPORTED_CASE_EXTRACTION'))
        return rows, declared, 0, 0, info.get('status') or 'UNSUPPORTED_CASE_EXTRACTION'

    events = list(outcome.cases)
    seen_static: dict[int, int] = {}
    for event in events:
        sid = int(event.get('static_id', -1))
        seen_static[sid] = seen_static.get(sid, 0) + 1
    instances: dict[int, int] = {}
    rows = []
    executed_static: set[int] = set()
    for event in events:
        sid = int(event.get('static_id', -1))
        manifest_case = manifest_cases.get(sid, {})
        instance = instances.get(sid, 0)
        instances[sid] = instance + 1
        executed_static.add(sid)
        case_id = f'a{sid}' + (f'#{instance}' if seen_static.get(sid, 0) > 1 else '')
        passed = bool(event.get('passed'))
        diagnostic_status = 'OK'
        if event.get('after_prior_failure'):
            diagnostic_status = 'AFTER_PRIOR_FAILURE'
        if not event.get('actual_ok') or not event.get('expected_ok'):
            diagnostic_status += ';VALUE_UNSUPPORTED'
        op = manifest_case.get('comparison_operator') or ''
        rows.append(_case_row(
            case_id=case_id,
            case_index=int(event.get('index') or 0),
            static_id=sid,
            case_source_span=_case_source_span(manifest_case),
            input_text=(manifest_case.get('lhs') or event.get('lhs')
                        or event.get('expression') or '')[:150000],
            expected_text=(manifest_case.get('rhs') or event.get('expected') or '')[:150000],
            expected_value_json=_value_json(event.get('expected'), bool(event.get('expected_ok'))),
            actual_text=(event.get('actual') or '')[:150000],
            actual_value_json=_value_json(event.get('actual'), bool(event.get('actual_ok'))),
            compare_expression=(event.get('expression') or '')[:150000],
            comparison_rule=(op or manifest_case.get('comparison_rule') or 'expression'),
            passed=int(passed),
            status='PASS' if passed else 'FAIL',
            exit_code=event.get('exit_code'),
            signal=event.get('signal'),
            stdout=str(event.get('stdout') or '')[:150000],
            stderr=str(event.get('stderr') or '')[:150000],
            stdout_truncated=int(bool(event.get('stdout_truncated'))),
            stderr_truncated=int(bool(event.get('stderr_truncated'))),
            duration_ms=int(event.get('duration_ms') or 0),
            oracle_source='assertion',
            diagnostic_status=diagnostic_status,
            error_type=_case_error_type(event),
        ))
    # Static, non-dynamic cases that never executed are reported explicitly.
    for sid, case in sorted(manifest_cases.items()):
        if sid in executed_static or case.get('dynamic'):
            continue
        rows.append(_case_row(
            case_id=f'a{sid}', case_index=sid + 1, static_id=sid,
            case_source_span=_case_source_span(case),
            input_text=(case.get('lhs') or case.get('expression') or '')[:150000],
            expected_text=(case.get('rhs') or '1')[:150000],
            compare_expression=(case.get('expression') or '')[:150000],
            comparison_rule=case.get('comparison_rule') or '',
            status='NOT_EXECUTED', oracle_source='assertion',
            diagnostic_status='NOT_EXECUTED', error_type='NOT_EXECUTED'))
    executed = [row for row in rows if row['status'] in ('PASS', 'FAIL')]
    status = outcome.diagnostic_status or ('OK' if executed else 'NO_CASES_EXECUTED')
    return rows, declared, len(executed), sum(r['passed'] for r in executed), status


def _exebench_cases(outcome, tests: list[dict]):
    rows = []
    for index, test in enumerate(tests):
        event = next((x for x in outcome.tests if int(x.get('index', -1)) == index), None)
        input_text = json.dumps(test.get('input'), ensure_ascii=False)
        expected = json.dumps(test.get('expected_output'), ensure_ascii=False)
        if event is None:
            rows.append(_case_row(
                case_id=f'e{index}', case_index=index, static_id=index,
                input_text=input_text, expected_text=expected,
                compare_expression='', comparison_rule='exebench_diff_io',
                oracle_source='exebench_wrapper', diagnostic_status='EXEBENCH',
                status='NOT_RUN_ORACLE_ERROR' if outcome.status == 'ORACLE_ERROR' else 'NOT_EXECUTED',
                error_type=outcome.status if outcome.status == 'ORACLE_ERROR' else 'NOT_EXECUTED'))
            continue
        actual = event.get('actual_output')
        rows.append(_case_row(
            case_id=f'e{index}', case_index=index, static_id=index,
            input_text=input_text, expected_text=expected,
            input_values_json=input_text, expected_value_json=expected,
            actual_text=json.dumps(actual, ensure_ascii=False) if actual is not None else '',
            actual_value_json=json.dumps(actual, ensure_ascii=False) if actual is not None else '',
            compare_expression='', comparison_rule='exebench_diff_io',
            passed=int(event.get('status') == 'PASS'),
            status='PASS' if event.get('status') == 'PASS' else 'FAIL',
            exit_code=event.get('exit_code'), stdout=str(event.get('stdout') or '')[:150000],
            stderr=str(event.get('stderr') or '')[:150000],
            duration_ms=int(event.get('duration_ms') or 0),
            oracle_source='exebench_wrapper', diagnostic_status='EXEBENCH',
            error_type=event.get('status') if event.get('status') != 'PASS' else 'NONE'))
    executed = [r for r in rows if r['status'] in ('PASS', 'FAIL')]
    if outcome.status in ('ORACLE_ERROR', 'INVALID_BENCHMARK'):
        status = outcome.status
    elif outcome.status in ('COMPILE_ERROR', 'LINK_ERROR', 'COMPILE_TIMEOUT'):
        status = 'NOT_RUN_COMPILE_ERROR'
    else:
        status = 'OK'
    return rows, len(tests), len(executed), sum(r['passed'] for r in executed), status


def _build_case_rows(outcome, sample: dict, info: dict):
    try:
        tests = json.loads(sample['tests_json'])
    except (TypeError, ValueError):
        return [], 0, 0, 0, 'INVALID_HARNESS'
    if sample.get('test_protocol', '').startswith('exebench'):
        return _exebench_cases(outcome, tests)
    try:
        tests = json.loads(sample['tests_json'])
    except (TypeError, ValueError):
        return [], 0, 0, 0, 'INVALID_HARNESS'
    compile_failed = outcome.status in ('COMPILE_ERROR', 'COMPILE_TIMEOUT')
    if info.get('status') == 'DISCRETE':
        return _discrete_cases(outcome, tests, compile_failed)
    return _program_cases(outcome, info, compile_failed)


def _replace_cases(db, run_id: int, sample_id: int, rows: list[dict]) -> None:
    db.execute('DELETE FROM test_case_results WHERE run_id=? AND sample_id=?',
               (run_id, sample_id))
    for row in rows:
        db.execute('''INSERT OR REPLACE INTO test_case_results
            (run_id,sample_id,case_id,case_index,static_id,case_source_span,
             input_text,input_values_json,expected_text,expected_value_json,
             actual_text,actual_value_json,compare_expression,comparison_rule,
             passed,status,exit_code,signal,stdout,stderr,stdout_truncated,
             stderr_truncated,duration_ms,oracle_source,diagnostic_status,
             error_type,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (run_id, sample_id, row['case_id'], row['case_index'], row['static_id'],
             row['case_source_span'], row['input_text'], row['input_values_json'],
             row['expected_text'], row['expected_value_json'], row['actual_text'],
             row['actual_value_json'], row['compare_expression'], row['comparison_rule'],
             int(row['passed']), row['status'], row['exit_code'], row['signal'],
             row['stdout'], row['stderr'], int(row['stdout_truncated']),
             int(row['stderr_truncated']), row['duration_ms'], row['oracle_source'],
             row['diagnostic_status'], row['error_type'], now()))


def _memory_available_mb() -> int | None:
    try:
        with open('/proc/meminfo', encoding='ascii') as fp:
            for line in fp:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _compile_limit(requested: int) -> int:
    """Cap Docker workers by available RAM and CPU, never above the request."""
    requested = max(1, min(int(requested), MAX_COMPILE_CONCURRENCY))
    available = _memory_available_mb()
    cpu = os.cpu_count() or 1
    if available is not None:
        memory_cap = max(1, int(available * COMPILE_MEMORY_FRACTION
                                 // COMPILE_MEMORY_PER_WORKER_MB))
        requested = min(requested, memory_cap)
    return max(1, min(requested, max(1, cpu)))


def _resolve_concurrency(cfg: dict) -> tuple[int, int]:
    raw_inference = cfg.get('inference_concurrency')
    inference = INFERENCE_CONCURRENCY if raw_inference is None else raw_inference
    inference = max(0, min(int(inference), MAX_INFERENCE_CONCURRENCY))
    compile_requested = int(cfg.get('compile_concurrency') or COMPILE_CONCURRENCY)
    compile_workers = _compile_limit(compile_requested)
    return inference, compile_workers


def _compile_requested(cfg: dict) -> int:
    return max(1, min(int(cfg.get('compile_concurrency') or COMPILE_CONCURRENCY),
                      MAX_COMPILE_CONCURRENCY))


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


def _insert_generated(db, run_id: int, sample: dict, gen: Generation,
                      submitted_at: str = '', generated_at: str = '') -> None:
    db.execute('''
            INSERT INTO results(run_id,sample_id,status,compile_success,tests_passed,tests_total,
                                model_output,inference_ms,latency_ms,prompt_tokens,completion_tokens,
                                prompt_text,generation_params_json,submitted_at,generated_at,created_at)
            VALUES(?,?,'GENERATED',NULL,0,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(run_id,sample_id) DO UPDATE SET
                status='GENERATED', compile_success=NULL, tests_passed=0,
                tests_total=excluded.tests_total, model_output=excluded.model_output,
                inference_ms=excluded.inference_ms, latency_ms=excluded.latency_ms,
                prompt_tokens=excluded.prompt_tokens, completion_tokens=excluded.completion_tokens,
                prompt_text=excluded.prompt_text,
                generation_params_json=excluded.generation_params_json,
                submitted_at=excluded.submitted_at, generated_at=excluded.generated_at,
                first_token_at='', compile_started_at='', compile_finished_at='',
                compile_log='', run_log='', compile_ms=0, test_ms=0,
                error='', detail_json='', created_at=excluded.created_at
            WHERE results.status IN ('GENERATED','INFERENCE_ERROR','INFRA_ERROR')
        ''', (run_id, sample['id'], len(json.loads(sample['tests_json'])),
              gen.text[:150000], gen.inference_ms, gen.inference_ms,
              gen.prompt_tokens, gen.completion_tokens, gen.prompt[:150000],
              json.dumps(gen.params, ensure_ascii=False), submitted_at, generated_at, now()))


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
                              compile_queue: asyncio.Queue,
                              submitted_at: str = '', generated_at: str = '') -> None:
    await _db_write_async(lambda db: _insert_generated(
        db, run_id, sample, gen, submitted_at, generated_at or now()))
    _progress_patch(run_id, inference_done=1, waiting_compile=1,
                    prompt_tokens=gen.prompt_tokens,
                    completion_tokens=gen.completion_tokens,
                    inference_ms_sum=gen.inference_ms)
    await compile_queue.put((sample, gen.text, gen.inference_ms,
                             submitted_at, generated_at or now()))


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
                     inference_ms: int, sandbox_ms: int, case_rows: list[dict],
                     cases_declared: int, cases_total: int, cases_passed: int,
                     test_case_status: str,
                     compile_started_at: str = '', compile_finished_at: str = '',
                     edit_sim_value: float | None = None) -> bool:
    latency = inference_ms + sandbox_ms
    source_ok = outcome.source_compile_success
    if source_ok is None and outcome.compile_success is not None:
        # Backward compatibility for synthetic events/tests that only emit the
        # full-harness compile event.
        source_ok = outcome.compile_success
    stage = outcome.compile_stage
    if not stage and outcome.compile_success is not None:
        stage = 'ok' if outcome.compile_success else 'harness'
    cur = db.execute('''
            UPDATE results SET status=?, compile_success=?, source_compile_ok=?,
                compile_stage=?, tests_passed=?, tests_total=?,
                compile_log=?, run_log=?, error=?, compile_ms=?, test_ms=?, sandbox_ms=?,
                latency_ms=?, cases_declared=?, cases_total=?, cases_passed=?,
                test_case_status=?, metric_version=?, compile_started_at=?,
                compile_finished_at=?, edit_similarity=?, detail_json=?, created_at=?
            WHERE run_id=? AND sample_id=? AND status='GENERATED'
        ''', (outcome.status,
              None if outcome.compile_success is None else int(outcome.compile_success),
              None if source_ok is None else int(source_ok), stage,
              outcome.tests_passed, outcome.tests_total,
              outcome.compile_log, outcome.run_log, outcome.error,
              outcome.compile_ms, outcome.test_ms, sandbox_ms, latency,
              cases_declared, cases_total, cases_passed, test_case_status,
              METRIC_VERSION, compile_started_at, compile_finished_at,
              edit_sim_value, outcome.detail(), now(), run_id, sample['id']))
    if cur.rowcount:
        _replace_cases(db, run_id, sample['id'], case_rows)
    return bool(cur.rowcount)


def _finalize(run_id: int, status: str) -> None:
    def write(db):
        excluded = db.execute(
            "SELECT COUNT(*) FROM results WHERE run_id=? AND status IN (?,?,?,?)",
            (run_id,'INFRA_ERROR','INFERENCE_ERROR','ORACLE_ERROR','INVALID_BENCHMARK')).fetchone()[0]
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
        db.execute('UPDATE runs SET status=?, completed=?, finished_at=?, error=?, '
                   'metric_version=? WHERE id=?',
                   (final, completed, now(), error, METRIC_VERSION, run_id))
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
def _exebench_signature(sample: dict) -> str:
    h = hashlib.sha256()
    for key in ('reference_source', 'dependencies', 'test_wrapper',
                'tests_json', 'real_iospec', 'language', 'compiler', 'optimization'):
        h.update(str(sample.get(key) or '').encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


async def _ensure_exebench_oracle(sample: dict) -> bool:
    signature = _exebench_signature(sample)
    if (sample.get('oracle_signature') == signature and
            sample.get('oracle_validated') in (1, -1)):
        return sample.get('oracle_validated') == 1
    tests = json.loads(sample['tests_json'])
    outcome = await evaluate_exebench_async(
        sample['reference_source'], sample['reference_source'],
        sample.get('dependencies') or '', sample.get('test_wrapper') or '',
        tests, function_signature=sample.get('function_signature') or '',
        language=sample.get('language') or 'c',
        compile_timeout=COMPILE_TIMEOUT, test_timeout=TEST_TIMEOUT,
        log_limit=LOG_LIMIT_BYTES)
    valid = outcome.status == 'TEST_PASS'
    passed = int(outcome.tests_passed or 0)
    failed = max(0, len(tests) - passed)
    error = '' if valid else (outcome.error or outcome.status)
    def write(db):
        db.execute('''UPDATE samples SET oracle_validated=?,oracle_test_count=?,
            oracle_passed_count=?,oracle_failed_count=?,oracle_error=?,
            oracle_detail_json=?,oracle_signature=?,oracle_checked_at=? WHERE id=?''',
            (1 if valid else -1, len(tests), passed, failed, error[:2000],
             json.dumps(outcome.tests, ensure_ascii=False), signature, now(), sample['id']))
    await _db_write_async(write)
    sample['oracle_validated'] = 1 if valid else -1
    sample['oracle_signature'] = signature
    return valid


async def _compile_one(run_id: int, sample: dict, prediction: str, inference_ms: int,
                       diagnostics: dict | None = None,
                       submitted_at: str = '', generated_at: str = '',
                       execution_mode: str = 'docker') -> None:
    tests = json.loads(sample['tests_json'])
    info = (diagnostics or {}).get(sample['id']) or {'status': 'DISABLED', 'manifest': {}}
    compile_started_at = now()
    started = time.perf_counter()
    try:
        if str(sample.get('test_protocol') or '').startswith('exebench'):
            if not await _ensure_exebench_oracle(sample):
                outcome = EvalOutcome(
                    'ORACLE_ERROR', False, 0, len(tests),
                    error=f'ExeBench reference Oracle failed: {sample.get("oracle_error") or "see sample oracle fields"}',
                    harness_mode='exebench', compile_stage='oracle')
            else:
                outcome = await evaluate_exebench_async(
                    prediction, sample['reference_source'],
                    sample.get('dependencies') or '', sample.get('test_wrapper') or '',
                    tests, function_signature=sample.get('function_signature') or '',
                    language=sample.get('language') or 'c',
                    compile_timeout=COMPILE_TIMEOUT, test_timeout=TEST_TIMEOUT,
                    log_limit=LOG_LIMIT_BYTES)
        elif execution_mode == 'host':
            outcome = await evaluate_code_host_async(
                prediction, sample['dependencies'], tests, language=sample['language'],
                compile_timeout=COMPILE_TIMEOUT, test_timeout=TEST_TIMEOUT,
                log_limit=LOG_LIMIT_BYTES, diagnostic=info.get('diagnostic'))
        else:
            outcome = await evaluate_code_async(
                prediction, sample['dependencies'], tests, language=sample['language'],
                compile_timeout=COMPILE_TIMEOUT, test_timeout=TEST_TIMEOUT,
                log_limit=LOG_LIMIT_BYTES, diagnostic=info.get('diagnostic'))
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # never let one sample kill the pipeline
        outcome = EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                              error=f'{type(exc).__name__}: {str(exc)[:400]}')
    if outcome.status == 'INFRA_ERROR' and outcome.error:
        outcome.error = outcome.error[:1000]
    sandbox_ms = round((time.perf_counter() - started) * 1000)
    case_rows, cases_declared, cases_total, cases_passed, case_status = \
        _build_case_rows(outcome, sample, info)
    compile_finished_at = now()
    edit_sim_value = round(edit_similarity(
        sample.get('reference_source') or '', extract_c(prediction)), 6)
    await _db_write_async(lambda db: _update_compiled(
        db, run_id, sample, outcome, inference_ms, sandbox_ms, case_rows,
        cases_declared, cases_total, cases_passed, case_status,
        compile_started_at, compile_finished_at, edit_sim_value))
    deltas = dict(compiling=-1, evaluated=1,
                  compile_ms_sum=outcome.compile_ms, test_ms_sum=outcome.test_ms,
                  sandbox_ms_sum=sandbox_ms, cases_declared=cases_declared,
                  cases_total=cases_total, cases_passed=cases_passed)
    if cases_total:
        deltas['diagnosed_samples'] = 1
    elif cases_declared:
        deltas['undiagnosed_samples'] = 1
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
                          cancel_event: asyncio.Event,
                          diagnostics: dict | None = None,
                          execution_mode: str = 'docker') -> None:
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            if cancel_event.is_set():
                return
            sample, prediction, inference_ms, submitted_at, generated_at = item
            _progress_patch(run_id, waiting_compile=-1, compiling=1)
            await _compile_one(run_id, sample, prediction, inference_ms, diagnostics,
                               submitted_at, generated_at, execution_mode=execution_mode)
        finally:
            queue.task_done()


# ---------------------------------------------------------------------------
# Run main loop.
# ---------------------------------------------------------------------------
async def _infer_one(run_id: int, model: dict, sample: dict, cfg: dict,
                     client: httpx.AsyncClient | None,
                     sem: asyncio.Semaphore | None,
                     compile_queue: asyncio.Queue, cancel_event: asyncio.Event) -> None:
    _progress_patch(run_id, inferring=1)
    try:
        if cancel_event.is_set():
            return
        submitted_at = now()
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
        await _persist_generation(run_id, sample, gen, compile_queue,
                                  submitted_at, now())
    finally:
        _progress_patch(run_id, inferring=-1)
        if sem is not None:
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
                await _persist_generation(run_id, sample, item, compile_queue,
                                          now(), now())


async def _auto_inference_producer(run_id: int, model: dict, samples: list[dict],
                                   cfg: dict, client: httpx.AsyncClient | None,
                                   compile_queue: asyncio.Queue,
                                   cancel_event: asyncio.Event,
                                   active: set[asyncio.Task],
                                   all_tasks: list[asyncio.Task]) -> None:
    """Continuous supply without a batch barrier (INFERENCE_CONCURRENCY=0).

    The ceiling starts small and grows while work is completing, but is always
    bounded by AUTO_MAX_INFLIGHT and by the bounded compile queue.  A finished
    request frees a slot immediately; there is no ``gather`` over a whole batch.
    """
    pending = list(samples)
    position = 0
    limit = max(1, min(AUTO_INITIAL_INFLIGHT, AUTO_MAX_INFLIGHT))
    _progress_set(run_id, auto_limit=limit, auto_max_inflight=AUTO_MAX_INFLIGHT)
    while (position < len(pending) or active) and not cancel_event.is_set():
        while (position < len(pending) and len(active) < limit
               and not cancel_event.is_set()):
            sample = pending[position]
            position += 1
            task = asyncio.create_task(_infer_one(
                run_id, model, sample, cfg, client, None, compile_queue, cancel_event))
            all_tasks.append(task)
            active.add(task)
            task.add_done_callback(active.discard)
        if not active:
            break
        done, _ = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
        active.difference_update(done)
        # Only raise the ceiling while there is still queued work; otherwise the
        # reported AUTO limit would keep growing after the dataset is drained.
        if position < len(pending) and limit < AUTO_MAX_INFLIGHT:
            limit = min(AUTO_MAX_INFLIGHT, limit + AUTO_STEP)
            _progress_set(run_id, auto_limit=limit)
    if active:
        await asyncio.gather(*list(active), return_exceptions=True)


async def _evaluate_run(run_id: int) -> None:
    loaded = await asyncio.to_thread(_load_run, run_id)
    if loaded is None:
        return
    run, model, cfg, samples, existing = loaded
    if run['status'] != 'queued':
        return
    if not await asyncio.to_thread(_mark_running, run_id):
        return

    diagnostics = await asyncio.to_thread(_build_diagnostics, samples)
    inference_conc, compile_conc = _resolve_concurrency(cfg)
    compile_requested = _compile_requested(cfg)
    auto_mode = inference_conc == 0
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
            to_compile.append((sample, row['model_output'], int(row['inference_ms'] or 0),
                               row.get('submitted_at') or '', row.get('generated_at') or ''))
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
    cases_declared = sum(int(r.get('cases_declared') or 0) for r in existing.values())
    cases_total = sum(int(r.get('cases_total') or 0) for r in existing.values())
    cases_passed = sum(int(r.get('cases_passed') or 0) for r in existing.values())
    diagnosed_samples = sum(1 for r in existing.values()
                            if int(r.get('cases_total') or 0) > 0
                            and (r.get('test_case_status') or '') in ('OK', 'DISCRETE', 'PARTIAL'))
    undiagnosed_samples = sum(1 for r in existing.values()
                              if int(r.get('cases_total') or 0) == 0
                              and (r.get('test_case_status') or '') not in ('', 'NOT_RUN_COMPILE_ERROR'))
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
        'auto_limit': 0,
        'auto_max_inflight': AUTO_MAX_INFLIGHT if auto_mode else 0,
        'compile_concurrency': compile_conc,
        'compile_concurrency_requested': compile_requested,
        'max_pending_compile': max_pending,
        'generation_mode': 'offline_batch' if batch_mode else 'online',
        'execution_mode': cfg.get('execution_mode', 'docker'),
        'batch_size': batch_size if batch_mode else 0,
        'max_model_len': int(cfg.get('max_model_len') or MAX_MODEL_LEN),
        'prompt_tokens': prompt_tokens,
        'completion_tokens': completion_tokens,
        'inference_ms_sum': inference_ms_sum,
        'compile_ms_sum': compile_ms_sum,
        'test_ms_sum': test_ms_sum,
        'sandbox_ms_sum': sandbox_ms_sum,
        'cases_declared': cases_declared,
        'cases_total': cases_total,
        'cases_passed': cases_passed,
        'diagnosed_samples': diagnosed_samples,
        'undiagnosed_samples': undiagnosed_samples,
        'status_counts': status_counts,
    })

    sem = None if auto_mode else asyncio.Semaphore(inference_conc)
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
            _compile_worker(run_id, compile_queue, cancel_event, diagnostics,
                            execution_mode=cfg.get('execution_mode', 'docker'))))
    watchdog = asyncio.create_task(_watch_cancel(run_id, cancel_event, active_inference))
    client: httpx.AsyncClient | None = None
    tokenizer_client: httpx.AsyncClient | None = None
    if not batch_mode and model['kind'] == 'openai':
        pool = (AUTO_MAX_INFLIGHT if auto_mode else inference_conc) + 32
        limits = httpx.Limits(max_connections=pool, max_keepalive_connections=pool)
        client = httpx.AsyncClient(follow_redirects=False, timeout=None, limits=limits)
        tokenizer_client = httpx.AsyncClient(
            follow_redirects=False, timeout=None,
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=64))
    if batch_mode:
        cfg['_batch_adapter'] = OfflineBatchAdapter(cfg)
    else:
        tokenizer = None
        if model['kind'] == 'openai' and tokenizer_client is not None:
            tokenizer = VllmTokenizer(tokenizer_client, model['base_url'], model['api_model'])
        cfg['_online_adapter'] = OnlineAdapter(model, cfg, client, tokenizer=tokenizer)
    try:
        for item in to_compile:
            if cancel_event.is_set():
                break
            # Init already counted these rows as waiting; workers decrement.
            await compile_queue.put(item)
        if batch_mode:
            await _batch_producer(run_id, cfg, to_infer, compile_queue, cancel_event)
            await asyncio.gather(*all_inference, return_exceptions=True)
        elif auto_mode:
            await _auto_inference_producer(run_id, model, to_infer, cfg, client,
                                           compile_queue, cancel_event,
                                           active_inference, all_inference)
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
        if tokenizer_client is not None:
            await tokenizer_client.aclose()
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
            'SELECT status,prompt_tokens,completion_tokens,inference_ms,compile_ms,test_ms,sandbox_ms,'
            'cases_declared,cases_total,cases_passed,test_case_status '
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
    cases_declared = sum(int(r.get('cases_declared') or 0) for r in rows)
    cases_total = sum(int(r.get('cases_total') or 0) for r in rows)
    cases_passed = sum(int(r.get('cases_passed') or 0) for r in rows)
    diagnosed_samples = sum(1 for r in rows if int(r.get('cases_total') or 0) > 0
                            and (r.get('test_case_status') or '') in ('OK', 'DISCRETE', 'PARTIAL'))
    undiagnosed_samples = sum(1 for r in rows if int(r.get('cases_total') or 0) == 0
                              and (r.get('test_case_status') or '') not in ('', 'NOT_RUN_COMPILE_ERROR'))
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
        'auto_limit': 0,
        'auto_max_inflight': AUTO_MAX_INFLIGHT if active and int(cfg.get('inference_concurrency') or 0) == 0 else 0,
        'compile_concurrency': _compile_limit(int(cfg.get('compile_concurrency') or COMPILE_CONCURRENCY)) if active else 0,
        'compile_concurrency_requested': int(cfg.get('compile_concurrency') or COMPILE_CONCURRENCY) if active else 0,
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
        'cases_declared': cases_declared,
        'cases_total': cases_total,
        'cases_passed': cases_passed,
        'diagnosed_samples': diagnosed_samples,
        'undiagnosed_samples': undiagnosed_samples,
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
