"""Pipeline tests: concurrency, backpressure, cancellation, resume, metrics.

Everything is mocked (inference + compiler), so these tests do not need a GPU
or Docker.  Real Docker behaviour is covered by the opt-in integration test.
"""
import asyncio
import json
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app import db, worker
from app.main import app, summarize
from app.adapters import InputTooLong
from app.sandbox import EvalOutcome
from app.worker import Generation

DATASET_NAME = 'pipeline-tests'
SAMPLE_COUNT = 24


_DATASET_ID = None


def ensure_dataset():
    global _DATASET_ID
    if _DATASET_ID is not None:
        return _DATASET_ID
    db.initialize()
    db.seed_demo()
    with db.connect() as conn:
        row = conn.execute('SELECT id FROM datasets WHERE name=?', (DATASET_NAME,)).fetchone()
        if row:
            _DATASET_ID = row['id']
            return _DATASET_ID
        did = conn.execute(
            'INSERT INTO datasets(name,version,description,created_at) VALUES(?,?,?,?)',
            (DATASET_NAME, 'v1', 'in-memory pipeline fixture', db.now())).lastrowid
        for index in range(SAMPLE_COUNT):
            key = f'pipe_{index}'
            conn.execute('''
                INSERT INTO samples(dataset_id,sample_key,problem_id,optimization,compiler,
                    architecture,language,assembly,reference_source,dependencies,tests_json,demo_prediction)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (did, key, key, 'O0', 'gcc', 'x86_64', 'c', 'ret',
                 'int f(void){return 0;}', '',
                 json.dumps([{'name': 't', 'code': 'f();'}]),
                 f'/*{key}*/ int f(void){{return 0;}}'))
        _DATASET_ID = did
        return did


_LARGE_DATASET_ID = None


def ensure_large_dataset(count=200):
    global _LARGE_DATASET_ID
    if _LARGE_DATASET_ID is not None:
        return _LARGE_DATASET_ID
    db.initialize()
    with db.connect() as conn:
        row = conn.execute('SELECT id FROM datasets WHERE name=?', ('pipeline-large',)).fetchone()
        if row:
            _LARGE_DATASET_ID = row['id']
            return _LARGE_DATASET_ID
        did = conn.execute(
            'INSERT INTO datasets(name,version,description,created_at) VALUES(?,?,?,?)',
            ('pipeline-large', 'v1', 'large in-memory fixture', db.now())).lastrowid
        for index in range(count):
            key = f'big_{index}'
            conn.execute('''
                INSERT INTO samples(dataset_id,sample_key,problem_id,optimization,compiler,
                    architecture,language,assembly,reference_source,dependencies,tests_json,demo_prediction)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                (did, key, key, 'O0', 'gcc', 'x86_64', 'c', 'ret',
                 'int f(void){return 0;}', '',
                 json.dumps([{'name': 't', 'code': 'f();'}]), ''))
        _LARGE_DATASET_ID = did
        return did


def fixture_model_id():
    with db.connect() as conn:
        return conn.execute("SELECT id FROM models WHERE kind='fixture' LIMIT 1").fetchone()[0]


def create_run(limit=8, ic=4, cc=2, did=None, generation_mode='online', batch_size=0):
    did = did or ensure_dataset()
    mid = fixture_model_id()
    cfg = {
        'dataset_id': did, 'model_id': mid, 'optimizations': ['O0'],
        'sample_limit': limit,
        'prompt_template': '# This is the assembly code:\n{assembly}\n# What is the source code?\n',
        'temperature': 0, 'max_tokens': 64,
        'inference_concurrency': ic, 'compile_concurrency': cc,
        'generation_mode': generation_mode, 'batch_size': batch_size,
        'max_model_len': 4096,
    }
    with db.connect() as conn:
        rid = conn.execute(
            'INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) '
            'VALUES(?,?,?,?,?,?)',
            (did, mid, 'queued', limit, json.dumps(cfg), db.now())).lastrowid
    return rid


def sampled_rows(run_id):
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(
            'SELECT r.*,s.sample_key FROM results r JOIN samples s ON s.id=r.sample_id '
            'WHERE r.run_id=? ORDER BY s.id', (run_id,))]


def run_row(run_id):
    with db.connect() as conn:
        return dict(conn.execute('SELECT * FROM runs WHERE id=?', (run_id,)).fetchone())


def set_result(run_id, sample_key, status, model_output='', inference_ms=0):
    did = ensure_dataset()
    with db.connect() as conn:
        sample = conn.execute('SELECT id FROM samples WHERE dataset_id=? AND sample_key=?',
                              (did, sample_key)).fetchone()
        conn.execute('''
            INSERT INTO results(run_id,sample_id,status,compile_success,tests_passed,tests_total,
                                model_output,inference_ms,latency_ms,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)''',
            (run_id, sample['id'], status,
             1 if status == 'TEST_PASS' else (None if status in ('GENERATED', 'INFRA_ERROR') else 0),
             1 if status == 'TEST_PASS' else 0, 1, model_output, inference_ms,
             inference_ms, db.now()))


@pytest.fixture(autouse=True)
def clean_pipeline_runs():
    yield
    dids = [ensure_dataset()]
    if _LARGE_DATASET_ID is not None:
        dids.append(_LARGE_DATASET_ID)
    marks_ds = ','.join('?' for _ in dids)
    with db.connect() as conn:
        ids = [r['id'] for r in conn.execute(
            f'SELECT id FROM runs WHERE dataset_id IN ({marks_ds})', dids)]
        if ids:
            marks = ','.join('?' for _ in ids)
            conn.execute(f'DELETE FROM results WHERE run_id IN ({marks})', ids)
            conn.execute(f'DELETE FROM runs WHERE id IN ({marks})', ids)
    for rid in ids:
        worker.clear_progress(rid)


class Recorder:
    """Simulated model + simulated compiler that record concurrency."""

    def __init__(self, infer_delay=0.03, compile_delay=0.04,
                 fail_inference=(), fail_compile=None, slow_compile=()):
        self.lock = threading.Lock()
        self.infer_active = 0
        self.compile_active = 0
        self.max_infer = 0
        self.max_compile = 0
        self.inference_keys = []
        self.compile_predictions = []
        self.compile_keys = []
        self.overlap = False
        self.infer_delay = infer_delay
        self.compile_delay = compile_delay
        self.fail_inference = set(fail_inference)
        self.fail_compile = dict(fail_compile or {})
        self.slow_compile = set(slow_compile)
        self.inference_starts = []
        self.inference_finishes = []
        self.delay_for = None

    async def infer(self, model, sample, cfg, client):
        key = sample['sample_key']
        index = int(key.rsplit('_', 1)[-1])
        delay = (self.delay_for(key) if self.delay_for else
                 self.infer_delay + (index % 5) * 0.01)
        with self.lock:
            self.infer_active += 1
            self.max_infer = max(self.max_infer, self.infer_active)
            self.inference_keys.append(key)
            self.inference_starts.append((key, time.monotonic()))
        try:
            await asyncio.sleep(delay)
            if key in self.fail_inference:
                raise RuntimeError('simulated model API failure')
            return Generation(f'/*{key}*/ int f(void){{return 0;}}',
                              inference_ms=10, prompt_tokens=5, completion_tokens=7,
                              prompt=f'PROMPT::{key}', params={'temperature': 0,
                                                               'max_tokens_effective': 64})
        finally:
            with self.lock:
                self.infer_active -= 1
                self.inference_finishes.append((key, time.monotonic()))

    async def compile(self, prediction, deps, tests, language='c', **kwargs):
        match = re.search(r'/\*(pipe_\d+)\*/', prediction)
        key = match.group(1) if match else ''
        with self.lock:
            if self.infer_active > 0:
                self.overlap = True
            self.compile_active += 1
            self.max_compile = max(self.max_compile, self.compile_active)
            self.compile_predictions.append(prediction)
            self.compile_keys.append(key)
        try:
            await asyncio.sleep(0.25 if key in self.slow_compile else self.compile_delay)
            if key in self.fail_compile:
                return self.fail_compile[key](tests)
            return EvalOutcome('TEST_PASS', True, len(tests), len(tests),
                               compile_ms=5, test_ms=2)
        finally:
            with self.lock:
                self.compile_active -= 1


def install(monkeypatch, recorder):
    monkeypatch.setattr(worker, 'generate_prediction_async', recorder.infer)
    monkeypatch.setattr(worker, 'evaluate_code_async', recorder.compile)


def test_concurrency_bounds_overlap_and_persistence(monkeypatch):
    rid = create_run(limit=20, ic=4, cc=2)
    rec = Recorder(infer_delay=0.02, compile_delay=0.03)
    install(monkeypatch, rec)
    worker.process_run(rid)

    run = run_row(rid)
    rows = sampled_rows(rid)
    assert run['status'] == 'completed'
    assert len(rows) == 20
    assert all(r['status'] == 'TEST_PASS' for r in rows)
    assert len({r['sample_id'] for r in rows}) == 20
    # Bounded concurrency: never above the configured values, but genuinely
    # parallel (not a serial loop).
    assert 2 <= rec.max_infer <= 4
    assert 1 <= rec.max_compile <= 2
    assert rec.overlap is True, 'compilation must overlap with inference'
    assert len(rec.inference_keys) == 20
    assert all(r['inference_ms'] == 10 and r['compile_ms'] == 5 for r in rows)
    assert all(r['prompt_tokens'] == 5 and r['completion_tokens'] == 7 for r in rows)

    progress = worker.get_progress(rid)
    assert progress['evaluated'] == 20
    assert progress['failed'] == 0
    assert progress['status'] == 'completed'
    assert progress['elapsed_seconds'] >= 0
    assert progress['inference_rate'] > 0


def test_inference_failure_does_not_block_other_samples(monkeypatch):
    rid = create_run(limit=6, ic=3, cc=2)
    rec = Recorder(fail_inference={'pipe_1', 'pipe_4'})
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = {r['sample_key']: r for r in sampled_rows(rid)}
    assert rows['pipe_1']['status'] == 'INFERENCE_ERROR'
    assert rows['pipe_4']['status'] == 'INFERENCE_ERROR'
    assert sum(1 for r in rows.values() if r['status'] == 'TEST_PASS') == 4
    assert run_row(rid)['status'] == 'completed_with_errors'
    metrics = summarize(list(rows.values()))
    assert metrics['evaluated'] == 4 and metrics['excluded'] == 2
    assert metrics['re'] == 100.0
    progress = worker.get_progress(rid)
    assert progress['inference_failed'] == 2 and progress['evaluated'] == 4


def test_compile_failure_does_not_block_queue(monkeypatch):
    def compile_error(tests):
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests), error='bad code')

    def runtime_error(tests):
        return EvalOutcome('RUNTIME_ERROR', True, 0, len(tests), error='SIGSEGV')

    rid = create_run(limit=6, ic=2, cc=2)
    rec = Recorder(fail_compile={'pipe_2': compile_error, 'pipe_3': runtime_error})
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = {r['sample_key']: r for r in sampled_rows(rid)}
    assert rows['pipe_2']['status'] == 'COMPILE_ERROR'
    assert rows['pipe_3']['status'] == 'RUNTIME_ERROR'
    assert sum(1 for r in rows.values() if r['status'] == 'TEST_PASS') == 4
    assert run_row(rid)['status'] == 'completed'


def test_out_of_order_completion_keeps_results_associated(monkeypatch):
    rid = create_run(limit=6, ic=3, cc=2)
    rec = Recorder(compile_delay=0.02, slow_compile={'pipe_0'})
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = sampled_rows(rid)
    assert len(rows) == 6
    for row in rows:
        assert f'/*{row["sample_key"]}*/' in row['model_output']
        assert row['status'] == 'TEST_PASS'


def test_cancel_stops_dispatching_new_samples(monkeypatch):
    rid = create_run(limit=20, ic=2, cc=2)
    rec = Recorder(infer_delay=0.06, compile_delay=0.2)
    install(monkeypatch, rec)
    thread = threading.Thread(target=worker.process_run, args=(rid,), daemon=True)
    thread.start()
    time.sleep(0.45)
    with db.connect() as conn:
        conn.execute("UPDATE runs SET status='cancel_requested' WHERE id=?", (rid,))
    thread.join(timeout=20)
    assert not thread.is_alive(), 'cancelled pipeline did not stop'
    run = run_row(rid)
    assert run['status'] == 'cancelled'
    assert len(rec.inference_keys) < 20, 'no new inference work may be dispatched after cancel'
    assert len(sampled_rows(rid)) < 20


def test_resume_skips_terminal_and_reuses_saved_generation(monkeypatch):
    rid = create_run(limit=4, ic=2, cc=2)
    set_result(rid, 'pipe_0', 'TEST_PASS', model_output='already-done', inference_ms=3)
    set_result(rid, 'pipe_1', 'GENERATED', model_output='/*pipe_1*/ stored', inference_ms=7)

    rec = Recorder()
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = {r['sample_key']: r for r in sampled_rows(rid)}
    assert 'pipe_0' not in rec.inference_keys
    assert 'pipe_0' not in rec.compile_keys
    assert rows['pipe_0']['model_output'] == 'already-done'
    assert 'pipe_1' not in rec.inference_keys
    assert '/*pipe_1*/ stored' in rec.compile_predictions
    assert rows['pipe_1']['status'] == 'TEST_PASS'
    assert 'pipe_2' in rec.inference_keys and 'pipe_3' in rec.inference_keys


def test_retry_does_not_reevaluate_terminal_samples(monkeypatch):
    rid = create_run(limit=3, ic=2, cc=1)
    install(monkeypatch, Recorder(compile_delay=0.01))
    worker.process_run(rid)
    before = {r['sample_key']: (r['status'], r['model_output']) for r in sampled_rows(rid)}

    class Forbidden(Recorder):
        async def infer(self, *args, **kwargs):
            raise AssertionError('terminal sample was re-inferred')

        async def compile(self, *args, **kwargs):
            raise AssertionError('terminal sample was re-compiled')

    install(monkeypatch, Forbidden())
    with db.connect() as conn:
        conn.execute("UPDATE runs SET status='queued', started_at=NULL, finished_at=NULL WHERE id=?",
                     (rid,))
    worker.process_run(rid)
    after = {r['sample_key']: (r['status'], r['model_output']) for r in sampled_rows(rid)}
    assert after == before
    assert run_row(rid)['status'] == 'completed'


def test_infra_errors_are_excluded_from_model_metrics(monkeypatch):
    def infra(tests):
        return EvalOutcome('INFRA_ERROR', None, 0, len(tests), error='docker denied')

    rid = create_run(limit=5, ic=2, cc=2)
    install(monkeypatch, Recorder(fail_compile={'pipe_1': infra}))
    worker.process_run(rid)

    rows = sampled_rows(rid)
    metrics = summarize(rows)
    assert metrics['evaluated'] == 4 and metrics['excluded'] == 1
    assert metrics['rc'] == 100.0 and metrics['re'] == 100.0
    assert all(r['status'] != 'INFRA_ERROR' for r in rows if r['sample_key'] == 'pipe_0')
    assert run_row(rid)['status'] == 'completed_with_errors'
    assert worker.get_progress(rid)['infra_errors'] == 1


def test_progress_endpoint_and_run_detail(monkeypatch):
    install(monkeypatch, Recorder(infer_delay=0.01, compile_delay=0.01))
    did = ensure_dataset()
    with TestClient(app) as client:
        mid = fixture_model_id()
        response = client.post('/api/runs', json={
            'dataset_id': did, 'model_id': mid, 'optimizations': ['O0'],
            'sample_limit': 6, 'inference_concurrency': 3, 'compile_concurrency': 2,
        })
        assert response.status_code == 201, response.text
        rid = response.json()['run_id']
        progress = {}
        for _ in range(200):
            progress = client.get(f'/api/runs/{rid}/progress').json()
            if progress.get('status') == 'completed':
                break
            time.sleep(0.02)
        assert progress['status'] == 'completed'
        assert progress['total'] == 6
        assert progress['evaluated'] == 6
        assert progress['inference_done'] == 6
        assert progress['inference_concurrency'] == 3
        assert progress['compile_concurrency'] == 2
        assert {'waiting_inference', 'inferring', 'waiting_compile', 'compiling',
                'failed', 'inference_rate', 'eval_rate'} <= set(progress)
        detail = client.get(f'/api/runs/{rid}').json()
        assert detail['progress']['evaluated'] == 6
        assert detail['metrics']['re'] == 100.0
        assert all('inference_ms' in row for row in detail['results'])


def test_reference_check_run_uses_same_sandbox(monkeypatch):
    install(monkeypatch, Recorder(compile_delay=0.01))
    did = ensure_dataset()
    with TestClient(app) as client:
        response = client.post(f'/api/datasets/{did}/reference-check',
                               json={'optimizations': ['O0'], 'sample_limit': 2})
        assert response.status_code == 201, response.text
        rid = response.json()['run_id']
        detail = {}
        for _ in range(200):
            detail = client.get(f'/api/runs/{rid}').json()
            if detail['run']['status'] in ('completed', 'completed_with_errors'):
                break
            time.sleep(0.02)
        assert detail['run']['status'] == 'completed'
        assert detail['run']['config']['reference'] is True
        assert detail['run']['model_name'] == worker.REFERENCE_MODEL_NAME
        assert len(detail['results']) == 2
        assert all(r['status'] == 'TEST_PASS' for r in detail['results'])
        models = client.get('/api/models').json()
        assert all(m['kind'] != 'reference' for m in models)
        client.delete(f'/api/runs/{rid}')


def test_inference_retry_on_429_and_transport_errors(monkeypatch):
    class Response:
        def __init__(self, status, body=None, text=''):
            self.status_code = status
            self._body = body or {}
            self.text = text

        def json(self):
            return self._body

    class Client:
        def __init__(self, items):
            self.items = list(items)
            self.calls = 0

        async def post(self, *args, **kwargs):
            self.calls += 1
            item = self.items.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    import httpx

    monkeypatch.setattr(worker.random, 'uniform', lambda a, b: 0.0)
    client = Client([Response(429, text='busy'), httpx.ConnectError('temporary'),
                     Response(200, {'choices': [{'text': 'int f(void){return 0;}'}]})])
    body = asyncio.run(worker._post_with_retries(
        client, 'http://x/v1/completions', {}, {}, timeout=5, retries=3, backoff_base=0.001))
    assert body['choices'][0]['text'].startswith('int f')
    assert client.calls == 3

    client = Client([Response(400, text='bad request')])
    with pytest.raises(worker.InferenceError):
        asyncio.run(worker._post_with_retries(
            client, 'http://x/v1/completions', {}, {}, timeout=5, retries=3, backoff_base=0.001))
    assert client.calls == 1, 'deterministic 4xx must not be retried'

    client = Client([httpx.ReadTimeout('slow')])
    with pytest.raises(worker.InferenceError):
        asyncio.run(worker._post_with_retries(
            client, 'http://x/v1/completions', {}, {}, timeout=5, retries=0, backoff_base=0.001))
    assert client.calls == 1


def test_db_write_lock_serialises_concurrent_results(monkeypatch):
    """Four compile workers writing at once must not lose rows or raise."""
    rid = create_run(limit=24, ic=4, cc=4)
    install(monkeypatch, Recorder(infer_delay=0.005, compile_delay=0.005))
    worker.process_run(rid)
    rows = sampled_rows(rid)
    assert len(rows) == 24
    assert len({r['sample_id'] for r in rows}) == 24
    assert all(r['status'] == 'TEST_PASS' for r in rows)


def test_sixteen_concurrent_inference_requests(monkeypatch):
    rid = create_run(limit=24, ic=16, cc=2)
    rec = Recorder(infer_delay=0.08, compile_delay=0.01)
    install(monkeypatch, rec)
    worker.process_run(rid)
    rows = sampled_rows(rid)
    assert len(rows) == 24 and all(r['status'] == 'TEST_PASS' for r in rows)
    # vLLM-style batching: many requests in flight at once, still bounded.
    assert 8 <= rec.max_infer <= 16
    assert rec.max_compile <= 2


def test_bounded_compile_queue_applies_backpressure(monkeypatch):
    monkeypatch.setattr(worker, 'MAX_PENDING_COMPILE', 2)
    rid = create_run(limit=12, ic=6, cc=1)
    rec = Recorder(infer_delay=0.01, compile_delay=0.05)
    install(monkeypatch, rec)
    worker.process_run(rid)
    rows = sampled_rows(rid)
    assert len(rows) == 12 and all(r['status'] == 'TEST_PASS' for r in rows)
    assert rec.max_compile == 1


def test_scheduler_refills_a_slot_without_waiting_for_slow_request(monkeypatch):
    rid = create_run(limit=6, ic=2, cc=2)
    rec = Recorder(infer_delay=0.02, compile_delay=0.01)
    rec.delay_for = lambda key: 0.6 if key == 'pipe_1' else 0.02
    install(monkeypatch, rec)
    worker.process_run(rid)

    starts = dict(rec.inference_starts)
    finishes = dict(rec.inference_finishes)
    # pipe_0 and pipe_1 start together; pipe_0 finishes first and the freed
    # slot is refilled immediately, while pipe_1 is still in flight.
    assert abs(starts['pipe_0'] - starts['pipe_1']) < 0.1
    assert starts['pipe_2'] - starts['pipe_0'] < 0.2
    assert finishes['pipe_1'] > starts['pipe_2']
    assert run_row(rid)['status'] == 'completed'
    assert len(sampled_rows(rid)) == 6


def test_128_in_flight_requests_are_bounded(monkeypatch):
    did = ensure_large_dataset(200)
    rid = create_run(limit=200, ic=128, cc=4, did=did)
    rec = Recorder(infer_delay=0.2, compile_delay=0.001)
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = sampled_rows(rid)
    assert len(rows) == 200
    assert all(r['status'] == 'TEST_PASS' for r in rows)
    assert 100 <= rec.max_infer <= 128, f'max in-flight was {rec.max_infer}'
    assert rec.max_compile <= 4
    progress = worker.get_progress(rid)
    assert progress['inference_done'] == 200 and progress['evaluated'] == 200


def test_input_too_long_is_recorded_and_not_excluded(monkeypatch):
    rid = create_run(limit=4, ic=2, cc=2)

    async def fake_infer(model, sample, cfg, client):
        if sample['sample_key'] == 'pipe_1':
            raise InputTooLong(4096, 4096, prompt='over-long prompt',
                               params={'max_tokens_requested': 64,
                                       'max_tokens_effective': 0})
        return Generation(f'/*{sample["sample_key"]}*/ int f(void){{return 0;}}',
                          inference_ms=1, prompt_tokens=2, completion_tokens=2,
                          prompt='ok')

    rec = Recorder(compile_delay=0.01)
    monkeypatch.setattr(worker, 'generate_prediction_async', fake_infer)
    monkeypatch.setattr(worker, 'evaluate_code_async', rec.compile)
    worker.process_run(rid)

    rows = {r['sample_key']: r for r in sampled_rows(rid)}
    assert rows['pipe_1']['status'] == 'INPUT_TOO_LONG'
    assert rows['pipe_1']['compile_success'] == 0
    assert 'over-long prompt' in rows['pipe_1']['prompt_text']
    metrics = summarize(rows.values())
    # INPUT_TOO_LONG is a valid failure: it stays in RC/RE denominators.
    assert metrics['evaluated'] == 4 and metrics['excluded'] == 0
    assert metrics['rc'] == 75.0 and metrics['re'] == 75.0
    assert run_row(rid)['status'] == 'completed'
    progress = worker.get_progress(rid)
    assert progress['input_too_long'] == 1
    assert progress['failed'] == 1 and progress['evaluated'] == 4


def test_token_and_prompt_parameters_are_persisted(monkeypatch):
    rid = create_run(limit=3, ic=2, cc=2)
    rec = Recorder(compile_delay=0.01)
    install(monkeypatch, rec)
    worker.process_run(rid)

    rows = sampled_rows(rid)
    assert all(r['prompt_text'] == f'PROMPT::{r["sample_key"]}' for r in rows)
    params = json.loads(rows[0]['generation_params_json'])
    assert params['max_tokens_effective'] == 64
    progress = worker.get_progress(rid)
    assert progress['completion_tokens'] == 21
    assert progress['prompt_tokens'] == 15
    assert progress['token_rate'] > 0
    assert progress['total_token_rate'] >= progress['token_rate']
    assert progress['inference_ms_sum'] == 30
    assert progress['compile_ms_sum'] == 15
    assert progress['test_ms_sum'] == 6
    assert progress['in_flight'] == 0


def test_offline_batch_mode_persists_and_uses_independent_prompts(monkeypatch):
    rid = create_run(limit=6, ic=4, cc=2, generation_mode='offline_batch', batch_size=2)
    calls = []

    class FakeBatchAdapter:
        def __init__(self, cfg):
            self.cfg = cfg

        def generate(self, samples):
            keys = [s['sample_key'] for s in samples]
            calls.append(keys)
            return [Generation(f'/*{key}*/ int f(void){{return 0;}}',
                               inference_ms=2, prompt_tokens=1, completion_tokens=2,
                               prompt=f'BATCH::{key}',
                               params={'generation_mode': 'offline_batch'})
                    for key in keys]

    class Compile:
        async def __call__(self, prediction, deps, tests, language='c', **kwargs):
            return EvalOutcome('TEST_PASS', True, len(tests), len(tests),
                               compile_ms=1, test_ms=1)

    monkeypatch.setattr(worker, 'OfflineBatchAdapter', FakeBatchAdapter)
    monkeypatch.setattr(worker, 'evaluate_code_async', Compile())
    worker.process_run(rid)

    rows = sampled_rows(rid)
    assert len(rows) == 6
    assert all(r['status'] == 'TEST_PASS' for r in rows)
    assert all(r['prompt_text'] == f'BATCH::{r["sample_key"]}' for r in rows)
    assert calls == [['pipe_0', 'pipe_1'], ['pipe_2', 'pipe_3'], ['pipe_4', 'pipe_5']]
    progress = worker.get_progress(rid)
    assert progress['generation_mode'] == 'offline_batch'
    assert progress['batch_size'] == 2


def test_batch_endpoint_validation(monkeypatch):
    from app import main as main_module
    monkeypatch.setattr(main_module.OfflineBatchAdapter, 'available',
                        staticmethod(lambda: False))
    with TestClient(app) as client:
        did = ensure_dataset()
        mid = fixture_model_id()
        missing = client.post('/api/runs/batch', json={
            'dataset_id': did, 'model_id': mid, 'sample_limit': 1})
        assert missing.status_code == 422 and 'model_path' in missing.text
        unavailable = client.post('/api/runs/batch', json={
            'dataset_id': did, 'model_id': mid, 'sample_limit': 1,
            'model_path': '/tmp/not-a-real-model'})
        assert unavailable.status_code == 503


def test_attach_vllm_metrics_is_cached(monkeypatch):
    from app import main as main_module
    main_module._METRICS_CACHE.clear()
    calls = []

    def fake_fetch(url):
        calls.append(url)
        return {'num_requests_running': 2.0, 'num_requests_waiting': 5.0}

    monkeypatch.setattr(main_module, 'fetch_vllm_metrics', fake_fetch)
    model = {'kind': 'openai', 'base_url': 'http://127.0.0.1:8001/v1'}
    first = main_module.attach_vllm_metrics({}, model, {})
    second = main_module.attach_vllm_metrics({}, model, {})
    assert first['vllm_metrics']['num_requests_running'] == 2.0
    assert second['vllm_metrics']['num_requests_waiting'] == 5.0
    assert len(calls) == 1, 'two-second cache should coalesce progress polls'


def test_default_concurrency_matches_config():
    from app.config import INFERENCE_CONCURRENCY, COMPILE_CONCURRENCY
    assert INFERENCE_CONCURRENCY == 128 and COMPILE_CONCURRENCY == 4
    assert worker._resolve_concurrency({}) == (INFERENCE_CONCURRENCY, COMPILE_CONCURRENCY)


def test_batch_endpoint_creates_and_runs(monkeypatch):
    from app import main as main_module
    monkeypatch.setattr(main_module.OfflineBatchAdapter, 'available',
                        staticmethod(lambda: True))

    class FakeBatchAdapter:
        def __init__(self, cfg):
            self.cfg = cfg

        def generate(self, samples):
            return [Generation(f'/*{s["sample_key"]}*/ int f(void){{return 0;}}',
                               inference_ms=1, prompt_tokens=1, completion_tokens=1,
                               prompt=f'BATCH::{s["sample_key"]}',
                               params={'generation_mode': 'offline_batch'})
                    for s in samples]

    class Compile:
        async def __call__(self, prediction, deps, tests, language='c', **kwargs):
            return EvalOutcome('TEST_PASS', True, len(tests), len(tests),
                               compile_ms=1, test_ms=1)

    monkeypatch.setattr(worker, 'OfflineBatchAdapter', FakeBatchAdapter)
    monkeypatch.setattr(worker, 'evaluate_code_async', Compile())
    with TestClient(app) as client:
        did = ensure_dataset()
        mid = fixture_model_id()
        response = client.post('/api/runs/batch', json={
            'dataset_id': did, 'model_id': mid, 'optimizations': ['O0'],
            'sample_limit': 4, 'model_path': '/fake/model', 'batch_size': 2})
        assert response.status_code == 201, response.text
        rid = response.json()['run_id']
        detail = {}
        for _ in range(200):
            detail = client.get(f'/api/runs/{rid}').json()
            if detail['run']['status'] == 'completed':
                break
            time.sleep(0.02)
        assert detail['run']['status'] == 'completed'
        assert detail['run']['config']['generation_mode'] == 'offline_batch'
        assert detail['run']['config']['batch_size'] == 2
        assert all(r['status'] == 'TEST_PASS' for r in detail['results'])
        client.delete(f'/api/runs/{rid}')
