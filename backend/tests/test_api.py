import io
import json
import os
import time
from fastapi.testclient import TestClient
from app.main import app, summarize
from app.sandbox import EvalOutcome
from app import worker


async def _fake_pass(prediction,deps,tests,language='c',**kwargs):
    return EvalOutcome('TEST_PASS',True,len(tests),len(tests),run_log='mock sandbox unit test')


def test_seed_and_endpoint():
    with TestClient(app) as client:
        assert client.get('/api/health').status_code==200
        ds=client.get('/api/datasets').json()
        assert len(ds)>=1
        assert any(d['sample_count']==4 for d in ds)
        assert len(client.get('/api/models').json())>=1


def test_dataset_upload_validation():
    with TestClient(app) as client:
        payload={'sample_id':'test_1','assembly':'ret','reference_source':'int x(void){return 0;}',
                 'tests':[{'name':'zero','code':'if(x()!=0)return 1;'}], 'optimization':'O0'}
        content=(json.dumps(payload)+'\n').encode()
        files={'file':('samples.jsonl',io.BytesIO(content),'application/octet-stream')}
        r=client.post('/api/datasets/import',files=files,data={'name':'Test Upload','version':'v1'})
        assert r.status_code==201,r.text
        did=r.json()['dataset_id']
        items=client.get(f'/api/datasets/{did}/samples').json()
        assert items[0]['sample_key']=='test_1'
        datasets=client.get('/api/datasets').json()
        uploaded=next(d for d in datasets if d['id']==did)
        assert len(uploaded['source_hash'])==64
        bad=client.post('/api/datasets/import',files={'file':('bad.jsonl',b'{not json}','text/plain')},data={'name':'Broken'})
        assert bad.status_code==422


def test_metrics_excludes_infra_errors():
    rows=[
        {'status':'TEST_PASS','compile_success':1,'tests_passed':2,'tests_total':2},
        {'status':'WRONG_ANSWER','compile_success':1,'tests_passed':1,'tests_total':2},
        {'status':'COMPILE_ERROR','compile_success':0,'tests_passed':0,'tests_total':3},
        {'status':'INFRA_ERROR','compile_success':None,'tests_passed':0,'tests_total':2},
    ]
    m=summarize(rows)
    assert m['evaluated']==3 and m['excluded']==1
    assert m['rc']==66.67 and m['re']==33.33
    # TCP is computed only over valid sample testcases (2+2+3).
    assert m['tcp']==42.86


def test_async_run_result_persisted(monkeypatch):
    monkeypatch.setattr(worker,'evaluate_code_async',_fake_pass)
    with TestClient(app) as client:
        did=next(d['id'] for d in client.get('/api/datasets').json() if d['sample_count']==4)
        mid=next(m['id'] for m in client.get('/api/models').json() if m['kind']=='fixture')
        r=client.post('/api/runs',json={'dataset_id':did,'model_id':mid,'sample_limit':2})
        assert r.status_code==201,r.text
        rid=r.json()['run_id']
        for i in range(80):
            detail=client.get(f'/api/runs/{rid}').json()
            if detail['run']['status']=='completed':break
            time.sleep(0.05)
        assert detail['run']['status']=='completed',detail
        assert detail['metrics']['re']==100.0
        assert len(detail['results'])==2
        row=client.get('/api/results/'+str(detail['results'][0]['id'])).json()
        assert row['assembly'] and row['reference_source']


def test_model_input_not_reference():
    from app.worker import generate_prediction
    from app.db import connect
    with TestClient(app):
        with connect() as db:
            model=db.execute("SELECT * FROM models WHERE kind='fixture'").fetchone()
            sample=db.execute('SELECT * FROM samples LIMIT 1').fetchone()
        assert generate_prediction(model,sample,{})==sample['demo_prediction']


def _fixture_ids(client):
    did=next(d['id'] for d in client.get('/api/datasets').json() if d['sample_count']==4)
    mid=next(m['id'] for m in client.get('/api/models').json() if m['kind']=='fixture')
    return did,mid


def _wait_for(client,run_id,timeout=15):
    detail=None
    for _ in range(int(timeout/0.05)):
        detail=client.get(f'/api/runs/{run_id}').json()
        if detail['run']['status'] in ('completed','completed_with_errors','cancelled','interrupted'):
            return detail
        time.sleep(0.05)
    raise AssertionError(f'run {run_id} did not finish: {detail}')


def test_delete_finished_run(monkeypatch):
    monkeypatch.setattr(worker,'evaluate_code_async',_fake_pass)
    with TestClient(app) as client:
        did,mid=_fixture_ids(client)
        rid=client.post('/api/runs',json={'dataset_id':did,'model_id':mid,'sample_limit':1}).json()['run_id']
        detail=_wait_for(client,rid)
        result_id=detail['results'][0]['id']
        assert client.delete('/api/runs/999999').status_code==404
        resp=client.delete(f'/api/runs/{rid}')
        assert resp.status_code==200 and resp.json()['deleted']==1
        assert client.get(f'/api/runs/{rid}').status_code==404
        assert client.get(f'/api/results/{result_id}').status_code==404
        assert all(r['id']!=rid for r in client.get('/api/runs').json())


def test_delete_active_run_rejected():
    from app.db import connect, now
    with TestClient(app) as client:
        did,mid=_fixture_ids(client)
        with connect() as db:
            rid=db.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) VALUES(?,?,?,?,?,?)',
                           (did,mid,'running',1,'{}',now())).lastrowid
        try:
            assert client.delete(f'/api/runs/{rid}').status_code==409
            assert client.get(f'/api/runs/{rid}').status_code==200
        finally:
            with connect() as db:
                db.execute('DELETE FROM runs WHERE id=?',(rid,))


def test_delete_finished_runs_bulk(monkeypatch):
    monkeypatch.setattr(worker,'evaluate_code_async',_fake_pass)
    from app.db import connect, now
    with TestClient(app) as client:
        did,mid=_fixture_ids(client)
        rid=client.post('/api/runs',json={'dataset_id':did,'model_id':mid,'sample_limit':1}).json()['run_id']
        _wait_for(client,rid)
        with connect() as db:
            active=db.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) VALUES(?,?,?,?,?,?)',
                              (did,mid,'queued',1,'{}',now())).lastrowid
        try:
            resp=client.delete('/api/runs')
            assert resp.status_code==200
            assert resp.json()['deleted']>=1
            assert client.get(f'/api/runs/{rid}').status_code==404
            # Queued/running runs must survive a history cleanup.
            assert client.get(f'/api/runs/{active}').status_code==200
        finally:
            with connect() as db:
                db.execute('DELETE FROM runs WHERE id=?',(active,))
        assert client.delete('/api/runs').json()['deleted']==0


def test_delete_offline_run_cleans_file_and_model(monkeypatch):
    monkeypatch.setattr(worker,'evaluate_code_async',_fake_pass)
    with TestClient(app) as client:
        did,mid=_fixture_ids(client)
        samples=client.get(f'/api/datasets/{did}/samples').json()
        payload='\n'.join(json.dumps({'sample_id':s['sample_key'],'prediction':'int f(void){return 0;}'})
                          for s in samples)
        files={'file':('preds.jsonl',io.BytesIO(payload.encode()),'application/octet-stream')}
        resp=client.post('/api/runs/offline',files=files,data={'dataset_id':did,'name':'Offline To Delete'})
        assert resp.status_code==201,resp.text
        rid=resp.json()['run_id']
        detail=_wait_for(client,rid)
        model_id=detail['run']['model_id']
        offline_file=detail['run']['config'].get('offline_file')
        assert offline_file and os.path.isfile(offline_file)
        assert client.delete(f'/api/runs/{rid}').status_code==200
        assert not os.path.exists(offline_file)
        assert all(m['id']!=model_id for m in client.get('/api/models').json())



def test_summarize_ignores_transient_rows():
    rows=[
        {'status':'TEST_PASS','compile_success':1,'tests_passed':2,'tests_total':2},
        {'status':'GENERATED','compile_success':None,'tests_passed':0,'tests_total':2},
        {'status':'COMPILING','compile_success':None,'tests_passed':0,'tests_total':2},
    ]
    m=summarize(rows)
    assert m['evaluated']==1 and m['excluded']==2
    assert m['re']==100.0 and m['tcp']==100.0


def test_input_too_long_counts_as_valid_failure():
    rows=[
        {'status':'TEST_PASS','compile_success':1,'tests_passed':2,'tests_total':2},
        {'status':'INPUT_TOO_LONG','compile_success':0,'tests_passed':0,'tests_total':2},
    ]
    m=summarize(rows)
    # It must not be silently excluded, otherwise RE would be inflated.
    assert m['evaluated']==2 and m['excluded']==0
    assert m['rc']==50.0 and m['re']==50.0 and m['tcp']==50.0


def test_db_migration_is_idempotent():
    from app import db
    db.initialize()
    db.initialize()
    with db.connect() as conn:
        result_cols={r[1] for r in conn.execute('PRAGMA table_info(results)')}
        run_cols={r[1] for r in conn.execute('PRAGMA table_info(runs)')}
    assert {'inference_ms','compile_ms','test_ms','prompt_tokens',
            'completion_tokens','prompt_text','generation_params_json',
            'sandbox_ms','cases_declared','cases_total','cases_passed',
            'test_case_status','metric_version','detail_json',
            'source_compile_ok','compile_stage','edit_similarity'} <= result_cols
    with db.connect() as conn:
        sample_cols={r[1] for r in conn.execute('PRAGMA table_info(samples)')}
    assert {'test_protocol','test_wrapper','real_iospec','function_name',
            'function_signature','metadata_json','oracle_validated',
            'oracle_test_count','oracle_passed_count','oracle_failed_count',
            'oracle_error','oracle_detail_json','oracle_signature','oracle_checked_at'} <= sample_cols
    assert 'started_at' in run_cols and 'metric_version' in run_cols
    with db.connect() as conn:
        tables={r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'case_manifests','test_case_results','deployments',
            'gpu_samples','vllm_samples'} <= tables
    with db.connect() as conn:
        model_cols={r[1] for r in conn.execute('PRAGMA table_info(models)')}
    assert 'deployment_id' in model_cols


def test_reference_model_cannot_be_used_as_normal_run():
    from app.db import connect
    with TestClient(app) as client:
        did=next(d['id'] for d in client.get('/api/datasets').json() if d['sample_count']==4)
        with connect() as db:
            mid=db.execute("INSERT INTO models(name,kind,created_at) VALUES(?,?,?)",
                           ('reference-test-model','reference',__import__('app.db',fromlist=['now']).now())).lastrowid
        try:
            response=client.post('/api/runs',json={'dataset_id':did,'model_id':mid,'sample_limit':1})
            assert response.status_code==422
            assert 'reference-check' in response.text
            assert all(m['id']!=mid for m in client.get('/api/models').json())
        finally:
            with connect() as db:
                db.execute('DELETE FROM models WHERE id=?',(mid,))


def test_dataset_jsonl_has_no_legacy_64mb_cap(monkeypatch):
    import app.main as main_module
    # Even if the legacy constant is tiny, dataset parsing must not apply it.
    monkeypatch.setattr(main_module, 'MAX_UPLOAD_BYTES', 1)
    raw = json.dumps({
        'sample_id': 'large-upload-test',
        'assembly': 'func0:\n ret',
        'reference_source': 'int func0(void){return 0;}',
        'tests': [{'code': 'if (func0()!=0) return 1;'}],
    }).encode()
    parsed = main_module._read_jsonl(raw)
    assert len(parsed) == 1
    assert parsed[0]['sample_key'] == 'large-upload-test'


def _jsonl_sample(sample_id='delete-test'):
    return json.dumps({
        'sample_id': sample_id,
        'assembly': 'func0:\n ret',
        'reference_source': 'int func0(void){return 0;}',
        'tests': [{'code': 'if (func0()!=0) return 1;'}],
    }).encode()


def test_delete_imported_dataset_cascades_samples():
    with TestClient(app) as client:
        imported = client.post('/api/datasets/import',
            data={'name': 'Delete me', 'version': 'v1', 'description': ''},
            files={'file': ('delete.jsonl', _jsonl_sample(), 'application/jsonl')})
        assert imported.status_code == 201, imported.text
        did = imported.json()['dataset_id']
        deleted = client.delete(f'/api/datasets/{did}')
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()['samples_deleted'] == 1
        assert all(d['id'] != did for d in client.get('/api/datasets').json())
    from app.db import connect
    with connect() as db:
        assert db.execute('SELECT COUNT(*) FROM samples WHERE dataset_id=?',
                          (did,)).fetchone()[0] == 0


def test_delete_dataset_with_tasks_is_rejected():
    from app.db import connect, now
    with TestClient(app) as client:
        imported = client.post('/api/datasets/import',
            data={'name': 'Protected', 'version': 'v1', 'description': ''},
            files={'file': ('protected.jsonl', _jsonl_sample('protected'), 'application/jsonl')})
        did = imported.json()['dataset_id']
        with connect() as db:
            mid = db.execute('SELECT id FROM models LIMIT 1').fetchone()['id']
            rid = db.execute('''INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at)
                                VALUES(?,?,?,?,?,?)''',
                             (did, mid, 'completed', 1, '{}', now())).lastrowid
        response = client.delete(f'/api/datasets/{did}')
        assert response.status_code == 409
        assert 'task' in response.text
        still = client.get(f'/api/datasets/{did}/samples')
        assert still.status_code == 200
        with connect() as db:
            db.execute('DELETE FROM runs WHERE id=?', (rid,))
        assert client.delete(f'/api/datasets/{did}').status_code == 200


def test_delete_missing_dataset_returns_404():
    with TestClient(app) as client:
        assert client.delete('/api/datasets/999999').status_code == 404


def test_delete_unreferenced_model():
    with TestClient(app) as client:
        created = client.post('/api/models', json={
            'name': 'delete-me-model', 'kind': 'openai',
            'base_url': 'http://127.0.0.1:9999/v1', 'api_model': 'm',
            'api_mode': 'completion'})
        mid = created.json()['model_id']
        response = client.delete(f'/api/models/{mid}')
        assert response.status_code == 200, response.text
        assert response.json()['deleted'] == 1
        assert all(m['id'] != mid for m in client.get('/api/models').json())


def test_delete_model_used_by_run_is_rejected():
    from app.db import connect, now
    with TestClient(app) as client:
        created = client.post('/api/models', json={
            'name': 'run-model', 'kind': 'openai',
            'base_url': 'http://127.0.0.1:9999/v1', 'api_model': 'm'})
        mid = created.json()['model_id']
        with connect() as db:
            did = db.execute('SELECT id FROM datasets LIMIT 1').fetchone()['id']
            rid = db.execute('''INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at)
                                VALUES(?,?,?,?,?,?)''',
                             (did, mid, 'completed', 0, '{}', now())).lastrowid
        assert client.delete(f'/api/models/{mid}').status_code == 409
        with connect() as db:
            db.execute('DELETE FROM runs WHERE id=?', (rid,))
        assert client.delete(f'/api/models/{mid}').status_code == 200


def test_delete_model_linked_to_active_deployment_is_rejected():
    from app.db import connect, now
    with TestClient(app) as client:
        created = client.post('/api/models', json={
            'name': 'deployed-model', 'kind': 'openai',
            'base_url': 'http://127.0.0.1:9999/v1', 'api_model': 'm'})
        mid = created.json()['model_id']
        with connect() as db:
            did = db.execute('''INSERT INTO deployments
                (name,served_name,model_path,gpu_ids,port,dtype,max_model_len,
                 max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,
                 managed,status,model_id,created_at,config_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                ('active-dep','m','', '0', 9999, 'auto', 4096, 8, 2048, 0.5,
                 0, 'external', mid, now(), '{}')).lastrowid
            db.execute('UPDATE models SET deployment_id=? WHERE id=?', (did, mid))
        assert client.delete(f'/api/models/{mid}').status_code == 409
        with connect() as db:
            db.execute("UPDATE deployments SET status='stopped' WHERE id=?", (did,))
        response = client.delete(f'/api/models/{mid}')
        assert response.status_code == 200, response.text
        with connect() as db:
            row = db.execute('SELECT model_id FROM deployments WHERE id=?', (did,)).fetchone()
            assert row['model_id'] is None
            db.execute('DELETE FROM deployments WHERE id=?', (did,))


def test_reference_model_cannot_be_deleted():
    from app.db import connect, now
    with TestClient(app) as client:
        with connect() as db:
            mid = db.execute("INSERT INTO models(name,kind,created_at) VALUES(?,?,?)",
                             ('reference-delete-test', 'reference', now())).lastrowid
        assert client.delete(f'/api/models/{mid}').status_code == 403
        with connect() as db:
            db.execute('DELETE FROM models WHERE id=?', (mid,))


def test_server_file_listing_and_import(tmp_path, monkeypatch):
    import app.main as main_module
    root = tmp_path / 'server-imports'; root.mkdir()
    (root / 'valid.jsonl').write_text(json.dumps({
        'sample_id': 'server-sample',
        'assembly': 'func0:\n ret',
        'reference_source': 'int func0(void){return 0;}',
        'tests': [{'code': 'if (func0()!=0) return 1;'}],
    }) + '\n', encoding='utf-8')
    (root / 'not-jsonl.txt').write_text('{}', encoding='utf-8')
    monkeypatch.setattr(main_module, 'SERVER_IMPORT_DIRS', [root])
    with TestClient(app) as client:
        listing = client.get('/api/datasets/server-files')
        assert listing.status_code == 200
        files = listing.json()['files']
        assert any(f['relative'] == 'valid.jsonl' for f in files)
        assert all(f['name'].endswith('.jsonl') for f in files)
        response = client.post('/api/datasets/import-server', json={
            'name': 'Server file dataset', 'version': 'v1',
            'description': '', 'path': 'valid.jsonl'})
        assert response.status_code == 201, response.text
        assert response.json()['sample_count'] == 1
        assert response.json()['source_path'] == str((root / 'valid.jsonl').resolve())
        did = response.json()['dataset_id']
        assert client.delete(f'/api/datasets/{did}').status_code == 200
        # Absolute paths outside the allow-list are rejected.
        assert client.post('/api/datasets/import-server', json={
            'name': 'Blocked', 'path': '/etc/passwd'}).status_code in (403, 422)
        # Path traversal is rejected.
        assert client.post('/api/datasets/import-server', json={
            'name': 'Blocked', 'path': '../outside.jsonl'}).status_code in (403, 404)


def test_host_execution_requires_server_switch_and_ack(monkeypatch):
    import app.main as main_module
    with TestClient(app) as client:
        ds = client.get('/api/datasets').json()[0]
        model = client.get('/api/models').json()[0]
        body = {'dataset_id': ds['id'], 'model_id': model['id'], 'sample_limit': 1,
                'execution_mode': 'host', 'host_execution_ack': True}
        monkeypatch.setattr(main_module, 'ALLOW_HOST_EXECUTION', False)
        assert client.post('/api/runs', json=body).status_code == 403
        monkeypatch.setattr(main_module, 'ALLOW_HOST_EXECUTION', True)
        assert client.post('/api/runs', json={**body, 'host_execution_ack': False}).status_code == 422
        assert client.get('/api/health').json()['allow_host_execution'] is True


def test_host_toolchain_endpoint_reports_gcc_version():
    with TestClient(app) as client:
        response = client.get('/api/host-toolchain')
        assert response.status_code == 200
        data = response.json()
        assert data['engine'] == 'host'
        assert 'cc_version' in data and 'cxx_version' in data
        if data['available']:
            assert data['cc_version']
            assert data['cc']['path']
