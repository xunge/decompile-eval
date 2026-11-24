import io
import json
import time
from fastapi.testclient import TestClient
from app.main import app, summarize
from app.sandbox import EvalOutcome
from app import worker


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
    def fake_eval(prediction,deps,tests):
        return EvalOutcome('TEST_PASS',True,len(tests),len(tests),run_log='mock sandbox unit test')
    monkeypatch.setattr(worker,'evaluate_code',fake_eval)
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
