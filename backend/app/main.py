"""DecompileEval MVP REST API, intentionally single-user and development-only."""
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from typing import Any
import json
import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from .config import DATA_DIR, CORS_ORIGINS, MAX_UPLOAD_BYTES, MAX_SAMPLES
from .db import connect, initialize, seed_demo, now
from .schemas import ModelCreate, RunCreate
from .worker import submit_run

EXCLUDED={'INFRA_ERROR','INFERENCE_ERROR'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    seed_demo()
    yield

app=FastAPI(title='DecompileEval MVP', version='0.1.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=CORS_ORIGINS,allow_credentials=False,
                   allow_methods=['*'],allow_headers=['*'])


def dictrow(row):
    return dict(row) if row is not None else None


def required(db, table:str, id:int):
    # table is a hardcoded constant in each caller (never user-supplied).
    row=db.execute(f'SELECT * FROM {table} WHERE id=?',(id,)).fetchone()
    if row is None:
        raise HTTPException(404,f'{table} #{id} not found')
    return row


def validate_sample(obj:dict, n:int):
    fields=('sample_id','assembly','reference_source','tests')
    for field in fields:
        if field not in obj:
            raise ValueError(f'Line {n}: required field {field} is missing')
    sid=str(obj['sample_id'])
    if not (1<=len(sid)<=160):
        raise ValueError(f'Line {n}: invalid sample_id')
    if not isinstance(obj['assembly'],str) or not obj['assembly'].strip():
        raise ValueError(f'Line {n}: assembly must be nonempty text')
    if not isinstance(obj['reference_source'],str):
        raise ValueError(f'Line {n}: reference_source must be a string')
    if len(obj['assembly'])>250000 or len(obj['reference_source'])>250000:
        raise ValueError(f'Line {n}: source/assembly is too long')
    tests=obj['tests']
    if not isinstance(tests,list) or not 1<=len(tests)<=50:
        raise ValueError(f'Line {n}: tests must contain 1–50 cases')
    normalized=[]
    for i,t in enumerate(tests):
        if isinstance(t,str):t={'name':f'case_{i+1}','code':t}
        if not isinstance(t,dict):
            raise ValueError(f'Line {n}: invalid test case {i+1}')
        field = 'program' if 'program' in t else 'code'
        if not isinstance(t.get(field),str) or not t[field].strip():
            raise ValueError(f'Line {n}: test case {i+1} must have code or program')
        if len(t[field])>150000:
            raise ValueError(f'Line {n}: test case {i+1} too long')
        normalized.append({'name':str(t.get('name',f'case_{i+1}'))[:100], field:t[field]})
    opt=str(obj.get('optimization','O0'))
    if opt not in ('O0','O1','O2','O3','Os','Ofast'):
        raise ValueError(f'Line {n}: invalid optimization {opt}')
    lang=str(obj.get('language','c')).lower()
    if lang not in ('c','cpp'):
        raise ValueError(f'Line {n}: only C/C++ supported')
    return {
      'sample_key':sid, 'problem_id':str(obj.get('problem_id',sid))[:160],
      'optimization':opt,'compiler':str(obj.get('compiler','gcc'))[:40],
      'architecture':str(obj.get('architecture','x86_64'))[:40],
      'language':lang,'assembly':obj['assembly'],
      'reference_source':obj['reference_source'],
      'dependencies':str(obj.get('dependencies',''))[:150000],
      'tests_json':json.dumps(normalized,ensure_ascii=False),
      'demo_prediction':str(obj.get('demo_prediction',''))[:250000],
    }


def _read_jsonl(raw:bytes):
    if len(raw)>MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'File exceeds 64 MB upload limit')
    try:
        lines=raw.decode('utf-8-sig').splitlines()
        if len(lines)>MAX_SAMPLES:
            raise ValueError(f'Max {MAX_SAMPLES} samples per file')
        items=[validate_sample(json.loads(line),n) for n,line in enumerate(lines,1) if line.strip()]
        if not items:
            raise ValueError('The JSONL file contains no samples')
        keys=[x['sample_key'] for x in items]
        if len(set(keys))!=len(keys):
            raise ValueError('Duplicate sample_id in uploaded dataset')
        return items
    except (UnicodeError,ValueError,TypeError,json.JSONDecodeError) as exc:
        raise HTTPException(422,f'Invalid dataset: {str(exc)[:500]}')


@app.get('/api/health')
def health():
    return {'ok':True,'version':'0.1.0','sandbox':'docker-required'}

@app.get('/api/overview')
def overview():
    with connect() as db:
        return {
            'datasets':db.execute('SELECT COUNT(*) FROM datasets').fetchone()[0],
            'samples':db.execute('SELECT COUNT(*) FROM samples').fetchone()[0],
            'models':db.execute('SELECT COUNT(*) FROM models').fetchone()[0],
            'runs':db.execute('SELECT COUNT(*) FROM runs').fetchone()[0],
            'active_runs':db.execute("SELECT COUNT(*) FROM runs WHERE status IN ('queued','running','cancel_requested')").fetchone()[0],
        }

@app.get('/api/datasets')
def datasets():
    with connect() as db:
        return [dict(row) for row in db.execute('''SELECT d.*,COUNT(s.id) AS sample_count FROM datasets d
             LEFT JOIN samples s ON s.dataset_id=d.id GROUP BY d.id ORDER BY d.id DESC''')]

@app.post('/api/datasets/import',status_code=201)
async def import_dataset(name:str=Form(...),version:str=Form('v1'),description:str=Form(''),file:UploadFile=File(...)):
    if not name.strip() or len(name)>100 or len(version)>40:
        raise HTTPException(422,'Invalid dataset name/version')
    if not file.filename or not file.filename.lower().endswith('.jsonl'):
        raise HTTPException(422,'Upload a .jsonl file')
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    parsed=_read_jsonl(raw)
    with connect() as db:
        cur=db.execute('INSERT INTO datasets(name,version,description,created_at) VALUES(?,?,?,?)',
                       (name.strip(),version,description[:1000],now()))
        did=cur.lastrowid
        for s in parsed:
            db.execute('''INSERT INTO samples(dataset_id,sample_key,problem_id,optimization,compiler,architecture,language,
                         assembly,reference_source,dependencies,tests_json,demo_prediction)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                       (did,s['sample_key'],s['problem_id'],s['optimization'],s['compiler'],s['architecture'],s['language'],
                        s['assembly'],s['reference_source'],s['dependencies'],s['tests_json'],s['demo_prediction']))
    return {'dataset_id':did,'sample_count':len(parsed)}

@app.get('/api/datasets/{dataset_id}/samples')
def list_samples(dataset_id:int,limit:int=Query(200,ge=1,le=1000)):
    with connect() as db:
        required(db,'datasets',dataset_id)
        return [dict(x) for x in db.execute('''SELECT id,sample_key,problem_id,optimization,compiler,architecture,language
                   FROM samples WHERE dataset_id=? ORDER BY id LIMIT ?''',(dataset_id,limit))]

@app.get('/api/samples/{sample_id}')
def sample_detail(sample_id:int):
    with connect() as db:
        item=dict(required(db,'samples',sample_id))
    item['tests']=json.loads(item.pop('tests_json'))
    # demo_prediction is not part of model inputs; only shown on detailed inspection.
    return item

@app.get('/api/models')
def models():
    with connect() as db:
        return [dict(m) for m in db.execute('SELECT * FROM models ORDER BY id DESC')]

@app.post('/api/models',status_code=201)
def create_model(body:ModelCreate):
    with connect() as db:
        cur=db.execute('''INSERT INTO models(name,kind,base_url,api_model,api_key_env,system_prompt,api_mode,created_at)
                          VALUES(?,?,?,?,?,?,?,?)''',
                       (body.name,body.kind,body.base_url,body.api_model,body.api_key_env,
                        body.system_prompt,body.api_mode,now()))
    return {'model_id':cur.lastrowid}

@app.post('/api/models/{model_id}/test')
def test_model(model_id:int):
    with connect() as db:
        m=required(db,'models',model_id)
    if m['kind']=='fixture':
        return {'ok':True,'message':'Fixture model is ready. Outputs are illustrative, not LLM-generated.'}
    if m['kind']=='offline':
        return {'ok':True,'message':'Offline predictions do not require a model endpoint.'}
    from .worker import generate_prediction
    try:
        content=generate_prediction(m,{'assembly':'add:\n  lea eax,[rdi+rsi]\n  ret'},
          {'prompt_template':'# This is the assembly code:\n{assembly}\n# What is the source code?\n', 'temperature':0,'max_tokens':128})
        return {'ok':True,'preview':content[:200]}
    except Exception as exc:
        return {'ok':False,'message':str(exc)[:400]}


def _select_samples(db, dataset_id:int, opts:list[str], limit:int):
    markers=','.join('?' for _ in opts)
    args=[dataset_id,*opts]
    q=f'SELECT id,sample_key FROM samples WHERE dataset_id=? AND optimization IN ({markers}) ORDER BY id'
    if limit:
        q+=' LIMIT ?'
        args.append(limit)
    return list(db.execute(q,args))

@app.post('/api/runs',status_code=201)
def create_run(body:RunCreate):
    with connect() as db:
        required(db,'datasets',body.dataset_id)
        selected_model=required(db,'models',body.model_id)
        if selected_model['kind']=='offline':
            raise HTTPException(422,'Offline models must be imported via /api/runs/offline')
        samples=_select_samples(db,body.dataset_id,body.optimizations,body.sample_limit)
        if not samples:
            raise HTTPException(422,'No samples match the selected conditions')
        cfg=body.model_dump()
        cur=db.execute('''INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at)
                VALUES(?,?,?,?,?,?)''',(body.dataset_id,body.model_id,'queued',len(samples),
                                        json.dumps(cfg,ensure_ascii=False),now()))
        rid=cur.lastrowid
    submit_run(rid)
    return {'run_id':rid,'status':'queued','total':len(samples)}

@app.post('/api/runs/offline',status_code=201)
async def import_offline_run(dataset_id:int=Form(...),name:str=Form('Offline predictions'),file:UploadFile=File(...)):
    """JSONL rows: {\"sample_id\":\"...\",\"prediction\":\"int f(...) {...}\"}."""
    raw=await file.read(MAX_UPLOAD_BYTES+1)
    if len(raw)>MAX_UPLOAD_BYTES:
        raise HTTPException(413,'Offline predictions file exceeds 64MB')
    mapping={}
    try:
        for n,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
            if not line.strip():continue
            item=json.loads(line)
            sid=str(item['sample_id']); pred=item['prediction']
            if not isinstance(pred,str) or len(pred)>120000 or sid in mapping:
                raise ValueError(f'Line {n}: invalid or duplicated prediction')
            mapping[sid]=pred
        if not mapping:
            raise ValueError('Empty predictions file')
    except (KeyError,UnicodeError,ValueError,TypeError) as exc:
        raise HTTPException(422,str(exc)[:400])
    with connect() as db:
        required(db,'datasets',dataset_id)
        samples=list(db.execute('SELECT sample_key FROM samples WHERE dataset_id=?',(dataset_id,)))
        allowed={s['sample_key'] for s in samples}
        if not set(mapping).issubset(allowed):
            raise HTTPException(422,'Some sample_id values are not present in the selected dataset')
        if set(mapping)!=allowed:
            raise HTTPException(422,f'Offline file must cover all dataset samples ({len(allowed)} expected, {len(mapping)} supplied)')
        mid=db.execute('INSERT INTO models(name,kind,created_at) VALUES(?,?,?)',
                       (name.strip()[:100] or 'Offline predictions','offline',now())).lastrowid
        cfg={'dataset_id':dataset_id,'model_id':mid,'optimizations':['O0','O1','O2','O3','Os','Ofast'],
             'sample_limit':0,'prompt_template':'offline predictions','temperature':0,'max_tokens':1024}
        rid=db.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) VALUES(?,?,?,?,?,?)',
                       (dataset_id,mid,'queued',len(samples),'{}',now())).lastrowid
        file_dir=DATA_DIR/'offline'
        file_dir.mkdir(parents=True,exist_ok=True)
        path=file_dir/f'run_{rid}.json'
        path.write_text(json.dumps(mapping,ensure_ascii=False),encoding='utf-8')
        cfg['offline_file']=str(path)
        db.execute('UPDATE runs SET config_json=? WHERE id=?',(json.dumps(cfg),rid))
    submit_run(rid)
    return {'run_id':rid,'status':'queued','total':len(samples)}


def summarize(rows:list[dict]):
    eligible=[x for x in rows if x['status'] not in EXCLUDED]
    n=len(eligible)
    cp=sum(x['compile_success']==1 for x in eligible)
    re=sum(x['status']=='TEST_PASS' for x in eligible)
    tp=sum(x['tests_passed'] for x in eligible)
    tt=sum(x['tests_total'] for x in eligible)
    status_counts={}
    for item in rows:
        status_counts[item['status']]=status_counts.get(item['status'],0)+1
    return {'evaluated':n,'excluded':len(rows)-n,'rc':round(100*cp/n,2) if n else None,
            're':round(100*re/n,2) if n else None,
            'tcp':round(100*tp/tt,2) if tt else None,
            'compile_pass':cp,'re_pass':re,'test_pass':tp,'test_total':tt,
            'status_counts':status_counts}

@app.get('/api/runs')
def list_runs():
    with connect() as db:
        return [dict(x) for x in db.execute('''SELECT r.*,m.name AS model_name,d.name AS dataset_name
            FROM runs r JOIN models m ON r.model_id=m.id JOIN datasets d ON r.dataset_id=d.id
            ORDER BY r.id DESC LIMIT 100''')]

@app.get('/api/runs/{run_id}')
def get_run(run_id:int):
    with connect() as db:
        run=dict(required(db,'runs',run_id))
        run['config']=json.loads(run.pop('config_json'))
        rows=[dict(x) for x in db.execute('''SELECT r.id,r.sample_id,r.status,r.compile_success,r.tests_passed,r.tests_total,
            r.latency_ms,r.error,s.sample_key,s.problem_id,s.optimization,s.compiler
            FROM results r JOIN samples s ON r.sample_id=s.id
            WHERE r.run_id=? ORDER BY s.id''',(run_id,))]
        model=required(db,'models',run['model_id'])
        dataset=required(db,'datasets',run['dataset_id'])
    run['model_name']=model['name']
    run['dataset_name']=dataset['name']
    grouped=defaultdict(list)
    for item in rows:
        grouped[item['optimization']].append(item)
    return {'run':run,'metrics':summarize(rows),
            'by_optimization':{opt:summarize(items) for opt,items in sorted(grouped.items())},
            'results':rows}

@app.get('/api/results/{result_id}')
def result_detail(result_id:int):
    with connect() as db:
        row=db.execute('''SELECT r.*,s.sample_key,s.problem_id,s.optimization,s.compiler,
             s.assembly,s.reference_source,s.dependencies,s.tests_json
             FROM results r JOIN samples s ON s.id=r.sample_id WHERE r.id=?''',(result_id,)).fetchone()
        if row is None:raise HTTPException(404,'Result not found')
        data=dict(row)
    data['tests']=json.loads(data.pop('tests_json'))
    return data

@app.post('/api/runs/{run_id}/cancel')
def cancel_run(run_id:int):
    with connect() as db:
        run=required(db,'runs',run_id)
        if run['status']=='queued':
            db.execute("UPDATE runs SET status='cancelled',finished_at=? WHERE id=?",(now(),run_id))
        elif run['status']=='running':
            db.execute("UPDATE runs SET status='cancel_requested' WHERE id=?",(run_id,))
        else:
            raise HTTPException(409,'Run cannot be cancelled in its current state')
    return {'ok':True}

@app.post('/api/runs/{run_id}/retry')
def retry_run(run_id:int):
    with connect() as db:
        run=required(db,'runs',run_id)
        if run['status'] not in ('interrupted','completed_with_errors','cancelled'):
            raise HTTPException(409,'Only interrupted, cancelled or infrastructure-failed runs can resume')
        db.execute("DELETE FROM results WHERE run_id=? AND status IN ('INFRA_ERROR','INFERENCE_ERROR')",(run_id,))
        count=db.execute('SELECT COUNT(*) FROM results WHERE run_id=?',(run_id,)).fetchone()[0]
        db.execute("UPDATE runs SET status='queued',finished_at=NULL,error='',completed=? WHERE id=?",(count,run_id))
    submit_run(run_id)
    return {'ok':True,'run_id':run_id}

@app.get('/api/compare')
def compare_runs(run_ids:str=Query(...,description='Comma-separated run IDs')):
    try:
        ids=[int(x) for x in run_ids.split(',')]
        if not 1<=len(ids)<=8:
            raise ValueError()
    except ValueError:
        raise HTTPException(422,'Specify 1 to 8 run IDs')
    reports=[]
    with connect() as db:
        for rid in ids:
            run=required(db,'runs',rid)
            model=required(db,'models',run['model_id'])
            rows=[dict(r) for r in db.execute('SELECT * FROM results WHERE run_id=?',(rid,))]
            reports.append({'id':rid,'model':model['name'],'dataset_id':run['dataset_id'],
                            'status':run['status'],**summarize(rows)})
    if len({r['dataset_id'] for r in reports})>1:
        raise HTTPException(422,'Compare only runs from the same dataset')
    return reports
