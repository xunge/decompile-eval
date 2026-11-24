"""DecompileEval MVP REST API, intentionally single-user and development-only."""
from contextlib import asynccontextmanager
from collections import defaultdict
from pathlib import Path
from typing import Any
import hashlib
import json
import threading
import time
import uuid
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from .config import (DATA_DIR, CORS_ORIGINS, MAX_UPLOAD_BYTES, MAX_SAMPLES,
                     INFERENCE_CONCURRENCY, COMPILE_CONCURRENCY, MAX_MODEL_LEN,
                     OFFLINE_BATCH_SIZE, VLLM_METRICS_ENABLED)
from .db import connect, initialize, seed_demo, now
from .schemas import ModelCreate, RunCreate, ReferenceCheckCreate, BatchRunCreate
from .worker import submit_run, get_progress, clear_progress, REFERENCE_MODEL_NAME
from .adapters import OfflineBatchAdapter, fetch_vllm_metrics
from .sandbox import extract_c
from .case_extract import METRIC_VERSION

EXCLUDED={'INFRA_ERROR','INFERENCE_ERROR'}
# Rows that exist before the sample reaches a final state are never counted in
# RC/RE/TCP denominators (old databases have none of them).
TRANSIENT={'GENERATED','COMPILING'}
ACTIVE_RUN_STATUSES=('queued','running','cancel_requested')
ACTIVE_RUN_MARKS=','.join('?' for _ in ACTIVE_RUN_STATUSES)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize()
    seed_demo()
    yield

app=FastAPI(title='DecompileEval MVP', version='0.5.0', lifespan=lifespan)
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


def offline_files(rows):
    """Resolve only the offline prediction files this app wrote under DATA_DIR."""
    paths=[]
    for row in rows:
        try:
            cfg=json.loads(row['config_json'] or '{}')
        except (TypeError,ValueError):
            continue
        raw=cfg.get('offline_file')
        if not raw:
            continue
        try:
            path=Path(raw).resolve()
        except (OSError,TypeError,ValueError):
            continue
        if DATA_DIR in path.parents:
            paths.append(path)
    return paths


def remove_files(paths):
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


_METRICS_CACHE: dict[str, tuple[float, dict]] = {}
_METRICS_LOCK = threading.Lock()
_METRICS_TTL = 2.0


def attach_vllm_metrics(progress: dict, model, cfg: dict) -> dict:
    """Best-effort vLLM /metrics scrape, cached for two seconds."""
    if not VLLM_METRICS_ENABLED or model['kind'] != 'openai':
        return progress
    url = cfg.get('vllm_metrics_url') or model['base_url']
    if not url:
        return progress
    now_ts = time.monotonic()
    with _METRICS_LOCK:
        cached = _METRICS_CACHE.get(url)
        if cached and now_ts - cached[0] < _METRICS_TTL:
            if cached[1]:
                progress['vllm_metrics'] = dict(cached[1])
            return progress
    metrics = fetch_vllm_metrics(url)
    with _METRICS_LOCK:
        _METRICS_CACHE[url] = (now_ts, metrics or {})
    if metrics:
        progress['vllm_metrics'] = metrics
    return progress


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
    return {'ok':True,'version':'0.5.0','sandbox':'docker-required','pipeline':'async'}

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
    source_hash=hashlib.sha256(raw).hexdigest()
    with connect() as db:
        cur=db.execute('INSERT INTO datasets(name,version,description,source_hash,created_at) '
                       'VALUES(?,?,?,?,?)',
                       (name.strip(),version,description[:1000],source_hash,now()))
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
        # Reference-source runs use an internal model row; keep it out of the
        # model picker while runs can still join its name.
        return [dict(m) for m in db.execute(
            "SELECT * FROM models WHERE kind!='reference' ORDER BY id DESC")]

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
    if m['kind']=='reference':
        return {'ok':True,'message':'Reference-source runs do not call a model endpoint.'}
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
        dataset=required(db,'datasets',body.dataset_id)
        selected_model=required(db,'models',body.model_id)
        if selected_model['kind']=='offline':
            raise HTTPException(422,'Offline models must be imported via /api/runs/offline')
        if selected_model['kind']=='reference':
            raise HTTPException(422,'Use /api/datasets/{id}/reference-check for reference runs')
        samples=_select_samples(db,body.dataset_id,body.optimizations,body.sample_limit)
        if not samples:
            raise HTTPException(422,'No samples match the selected conditions')
        cfg=body.model_dump()
        cfg['inference_concurrency']=(body.inference_concurrency
                                      if body.inference_concurrency is not None
                                      else INFERENCE_CONCURRENCY)
        cfg['compile_concurrency']=(body.compile_concurrency
                                    if body.compile_concurrency is not None
                                    else COMPILE_CONCURRENCY)
        cfg['generation_mode']='online'
        cfg['batch_size']=0
        cfg['max_model_len']=MAX_MODEL_LEN
        cfg['dataset_hash']=dataset['source_hash']
        cfg['dataset_version']=dataset['version']
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
        dataset=required(db,'datasets',dataset_id)
        samples=list(db.execute('SELECT sample_key FROM samples WHERE dataset_id=?',(dataset_id,)))
        allowed={s['sample_key'] for s in samples}
        if not set(mapping).issubset(allowed):
            raise HTTPException(422,'Some sample_id values are not present in the selected dataset')
        if set(mapping)!=allowed:
            raise HTTPException(422,f'Offline file must cover all dataset samples ({len(allowed)} expected, {len(mapping)} supplied)')
        mid=db.execute('INSERT INTO models(name,kind,created_at) VALUES(?,?,?)',
                       (name.strip()[:100] or 'Offline predictions','offline',now())).lastrowid
        cfg={'dataset_id':dataset_id,'model_id':mid,'optimizations':['O0','O1','O2','O3','Os','Ofast'],
             'sample_limit':0,'prompt_template':'offline predictions','temperature':0,'max_tokens':1024,
             'inference_concurrency':1,'compile_concurrency':COMPILE_CONCURRENCY,
             'generation_mode':'online','batch_size':0,'max_model_len':MAX_MODEL_LEN,
             'dataset_hash':dataset['source_hash'],'dataset_version':dataset['version']}
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


@app.post('/api/runs/batch',status_code=201)
def create_batch_run(body:BatchRunCreate):
    """Optional offline generation through the local vLLM Python API."""
    model_path=(body.model_path or '').strip()
    if not model_path:
        raise HTTPException(422,'model_path is required (or set DEC_EVAL_BATCH_MODEL_PATH)')
    if not OfflineBatchAdapter.available():
        raise HTTPException(503,'The vLLM Python package is not installed in the backend '
                                'environment; use the online adapter or install vllm.')
    with connect() as db:
        dataset=required(db,'datasets',body.dataset_id)
        selected_model=required(db,'models',body.model_id)
        if selected_model['kind'] in ('offline','reference'):
            raise HTTPException(422,'Offline-batch runs require a normal model entry')
        samples=_select_samples(db,body.dataset_id,body.optimizations,body.sample_limit)
        if not samples:
            raise HTTPException(422,'No samples match the selected conditions')
        cfg=body.model_dump()
        cfg['generation_mode']='offline_batch'
        cfg['model_path']=model_path
        cfg['batch_size']=body.batch_size or OFFLINE_BATCH_SIZE
        cfg['inference_concurrency']=cfg['batch_size']
        cfg['compile_concurrency']=(body.compile_concurrency
                                    if body.compile_concurrency is not None
                                    else COMPILE_CONCURRENCY)
        cfg['max_model_len']=MAX_MODEL_LEN
        cfg['dataset_hash']=dataset['source_hash']
        cfg['dataset_version']=dataset['version']
        rid=db.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) '
                       'VALUES(?,?,?,?,?,?)',(body.dataset_id,body.model_id,'queued',len(samples),
                                              json.dumps(cfg,ensure_ascii=False),now())).lastrowid
    submit_run(rid)
    return {'run_id':rid,'status':'queued','total':len(samples),'generation_mode':'offline_batch'}


@app.post('/api/datasets/{dataset_id}/reference-check',status_code=201)
def reference_check(dataset_id:int,body:ReferenceCheckCreate|None=None):
    """Evaluate the dataset's own reference sources with the exact same sandbox.

    Results are stored in a normal run (using an internal model row) so they can
    be inspected, cancelled and exported like any other evaluation.  Reference
    failures indicate a dataset/environment problem, not a model failure.
    """
    body=body or ReferenceCheckCreate()
    with connect() as db:
        dataset=required(db,'datasets',dataset_id)
        model=db.execute("SELECT id FROM models WHERE kind='reference' AND name=?",
                         (REFERENCE_MODEL_NAME,)).fetchone()
        mid=model['id'] if model else db.execute(
            "INSERT INTO models(name,kind,system_prompt,created_at) VALUES(?,?,?,?)",
            (REFERENCE_MODEL_NAME,'reference','',now())).lastrowid
        samples=_select_samples(db,dataset_id,body.optimizations,body.sample_limit)
        if not samples:
            raise HTTPException(422,'No samples match the selected conditions')
        cfg={'dataset_id':dataset_id,'model_id':mid,'optimizations':body.optimizations,
             'sample_limit':body.sample_limit,
             'prompt_template':'reference source (no model call)','temperature':0,'max_tokens':128,
             'inference_concurrency':1,
             'compile_concurrency':(body.compile_concurrency
                                    if body.compile_concurrency is not None
                                    else COMPILE_CONCURRENCY),
             'generation_mode':'online','batch_size':0,'max_model_len':MAX_MODEL_LEN,
             'dataset_hash':dataset['source_hash'],'dataset_version':dataset['version'],
             'reference':True}
        rid=db.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) '
                       'VALUES(?,?,?,?,?,?)',(dataset_id,mid,'queued',len(samples),
                                              json.dumps(cfg,ensure_ascii=False),now())).lastrowid
    submit_run(rid)
    return {'run_id':rid,'status':'queued','total':len(samples),'reference':True}


@app.get('/api/runs/{run_id}/results/{result_id}/test-cases')
def result_test_cases(run_id:int, result_id:int,
                      status:str=Query('',description='PASS/FAIL/unsupported or empty'),
                      page:int=Query(1,ge=1), page_size:int=Query(50,ge=1,le=500),
                      export:str=Query('',description='json or csv')):
    with connect() as db:
        result=db.execute('SELECT * FROM results WHERE id=? AND run_id=?',
                          (result_id,run_id)).fetchone()
        if result is None:
            raise HTTPException(404,'Result not found')
        all_status=[dict(r) for r in db.execute(
            'SELECT status,COUNT(*) AS n FROM test_case_results '
            'WHERE run_id=? AND sample_id=? GROUP BY status',(run_id,result['sample_id']))]
        where='run_id=? AND sample_id=?'
        args=[run_id,result['sample_id']]
        filt=(status or '').strip()
        if filt.lower()=='pass':
            where+=' AND status=?'; args.append('PASS')
        elif filt.lower()=='fail':
            where+=' AND status=?'; args.append('FAIL')
        elif filt.lower()=='unsupported':
            where+=" AND status NOT IN ('PASS','FAIL')"
        elif filt:
            where+=' AND status=?'; args.append(filt.upper())
        total=db.execute(f'SELECT COUNT(*) FROM test_case_results WHERE {where}',args).fetchone()[0]
        params=list(args)
        query=f'SELECT * FROM test_case_results WHERE {where} ORDER BY case_index,id'
        if export.lower() not in ('json','csv'):
            query+=' LIMIT ? OFFSET ?'; params += [page_size,(page-1)*page_size]
        rows=[dict(r) for r in db.execute(query,params)]
    for row in rows:
        for key in ('case_source_span',):
            raw=row.get(key) or ''
            if raw:
                try: row[key+'_json']=json.loads(raw)
                except ValueError: row[key+'_json']={}
        for key in ('expected_value_json','actual_value_json','input_values_json'):
            raw=row.get(key) or ''
            if raw:
                try: row[key+'_parsed']=json.loads(raw)
                except ValueError: row[key+'_parsed']=None
        for key in ('stdout','stderr'):
            row[key]=row.get(key) or ''
    counts={item['status']:item['n'] for item in all_status}
    executed=sum(counts.get(s,0) for s in ('PASS','FAIL'))
    passed=counts.get('PASS',0)
    official=result['status']
    consistent=None
    if executed:
        consistent=(official=='TEST_PASS')==(passed==executed)
    meta={
        'run_id':run_id,'result_id':result_id,'sample_id':result['sample_id'],
        'official_status':official,'official_tests_passed':result['tests_passed'],
        'official_tests_total':result['tests_total'],
        'test_case_status':result['test_case_status'],
        'cases_declared':result['cases_declared'],
        'cases_total':result['cases_total'],'cases_passed':result['cases_passed'],
        'coverage':round(min(100.0, 100*result['cases_total'] /
                             max(result['cases_declared'], result['cases_total'])),2)
                    if result['cases_declared'] else None,
        'diagnostic_consistent':consistent,
        'case_status_counts':counts,'metric_version':result['metric_version'],
    }
    if export.lower()=='csv':
        import csv as _csv, io as _io
        out=_io.StringIO()
        columns=['case_id','case_index','static_id','status','passed','error_type',
                 'input_text','expected_text','actual_text','compare_expression',
                 'comparison_rule','oracle_source','diagnostic_status','exit_code',
                 'signal','duration_ms','stdout','stderr','case_source_span']
        writer=_csv.writer(out)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c,'') for c in columns])
        return Response(content=out.getvalue(), media_type='text/csv',
                        headers={'Content-Disposition':
                                 f'attachment; filename="run{run_id}_result{result_id}_cases.csv"'})
    if export.lower()=='json':
        return {'summary':meta,'total':total,'items':rows,'export':'json'}
    return {'summary':meta,'total':total,'page':page,'page_size':page_size,
            'items':rows}


def summarize(rows:list[dict]):
    eligible=[x for x in rows if x['status'] not in EXCLUDED and x['status'] not in TRANSIENT]
    n=len(eligible)
    cp=sum(x['compile_success']==1 for x in eligible)
    re=sum(x['status']=='TEST_PASS' for x in eligible)
    tp=sum(x['tests_passed'] for x in eligible)
    tt=sum(x['tests_total'] for x in eligible)
    status_counts={}
    for item in rows:
        status_counts[item['status']]=status_counts.get(item['status'],0)+1
    # Case-level TCP is only meaningful when the result rows carry case data.
    has_case_columns=any('cases_total' in x for x in rows)
    case_rows=[x for x in eligible if int(x.get('cases_total') or 0)>0]
    case_total=sum(int(x.get('cases_total') or 0) for x in case_rows)
    case_passed=sum(int(x.get('cases_passed') or 0) for x in case_rows)
    case_declared=sum(int(x.get('cases_declared') or 0) for x in eligible)
    tcp_micro=round(100*case_passed/case_total,2) if case_total else None
    ratios=[int(x.get('cases_passed') or 0)/int(x.get('cases_total') or 1)
            for x in case_rows]
    tcp_macro=round(100*sum(ratios)/len(ratios),2) if ratios else None
    if has_case_columns:
        tcp=tcp_micro
        metric_version=METRIC_VERSION
    else:
        # Legacy rows (or unit-test rows without diagnostics) keep the old
        # tests_passed/tests_total definition; never invent case-level TCP.
        tcp=round(100*tp/tt,2) if tt else None
        metric_version='legacy'
    diagnosed_samples=len(case_rows)
    coverage=round(100*diagnosed_samples/n,2) if n else None
    unsupported=sum(1 for x in eligible
                    if (x.get('test_case_status') or '') not in
                    ('','OK','DISCRETE','PARTIAL','NOT_RUN_COMPILE_ERROR'))
    return {'evaluated':n,'excluded':len(rows)-n,'rc':round(100*cp/n,2) if n else None,
            're':round(100*re/n,2) if n else None,
            'tcp':tcp,'tcp_micro':tcp_micro,'tcp_macro':tcp_macro,
            'compile_pass':cp,'re_pass':re,'test_pass':tp,'test_total':tt,
            'cases_declared':case_declared,'cases_total':case_total,
            'cases_passed':case_passed,'diagnosed_samples':diagnosed_samples,
            'case_coverage':coverage,'unsupported_samples':unsupported,
            'official_re_denominator':n,'tcp_valid_cases':case_total,
            'uncovered_cases':max(0,case_declared-case_total),
            'uncovered_samples':max(0,n-diagnosed_samples),
            'failure_type_counts':{k:v for k,v in status_counts.items() if k!='TEST_PASS'},
            'metric_version':metric_version,'status_counts':status_counts}

@app.get('/api/runs')
def list_runs():
    with connect() as db:
        items=[dict(x) for x in db.execute('''SELECT r.*,m.name AS model_name,d.name AS dataset_name
            FROM runs r JOIN models m ON r.model_id=m.id JOIN datasets d ON r.dataset_id=d.id
            ORDER BY r.id DESC LIMIT 100''')]
    # Live counters are in-memory and cheap to attach for the few active runs.
    for item in items:
        if item['status'] in ACTIVE_RUN_STATUSES:
            item['progress']=get_progress(item['id'])
    return items

@app.get('/api/runs/{run_id}')
def get_run(run_id:int):
    with connect() as db:
        run=dict(required(db,'runs',run_id))
        run['config']=json.loads(run.pop('config_json'))
        rows=[dict(x) for x in db.execute('''SELECT r.id,r.sample_id,r.status,r.compile_success,
            r.tests_passed,r.tests_total,r.inference_ms,r.compile_ms,r.test_ms,
            r.sandbox_ms,r.latency_ms,
            r.prompt_tokens,r.completion_tokens,r.error,
            r.cases_declared,r.cases_total,r.cases_passed,r.test_case_status,
            r.metric_version,r.submitted_at,r.generated_at,r.first_token_at,
            r.compile_started_at,r.compile_finished_at,
            s.sample_key,s.problem_id,s.optimization,s.compiler
            FROM results r JOIN samples s ON r.sample_id=s.id
            WHERE r.run_id=? ORDER BY s.id''',(run_id,))]
        model=required(db,'models',run['model_id'])
        dataset=required(db,'datasets',run['dataset_id'])
    run['model_name']=model['name']
    run['dataset_name']=dataset['name']
    progress=attach_vllm_metrics(get_progress(run_id),model,run['config'])
    grouped=defaultdict(list)
    for item in rows:
        grouped[item['optimization']].append(item)
    return {'run':run,'progress':progress,'metrics':summarize(rows),
            'by_optimization':{opt:summarize(items) for opt,items in sorted(grouped.items())},
            'results':rows}


@app.get('/api/runs/{run_id}/progress')
def run_progress(run_id:int):
    with connect() as db:
        run=required(db,'runs',run_id)
        model=required(db,'models',run['model_id'])
    try:
        cfg=json.loads(run['config_json'] or '{}')
    except ValueError:
        cfg={}
    progress=get_progress(run_id)
    return attach_vllm_metrics(progress,model,cfg)

@app.get('/api/results/{result_id}')
def result_detail(result_id:int):
    with connect() as db:
        row=db.execute('''SELECT r.*,s.sample_key,s.problem_id,s.optimization,s.compiler,
             s.assembly,s.reference_source,s.dependencies,s.tests_json
             FROM results r JOIN samples s ON s.id=r.sample_id WHERE r.id=?''',(result_id,)).fetchone()
        if row is None:raise HTTPException(404,'Result not found')
        data=dict(row)
    data['tests']=json.loads(data.pop('tests_json'))
    data['extracted_source']=extract_c(data.get('model_output') or '')
    try:
        detail=json.loads(data.get('detail_json') or '{}')
    except ValueError:
        detail={}
    data['detail']=detail
    data['tests_detail']=detail.get('tests') or []
    data['harness_mode']=detail.get('harness_mode') or ('program' if len(data['tests'])==1 and 'program' in data['tests'][0] else 'cases')
    data['harness_note']=detail.get('harness_note') or ''
    try:
        data['generation_params']=json.loads(data.get('generation_params_json') or '{}')
    except ValueError:
        data['generation_params']={}
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
        count=db.execute("SELECT COUNT(*) FROM results WHERE run_id=? AND status NOT IN ('GENERATED','COMPILING')",
                         (run_id,)).fetchone()[0]
        db.execute("UPDATE runs SET status='queued',started_at=NULL,finished_at=NULL,error='',completed=? WHERE id=?",
                   (count,run_id))
    clear_progress(run_id)
    submit_run(run_id)
    return {'ok':True,'run_id':run_id}

@app.delete('/api/runs/{run_id}')
def delete_run(run_id:int):
    """Delete one finished run together with its results and offline prediction file."""
    with connect() as db:
        run=required(db,'runs',run_id)
        if run['status'] in ACTIVE_RUN_STATUSES:
            raise HTTPException(409,'Stop the run before deleting it')
        db.execute('DELETE FROM results WHERE run_id=?',(run_id,))
        db.execute('DELETE FROM runs WHERE id=?',(run_id,))
        # Offline imports create one model per run; drop it once no run references it.
        db.execute("DELETE FROM models WHERE kind='offline' AND NOT EXISTS (SELECT 1 FROM runs WHERE runs.model_id=models.id)")
        clear_progress(run_id)
        files=offline_files([run])
    remove_files(files)
    return {'ok':True,'deleted':1}


@app.delete('/api/runs')
def delete_finished_runs():
    """Delete every finished run (completed, failed, cancelled or interrupted)."""
    with connect() as db:
        rows=[dict(x) for x in db.execute(
            f'SELECT id,config_json FROM runs WHERE status NOT IN ({ACTIVE_RUN_MARKS})',ACTIVE_RUN_STATUSES)]
        if not rows:
            return {'ok':True,'deleted':0}
        ids=[x['id'] for x in rows]
        marks=','.join('?' for _ in ids)
        db.execute(f'DELETE FROM results WHERE run_id IN ({marks})',ids)
        db.execute(f'DELETE FROM runs WHERE id IN ({marks})',ids)
        db.execute("DELETE FROM models WHERE kind='offline' AND NOT EXISTS (SELECT 1 FROM runs WHERE runs.model_id=models.id)")
        for item in ids:
            clear_progress(item)
        files=offline_files(rows)
    remove_files(files)
    return {'ok':True,'deleted':len(ids)}


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
