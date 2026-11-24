"""Single-host asynchronous run processor. API requests never execute C themselves."""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import json
import os
import httpx
from .db import connect, now
from .sandbox import evaluate_code

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='eval-worker')


def submit_run(run_id: int):
    return _executor.submit(process_run, run_id)


def generate_prediction(model, sample, cfg: dict) -> str:
    if model['kind']=='fixture':
        if not sample['demo_prediction']:
            raise ValueError('Fixture model can only be used on samples with demo_prediction.')
        return sample['demo_prediction']
    if model['kind'] == 'offline':
        lookup = cfg.get('_predictions', {})
        if sample['sample_key'] not in lookup:
            raise ValueError(f'Missing offline prediction for {sample["sample_key"]}')
        return lookup[sample['sample_key']]
    if model['kind'] != 'openai':
        raise ValueError(f'Unsupported model kind: {model["kind"]}')
    key_env = model['api_key_env']
    api_key = os.getenv(key_env, '') if key_env else ''
    if key_env and not api_key:
        raise RuntimeError(f'Missing environment variable {key_env}')
    api_mode = model['api_mode'] if 'api_mode' in model.keys() else 'chat'
    url = model['base_url'].rstrip('/') + ('/completions' if api_mode == 'completion' else '/chat/completions')
    prompt = cfg['prompt_template'].replace('{assembly}', sample['assembly'])
    headers = {'Content-Type':'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {'model': model['api_model'], 'temperature': cfg['temperature'],
               'max_tokens': cfg['max_tokens'], 'stream': False}
    if api_mode == 'completion':
        payload['prompt'] = prompt
    else:
        payload['messages'] = [
            {'role':'system','content':model['system_prompt']},
            {'role':'user','content':prompt},
        ]
    with httpx.Client(timeout=300,follow_redirects=False) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        body = response.json()
    choice=body['choices'][0]
    content = choice.get('text','') if api_mode == 'completion' else choice['message']['content']
    if isinstance(content,list):
        return '\n'.join(str(part.get('text','')) for part in content if isinstance(part,dict))
    return content or ''


def _get_samples(db, run_id: int, cfg:dict):
    args=[cfg['dataset_id'], *cfg['optimizations']]
    markers=','.join('?' for _ in cfg['optimizations'])
    query=f'SELECT * FROM samples WHERE dataset_id=? AND optimization IN ({markers}) ORDER BY id'
    if cfg['sample_limit']:
        query += ' LIMIT ?'
        args.append(cfg['sample_limit'])
    return db.execute(query,args).fetchall()


def process_run(run_id: int):
    with connect() as db:
        run=db.execute('SELECT * FROM runs WHERE id=?',(run_id,)).fetchone()
        if not run or run['status'] != 'queued':
            return
        model=db.execute('SELECT * FROM models WHERE id=?',(run['model_id'],)).fetchone()
        cfg=json.loads(run['config_json'])
        samples=_get_samples(db,run_id,cfg)
        if cfg.get('offline_file'):
            with open(cfg['offline_file'],encoding='utf-8') as fp:
                cfg['_predictions']=json.load(fp)
        db.execute("UPDATE runs SET status='running', error='' WHERE id=?",(run_id,))

    try:
        for sample in samples:
            with connect() as db:
                status=db.execute('SELECT status FROM runs WHERE id=?',(run_id,)).fetchone()['status']
                if status=='cancel_requested':
                    db.execute('UPDATE runs SET status=?,finished_at=? WHERE id=?',('cancelled',now(),run_id))
                    return
                done=db.execute('SELECT id FROM results WHERE run_id=? AND sample_id=?',(run_id,sample['id'])).fetchone()
                if done:
                    continue
            started=perf_counter()
            prediction=''
            try:
                prediction=generate_prediction(model,sample,cfg)
                outcome=(evaluate_code(prediction,sample['dependencies'],json.loads(sample['tests_json']), language='cpp')
                         if sample['language'] == 'cpp' else
                         evaluate_code(prediction,sample['dependencies'],json.loads(sample['tests_json'])))
            except Exception as exc:
                # Network/service outages are not scored as incorrect decompilation.
                from .sandbox import EvalOutcome
                outcome=EvalOutcome('INFERENCE_ERROR',None,0,len(json.loads(sample['tests_json'])),
                                    error=f'{type(exc).__name__}: {str(exc)[:350]}')
            latency_ms=round((perf_counter()-started)*1000)
            with connect() as db:
                db.execute('''INSERT OR IGNORE INTO results
                    (run_id,sample_id,status,compile_success,tests_passed,tests_total,
                     model_output,compile_log,run_log,latency_ms,error,created_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                   (run_id,sample['id'],outcome.status,
                    None if outcome.compile_success is None else int(outcome.compile_success),
                    outcome.tests_passed,outcome.tests_total,prediction[:150000],
                    outcome.compile_log,outcome.run_log,latency_ms,outcome.error,now()))
                db.execute('UPDATE runs SET completed=(SELECT COUNT(*) FROM results WHERE run_id=?) WHERE id=?',
                           (run_id,run_id))
        with connect() as db:
            excluded=db.execute("SELECT COUNT(*) AS n FROM results WHERE run_id=? AND status IN ('INFRA_ERROR','INFERENCE_ERROR')",(run_id,)).fetchone()['n']
            db.execute('UPDATE runs SET status=?,finished_at=? WHERE id=?',
                       ('completed_with_errors' if excluded else 'completed',now(),run_id))
    except Exception as exc:
        with connect() as db:
            db.execute('UPDATE runs SET status=?,error=?,finished_at=? WHERE id=?',
                       ('interrupted',f'{type(exc).__name__}: {exc}'[:500],now(),run_id))
        raise
