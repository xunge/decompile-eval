"""Local vLLM deployment manager.

Only the path to the trusted vLLM executable is used, always with
``subprocess.Popen(list, shell=False)``.  Arguments are built from validated
fields; no user string is ever passed to a shell.

A managed deployment is started in its own session so it survives an API
restart.  Recovery re-reads ``/proc/<pid>/cmdline`` before trusting a PID, and
stop/restart only signals a process that still matches the recorded model/port.
External deployments are health-checked and monitored but never stopped.
"""
from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from . import gpu
from .config import (
    ADMIN_TOKEN, ALLOW_DEPLOY, DATA_DIR, DEPLOY_LOG_TAIL_BYTES,
    DEPLOY_MAX_RUNNING, DEPLOY_RESERVED_PORTS, DEPLOY_START_TIMEOUT,
    DEPLOY_STOP_TIMEOUT, VLLM_BIN,
)
from .db import connect, now

DTYPES = {'auto', 'half', 'float16', 'bfloat16', 'float32'}
API_MODES = {'completion', 'chat'}
SERVED_NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$')
_ACTIVE_STATUSES = {'starting', 'running', 'unreachable'}
_DEPLOY_LOCK = threading.RLock()


class DeploymentError(RuntimeError):
    def __init__(self, message: str, status_code: int = 422):
        self.status_code = status_code
        super().__init__(message)


def _log_dir() -> Path:
    path = DATA_DIR / 'deployments'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _as_bool(value) -> bool:
    return bool(int(value)) if isinstance(value, int) else bool(value)


def _parse_gpu_ids(raw) -> list[int]:
    if isinstance(raw, int):
        items = [raw]
    else:
        text = str(raw or '').strip()
        if not text:
            raise DeploymentError('gpu_ids is required (e.g. "0" or "0,1")')
        items = []
        for part in text.split(','):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise DeploymentError(f'invalid GPU id: {part!r}')
            items.append(int(part))
    items = sorted(set(items))
    if not items:
        raise DeploymentError('select at least one GPU')
    if len(items) > 8:
        raise DeploymentError('select at most 8 GPUs')
    return items


def _port_available(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _running_deployments() -> list[dict]:
    with connect() as db:
        return [dict(row) for row in db.execute(
            "SELECT * FROM deployments WHERE status IN ('starting','running','unreachable')")]


def validate_deployment(data: dict) -> dict:
    model_path = str(data.get('model_path') or '').strip()
    if not model_path:
        raise DeploymentError('model_path is required')
    if not os.path.isabs(model_path):
        raise DeploymentError('model_path must be an absolute local path')
    real = os.path.realpath(model_path)
    if not os.path.isdir(real):
        raise DeploymentError(f'model_path does not exist or is not a directory: {model_path}')
    if not os.path.isfile(os.path.join(real, 'config.json')):
        raise DeploymentError('model_path must contain a Hugging Face config.json')
    if not os.access(real, os.R_OK | os.X_OK):
        raise DeploymentError('model_path is not readable')

    name = str(data.get('name') or '').strip()
    if not name or len(name) > 100:
        raise DeploymentError('name must be 1–100 characters')
    served_name = str(data.get('served_name') or name).strip()
    if not SERVED_NAME_RE.fullmatch(served_name):
        raise DeploymentError('served_name may only contain letters, digits, dot, dash, underscore')

    port = int(data.get('port') or 0)
    if not (1024 <= port <= 65535):
        raise DeploymentError('port must be between 1024 and 65535')
    if port in DEPLOY_RESERVED_PORTS:
        raise DeploymentError(f'port {port} is reserved by the DeployEval stack')
    if not _port_available(port):
        raise DeploymentError(f'port {port} is already in use')

    dtype = str(data.get('dtype') or 'auto').strip().lower()
    if dtype not in DTYPES:
        raise DeploymentError(f'dtype must be one of {sorted(DTYPES)}')
    api_mode = str(data.get('api_mode') or 'completion').strip().lower()
    if api_mode not in API_MODES:
        raise DeploymentError('api_mode must be completion or chat')

    def _number(key, default):
        value = data.get(key)
        return default if value is None or value == '' else value

    max_model_len = int(_number('max_model_len', 4096))
    max_num_seqs = int(_number('max_num_seqs', 64))
    max_num_batched_tokens = int(_number('max_num_batched_tokens', 8192))
    utilization = float(_number('gpu_memory_utilization', 0.7))
    if not (128 <= max_model_len <= 1_000_000):
        raise DeploymentError('max_model_len must be 128–1,000,000')
    if not (1 <= max_num_seqs <= 4096):
        raise DeploymentError('max_num_seqs must be 1–4096')
    if not (1 <= max_num_batched_tokens <= 1_000_000):
        raise DeploymentError('max_num_batched_tokens must be 1–1,000,000')
    if not (0.05 <= utilization <= 0.99):
        raise DeploymentError('gpu_memory_utilization must be 0.05–0.99')
    seed = int(_number('seed', 0))
    if not (0 <= seed <= 2_147_483_647):
        raise DeploymentError('seed must be between 0 and 2147483647')
    deterministic = bool(data.get('deterministic', True))
    disable_prefix_caching = bool(data.get('disable_prefix_caching', True))
    enforce_eager = bool(data.get('enforce_eager', False))

    gpu_ids = _parse_gpu_ids(data.get('gpu_ids'))
    count = gpu.device_count()
    if count is not None:
        bad = [item for item in gpu_ids if item >= count]
        if bad:
            raise DeploymentError(f'GPU id(s) {bad} out of range; detected {count} GPU(s)')
    snap = gpu.safe_snapshot()
    if snap['available']:
        by_index = {item['index']: item for item in snap['devices']}
        for index in gpu_ids:
            device = by_index.get(index)
            if device is None or not device.get('memory_total_mb'):
                continue
            required = device['memory_total_mb'] * utilization
            free = device['memory_total_mb'] - (device.get('memory_used_mb') or 0)
            if required > free + 256:
                raise DeploymentError(
                    f'GPU {index} has {free:.0f} MiB free but this deployment '
                    f'needs about {required:.0f} MiB; lower gpu_memory_utilization '
                    f'or free another model')

    running = _running_deployments()
    if len(running) >= DEPLOY_MAX_RUNNING:
        raise DeploymentError(
            f'maximum of {DEPLOY_MAX_RUNNING} active deployments reached', 409)
    if not os.path.isfile(VLLM_BIN) or not os.access(VLLM_BIN, os.X_OK):
        raise DeploymentError(f'vLLM executable not found: {VLLM_BIN}', 503)

    return {
        'name': name, 'served_name': served_name, 'model_path': real,
        'gpu_ids': ','.join(str(item) for item in gpu_ids),
        'port': port, 'dtype': dtype, 'api_mode': api_mode,
        'max_model_len': max_model_len, 'max_num_seqs': max_num_seqs,
        'max_num_batched_tokens': max_num_batched_tokens,
        'gpu_memory_utilization': utilization,
        'deterministic': deterministic, 'seed': seed,
        'disable_prefix_caching': disable_prefix_caching,
        'enforce_eager': enforce_eager,
        'base_url': f'http://127.0.0.1:{port}/v1',
    }


def build_command(spec: dict) -> list[str]:
    """Build the argv list; there is intentionally no shell string anywhere."""
    gpu_ids = [item for item in str(spec['gpu_ids']).split(',') if item]
    command = [
        VLLM_BIN, 'serve', spec['model_path'],
        '--served-model-name', spec['served_name'],
        '--host', '127.0.0.1', '--port', str(spec['port']),
        '--dtype', spec['dtype'],
        '--max-model-len', str(spec['max_model_len']),
        '--max-num-seqs', str(spec['max_num_seqs']),
        '--max-num-batched-tokens', str(spec['max_num_batched_tokens']),
        '--gpu-memory-utilization', str(spec['gpu_memory_utilization']),
    ]
    if len(gpu_ids) > 1:
        command += ['--tensor-parallel-size', str(len(gpu_ids))]
    command += ['--seed', str(spec.get('seed', 0))]
    if spec.get('disable_prefix_caching'):
        command += ['--no-enable-prefix-caching']
    if spec.get('enforce_eager'):
        command += ['--enforce-eager']
    return command


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A terminated but not-yet-reaped child is a zombie; it is not running.
    try:
        with open(f'/proc/{pid}/stat', encoding='ascii', errors='replace') as fp:
            data = fp.read()
        state = data[data.rfind(')') + 2:][:1]
        if state == 'Z':
            return False
    except OSError:
        pass
    return True


def _process_cmdline(pid: int) -> list[str]:
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as fp:
            raw = fp.read()
        return [part.decode('utf-8', 'replace') for part in raw.split(b'\0') if part]
    except OSError:
        return []


def _pid_matches(pid: int | None, port: int, model_path: str) -> bool:
    if not _pid_alive(pid):
        return False
    argv = _process_cmdline(int(pid))
    if not argv:
        return False
    joined = ' '.join(argv)
    return ('vllm' in joined and 'serve' in argv
            and str(port) in argv and model_path in argv)


def _log_path(deployment_id: int) -> Path:
    return _log_dir() / f'deployment_{deployment_id}.log'


def _write_log(path: Path, message: str) -> None:
    try:
        with path.open('ab') as fp:
            fp.write((message.rstrip() + '\n').encode('utf-8', 'replace'))
    except OSError:
        pass


def log_tail(deployment_id: int, limit: int = DEPLOY_LOG_TAIL_BYTES) -> str:
    with connect() as db:
        row = db.execute('SELECT log_path FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
    if row is None or not row['log_path']:
        return ''
    path = Path(row['log_path'])
    try:
        size = path.stat().st_size
        with path.open('rb') as fp:
            if size > limit:
                fp.seek(size - limit)
            raw = fp.read()
        return raw.decode('utf-8', 'replace')
    except OSError:
        return ''


def _health_ok(base_url: str, served_name: str = '') -> bool:
    try:
        with httpx.Client(timeout=2.0, follow_redirects=False) as client:
            response = client.get(base_url.rstrip('/') + '/models')
            if response.status_code != 200:
                return False
            payload = response.json()
            ids = [item.get('id') for item in payload.get('data', [])
                   if isinstance(item, dict)]
            return not served_name or served_name in ids or bool(ids)
    except (httpx.HTTPError, ValueError, OSError, AttributeError, TypeError):
        return False


def _register_model(deployment: dict) -> int:
    with connect() as db:
        # The deployment may have been cancelled while the health check was
        # running; never resurrect a deleted deployment's model.
        exists = db.execute('SELECT id FROM deployments WHERE id=?',
                            (deployment['id'],)).fetchone()
        if exists is None:
            return 0
        row = db.execute('SELECT id FROM models WHERE deployment_id=?',
                         (deployment['id'],)).fetchone()
        if row is None:
            row = db.execute('SELECT id FROM models WHERE base_url=? AND api_model=?',
                             (deployment['base_url'], deployment['served_name'])).fetchone()
        if row is None:
            model_id = db.execute('''INSERT INTO models
                (name,kind,base_url,api_model,api_mode,api_key_env,system_prompt,
                 deployment_id,created_at)
                VALUES(?,?,?,?,?,?,?,?,?)''',
                (deployment['name'], 'openai', deployment['base_url'],
                 deployment['served_name'], deployment['api_mode'], '',
                 'You are an expert binary decompiler. Output only valid C source code.',
                 deployment['id'], now())).lastrowid
        else:
            model_id = row['id']
            db.execute('''UPDATE models SET name=?,kind='openai',base_url=?,api_model=?,
                          api_mode=?,deployment_id=? WHERE id=?''',
                       (deployment['name'], deployment['base_url'],
                        deployment['served_name'], deployment['api_mode'],
                        deployment['id'], model_id))
        db.execute('UPDATE deployments SET model_id=? WHERE id=?',
                   (model_id, deployment['id']))
        return int(model_id)


def _watch_start(deployment_id: int) -> None:
    deadline = time.monotonic() + DEPLOY_START_TIMEOUT
    while time.monotonic() < deadline:
        with connect() as db:
            row = db.execute('SELECT * FROM deployments WHERE id=?',
                             (deployment_id,)).fetchone()
        if row is None:
            return
        deployment = dict(row)
        if deployment['status'] in ('failed', 'stopped'):
            return
        if not _pid_alive(deployment['pid']):
            _fail(deployment_id, 'vLLM process exited during startup')
            return
        if _health_ok(deployment['base_url'], deployment['served_name']):
            # The GPU monitor may already have flipped starting->running while
            # metrics became available; register the model either way.
            _register_model(deployment)
            _mark_running(deployment_id)
            return
        time.sleep(1.0)
    message = f'vLLM did not become healthy within {DEPLOY_START_TIMEOUT}s'
    try:
        stop_deployment(deployment_id, silent=True)
    except Exception:
        pass
    _fail(deployment_id, message)


def _mark_running(deployment_id: int) -> None:
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
        if row is None:
            return
        db.execute("UPDATE deployments SET status='running', error='', started_at=?, "
                   'last_check_at=? WHERE id=?',
                   (now(), now(), deployment_id))
        deployment = dict(row)
        deployment['status'] = 'running'
    _register_model(deployment)


def _fail(deployment_id: int, message: str) -> None:
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
        if row is None:
            return
        db.execute("UPDATE deployments SET status='failed', error=?, stopped_at=?, "
                   'last_check_at=? WHERE id=?',
                   (message[:1000], now(), now(), deployment_id))
    _write_log(_log_path(deployment_id), f'[deploye] {message}')


def _spawn(deployment: dict) -> subprocess.Popen:
    log_path = Path(deployment['log_path'])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(deployment)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(deployment['gpu_ids'])
    env['PYTHONUNBUFFERED'] = '1'
    if deployment.get('deterministic'):
        env['VLLM_BATCH_INVARIANT'] = '1'
    _write_log(log_path, f'[deploye] starting: {json.dumps(command, ensure_ascii=False)}')
    _write_log(log_path, f'[deploye] CUDA_VISIBLE_DEVICES={env["CUDA_VISIBLE_DEVICES"]}')
    if env.get('VLLM_BATCH_INVARIANT'):
        _write_log(log_path, f'[deploye] VLLM_BATCH_INVARIANT={env["VLLM_BATCH_INVARIANT"]}')
    with log_path.open('ab') as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
            shell=False, env=env, cwd=str(_log_dir()))
    return process


def start_deployment(data: dict, deployment_id: int | None = None) -> dict:
    spec = validate_deployment(data)
    with _DEPLOY_LOCK:
        if deployment_id is None:
            with connect() as db:
                deployment_id = db.execute('''INSERT INTO deployments
                    (name,served_name,model_path,gpu_ids,port,dtype,max_model_len,
                     max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,
                     deterministic,seed,disable_prefix_caching,enforce_eager,
                     api_mode,managed,base_url,status,created_at,config_json)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                    (spec['name'], spec['served_name'], spec['model_path'],
                     spec['gpu_ids'], spec['port'], spec['dtype'],
                     spec['max_model_len'], spec['max_num_seqs'],
                     spec['max_num_batched_tokens'], spec['gpu_memory_utilization'],
                     int(spec['deterministic']), spec['seed'],
                     int(spec['disable_prefix_caching']), int(spec['enforce_eager']),
                     spec['api_mode'], 1, spec['base_url'], 'starting', now(),
                     json.dumps(spec, ensure_ascii=False))).lastrowid
        log_path = str(_log_path(deployment_id))
        with connect() as db:
            db.execute('''UPDATE deployments SET name=?,served_name=?,model_path=?,
                gpu_ids=?,port=?,dtype=?,max_model_len=?,max_num_seqs=?,
                max_num_batched_tokens=?,gpu_memory_utilization=?,
                deterministic=?,seed=?,disable_prefix_caching=?,enforce_eager=?,
                api_mode=?,base_url=?,managed=1,status='starting',error='',log_path=?,
                stopped_at=NULL,config_json=?,created_at=COALESCE(created_at,?)
                WHERE id=?''',
                (spec['name'], spec['served_name'], spec['model_path'],
                 spec['gpu_ids'], spec['port'], spec['dtype'], spec['max_model_len'],
                 spec['max_num_seqs'], spec['max_num_batched_tokens'],
                 spec['gpu_memory_utilization'],
                 int(spec['deterministic']), spec['seed'],
                 int(spec['disable_prefix_caching']), int(spec['enforce_eager']),
                 spec['api_mode'], spec['base_url'],
                 log_path, json.dumps(spec, ensure_ascii=False), now(), deployment_id))
            deployment = dict(db.execute('SELECT * FROM deployments WHERE id=?',
                                         (deployment_id,)).fetchone())
        try:
            process = _spawn(deployment)
        except Exception as exc:
            _fail(deployment_id, f'failed to start vLLM: {type(exc).__name__}: {exc}')
            raise DeploymentError(f'failed to start vLLM: {exc}', 500) from exc
        with connect() as db:
            db.execute('UPDATE deployments SET pid=?,pgid=? WHERE id=?',
                       (process.pid, process.pid, deployment_id))
        threading.Thread(target=_watch_start, args=(deployment_id,),
                         name=f'deploy-{deployment_id}', daemon=True).start()
        with connect() as db:
            return dict(db.execute('SELECT * FROM deployments WHERE id=?',
                                   (deployment_id,)).fetchone())


def stop_deployment(deployment_id: int, silent: bool = False) -> dict:
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
    if row is None:
        raise DeploymentError('deployment not found', 404)
    deployment = dict(row)
    if not deployment['managed']:
        raise DeploymentError('external deployments are monitored, not stopped', 409)
    pid = deployment['pid']
    if not _pid_alive(pid):
        with connect() as db:
            db.execute("UPDATE deployments SET status='stopped', stopped_at=?, "
                       "error='' WHERE id=?", (now(), deployment_id))
        return deployment
    if not _pid_matches(pid, deployment['port'], deployment['model_path']):
        with connect() as db:
            db.execute("UPDATE deployments SET status='stopped', stopped_at=?, "
                       "error='PID no longer matches this deployment; not signalled' "
                       'WHERE id=?', (now(), deployment_id))
        return deployment
    pgid = deployment['pgid'] or pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + DEPLOY_STOP_TIMEOUT
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.2)
    if _pid_alive(pid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    with connect() as db:
        db.execute("UPDATE deployments SET status='stopped', stopped_at=?, "
                   "error='' WHERE id=?", (now(), deployment_id))
    _write_log(_log_path(deployment_id), '[deploye] stopped by API')
    with connect() as db:
        return dict(db.execute('SELECT * FROM deployments WHERE id=?',
                               (deployment_id,)).fetchone())


def cancel_deployment(deployment_id: int) -> dict:
    """Undeploy: stop a managed process and remove the deployment record.

    For external services only the monitoring record is removed; the external
    process is never signalled.  An auto-registered model is deleted only when
    no evaluation run references it; otherwise it is detached from the
    deployment and kept for historical run views.
    """
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
    if row is None:
        raise DeploymentError('deployment not found', 404)
    deployment = dict(row)
    stopped = False
    if deployment['managed']:
        stop_deployment(deployment_id, silent=True)
        stopped = True
    with connect() as db:
        models = [dict(item) for item in db.execute(
            'SELECT id FROM models WHERE deployment_id=?', (deployment_id,))]
        for model in models:
            used = db.execute('SELECT COUNT(*) FROM runs WHERE model_id=?',
                              (model['id'],)).fetchone()[0]
            if used:
                db.execute('UPDATE models SET deployment_id=NULL WHERE id=?',
                           (model['id'],))
            else:
                db.execute('DELETE FROM models WHERE id=?', (model['id'],))
        db.execute('DELETE FROM deployments WHERE id=?', (deployment_id,))
    _write_log(_log_path(deployment_id), '[deploye] cancelled/undeployed by API')
    return {'ok': True, 'deleted': True, 'deployment_id': deployment_id,
            'managed': bool(deployment['managed']), 'stopped': stopped,
            'external_untouched': not bool(deployment['managed'])}


def restart_deployment(deployment_id: int) -> dict:
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
    if row is None:
        raise DeploymentError('deployment not found', 404)
    deployment = dict(row)
    if not deployment['managed']:
        raise DeploymentError('external deployments cannot be restarted', 409)
    if deployment['status'] in _ACTIVE_STATUSES:
        stop_deployment(deployment_id, silent=True)
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
        data = json.loads(row['config_json'])
    return start_deployment(data, deployment_id)


def recover_deployments() -> None:
    """Reattach to vLLM children left running across an API restart."""
    with connect() as db:
        rows = [dict(row) for row in db.execute(
            "SELECT * FROM deployments WHERE managed=1 AND status IN "
            "('starting','running','unreachable')")]
    for deployment in rows:
        if _pid_matches(deployment['pid'], deployment['port'], deployment['model_path']):
            with connect() as db:
                db.execute("UPDATE deployments SET status='starting', error='' WHERE id=?",
                           (deployment['id'],))
            threading.Thread(target=_watch_start, args=(deployment['id'],),
                             name=f'deploy-recover-{deployment["id"]}', daemon=True).start()
        else:
            with connect() as db:
                db.execute("UPDATE deployments SET status='stopped', stopped_at=?, "
                           "error=? WHERE id=?",
                           (now(), 'process not found after API restart; not signalled',
                            deployment['id']))


def register_external(data: dict) -> dict:
    name = str(data.get('name') or '').strip()
    if not name or len(name) > 100:
        raise DeploymentError('name must be 1–100 characters')
    base_url = str(data.get('base_url') or '').strip().rstrip('/')
    if not re.match(r'^https?://[A-Za-z0-9._:\-\[\]/]+$', base_url):
        raise DeploymentError('base_url must be an http(s) URL')
    if not base_url.endswith('/v1'):
        base_url += '/v1'
    api_mode = str(data.get('api_mode') or 'completion').strip().lower()
    if api_mode not in API_MODES:
        raise DeploymentError('api_mode must be completion or chat')
    try:
        with httpx.Client(timeout=5.0, follow_redirects=False) as client:
            response = client.get(base_url + '/models')
            response.raise_for_status()
            models = [item.get('id') for item in response.json().get('data', [])
                      if isinstance(item, dict)]
    except (httpx.HTTPError, ValueError, AttributeError, TypeError) as exc:
        raise DeploymentError(f'cannot reach external vLLM service: {exc}') from exc
    served_name = str(data.get('served_name') or '').strip()
    if not served_name:
        if not models:
            raise DeploymentError('external service returned no models')
        served_name = models[0]
    elif served_name not in models and models:
        raise DeploymentError(f'served_name {served_name!r} not present in external /v1/models')
    if not SERVED_NAME_RE.fullmatch(served_name):
        raise DeploymentError('invalid served_name')
    port = 0
    try:
        port = int(base_url.split('//', 1)[1].split('/', 1)[0].rsplit(':', 1)[1])
    except (IndexError, ValueError):
        pass
    with connect() as db:
        deployment_id = db.execute('''INSERT INTO deployments
            (name,served_name,model_path,gpu_ids,port,dtype,max_model_len,
             max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,api_mode,
             managed,base_url,status,created_at,config_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (name, served_name, '', '', port, 'auto', 4096, 64, 8192,
             0.0, api_mode, 0, base_url, 'external', now(),
             json.dumps({'name': name, 'served_name': served_name,
                         'base_url': base_url, 'api_mode': api_mode}))).lastrowid
        deployment = dict(db.execute('SELECT * FROM deployments WHERE id=?',
                                     (deployment_id,)).fetchone())
    _register_model(deployment)
    return deployment


def list_deployments() -> list[dict]:
    with connect() as db:
        rows = [dict(row) for row in db.execute(
            'SELECT * FROM deployments ORDER BY id DESC')]
    for row in rows:
        row['metrics'] = gpu.latest_metrics(row['id'])
        row['alive'] = _pid_alive(row['pid']) if row['managed'] else None
    return rows


def get_deployment(deployment_id: int) -> dict:
    with connect() as db:
        row = db.execute('SELECT * FROM deployments WHERE id=?',
                         (deployment_id,)).fetchone()
    if row is None:
        raise DeploymentError('deployment not found', 404)
    deployment = dict(row)
    deployment['metrics'] = gpu.latest_metrics(deployment_id)
    deployment['alive'] = _pid_alive(deployment['pid']) if deployment['managed'] else None
    return deployment


def require_admin(token: str) -> None:
    if not ALLOW_DEPLOY:
        raise DeploymentError('deployment management is disabled by configuration', 403)
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise DeploymentError('admin token required', 403)
