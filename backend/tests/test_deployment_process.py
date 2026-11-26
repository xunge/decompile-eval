"""Real process lifecycle test using a fake local vLLM executable.

This exercises argv-only spawning, health checks, auto model registration,
stop (SIGTERM) and restart recovery without requiring a GPU.
"""
import os
import stat
import time

import pytest

from app import db, deployments, gpu
from app.deployments import DeploymentError

FAKE_VLLM = r'''#!/usr/bin/env python3
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

port = 0
name = 'fake-model'
for i, arg in enumerate(sys.argv):
    if arg == '--port' and i + 1 < len(sys.argv):
        port = int(sys.argv[i + 1])
    if arg == '--served-model-name' and i + 1 < len(sys.argv):
        name = sys.argv[i + 1]

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.rstrip('/').endswith('/models'):
            body = json.dumps({'data': [{'id': name}]}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, *args):
        pass

print('fake vllm serving', name, 'on', port, flush=True)
ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()
'''


@pytest.fixture
def fake_vllm(tmp_path, monkeypatch):
    script = tmp_path / 'fake_vllm.py'
    script.write_text(FAKE_VLLM)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(deployments, 'VLLM_BIN', str(script))
    monkeypatch.setattr(gpu, 'device_count', lambda: None)
    monkeypatch.setattr(gpu, 'safe_snapshot', lambda: {
        'available': False, 'error': 'no driver', 'devices': [], 'sampled_at': db.now()})
    db.initialize()
    with db.connect() as conn:
        conn.execute('DELETE FROM deployments')
    yield script
    # Always reap a leaked fake process.
    with db.connect() as conn:
        rows = [dict(r) for r in conn.execute('SELECT * FROM deployments')]
    for row in rows:
        if row['managed'] and deployments._pid_alive(row['pid']):
            try:
                os.killpg(row['pgid'] or row['pid'], 9)
            except OSError:
                pass
    with db.connect() as conn:
        conn.execute('DELETE FROM deployments')
        conn.execute('DELETE FROM models WHERE deployment_id IS NOT NULL')


def _payload(tmp_path):
    model = tmp_path / 'model'
    model.mkdir(exist_ok=True)
    (model / 'config.json').write_text('{"model_type":"fake"}')
    # Pick a free port.
    import socket
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return {
        'name': 'Process Test', 'served_name': 'fake-model',
        'model_path': str(model), 'gpu_ids': '0', 'port': port,
        'dtype': 'auto', 'max_model_len': 512, 'max_num_seqs': 4,
        'max_num_batched_tokens': 512, 'gpu_memory_utilization': 0.1,
        'api_mode': 'completion',
    }


def _wait_status(deployment_id, status, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = deployments.get_deployment(deployment_id)
        if row['status'] == status:
            return row
        time.sleep(0.1)
    raise AssertionError(f'deployment {deployment_id} did not reach {status}: '
                         f'{deployments.get_deployment(deployment_id)}')


def test_real_process_start_register_stop_and_recover(tmp_path, fake_vllm):
    row = deployments.start_deployment(_payload(tmp_path))
    running = _wait_status(row['id'], 'running')
    assert running['pid'] and running['alive'] is True
    assert running['model_id']
    with db.connect() as conn:
        model = conn.execute('SELECT * FROM models WHERE deployment_id=?',
                             (row['id'],)).fetchone()
    assert model is not None and model['api_model'] == 'fake-model'
    assert deployments.log_tail(row['id'])

    # Simulate an API restart: the child stays alive, recovery reattaches.
    with db.connect() as conn:
        conn.execute("UPDATE deployments SET status='running' WHERE id=?", (row['id'],))
    deployments.recover_deployments()
    recovered = _wait_status(row['id'], 'running')
    assert recovered['pid'] == running['pid']

    stopped = deployments.stop_deployment(row['id'])
    assert stopped['status'] == 'stopped'
    time.sleep(0.2)
    assert deployments.get_deployment(row['id'])['alive'] is False

    restarted = deployments.restart_deployment(row['id'])
    assert restarted['id'] == row['id']
    _wait_status(row['id'], 'running')
    deployments.stop_deployment(row['id'])


def test_external_process_cannot_be_stopped(tmp_path, fake_vllm):
    row = deployments.start_deployment(_payload(tmp_path))
    _wait_status(row['id'], 'running')
    # A deployment row created by an external service (managed=0).
    with db.connect() as conn:
        conn.execute("UPDATE deployments SET managed=0 WHERE id=?", (row['id'],))
    with pytest.raises(DeploymentError, match='external'):
        deployments.stop_deployment(row['id'])
    # restore managed and stop the real child
    with db.connect() as conn:
        conn.execute("UPDATE deployments SET managed=1 WHERE id=?", (row['id'],))
    deployments.stop_deployment(row['id'])


def test_cancel_deployment_stops_real_process_and_removes_record(tmp_path, fake_vllm):
    row = deployments.start_deployment(_payload(tmp_path))
    _wait_status(row['id'], 'running')
    pid = row['pid']
    result = deployments.cancel_deployment(row['id'])
    assert result['deleted'] and result['stopped']
    time.sleep(0.2)
    assert deployments._pid_alive(pid) is False
    with pytest.raises(DeploymentError):
        deployments.get_deployment(row['id'])
