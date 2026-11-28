"""Deployment validation, lifecycle, recovery and permission tests.

No real vLLM process is started: Popen/spawn and PID checks are mocked.
"""
import json
import os
import socket
import signal
import time

import pytest
from fastapi.testclient import TestClient

from app import db, deployments, gpu
from app.deployments import DeploymentError
from app.main import app


@pytest.fixture(autouse=True)
def clean_deployments():
    db.initialize()
    with db.connect() as conn:
        conn.execute('DELETE FROM deployments')
        conn.execute('DELETE FROM models WHERE deployment_id IS NOT NULL')
    yield
    with db.connect() as conn:
        conn.execute('DELETE FROM deployments')
        conn.execute('DELETE FROM models WHERE deployment_id IS NOT NULL')
        conn.execute('DELETE FROM vllm_samples')


def free_port():
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def model_dir(tmp_path):
    path = tmp_path / 'model'
    path.mkdir(exist_ok=True)
    (path / 'config.json').write_text('{"model_type":"test"}')
    return str(path)


def payload(tmp_path, **overrides):
    data = {
        'name': 'Test Deploy', 'served_name': 'test-deploy',
        'model_path': model_dir(tmp_path), 'gpu_ids': '0',
        'port': free_port(), 'dtype': 'auto',
        'max_model_len': 4096, 'max_num_seqs': 8,
        'max_num_batched_tokens': 2048,
        'gpu_memory_utilization': 0.5, 'api_mode': 'completion',
    }
    data.update(overrides)
    return data


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(gpu, 'device_count', lambda: None)
    monkeypatch.setattr(gpu, 'safe_snapshot', lambda: {
        'available': False, 'error': 'no driver', 'devices': [], 'sampled_at': db.now()})


def test_validate_accepts_valid_spec(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    spec = deployments.validate_deployment(payload(tmp_path))
    assert spec['served_name'] == 'test-deploy'
    assert spec['gpu_ids'] == '0'
    assert spec['base_url'].endswith('/v1')


def test_validate_rejects_bad_paths_and_ports(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    with pytest.raises(DeploymentError, match='absolute'):
        deployments.validate_deployment(payload(tmp_path, model_path='relative/model'))
    with pytest.raises(DeploymentError, match='does not exist'):
        deployments.validate_deployment(payload(tmp_path, model_path='/tmp/does-not-exist-xyz'))
    missing = tmp_path / 'empty'
    missing.mkdir()
    with pytest.raises(DeploymentError, match='config.json'):
        deployments.validate_deployment(payload(tmp_path, model_path=str(missing)))
    with pytest.raises(DeploymentError, match='reserved'):
        deployments.validate_deployment(payload(tmp_path, port=8001))
    with pytest.raises(DeploymentError, match='between 1024'):
        deployments.validate_deployment(payload(tmp_path, port=80))
    with pytest.raises(DeploymentError, match='dtype'):
        deployments.validate_deployment(payload(tmp_path, dtype='int8'))
    with pytest.raises(DeploymentError, match='gpu_memory_utilization'):
        deployments.validate_deployment(payload(tmp_path, gpu_memory_utilization=1.5))
    with pytest.raises(DeploymentError, match='max_num_seqs'):
        deployments.validate_deployment(payload(tmp_path, max_num_seqs=0))


def test_validate_rejects_port_in_use(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('127.0.0.1', 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    try:
        with pytest.raises(DeploymentError, match='in use'):
            deployments.validate_deployment(payload(tmp_path, port=port))
    finally:
        sock.close()


def test_validate_gpu_ids_against_nvml(tmp_path, monkeypatch):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(gpu, 'device_count', lambda: 1)
    monkeypatch.setattr(gpu, 'safe_snapshot', lambda: {
        'available': False, 'error': '', 'devices': [], 'sampled_at': db.now()})
    with pytest.raises(DeploymentError, match='out of range'):
        deployments.validate_deployment(payload(tmp_path, gpu_ids='0,1'))
    with pytest.raises(DeploymentError, match='invalid GPU'):
        deployments.validate_deployment(payload(tmp_path, gpu_ids='x'))


class FakeProcess:
    pid = 424242


def test_command_is_argv_without_shell(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    captured = {}

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return FakeProcess()

    monkeypatch.setattr(deployments.subprocess, 'Popen', fake_popen)
    row = deployments.start_deployment(payload(tmp_path, gpu_ids='0,1'))
    assert row['status'] == 'starting' and row['pid'] == 424242
    assert isinstance(captured['args'], list)
    assert captured['kwargs']['shell'] is False
    assert '--tensor-parallel-size' in captured['args']
    assert captured['kwargs']['env']['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert captured['kwargs']['env']['VLLM_BATCH_INVARIANT'] == '1'
    assert '--seed' in captured['args']
    assert '--no-enable-prefix-caching' in captured['args']
    assert captured['args'][0] == '/bin/true'


def test_mark_running_registers_model(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path, served_name='served-a'))
    deployments._mark_running(row['id'])
    updated = deployments.get_deployment(row['id'])
    assert updated['status'] == 'running'
    with db.connect() as conn:
        model = conn.execute('SELECT * FROM models WHERE deployment_id=?',
                             (row['id'],)).fetchone()
    assert model is not None
    assert model['base_url'] == row['base_url']
    assert model['api_model'] == 'served-a'
    assert model['kind'] == 'openai'
    deployments.stop_deployment(row['id'])
    assert deployments.get_deployment(row['id'])['status'] == 'stopped'


def test_stop_sends_sigterm_and_does_not_touch_mismatched_pid(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    alive = {'value': True}
    signals = []
    monkeypatch.setattr(deployments, '_pid_alive', lambda pid: alive['value'])
    monkeypatch.setattr(deployments, '_pid_matches', lambda *a: True)

    def fake_killpg(pgid, sig):
        signals.append(sig)
        alive['value'] = False

    monkeypatch.setattr(deployments.os, 'killpg', fake_killpg)
    stopped = deployments.stop_deployment(row['id'])
    assert stopped['status'] == 'stopped'
    assert signals == [signal.SIGTERM]

    row2 = deployments.start_deployment(payload(tmp_path, name='Mismatch'))
    alive['value'] = True
    monkeypatch.setattr(deployments, '_pid_matches', lambda *a: False)
    deployments.stop_deployment(row2['id'])
    assert deployments.get_deployment(row2['id'])['status'] == 'stopped'
    assert 'not signalled' in deployments.get_deployment(row2['id'])['error']


def test_restart_uses_persisted_config(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    monkeypatch.setattr(deployments, '_pid_alive', lambda pid: False)
    restarted = deployments.restart_deployment(row['id'])
    assert restarted['id'] == row['id']
    assert restarted['status'] == 'starting'
    assert restarted['name'] == row['name']


def test_external_registration_is_monitored_not_stopped(tmp_path, monkeypatch, no_gpu):
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {'data': [{'id': 'remote-model'}]}

    class Client:
        def __init__(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def get(self, url):
            return Response()

    monkeypatch.setattr(deployments.httpx, 'Client', Client)
    row = deployments.register_external({
        'name': 'Remote', 'base_url': 'http://127.0.0.1:9999',
        'served_name': '', 'api_mode': 'completion'})
    assert row['managed'] == 0 and row['status'] == 'external'
    assert row['served_name'] == 'remote-model'
    with db.connect() as conn:
        model = conn.execute('SELECT * FROM models WHERE deployment_id=?',
                             (row['id'],)).fetchone()
    assert model is not None and model['base_url'].endswith('/v1')
    with pytest.raises(DeploymentError, match='external'):
        deployments.stop_deployment(row['id'])


def test_recovery_reattaches_matching_pid_and_marks_missing_stopped(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    deployments._mark_running(row['id'])
    calls = []
    monkeypatch.setattr(deployments, '_pid_matches',
                        lambda pid, port, path: port == row['port'])
    monkeypatch.setattr(deployments, '_watch_start',
                        lambda deployment_id: calls.append(deployment_id))
    deployments.recover_deployments()
    assert calls == [row['id']]
    assert deployments.get_deployment(row['id'])['status'] == 'starting'

    row2 = deployments.start_deployment(payload(tmp_path, name='Lost'))
    monkeypatch.setattr(deployments, '_pid_matches', lambda *a: False)
    deployments.recover_deployments()
    assert deployments.get_deployment(row2['id'])['status'] == 'stopped'
    assert 'not signalled' in deployments.get_deployment(row2['id'])['error']


def test_deployment_api_admin_token_and_gpu_endpoint(monkeypatch):
    monkeypatch.setattr(deployments, 'ADMIN_TOKEN', 'secret')
    monkeypatch.setattr(deployments, 'ALLOW_DEPLOY', True)
    monkeypatch.setattr(deployments, 'start_deployment',
                        lambda data: {'id': 1, **data, 'status': 'starting'})
    monkeypatch.setattr(gpu, 'overview', lambda seconds=300: {
        'available': True, 'error': '', 'devices': [], 'sampled_at': db.now()})
    with TestClient(app) as client:
        body = {'name': 'x', 'model_path': '/tmp/x', 'port': 12345,
                'max_model_len': 4096, 'max_num_seqs': 8,
                'max_num_batched_tokens': 2048, 'gpu_memory_utilization': 0.5}
        assert client.post('/api/deployments', json=body).status_code == 403
        created = client.post('/api/deployments', json=body,
                              headers={'X-Admin-Token': 'secret'})
        assert created.status_code == 201, created.text
        assert created.json()['status'] == 'starting'
        gpus = client.get('/api/gpus').json()
        assert gpus['available'] is True
    monkeypatch.setattr(deployments, 'ALLOW_DEPLOY', False)
    with TestClient(app) as client:
        assert client.post('/api/deployments', json=body).status_code == 403


def test_log_tail_returns_recent_bytes(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    path = deployments._log_path(row['id'])
    path.write_text('line1\nline2\nline3\n', encoding='utf-8')
    assert 'line3' in deployments.log_tail(row['id'])


def test_cancel_managed_deployment_stops_and_removes(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    deployments._mark_running(row['id'])
    alive = {'value': True}
    monkeypatch.setattr(deployments, '_pid_alive', lambda pid: alive['value'])
    monkeypatch.setattr(deployments, '_pid_matches', lambda *a: True)
    monkeypatch.setattr(deployments.os, 'killpg',
                        lambda pgid, sig: alive.__setitem__('value', False))
    result = deployments.cancel_deployment(row['id'])
    assert result['deleted'] and result['stopped'] and result['managed']
    with pytest.raises(DeploymentError):
        deployments.get_deployment(row['id'])
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM models WHERE deployment_id=?',
                            (row['id'],)).fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM deployments WHERE id=?',
                            (row['id'],)).fetchone()[0] == 0


def test_cancel_external_does_not_signal_and_removes_monitoring(monkeypatch, no_gpu):
    with db.connect() as conn:
        deployment_id = conn.execute('''INSERT INTO deployments
            (name,served_name,model_path,gpu_ids,port,dtype,max_model_len,
             max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,api_mode,
             managed,base_url,status,created_at,config_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            ('Remote cancel', 'remote-model', '', '', 9999, 'auto', 4096, 8,
             2048, 0.0, 'completion', 0,
             'http://127.0.0.1:9999/v1', 'external', db.now(), '{}')).lastrowid
    signals = []
    monkeypatch.setattr(deployments, '_pid_matches', lambda *a: True)
    monkeypatch.setattr(deployments.os, 'killpg', lambda *a: signals.append(a))
    result = deployments.cancel_deployment(deployment_id)
    assert result['external_untouched'] is True
    assert signals == []
    assert result['deleted'] is True


def test_cancel_keeps_model_referenced_by_historical_run(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    monkeypatch.setattr(deployments.subprocess, 'Popen', lambda *a, **k: FakeProcess())
    row = deployments.start_deployment(payload(tmp_path))
    deployments._mark_running(row['id'])
    row = deployments.get_deployment(row['id'])
    # Create a historical run referencing the auto-registered model.
    with db.connect() as conn:
        dataset = conn.execute('SELECT id FROM datasets LIMIT 1').fetchone()
        if dataset is None:
            db.seed_demo()
            with db.connect() as conn2:
                dataset = conn2.execute('SELECT id FROM datasets LIMIT 1').fetchone()
        conn.execute('''INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at)
                        VALUES(?,?,?,?,?,?)''',
                     (dataset[0], row['model_id'], 'completed', 0, '{}', db.now()))
    deployments.cancel_deployment(row['id'])
    with db.connect() as conn:
        model = conn.execute('SELECT * FROM models WHERE id=?', (row['model_id'],)).fetchone()
    assert model is not None
    assert model['deployment_id'] is None


def test_cancel_api_requires_admin(monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'ADMIN_TOKEN', 'secret')
    monkeypatch.setattr(deployments, 'ALLOW_DEPLOY', True)
    called = []
    monkeypatch.setattr(deployments, 'cancel_deployment',
                        lambda deployment_id: called.append(deployment_id) or {'ok': True})
    with TestClient(app) as client:
        assert client.delete('/api/deployments/1').status_code == 403
        response = client.delete('/api/deployments/1', headers={'X-Admin-Token': 'secret'})
        assert response.status_code == 200
        assert called == [1]


def test_deterministic_deployment_flags_are_configurable(tmp_path, monkeypatch, no_gpu):
    monkeypatch.setattr(deployments, 'VLLM_BIN', '/bin/true')
    monkeypatch.setattr(deployments, '_watch_start', lambda deployment_id: None)
    captured = {}

    def fake_popen(args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return FakeProcess()

    monkeypatch.setattr(deployments.subprocess, 'Popen', fake_popen)
    row = deployments.start_deployment(payload(
        tmp_path, deterministic=False, seed=42,
        disable_prefix_caching=False, enforce_eager=True))
    assert row['deterministic'] == 0 and row['seed'] == 42
    assert row['disable_prefix_caching'] == 0 and row['enforce_eager'] == 1
    args = captured['args']
    assert args[args.index('--seed') + 1] == '42'
    assert '--no-enable-prefix-caching' not in args
    assert '--enforce-eager' in args
    assert 'VLLM_BATCH_INVARIANT' not in captured['kwargs']['env']
    # Restart reuses the persisted deterministic parameters.
    monkeypatch.setattr(deployments, '_pid_alive', lambda pid: False)
    again = deployments.restart_deployment(row['id'])
    assert again['seed'] == 42 and again['enforce_eager'] == 1
