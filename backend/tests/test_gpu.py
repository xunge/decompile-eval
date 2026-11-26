"""GPU monitor tests with a fake NVML module (no driver required)."""
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import db, gpu
from app.main import app


class FakeNVML:
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, count=2, fail=False):
        self.count = count
        self.fail = fail

    def nvmlInit(self):
        if self.fail:
            raise RuntimeError('driver not loaded')

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, index):
        return index

    def nvmlDeviceGetName(self, handle):
        return f'Fake GPU {handle}'.encode()

    def nvmlDeviceGetUUID(self, handle):
        return f'GPU-fake-{handle}'.encode()

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=10 + handle)

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(total=24 * 1024**3, used=(2 + handle) * 1024**3)

    def nvmlDeviceGetTemperature(self, handle, kind):
        return 40 + handle

    def nvmlDeviceGetPowerUsage(self, handle):
        return 100_000 + handle

    def nvmlDeviceGetEnforcedPowerLimit(self, handle):
        return 300_000

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [SimpleNamespace(pid=1000 + handle, usedGpuMemory=1024**3)]

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        return []


@pytest.fixture(autouse=True)
def clean_gpu_tables():
    db.initialize()
    with db.connect() as conn:
        conn.execute('DELETE FROM vllm_samples')
        conn.execute('DELETE FROM gpu_samples')
        conn.execute('DELETE FROM deployments')
    yield
    gpu.reset_nvml_state()
    with db.connect() as conn:
        conn.execute('DELETE FROM vllm_samples')
        conn.execute('DELETE FROM gpu_samples')
        conn.execute('DELETE FROM deployments')


def use_fake_nvml(monkeypatch, fake):
    monkeypatch.setattr(gpu, 'pynvml', fake)
    gpu.reset_nvml_state()


def test_snapshot_fields_and_processes(monkeypatch):
    use_fake_nvml(monkeypatch, FakeNVML(count=2))
    devices = gpu.snapshot()
    assert len(devices) == 2
    first = devices[0]
    assert first['index'] == 0 and first['name'] == 'Fake GPU 0'
    assert first['utilization'] == 10.0
    assert first['memory_total_mb'] == 24 * 1024
    assert first['memory_used_mb'] == 2 * 1024
    assert first['temperature_c'] == 40.0
    assert first['power_w'] == 100.0
    assert first['power_limit_w'] == 300.0
    assert first['processes'][0]['pid'] == 1000
    assert gpu.device_count() == 2


def test_snapshot_unavailable_is_soft(monkeypatch):
    use_fake_nvml(monkeypatch, FakeNVML(fail=True))
    result = gpu.safe_snapshot()
    assert result['available'] is False
    assert 'driver not loaded' in result['error']
    assert gpu.device_count() is None


def _insert_external_deployment():
    with db.connect() as conn:
        return conn.execute('''INSERT INTO deployments
            (name,served_name,model_path,gpu_ids,port,dtype,max_model_len,
             max_num_seqs,max_num_batched_tokens,gpu_memory_utilization,api_mode,
             managed,base_url,status,created_at,config_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            ('ext', 'm', '', '', 9000, 'auto', 4096, 8, 2048, 0.0,
             'completion', 0, 'http://127.0.0.1:9000/v1', 'external',
             db.now(), '{}')).lastrowid


def test_monitor_stores_gpu_and_vllm_rates(monkeypatch):
    use_fake_nvml(monkeypatch, FakeNVML(count=2))
    did = _insert_external_deployment()
    devices = [
        {'index': 0, 'uuid': 'u0', 'name': 'GPU0', 'utilization': 50.0,
         'memory_used_mb': 1000.0, 'memory_total_mb': 24000.0,
         'temperature_c': 60.0, 'power_w': 120.0, 'power_limit_w': 300.0,
         'processes': []},
        {'index': 1, 'uuid': 'u1', 'name': 'GPU1', 'utilization': 20.0,
         'memory_used_mb': 500.0, 'memory_total_mb': 24000.0,
         'temperature_c': 50.0, 'power_w': 100.0, 'power_limit_w': 300.0,
         'processes': []},
    ]
    monkeypatch.setattr(gpu, 'safe_snapshot', lambda: {
        'available': True, 'error': '', 'devices': devices,
        'sampled_at': db.now()})
    responses = iter([
        {'num_requests_running': 2, 'num_requests_waiting': 1,
         'prompt_tokens_total': 100, 'generation_tokens_total': 50},
        {'num_requests_running': 3, 'num_requests_waiting': 0,
         'prompt_tokens_total': 200, 'generation_tokens_total': 150},
    ])
    monkeypatch.setattr(gpu, 'fetch_vllm_metrics', lambda url: next(responses))
    monitor = gpu.GPUMonitor()
    monitor.sample_once(epoch=1000.0)
    monitor.sample_once(epoch=1001.0)
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM gpu_samples').fetchone()[0] == 4
        rows = [dict(r) for r in conn.execute(
            'SELECT * FROM vllm_samples ORDER BY ts_epoch')]
    assert len(rows) == 2
    assert rows[0]['input_tokens_per_sec'] == 0.0
    assert rows[1]['input_tokens_per_sec'] == 100.0
    assert rows[1]['output_tokens_per_sec'] == 100.0
    assert gpu.latest_metrics(did)['running'] == 3
    with db.connect() as conn:
        assert conn.execute('SELECT status FROM deployments WHERE id=?',
                            (did,)).fetchone()[0] == 'external'


def test_overview_and_history(monkeypatch):
    use_fake_nvml(monkeypatch, FakeNVML(count=1))
    with db.connect() as conn:
        conn.execute('''INSERT INTO gpu_samples
            (ts,ts_epoch,gpu_index,gpu_uuid,name,utilization,memory_used_mb,
             memory_total_mb,temperature_c,power_w,power_limit_w)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (db.now(), time.time(), 0, 'u', 'GPU', 30.0, 1000, 24000, 55.0, 100, 300))
    history = gpu.gpu_history(300)
    assert history[0][0]['utilization'] == 30.0


def test_gpu_api_uses_service(monkeypatch):
    from app import main as main_module
    monkeypatch.setattr(main_module.gpu_service, 'overview', lambda seconds=300: {
        'available': True, 'error': '', 'devices': [{'index': 0, 'name': 'GPU0',
        'utilization': 42.0, 'history': []}], 'sampled_at': db.now()})
    with TestClient(app) as client:
        response = client.get('/api/gpus?history_seconds=60')
        assert response.status_code == 200
        body = response.json()
        assert body['available'] is True
        assert body['devices'][0]['utilization'] == 42.0
