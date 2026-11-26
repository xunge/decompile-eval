"""NVIDIA GPU monitoring through nvidia-ml-py and vLLM /metrics.

The monitor thread samples all GPUs every ``GPU_SAMPLE_INTERVAL`` seconds and
stores a bounded history in SQLite.  It also polls each deployed vLLM service's
``/metrics`` for running/waiting requests and token throughput.  NVML is
optional: machines without a driver simply report ``available=false`` and the
rest of DecompileEval keeps working.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

try:  # nvidia-ml-py exposes the pynvml module.
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - driver/package absent
    pynvml = None  # type: ignore

from .adapters import fetch_vllm_metrics
from .config import (
    GPU_HISTORY_SECONDS, GPU_METRICS_INTERVAL, GPU_MONITOR_ENABLED,
    GPU_SAMPLE_INTERVAL,
)
from .db import connect, now

_NVML_LOCK = threading.Lock()
_NVML_READY = False
_NVML_ERROR = ''


def _ensure_nvml() -> None:
    global _NVML_READY, _NVML_ERROR
    if pynvml is None:
        raise RuntimeError('nvidia-ml-py (pynvml) is not installed')
    with _NVML_LOCK:
        if _NVML_READY:
            return
        try:
            pynvml.nvmlInit()
            _NVML_READY = True
            _NVML_ERROR = ''
        except Exception as exc:
            _NVML_ERROR = f'{type(exc).__name__}: {exc}'
            raise RuntimeError(_NVML_ERROR) from exc


def reset_nvml_state() -> None:
    """Test helper: forget the driver state after swapping the fake module."""
    global _NVML_READY, _NVML_ERROR
    with _NVML_LOCK:
        _NVML_READY = False
        _NVML_ERROR = ''


def _ts(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec='seconds')


def _read_proc_name(pid: int) -> str:
    try:
        with open(f'/proc/{pid}/comm', encoding='utf-8', errors='replace') as fp:
            return fp.read().strip()[:80]
    except OSError:
        return ''


def _device_processes(handle) -> list[dict]:
    processes = []
    seen = set()
    for getter in ('nvmlDeviceGetComputeRunningProcesses',
                   'nvmlDeviceGetGraphicsRunningProcesses'):
        fn = getattr(pynvml, getter, None)
        if fn is None:
            continue
        try:
            items = fn(handle)
        except Exception:
            continue
        for item in items:
            pid = int(getattr(item, 'pid', 0) or 0)
            if pid <= 0 or (pid, getter) in seen:
                continue
            seen.add((pid, getter))
            used = getattr(item, 'usedGpuMemory', None)
            processes.append({
                'pid': pid,
                'name': _read_proc_name(pid),
                'used_memory_mb': round(used / (1024 * 1024), 1) if used else None,
                'kind': 'compute' if 'Compute' in getter else 'graphics',
            })
    return processes


def snapshot() -> list[dict]:
    """Current NVML snapshot for every GPU."""
    _ensure_nvml()
    devices = []
    for index in range(int(pynvml.nvmlDeviceGetCount())):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode('utf-8', 'replace')
        uuid = ''
        try:
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode('utf-8', 'replace')
        except Exception:
            pass
        utilization = memory_used = memory_total = None
        temperature = power = power_limit = None
        try:
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = float(rates.gpu)
        except Exception:
            pass
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = round(info.used / (1024 * 1024), 1)
            memory_total = round(info.total / (1024 * 1024), 1)
        except Exception:
            pass
        try:
            temperature = float(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        try:
            power = round(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0, 1)
        except Exception:
            pass
        try:
            power_limit = round(pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0, 1)
        except Exception:
            pass
        devices.append({
            'index': index,
            'uuid': uuid,
            'name': name,
            'utilization': utilization,
            'memory_used_mb': memory_used,
            'memory_total_mb': memory_total,
            'temperature_c': temperature,
            'power_w': power,
            'power_limit_w': power_limit,
            'processes': _device_processes(handle),
        })
    return devices


def safe_snapshot() -> dict[str, Any]:
    try:
        devices = snapshot()
        return {'available': True, 'error': '', 'devices': devices,
                'sampled_at': now()}
    except Exception as exc:
        return {'available': False, 'error': f'{type(exc).__name__}: {exc}',
                'devices': [], 'sampled_at': now()}


def device_count() -> int | None:
    try:
        _ensure_nvml()
        return int(pynvml.nvmlDeviceGetCount())
    except Exception:
        return None


def gpu_history(seconds: int = 300) -> dict[int, list[dict]]:
    cutoff = time.time() - max(10, min(int(seconds), GPU_HISTORY_SECONDS))
    with connect() as db:
        rows = db.execute('''
            SELECT ts_epoch,gpu_index,utilization,memory_used_mb,memory_total_mb,
                   temperature_c,power_w
            FROM gpu_samples WHERE ts_epoch >= ? ORDER BY ts_epoch''', (cutoff,)).fetchall()
    history: dict[int, list[dict]] = {}
    for row in rows:
        history.setdefault(int(row['gpu_index']), []).append({
            't': round(float(row['ts_epoch']), 2),
            'utilization': row['utilization'],
            'memory_used_mb': row['memory_used_mb'],
            'memory_total_mb': row['memory_total_mb'],
            'temperature_c': row['temperature_c'],
            'power_w': row['power_w'],
        })
    return history


def overview(history_seconds: int = 300) -> dict[str, Any]:
    current = safe_snapshot()
    history = gpu_history(history_seconds)
    for device in current['devices']:
        device['history'] = history.get(device['index'], [])
    return current


def latest_metrics(deployment_id: int) -> dict | None:
    with connect() as db:
        row = db.execute('''SELECT * FROM vllm_samples WHERE deployment_id=?
                            ORDER BY ts_epoch DESC LIMIT 1''',
                         (deployment_id,)).fetchone()
    return dict(row) if row is not None else None


def deployment_metrics_history(deployment_id: int, seconds: int = 300) -> list[dict]:
    cutoff = time.time() - max(10, min(int(seconds), GPU_HISTORY_SECONDS))
    with connect() as db:
        rows = db.execute('''SELECT * FROM vllm_samples
                             WHERE deployment_id=? AND ts_epoch>=?
                             ORDER BY ts_epoch''', (deployment_id, cutoff)).fetchall()
    return [dict(row) for row in rows]


class GPUMonitor:
    def __init__(self):
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_health: dict[int, float] = {}
        self._previous_tokens: dict[int, tuple[float, float, float]] = {}

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if not GPU_MONITOR_ENABLED or self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name='gpu-monitor',
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.sample_once()
            except Exception:
                pass
            self._stop.wait(GPU_SAMPLE_INTERVAL)

    def sample_once(self, epoch: float | None = None) -> None:
        epoch = time.time() if epoch is None else epoch
        snap = safe_snapshot()
        if snap['available']:
            self._store_gpu(epoch, snap['devices'])
        self._sample_vllm(epoch)

    def _store_gpu(self, epoch: float, devices: list[dict]) -> None:
        with connect() as db:
            for device in devices:
                db.execute('''INSERT INTO gpu_samples
                    (ts,ts_epoch,gpu_index,gpu_uuid,name,utilization,memory_used_mb,
                     memory_total_mb,temperature_c,power_w,power_limit_w)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
                    (_ts(epoch), epoch, device['index'], device['uuid'],
                     device['name'], device['utilization'],
                     device['memory_used_mb'], device['memory_total_mb'],
                     device['temperature_c'], device['power_w'],
                     device['power_limit_w']))
            cutoff = epoch - GPU_HISTORY_SECONDS
            db.execute('DELETE FROM gpu_samples WHERE ts_epoch < ?', (cutoff,))

    def _sample_vllm(self, epoch: float) -> None:
        with connect() as db:
            deployments = [dict(row) for row in db.execute(
                'SELECT * FROM deployments WHERE base_url!="" '
                "AND status IN ('starting','running','unreachable','external')")]
        for deployment in deployments:
            metrics = fetch_vllm_metrics(deployment['base_url'])
            did = deployment['id']
            if metrics is None:
                self._maybe_health(epoch, deployment)
                continue
            running = int(metrics.get('num_requests_running') or 0)
            waiting = int(metrics.get('num_requests_waiting') or 0)
            prompt_total = float(metrics.get('prompt_tokens_total') or 0)
            generation_total = float(metrics.get('generation_tokens_total') or 0)
            previous = self._previous_tokens.get(did)
            input_rate = output_rate = 0.0
            if previous is not None:
                prev_epoch, prev_prompt, prev_generation = previous
                elapsed = max(0.001, epoch - prev_epoch)
                if prompt_total >= prev_prompt:
                    input_rate = (prompt_total - prev_prompt) / elapsed
                if generation_total >= prev_generation:
                    output_rate = (generation_total - prev_generation) / elapsed
            self._previous_tokens[did] = (epoch, prompt_total, generation_total)
            with connect() as db:
                db.execute('''INSERT INTO vllm_samples
                    (ts,ts_epoch,deployment_id,running,waiting,prompt_tokens_total,
                     generation_tokens_total,input_tokens_per_sec,output_tokens_per_sec)
                    VALUES(?,?,?,?,?,?,?,?,?)''',
                    (_ts(epoch), epoch, did, running, waiting, prompt_total,
                     generation_total, round(input_rate, 2), round(output_rate, 2)))
                status = 'running' if deployment['managed'] else 'external'
                db.execute("UPDATE deployments SET status=?, last_check_at=? "
                           "WHERE id=? AND status NOT IN ('stopped','failed')",
                           (status, _ts(epoch), did))
                cutoff = epoch - GPU_HISTORY_SECONDS
                db.execute('DELETE FROM vllm_samples WHERE ts_epoch < ?', (cutoff,))
            self._last_health[did] = epoch

    def _maybe_health(self, epoch: float, deployment: dict) -> None:
        last = self._last_health.get(deployment['id'], 0)
        if epoch - last < GPU_METRICS_INTERVAL:
            return
        self._last_health[deployment['id']] = epoch
        healthy = False
        try:
            import httpx
            base = deployment['base_url'].rstrip('/')
            with httpx.Client(timeout=2.0, follow_redirects=False) as client:
                response = client.get(base + '/models')
                healthy = response.status_code == 200
        except Exception:
            healthy = False
        status = 'running' if healthy else 'unreachable'
        if deployment['managed'] == 0:
            status = 'external' if healthy else 'unreachable'
        with connect() as db:
            db.execute("UPDATE deployments SET status=?, last_check_at=? "
                       "WHERE id=? AND status NOT IN ('stopped','failed')",
                       (status, _ts(epoch), deployment['id']))


_MONITOR = GPUMonitor()


def start_monitor() -> None:
    _MONITOR.start()


def stop_monitor() -> None:
    _MONITOR.stop()


def monitor() -> GPUMonitor:
    return _MONITOR
