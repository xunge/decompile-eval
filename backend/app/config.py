"""Settings for the single-machine MVP; do not publicly expose this deployment."""
from pathlib import Path
import os


def _env_float(name: str, default: float, minimum: float = 0.0,
               maximum: float = 1_000_000.0) -> float:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


def _env_int(name: str, default: int, minimum: int = 1, maximum: int = 4096) -> int:
    raw = os.getenv(name, '').strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv('DEC_EVAL_DATA_DIR', str(ROOT / 'storage'))).resolve()
DB_FILE = DATA_DIR / 'decompile_eval.sqlite3'
SANDBOX_IMAGE = os.getenv('DEC_EVAL_SANDBOX_IMAGE', 'gcc:14')
CORS_ORIGINS = os.getenv('DEC_EVAL_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',')
MAX_UPLOAD_BYTES = 64 * 1024 * 1024  # offline predictions only
# Dataset uploads are unlimited by default; set a byte cap only if the
# operator explicitly wants one.  0 means no cap.
MAX_DATASET_UPLOAD_BYTES = _env_int('DEC_EVAL_MAX_DATASET_UPLOAD_BYTES', 0, 0,
                                    10 * 1024 * 1024 * 1024)
MAX_SAMPLES = 100000
MAX_SOURCE_CHARS = 500_000

# Evaluation pipeline. All of them can be overridden through the environment;
# per-run values for the two concurrency limits are stored in the run config.
# 128 in-flight requests is the vLLM-friendly default: the scheduler refills a
# slot as soon as any request finishes, so the GPU never waits for the batch.
INFERENCE_CONCURRENCY = _env_int('INFERENCE_CONCURRENCY', 0, 0, 512)
COMPILE_CONCURRENCY = _env_int('COMPILE_CONCURRENCY', 64, 1, 128)
MAX_PENDING_COMPILE = _env_int('MAX_PENDING_COMPILE', 128, 1, 4096)
MAX_INFERENCE_CONCURRENCY = 512

# AUTO mode (INFERENCE_CONCURRENCY=0): bounded continuous supply with a
# dynamically growing in-flight ceiling.  The ceiling is a resource guard, not
# a user-specified batch size; every completed request is replaced immediately.
AUTO_INITIAL_INFLIGHT = _env_int('AUTO_INITIAL_INFLIGHT', 32, 1, 512)
AUTO_MAX_INFLIGHT = _env_int('AUTO_MAX_INFLIGHT', 256, 1, 1024)
AUTO_STEP = _env_int('AUTO_STEP', 16, 1, 256)
COMPILE_MEMORY_FRACTION = float(os.getenv('COMPILE_MEMORY_FRACTION', '0.6') or 0.6)
COMPILE_MEMORY_PER_WORKER_MB = _env_int('COMPILE_MEMORY_PER_WORKER_MB', 1024, 128, 8192)
MAX_COMPILE_CONCURRENCY = 128

# Context safety. The tokenizer is the real model tokenizer exposed by vLLM at
# POST /tokenize; max_tokens is reduced only when prompt+output would cross the
# model window. Prompt text and generation parameters are stored per result.
MAX_MODEL_LEN = _env_int('MAX_MODEL_LEN', 4096, 128, 1_000_000)
TOKENIZER_TIMEOUT = _env_int('TOKENIZER_TIMEOUT', 10, 1, 300)
TOKENIZER_CACHE_SIZE = _env_int('TOKENIZER_CACHE_SIZE', 8192, 16, 1_000_000)

# Optional offline batch generation through the vLLM Python API.
OFFLINE_BATCH_MODEL_PATH = os.getenv('DEC_EVAL_BATCH_MODEL_PATH', '')
OFFLINE_BATCH_SIZE = _env_int('DEC_EVAL_BATCH_SIZE', 32, 1, 4096)

# Optional vLLM Prometheus metrics (/metrics) attached to the progress API.
VLLM_METRICS_ENABLED = os.getenv('DEC_EVAL_VLLM_METRICS', '1').strip().lower() not in ('0', 'false', 'no')
VLLM_METRICS_TIMEOUT = 0.8

# Per-case diagnostics extracted from the official harness.  The official
# harness remains the only RE oracle; diagnostics only add case-level TCP.
DIAGNOSTICS_ENABLED = os.getenv('DEC_EVAL_DIAGNOSTICS', '1').strip().lower() not in ('0', 'false', 'no')
MAX_DIAGNOSTIC_CASES = _env_int('MAX_DIAGNOSTIC_CASES', 200, 1, 2000)

INFERENCE_TIMEOUT = _env_int('INFERENCE_TIMEOUT', 180, 5, 3600)
INFERENCE_RETRIES = _env_int('INFERENCE_RETRIES', 2, 0, 10)
COMPILE_TIMEOUT = _env_int('COMPILE_TIMEOUT', 60, 5, 900)
TEST_TIMEOUT = _env_int('TEST_TIMEOUT', 10, 1, 300)

# Per-stream log cap that is sent back from the sandbox container.
LOG_LIMIT_BYTES = _env_int('DEC_EVAL_LOG_LIMIT_BYTES', 32 * 1024, 1024, 1024 * 1024)

# ---------------------------------------------------------------------------
# Local vLLM deployment management and GPU monitoring.
# ---------------------------------------------------------------------------
VLLM_BIN = os.getenv('DEC_EVAL_VLLM_BIN', '/home/jiang/venvs/vllm/bin/vllm')
ALLOW_DEPLOY = os.getenv('DEC_EVAL_ALLOW_DEPLOY', '1').strip().lower() not in ('0', 'false', 'no')
ADMIN_TOKEN = os.getenv('DEC_EVAL_ADMIN_TOKEN', '').strip()
DEPLOY_START_TIMEOUT = _env_int('DEC_EVAL_DEPLOY_START_TIMEOUT', 240, 10, 3600)
DEPLOY_STOP_TIMEOUT = _env_int('DEC_EVAL_DEPLOY_STOP_TIMEOUT', 30, 1, 300)
# Ports the evaluation stack itself uses; never let a deployment take them.
DEPLOY_RESERVED_PORTS = sorted({
    int(item) for item in os.getenv(
        'DEC_EVAL_RESERVED_PORTS', '22,80,443,3080,5173,8001,8010'
    ).split(',') if item.strip().isdigit()
})
DEPLOY_MAX_RUNNING = _env_int('DEC_EVAL_DEPLOY_MAX_RUNNING', 4, 1, 32)
DEPLOY_LOG_TAIL_BYTES = _env_int('DEC_EVAL_DEPLOY_LOG_TAIL', 32768, 1024, 1_048_576)

GPU_MONITOR_ENABLED = os.getenv('DEC_EVAL_GPU_MONITOR', '1').strip().lower() not in ('0', 'false', 'no')
GPU_SAMPLE_INTERVAL = _env_float('DEC_EVAL_GPU_INTERVAL', 1.5, 0.5, 10.0)
GPU_HISTORY_SECONDS = _env_int('DEC_EVAL_GPU_HISTORY', 1800, 60, 86400)
GPU_METRICS_INTERVAL = _env_float('DEC_EVAL_VLLM_METRICS_INTERVAL', 5.0, 1.0, 60.0)
