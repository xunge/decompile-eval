"""Settings for single-machine MVP; do not publicly expose this deployment."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv('DEC_EVAL_DATA_DIR', str(ROOT / 'storage'))).resolve()
DB_FILE = DATA_DIR / 'decompile_eval.sqlite3'
SANDBOX_IMAGE = os.getenv('DEC_EVAL_SANDBOX_IMAGE', 'gcc:14')
CORS_ORIGINS = os.getenv('DEC_EVAL_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',')
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_SAMPLES = 12000
MAX_SOURCE_CHARS = 500_000
