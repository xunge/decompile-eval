"""Small, inspectable SQLite database. One local process owns run workers."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sqlite3
from .config import DATA_DIR, DB_FILE


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


@contextmanager
def connect():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA busy_timeout = 30000')
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize():
    with connect() as db:
        db.executescript('''
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS datasets (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL, version TEXT NOT NULL DEFAULT 'v1',
          description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS samples (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
          sample_key TEXT NOT NULL, problem_id TEXT NOT NULL,
          optimization TEXT NOT NULL, compiler TEXT NOT NULL,
          architecture TEXT NOT NULL, language TEXT NOT NULL,
          assembly TEXT NOT NULL, reference_source TEXT NOT NULL,
          dependencies TEXT NOT NULL DEFAULT '', tests_json TEXT NOT NULL,
          demo_prediction TEXT NOT NULL DEFAULT '',
          UNIQUE(dataset_id, sample_key)
        );
        CREATE TABLE IF NOT EXISTS models (
          id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
          kind TEXT NOT NULL, base_url TEXT NOT NULL DEFAULT '',
          api_model TEXT NOT NULL DEFAULT '', api_key_env TEXT NOT NULL DEFAULT '',
          system_prompt TEXT NOT NULL DEFAULT '', api_mode TEXT NOT NULL DEFAULT 'chat', created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS runs (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          dataset_id INTEGER NOT NULL REFERENCES datasets(id),
          model_id INTEGER NOT NULL REFERENCES models(id),
          status TEXT NOT NULL, total INTEGER NOT NULL, completed INTEGER NOT NULL DEFAULT 0,
          config_json TEXT NOT NULL, created_at TEXT NOT NULL,
          finished_at TEXT, error TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
          sample_id INTEGER NOT NULL REFERENCES samples(id),
          status TEXT NOT NULL, compile_success INTEGER,
          tests_passed INTEGER NOT NULL DEFAULT 0,
          tests_total INTEGER NOT NULL DEFAULT 0,
          model_output TEXT NOT NULL DEFAULT '',
          compile_log TEXT NOT NULL DEFAULT '', run_log TEXT NOT NULL DEFAULT '',
          latency_ms INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          UNIQUE(run_id, sample_id)
        );
        CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
        CREATE INDEX IF NOT EXISTS idx_samples_ds ON samples(dataset_id);
        ''')
        # Preserve pre-existing SQLite installations when adding model API mode.
        model_cols={r[1] for r in db.execute('PRAGMA table_info(models)')}
        if 'api_mode' not in model_cols:
            db.execute("ALTER TABLE models ADD COLUMN api_mode TEXT NOT NULL DEFAULT 'chat'")
        # A run that was interrupted by a server restart must never be shown as actively running.
        db.execute("UPDATE runs SET status='interrupted', error='Server restarted; select Retry to resume.' WHERE status IN ('running','queued')")


def seed_demo():
    from .seed import EXAMPLES
    with connect() as db:
        d = db.execute("SELECT id FROM datasets WHERE name=? AND version=?", ('演示数据集（4个函数）', 'v1')).fetchone()
        if not d:
            cur = db.execute('INSERT INTO datasets(name,version,description,created_at) VALUES(?,?,?,?)',
                             ('演示数据集（4个函数）','v1','内置最小演示数据，全部代码和测试均为示例。',now()))
            dataset_id = cur.lastrowid
            for ex in EXAMPLES:
                db.execute('''INSERT INTO samples(dataset_id,sample_key,problem_id,optimization,compiler,architecture,language,
                            assembly,reference_source,dependencies,tests_json,demo_prediction) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)''',
                           (dataset_id,ex['sample_id'],ex['problem_id'],ex['optimization'],ex['compiler'],ex['architecture'],
                            ex['language'],ex['assembly'],ex['reference_source'],ex.get('dependencies',''),
                            json.dumps(ex['tests'],ensure_ascii=False),ex['demo_prediction']))
        m=db.execute("SELECT id FROM models WHERE name=?",('演示固定输出（非真实LLM）',)).fetchone()
        if not m:
            db.execute('INSERT INTO models(name,kind,system_prompt,created_at) VALUES(?,?,?,?)',
                       ('演示固定输出（非真实LLM）','fixture','',now()))
