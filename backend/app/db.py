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
          description TEXT NOT NULL DEFAULT '', source_hash TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL
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
          test_protocol TEXT NOT NULL DEFAULT '',
          test_wrapper TEXT NOT NULL DEFAULT '',
          real_iospec TEXT NOT NULL DEFAULT '',
          function_name TEXT NOT NULL DEFAULT '',
          function_signature TEXT NOT NULL DEFAULT '',
          metadata_json TEXT NOT NULL DEFAULT '',
          oracle_validated INTEGER NOT NULL DEFAULT 0,
          oracle_test_count INTEGER NOT NULL DEFAULT 0,
          oracle_passed_count INTEGER NOT NULL DEFAULT 0,
          oracle_failed_count INTEGER NOT NULL DEFAULT 0,
          oracle_error TEXT NOT NULL DEFAULT '',
          oracle_detail_json TEXT NOT NULL DEFAULT '',
          oracle_signature TEXT NOT NULL DEFAULT '',
          oracle_checked_at TEXT NOT NULL DEFAULT '',
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
          started_at TEXT, finished_at TEXT, error TEXT NOT NULL DEFAULT '',
          metric_version TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
          sample_id INTEGER NOT NULL REFERENCES samples(id),
          status TEXT NOT NULL, compile_success INTEGER,
          source_compile_ok INTEGER,
          compile_stage TEXT NOT NULL DEFAULT '',
          edit_similarity REAL,
          tests_passed INTEGER NOT NULL DEFAULT 0,
          tests_total INTEGER NOT NULL DEFAULT 0,
          model_output TEXT NOT NULL DEFAULT '',
          compile_log TEXT NOT NULL DEFAULT '', run_log TEXT NOT NULL DEFAULT '',
          latency_ms INTEGER NOT NULL DEFAULT 0, error TEXT NOT NULL DEFAULT '',
          inference_ms INTEGER NOT NULL DEFAULT 0,
          compile_ms INTEGER NOT NULL DEFAULT 0,
          test_ms INTEGER NOT NULL DEFAULT 0,
          sandbox_ms INTEGER NOT NULL DEFAULT 0,
          prompt_tokens INTEGER NOT NULL DEFAULT 0,
          completion_tokens INTEGER NOT NULL DEFAULT 0,
          prompt_text TEXT NOT NULL DEFAULT '',
          generation_params_json TEXT NOT NULL DEFAULT '',
          cases_declared INTEGER NOT NULL DEFAULT 0,
          cases_total INTEGER NOT NULL DEFAULT 0,
          cases_passed INTEGER NOT NULL DEFAULT 0,
          test_case_status TEXT NOT NULL DEFAULT '',
          metric_version TEXT NOT NULL DEFAULT '',
          submitted_at TEXT NOT NULL DEFAULT '',
          generated_at TEXT NOT NULL DEFAULT '',
          first_token_at TEXT NOT NULL DEFAULT '',
          compile_started_at TEXT NOT NULL DEFAULT '',
          compile_finished_at TEXT NOT NULL DEFAULT '',
          detail_json TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          UNIQUE(run_id, sample_id)
        );
        CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
        CREATE INDEX IF NOT EXISTS idx_samples_ds ON samples(dataset_id);
        CREATE INDEX IF NOT EXISTS idx_results_run_status ON results(run_id,status);
        CREATE TABLE IF NOT EXISTS deployments (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL,
          served_name TEXT NOT NULL,
          model_path TEXT NOT NULL,
          gpu_ids TEXT NOT NULL DEFAULT '0',
          port INTEGER NOT NULL,
          dtype TEXT NOT NULL DEFAULT 'auto',
          max_model_len INTEGER NOT NULL,
          max_num_seqs INTEGER NOT NULL,
          max_num_batched_tokens INTEGER NOT NULL,
          gpu_memory_utilization REAL NOT NULL,
          api_mode TEXT NOT NULL DEFAULT 'completion',
          managed INTEGER NOT NULL DEFAULT 1,
          base_url TEXT NOT NULL DEFAULT '',
          pid INTEGER,
          pgid INTEGER,
          status TEXT NOT NULL DEFAULT 'starting',
          error TEXT NOT NULL DEFAULT '',
          log_path TEXT NOT NULL DEFAULT '',
          model_id INTEGER,
          config_json TEXT NOT NULL DEFAULT '{}',
          created_at TEXT NOT NULL,
          started_at TEXT,
          stopped_at TEXT,
          last_check_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_deployments_status ON deployments(status);
        CREATE TABLE IF NOT EXISTS gpu_samples (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          ts_epoch REAL NOT NULL,
          gpu_index INTEGER NOT NULL,
          gpu_uuid TEXT NOT NULL DEFAULT '',
          name TEXT NOT NULL DEFAULT '',
          utilization REAL,
          memory_used_mb INTEGER,
          memory_total_mb INTEGER,
          temperature_c REAL,
          power_w REAL,
          power_limit_w REAL
        );
        CREATE INDEX IF NOT EXISTS idx_gpu_samples_ts ON gpu_samples(ts_epoch);
        CREATE TABLE IF NOT EXISTS vllm_samples (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          ts_epoch REAL NOT NULL,
          deployment_id INTEGER NOT NULL REFERENCES deployments(id) ON DELETE CASCADE,
          running INTEGER,
          waiting INTEGER,
          prompt_tokens_total REAL,
          generation_tokens_total REAL,
          input_tokens_per_sec REAL,
          output_tokens_per_sec REAL
        );
        CREATE INDEX IF NOT EXISTS idx_vllm_samples_deploy ON vllm_samples(deployment_id,ts_epoch);
        CREATE TABLE IF NOT EXISTS case_manifests (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          sample_id INTEGER NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
          harness_hash TEXT NOT NULL,
          parser_version TEXT NOT NULL,
          status TEXT NOT NULL,
          manifest_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          UNIQUE(sample_id, harness_hash, parser_version)
        );
        CREATE TABLE IF NOT EXISTS test_case_results (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
          sample_id INTEGER NOT NULL REFERENCES samples(id),
          case_id TEXT NOT NULL,
          case_index INTEGER NOT NULL,
          static_id INTEGER NOT NULL DEFAULT -1,
          case_source_span TEXT NOT NULL DEFAULT '',
          input_text TEXT NOT NULL DEFAULT '',
          input_values_json TEXT NOT NULL DEFAULT '',
          expected_text TEXT NOT NULL DEFAULT '',
          expected_value_json TEXT NOT NULL DEFAULT '',
          actual_text TEXT NOT NULL DEFAULT '',
          actual_value_json TEXT NOT NULL DEFAULT '',
          compare_expression TEXT NOT NULL DEFAULT '',
          comparison_rule TEXT NOT NULL DEFAULT '',
          passed INTEGER NOT NULL,
          status TEXT NOT NULL,
          exit_code INTEGER,
          signal INTEGER,
          stdout TEXT NOT NULL DEFAULT '',
          stderr TEXT NOT NULL DEFAULT '',
          stdout_truncated INTEGER NOT NULL DEFAULT 0,
          stderr_truncated INTEGER NOT NULL DEFAULT 0,
          duration_ms INTEGER NOT NULL DEFAULT 0,
          oracle_source TEXT NOT NULL DEFAULT '',
          diagnostic_status TEXT NOT NULL DEFAULT '',
          error_type TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          UNIQUE(run_id, sample_id, case_id)
        );
        CREATE INDEX IF NOT EXISTS idx_case_results_result ON test_case_results(run_id,sample_id);
        CREATE INDEX IF NOT EXISTS idx_case_results_status ON test_case_results(run_id,status);
        ''')
        # Preserve pre-existing SQLite installations when adding model API mode.
        model_cols={r[1] for r in db.execute('PRAGMA table_info(models)')}
        if 'api_mode' not in model_cols:
            db.execute("ALTER TABLE models ADD COLUMN api_mode TEXT NOT NULL DEFAULT 'chat'")
        _ensure_column(db, 'models', 'deployment_id', 'INTEGER')
        db.execute('CREATE INDEX IF NOT EXISTS idx_models_deployment ON models(deployment_id)')
        # Automatic, additive migration for existing evaluation databases.
        _ensure_column(db, 'runs', 'started_at', 'TEXT')
        for column, ddl in (
            ('inference_ms', 'INTEGER NOT NULL DEFAULT 0'),
            ('compile_ms', 'INTEGER NOT NULL DEFAULT 0'),
            ('test_ms', 'INTEGER NOT NULL DEFAULT 0'),
            ('sandbox_ms', 'INTEGER NOT NULL DEFAULT 0'),
            ('prompt_tokens', 'INTEGER NOT NULL DEFAULT 0'),
            ('completion_tokens', 'INTEGER NOT NULL DEFAULT 0'),
            ('prompt_text', "TEXT NOT NULL DEFAULT ''"),
            ('generation_params_json', "TEXT NOT NULL DEFAULT ''"),
            ('cases_declared', 'INTEGER NOT NULL DEFAULT 0'),
            ('cases_total', 'INTEGER NOT NULL DEFAULT 0'),
            ('cases_passed', 'INTEGER NOT NULL DEFAULT 0'),
            ('test_case_status', "TEXT NOT NULL DEFAULT ''"),
            ('metric_version', "TEXT NOT NULL DEFAULT ''"),
            ('submitted_at', "TEXT NOT NULL DEFAULT ''"),
            ('generated_at', "TEXT NOT NULL DEFAULT ''"),
            ('first_token_at', "TEXT NOT NULL DEFAULT ''"),
            ('compile_started_at', "TEXT NOT NULL DEFAULT ''"),
            ('compile_finished_at', "TEXT NOT NULL DEFAULT ''"),
            ('detail_json', "TEXT NOT NULL DEFAULT ''"),
        ):
            _ensure_column(db, 'results', column, ddl)
        for column, ddl in (
            ('test_protocol', "TEXT NOT NULL DEFAULT ''"),
            ('test_wrapper', "TEXT NOT NULL DEFAULT ''"),
            ('real_iospec', "TEXT NOT NULL DEFAULT ''"),
            ('function_name', "TEXT NOT NULL DEFAULT ''"),
            ('function_signature', "TEXT NOT NULL DEFAULT ''"),
            ('metadata_json', "TEXT NOT NULL DEFAULT ''"),
            ('oracle_validated', 'INTEGER NOT NULL DEFAULT 0'),
            ('oracle_test_count', 'INTEGER NOT NULL DEFAULT 0'),
            ('oracle_passed_count', 'INTEGER NOT NULL DEFAULT 0'),
            ('oracle_failed_count', 'INTEGER NOT NULL DEFAULT 0'),
            ('oracle_error', "TEXT NOT NULL DEFAULT ''"),
            ('oracle_detail_json', "TEXT NOT NULL DEFAULT ''"),
            ('oracle_signature', "TEXT NOT NULL DEFAULT ''"),
            ('oracle_checked_at', "TEXT NOT NULL DEFAULT ''"),
        ):
            _ensure_column(db, 'samples', column, ddl)
        _ensure_column(db, 'results', 'source_compile_ok', 'INTEGER')
        _ensure_column(db, 'results', 'compile_stage', "TEXT NOT NULL DEFAULT ''")
        _ensure_column(db, 'results', 'edit_similarity', 'REAL')
        db.execute('CREATE INDEX IF NOT EXISTS idx_results_run_status ON results(run_id,status)')
        _ensure_column(db, 'runs', 'metric_version', "TEXT NOT NULL DEFAULT ''")
        _ensure_column(db, 'datasets', 'source_hash', "TEXT NOT NULL DEFAULT ''")
        db.execute('CREATE INDEX IF NOT EXISTS idx_results_case_status ON results(run_id,test_case_status)')
        # A run that was interrupted by a server restart must never be shown as actively running.
        db.execute("UPDATE runs SET status='interrupted', error='Server restarted; select Retry to resume.' WHERE status IN ('running','queued')")


def _ensure_column(db, table: str, column: str, ddl: str):
    columns = {row[1] for row in db.execute(f'PRAGMA table_info({table})')}
    if column not in columns:
        db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {ddl}')


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
