"""Official Edit Similarity tests."""
import pytest

from app import db, edit_sim
from app.edit_sim import backfill_edit_similarity, edit_similarity
from app.main import summarize


@pytest.fixture(autouse=True)
def ensure_db():
    db.initialize()
    yield


def test_official_es_normalizes_lines_and_blank_lines():
    reference = 'int f(void) {\n  return 1;\n}\n'
    prediction = '\nint f(void) {\nreturn 1;\n}\n\n'
    assert edit_similarity(reference, prediction) == 1.0


def test_official_es_known_ratio():
    # One substitution over a length-3 string -> 2/3.
    value = edit_similarity('abc', 'abd')
    assert round(value, 6) == 0.666667
    assert edit_similarity('abc', '') == 0.0
    assert edit_similarity('', '') == 1.0


def test_pure_python_fallback_matches(monkeypatch):
    monkeypatch.setattr(edit_sim, '_editdistance', None)
    assert round(edit_similarity('abc', 'abd'), 6) == 0.666667
    assert edit_similarity('int f(){return 1;}', 'int f(){return 1;}') == 1.0


def test_summarize_edit_similarity_mean():
    metrics = summarize([
        {'status': 'TEST_PASS', 'compile_success': 1, 'tests_passed': 1,
         'tests_total': 1, 'edit_similarity': 1.0},
        {'status': 'COMPILE_ERROR', 'compile_success': 0, 'tests_passed': 0,
         'tests_total': 1, 'edit_similarity': 0.5},
        {'status': 'INPUT_TOO_LONG', 'compile_success': 0, 'tests_passed': 0,
         'tests_total': 1, 'edit_similarity': None},
    ])
    # INPUT_TOO_LONG is excluded from the evaluated denominator and has no ES.
    assert metrics['edit_similarity'] == 75.0
    assert metrics['edit_similarity_samples'] == 2


def _make_result(prediction, reference, status='COMPILE_ERROR'):
    with db.connect() as conn:
        sample = conn.execute('SELECT * FROM samples ORDER BY id LIMIT 1').fetchone()
        model = conn.execute('SELECT id FROM models ORDER BY id LIMIT 1').fetchone()
        dataset = conn.execute('SELECT id FROM datasets ORDER BY id LIMIT 1').fetchone()
        if sample is None or model is None or dataset is None:
            db.seed_demo()
            sample = conn.execute('SELECT * FROM samples ORDER BY id LIMIT 1').fetchone()
            model = conn.execute('SELECT id FROM models ORDER BY id LIMIT 1').fetchone()
            dataset = conn.execute('SELECT id FROM datasets ORDER BY id LIMIT 1').fetchone()
        run_id = conn.execute('''INSERT INTO runs
            (dataset_id,model_id,status,total,config_json,created_at)
            VALUES(?,?,?,?,?,?)''',
            (dataset['id'], model['id'], 'completed', 1, '{}', db.now())).lastrowid
        result_id = conn.execute('''INSERT INTO results
            (run_id,sample_id,status,model_output,created_at)
            VALUES(?,?,?,?,?)''',
            (run_id, sample['id'], status, prediction, db.now())).lastrowid
        conn.execute('UPDATE samples SET reference_source=? WHERE id=?',
                     (reference, sample['id']))
        return result_id, sample['id']


def test_backfill_computes_missing_es_without_changing_status():
    result_id, _ = _make_result('int f(void){return 2;}', 'int f(void){return 1;}')
    with db.connect() as conn:
        assert conn.execute('SELECT edit_similarity FROM results WHERE id=?',
                            (result_id,)).fetchone()[0] is None
    updated = backfill_edit_similarity()
    assert updated >= 1
    with db.connect() as conn:
        row = conn.execute('SELECT edit_similarity,status FROM results WHERE id=?',
                           (result_id,)).fetchone()
    assert row['status'] == 'COMPILE_ERROR'       # derived metric only
    assert 0.0 <= row['edit_similarity'] < 1.0
