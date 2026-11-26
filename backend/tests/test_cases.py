"""TCP micro/macro, manifest merging and test-case API tests."""
import json
import time

from fastapi.testclient import TestClient

from app import db, worker
from app.main import app, summarize
from app.sandbox import EvalOutcome


def row(**overrides):
    base = {
        'status': 'TEST_PASS', 'compile_success': 1, 'tests_passed': 1,
        'tests_total': 1, 'cases_declared': 0, 'cases_total': 0,
        'cases_passed': 0, 'test_case_status': 'OK',
    }
    base.update(overrides)
    return base


def test_five_cases_three_pass_micro_and_re():
    metrics = summarize([row(status='WRONG_ANSWER', tests_passed=3, tests_total=5,
                             cases_declared=5, cases_total=5, cases_passed=3)])
    assert metrics['rc'] == 100.0
    assert metrics['re'] == 0.0
    assert metrics['tcp_micro'] == 60.0
    assert metrics['tcp_macro'] == 60.0
    assert metrics['tcp'] == 60.0


def test_micro_and_macro_differ_between_functions():
    metrics = summarize([
        row(cases_declared=2, cases_total=2, cases_passed=2),
        row(status='WRONG_ANSWER', cases_declared=5, cases_total=5, cases_passed=3),
    ])
    # micro = (2+3)/(2+5), macro = mean(1.0, 0.6)
    assert metrics['tcp_micro'] == 71.43
    assert metrics['tcp_macro'] == 80.0
    assert metrics['cases_total'] == 7 and metrics['cases_passed'] == 5
    assert metrics['case_coverage'] == 100.0


def test_all_pass_and_one_failure_do_not_inflate_re():
    all_pass = summarize([row(cases_declared=5, cases_total=5, cases_passed=5)])
    one_fail = summarize([
        row(cases_declared=5, cases_total=5, cases_passed=5),
        row(status='WRONG_ANSWER', cases_declared=5, cases_total=5, cases_passed=4),
    ])
    assert all_pass['re'] == 100.0 and all_pass['tcp_micro'] == 100.0
    assert one_fail['re'] == 50.0
    assert one_fail['tcp_micro'] == 90.0


def test_legacy_rows_keep_old_tcp_but_have_no_case_variants():
    metrics = summarize([{'status': 'TEST_PASS', 'compile_success': 1,
                          'tests_passed': 2, 'tests_total': 4}])
    assert metrics['tcp'] == 50.0
    assert metrics['tcp_micro'] is None and metrics['tcp_macro'] is None
    assert metrics['metric_version'] == 'legacy'


def _manifest(cases):
    return {'status': 'OK', 'cases': cases}


def _case(sid, line, expr='f(x)==1', lhs='f(x)', rhs='1', op='=='):
    return {'static_id': sid, 'line': line, 'column': 5, 'end_line': line,
            'end_column': 20, 'span': [0, 10], 'expression': expr,
            'comparison_operator': op, 'lhs': lhs, 'rhs': rhs,
            'comparison_rule': 'operator', 'dynamic': False,
            'conditional': False, 'semantics_risk': False, 'function': 'main'}


def _event(index, sid, passed, actual='1', expected='1', after=False):
    return {'type': 'case', 'index': index, 'static_id': sid, 'line': 6,
            'expression': 'f(x)==1', 'lhs': 'f(x)', 'rhs': '1',
            'actual': actual, 'expected': expected, 'actual_ok': True,
            'expected_ok': True, 'passed': passed, 'after_prior_failure': after,
            'exit_code': 0 if passed else 3, 'signal': None, 'timed_out': False,
            'duration_ms': 1, 'stdout': '', 'stderr': '',
            'stdout_truncated': False, 'stderr_truncated': False,
            'stdout_bytes': 0, 'stderr_bytes': 0}


def test_failed_case_does_not_hide_later_pass():
    outcome = EvalOutcome('WRONG_ANSWER', True, 0, 1)
    outcome.cases = [_event(1, 0, False, actual='2', expected='1'),
                     _event(2, 1, True, after=True)]
    info = {'status': 'OK', 'manifest': _manifest([
        _case(0, 6), _case(1, 7)])}
    rows, declared, total, passed, status = worker._program_cases(outcome, info, False)
    assert declared == 2 and total == 2 and passed == 1
    assert [r['status'] for r in rows] == ['FAIL', 'PASS']
    assert rows[0]['actual_text'] == '2' and rows[0]['expected_text'] == '1'
    assert rows[1]['diagnostic_status'].startswith('AFTER_PRIOR_FAILURE')


def test_dynamic_loop_gets_instance_case_ids():
    outcome = EvalOutcome('WRONG_ANSWER', True, 0, 1)
    outcome.cases = [_event(1, 0, False, actual='0', expected='1'),
                     _event(2, 0, False, actual='1', expected='2', after=True),
                     _event(3, 1, True, actual='9', expected='9', after=True)]
    info = {'status': 'OK', 'manifest': _manifest([_case(0, 6), _case(1, 7)])}
    rows, declared, total, passed, _ = worker._program_cases(outcome, info, False)
    assert declared == 2 and total == 3 and passed == 1
    assert [r['case_id'] for r in rows] == ['a0#0', 'a0#1', 'a1']
    assert rows[0]['case_index'] == 1


def test_compile_failure_marks_known_cases_not_run():
    outcome = EvalOutcome('COMPILE_ERROR', False, 0, 1)
    info = {'status': 'OK', 'manifest': _manifest([_case(0, 6), _case(1, 7)])}
    rows, declared, total, passed, status = worker._program_cases(outcome, info, True)
    assert declared == 2 and total == 0 and passed == 0
    assert status == 'NOT_RUN_COMPILE_ERROR'
    assert all(r['status'] == 'NOT_RUN_COMPILE_ERROR' for r in rows)


def test_unsupported_manifest_does_not_fabricate_cases():
    outcome = EvalOutcome('TEST_PASS', True, 1, 1)
    info = {'status': 'NO_ASSERT_CASES', 'manifest': {'status': 'NO_ASSERT_CASES', 'cases': []}}
    rows, declared, total, passed, status = worker._program_cases(outcome, info, False)
    assert rows == [] and declared == 0 and total == 0
    assert status == 'NO_ASSERT_CASES'


def test_discrete_tests_become_cases():
    tests = [{'name': 'a', 'code': 'if (f(1)!=1) return 1;'},
             {'name': 'b', 'code': 'if (f(2)!=2) return 1;'}]
    outcome = EvalOutcome('WRONG_ANSWER', True, 1, 2)
    outcome.tests = [
        {'index': 0, 'status': 'PASS', 'exit_code': 0, 'signal': None,
         'timed_out': False, 'duration_ms': 1, 'stdout': '', 'stderr': '',
         'stdout_truncated': False, 'stderr_truncated': False},
        {'index': 1, 'status': 'WRONG_ANSWER', 'exit_code': 1, 'signal': None,
         'timed_out': False, 'duration_ms': 1, 'stdout': 'out', 'stderr': '',
         'stdout_truncated': False, 'stderr_truncated': False},
    ]
    rows, declared, total, passed, status = worker._discrete_cases(outcome, tests, False)
    assert declared == 2 and total == 2 and passed == 1
    assert [r['status'] for r in rows] == ['PASS', 'FAIL']
    assert rows[1]['stdout'] == 'out'


def test_expected_and_actual_value_json_are_typed():
    outcome = EvalOutcome('TEST_PASS', True, 1, 1)
    outcome.cases = [_event(1, 0, True, actual='42', expected='42')]
    info = {'status': 'OK', 'manifest': _manifest([_case(0, 6)])}
    rows, *_ = worker._program_cases(outcome, info, False)
    assert json.loads(rows[0]['actual_value_json']) == 42
    assert json.loads(rows[0]['expected_value_json']) == 42


def _prepare_case_db():
    db.initialize()
    db.seed_demo()
    with db.connect() as conn:
        dataset = conn.execute("SELECT id FROM datasets WHERE name LIKE '演示%'").fetchone()[0]
        sample = conn.execute('SELECT * FROM samples WHERE dataset_id=? LIMIT 1',
                              (dataset,)).fetchone()
        model = conn.execute("SELECT id FROM models WHERE kind='fixture' LIMIT 1").fetchone()[0]
        cfg = {'dataset_id': dataset, 'model_id': model, 'optimizations': ['O0'],
               'sample_limit': 1, 'prompt_template': 'x{assembly}', 'temperature': 0,
               'max_tokens': 64, 'inference_concurrency': 1, 'compile_concurrency': 1,
               'generation_mode': 'online', 'batch_size': 0, 'max_model_len': 4096}
        rid = conn.execute('INSERT INTO runs(dataset_id,model_id,status,total,config_json,created_at) '
                           'VALUES(?,?,?,?,?,?)',
                           (dataset, model, 'completed', 1, json.dumps(cfg), db.now())).lastrowid
        result_id = conn.execute('''INSERT INTO results
            (run_id,sample_id,status,compile_success,tests_passed,tests_total,model_output,
             cases_declared,cases_total,cases_passed,test_case_status,metric_version,created_at)
            VALUES(?,?,'TEST_PASS',1,1,1,'int f(void){return 0;}',5,5,3,'OK',
                   'tcp_micro_macro_v1',?)''', (rid, sample['id'], db.now())).lastrowid
        rows = []
        for index in range(5):
            rows.append(worker._case_row(
                case_id=f'a{index}', case_index=index, static_id=index,
                input_text=f'f({index})', expected_text='1',
                actual_text='1' if index < 3 else '0',
                compare_expression=f'f({index})==1', comparison_rule='==',
                passed=1 if index < 3 else 0,
                status='PASS' if index < 3 else 'FAIL',
                exit_code=0 if index < 3 else 3,
                oracle_source='assertion', diagnostic_status='OK',
                error_type='NONE' if index < 3 else 'ASSERTION'))
    worker._db_write(lambda conn: worker._replace_cases(conn, rid, sample['id'], rows))
    return rid, result_id


def test_test_cases_api_pagination_filters_and_exports():
    rid, result_id = _prepare_case_db()
    with TestClient(app) as client:
        page = client.get(f'/api/runs/{rid}/results/{result_id}/test-cases?page=2&page_size=2')
        assert page.status_code == 200
        body = page.json()
        assert body['total'] == 5 and len(body['items']) == 2
        assert body['summary']['cases_passed'] == 3
        assert body['summary']['diagnostic_consistent'] is False
        failed = client.get(f'/api/runs/{rid}/results/{result_id}/test-cases?status=fail').json()
        assert failed['total'] == 2
        assert all(item['status'] == 'FAIL' for item in failed['items'])
        csv = client.get(f'/api/runs/{rid}/results/{result_id}/test-cases?export=csv')
        assert csv.status_code == 200 and csv.text.count('\n') == 6
        exported = client.get(f'/api/runs/{rid}/results/{result_id}/test-cases?export=json').json()
        assert len(exported['items']) == 5
        assert exported['summary']['official_status'] == 'TEST_PASS'


def test_official_re_is_not_overridden_by_case_results():
    metrics = summarize([row(status='TEST_PASS', cases_declared=5,
                             cases_total=5, cases_passed=3)])
    assert metrics['re'] == 100.0
    assert metrics['tcp_micro'] == 60.0
    assert metrics['cases_passed'] == 3


def test_uncovered_cases_and_samples_are_reported():
    metrics = summarize([
        row(cases_declared=10, cases_total=7, cases_passed=7),
        row(status='COMPILE_ERROR', compile_success=0, cases_declared=5,
            cases_total=0, cases_passed=0, test_case_status='NOT_RUN_COMPILE_ERROR'),
    ])
    assert metrics['cases_declared'] == 15
    assert metrics['cases_total'] == 7
    assert metrics['uncovered_cases'] == 8
    assert metrics['official_re_denominator'] == 2
    assert metrics['uncovered_samples'] == 1
    assert metrics['case_coverage'] == 50.0


def test_rc_uses_official_source_compile_and_keeps_harness_rate():
    metrics = summarize([
        {'status': 'COMPILE_ERROR', 'compile_success': 0, 'source_compile_ok': 1,
         'compile_stage': 'harness', 'tests_passed': 0, 'tests_total': 1},
        {'status': 'TEST_PASS', 'compile_success': 1, 'source_compile_ok': 1,
         'compile_stage': 'ok', 'tests_passed': 1, 'tests_total': 1},
        {'status': 'COMPILE_ERROR', 'compile_success': 0, 'source_compile_ok': 0,
         'compile_stage': 'source', 'tests_passed': 0, 'tests_total': 1},
    ])
    assert metrics['rc'] == 66.67
    assert metrics['rc_source'] == 66.67
    assert metrics['rc_harness'] == 33.33
    assert metrics['source_compile_pass'] == 2
    assert metrics['rc_mode'] == 'official_source_compile'


def test_legacy_rc_mode_is_reported():
    metrics = summarize([{'status': 'TEST_PASS', 'compile_success': 1,
                          'tests_passed': 1, 'tests_total': 1}])
    assert metrics['rc'] == 100.0
    assert metrics['rc_mode'] == 'legacy_harness_compile'
    assert metrics['rc_source'] is None


def test_oracle_and_invalid_benchmark_are_excluded_from_model_metrics():
    metrics = summarize([
        {'status': 'TEST_PASS', 'compile_success': 1, 'source_compile_ok': 1,
         'tests_passed': 2, 'tests_total': 2, 'cases_declared': 2,
         'cases_total': 2, 'cases_passed': 2},
        {'status': 'ORACLE_ERROR', 'compile_success': 0, 'source_compile_ok': 0,
         'tests_passed': 0, 'tests_total': 3},
        {'status': 'INVALID_BENCHMARK', 'compile_success': None,
         'source_compile_ok': None, 'tests_passed': 0, 'tests_total': 1},
    ])
    assert metrics['evaluated'] == 1
    assert metrics['excluded'] == 2
    assert metrics['rc'] == 100.0 and metrics['re'] == 100.0
    assert metrics['tcp_micro'] == 100.0
