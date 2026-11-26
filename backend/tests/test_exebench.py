"""ExeBench adapter, import and official diff_io tests."""
import json
import os

import pytest
from fastapi.testclient import TestClient

from app import db
from app.exebench import (diff_io, diff_detail, normalize_tests, rewrite_wrapper,
                          sample_signature, _classify, _record)
from app.main import app


def _tests():
    return [
        {'name': 'test_0', 'input': {'x': 1}, 'expected_output': {'value': 2},
         'dummy_funcs': '', 'dummy_funcs_seed': None, 'case_index': 0},
        {'name': 'test_1', 'input': {'x': [1, 2]}, 'expected_output': {'value': 3.0},
         'dummy_funcs': '', 'dummy_funcs_seed': None, 'case_index': 1},
    ]


def test_diff_io_official_nested_and_float():
    assert diff_io({'a': [1, {'b': 2.0}]}, {'a': [1, {'b': 2.0}]})
    assert diff_io({'a': 2.0}, {'a': 2.0 + 1e-12})
    assert not diff_io({'a': 1}, {'a': True})       # official type strictness
    assert not diff_io({'a': [1, 2]}, {'a': [1]})
    assert not diff_io({'a': 1, 'extra': 2}, {'a': 1})
    assert 'length' in diff_detail({'a': [1]}, {'a': [1, 2]}) or 'length' in diff_detail([1], [1, 2])


def test_normalize_legacy_and_unified_tests():
    legacy = {'var': ['x'], 'value': [1]}
    out = normalize_tests([{'input': legacy, 'output': {'var': ['y'], 'value': [2]}}])
    assert out[0]['input'] == {'x': 1}
    assert out[0]['expected_output'] == {'y': 2}


def test_wrapper_rewrite_candidate_include():
    wrapper = 'extern "C" {\n#include "/tmp/pytmpfile_abc.c"\n}\nint main(){return 0;}'
    assert '#include "candidate.c"' in rewrite_wrapper(wrapper)
    assert '/tmp/pytmpfile' not in rewrite_wrapper(wrapper)
    no_include = 'int main(){return 0;}'
    assert 'candidate.c' in rewrite_wrapper(no_include)


def test_signature_changes_with_tests_and_dummy():
    sample = {'reference_source': 'int f(void){return 1;}', 'dependencies': '',
              'test_wrapper': 'w', 'tests_json': '[]', 'real_iospec': '{}',
              'language': 'c', 'compiler': 'gcc', 'optimization': 'O0'}
    first = sample_signature(sample)
    sample['tests_json'] = '[{"input":1}]'
    assert sample_signature(sample) != first


def _compile(group, rc=0, stderr=''):
    return {'type': 'compile', 'group': group, 'returncode': rc,
            'stdout': '', 'stderr': stderr, 'duration_ms': 5, 'timed_out': False}


def test_classify_pass_wrong_and_invalid_output():
    tests = _tests()
    events = [_compile(0),
              {'type': 'test', 'group': 0, 'index': 0, 'actual': {'value': 2},
               'returncode': 0, 'stdout': '', 'stderr': '', 'duration_ms': 2,
               'timed_out': False, 'output_error': ''},
              {'type': 'test', 'group': 0, 'index': 1, 'actual': {'value': 4.0},
               'returncode': 0, 'stdout': '', 'stderr': '', 'duration_ms': 2,
               'timed_out': False, 'output_error': ''}]
    outcome = _classify(events, tests, 'int f(void){return 1;}')
    assert outcome.status == 'WRONG_ANSWER'
    assert outcome.tests_passed == 1
    assert outcome.tests_total == 2
    assert outcome.tests[0]['status'] == 'PASS'
    assert outcome.tests[1]['status'] == 'WRONG_ANSWER'


def test_classify_compile_link_and_timeout():
    tests = _tests()
    link = _classify([_compile(0, 1, 'undefined reference to helper')],
                     tests, 'int f(void){return 1;}')
    assert link.status == 'LINK_ERROR' and link.compile_success is False
    timeout = _classify([_compile(0),
                         {'type': 'test', 'group': 0, 'index': 0, 'actual': None,
                          'returncode': None, 'timed_out': True, 'duration_ms': 999,
                          'stdout': '', 'stderr': '', 'output_error': ''},
                         {'type': 'test', 'group': 0, 'index': 1, 'actual': None,
                          'returncode': 0, 'timed_out': False, 'duration_ms': 1,
                          'stdout': '', 'stderr': '', 'output_error': 'bad json'}],
                        tests, 'int f(void){return 1;}')
    assert timeout.status == 'TIMEOUT'


def test_import_exebench_streaming_and_delete():
    raw = json.dumps({
        'sample_id': 'exebench_test_import_1',
        'problem_id': 'p1', 'assembly': 'func0:\n ret', 'reference_source': 'int f(void){return 1;}',
        'dependencies': '', 'test_protocol': 'exebench_io',
        'test_wrapper': 'extern "C" {\n#include "/tmp/x.c"\n}\nint main(){return 0;}',
        'real_iospec': {'typemap': {}},
        'function_name': 'f', 'function_signature': 'int f(void)',
        'tests': [{'name': 'test_0', 'input': {}, 'expected_output': {'x': 1},
                   'dummy_funcs': '', 'dummy_funcs_seed': None}],
    }).encode()
    with TestClient(app) as client:
        response = client.post('/api/datasets/import',
            data={'name': 'ExeBench import', 'version': 'v1'},
            files={'file': ('eb.jsonl', raw, 'application/jsonl')})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body['sample_count'] == 1
        assert body['by_compiler'] == {'gcc': 1}
        did = body['dataset_id']
        samples = client.get(f'/api/datasets/{did}/samples').json()
        assert samples[0]['sample_key'] == 'exebench_test_import_1'
        assert client.delete(f'/api/datasets/{did}').status_code == 200


def test_import_invalid_exebench_reports_line():
    bad = json.dumps({'sample_id': 'bad', 'assembly': 'x', 'reference_source': 'int f(){}',
                      'test_protocol': 'exebench_io', 'tests': [{'input': {}}]}).encode()
    with TestClient(app) as client:
        response = client.post('/api/datasets/import',
            data={'name': 'Bad', 'version': 'v1'},
            files={'file': ('bad.jsonl', bad, 'application/jsonl')})
        assert response.status_code == 422
        detail = response.json()['detail']
        assert detail['line'] == 1
        assert 'expected_output' in json.dumps(detail)


def test_import_legacy_exebench_testcases():
    raw = json.dumps({
        'sample_id': 'legacy_eb', 'assembly': 'x', 'reference_source': 'int f(){}',
        'testcases': [{'input': {'x': 1}, 'output': {'x': 1}}],
        'real_exe_wrapper': 'int main(){return 0;}',
    }).encode()
    with TestClient(app) as client:
        response = client.post('/api/datasets/import',
            data={'name': 'Legacy EB', 'version': 'v1'},
            files={'file': ('legacy.jsonl', raw, 'application/jsonl')})
        assert response.status_code == 201, response.text
        did = response.json()['dataset_id']
        assert client.delete(f'/api/datasets/{did}').status_code == 200
