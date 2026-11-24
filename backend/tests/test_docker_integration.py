"""Opt-in real Docker E2E check, run only on a trusted developer machine."""
import os
import pytest
from app.sandbox import evaluate_code


@pytest.mark.skipif(os.getenv('RUN_DOCKER_TESTS') != '1', reason='Set RUN_DOCKER_TESTS=1 to test real Docker')
def test_real_sandbox_compiles_and_runs():
    src='int sum_two(int a, int b) { return a+b; }'
    tests=[
        {'name':'positive','code':'if (sum_two(2,3)!=5) return 1;'},
        {'name':'negative','code':'if (sum_two(-6,2)!=-4) return 1;'},
    ]
    out=evaluate_code(src, '', tests)
    assert out.status=='TEST_PASS',f'{out.status}: {out.error}; {out.compile_log}; {out.run_log}'
    assert out.compile_success and out.tests_passed==2
