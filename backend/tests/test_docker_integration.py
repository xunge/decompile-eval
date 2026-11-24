"""Opt-in real Docker E2E checks, run only on a trusted developer machine.

    RUN_DOCKER_TESTS=1 pytest -q tests/test_docker_integration.py

The sandbox bind-mounts a temporary directory into the container.  If the
backend itself runs inside an isolated PID/mount namespace whose /tmp is not
visible to the Docker daemon, point TMPDIR at a directory the daemon can see:

    TMPDIR=$PWD/storage/.tmp-docker RUN_DOCKER_TESTS=1 pytest -q tests/test_docker_integration.py
"""
import os

import pytest

from app.sandbox import evaluate_code

pytestmark = pytest.mark.skipif(os.getenv('RUN_DOCKER_TESTS') != '1',
                                reason='Set RUN_DOCKER_TESTS=1 to test real Docker')


def test_real_sandbox_compiles_and_runs():
    src = 'int sum_two(int a, int b) { return a+b; }'
    tests = [
        {'name': 'positive', 'code': 'if (sum_two(2,3)!=5) return 1;'},
        {'name': 'negative', 'code': 'if (sum_two(-6,2)!=-4) return 1;'},
    ]
    out = evaluate_code(src, '', tests)
    assert out.status == 'TEST_PASS', f'{out.status}: {out.error}; {out.compile_log}; {out.run_log}'
    assert out.compile_success and out.tests_passed == 2
    assert out.tests_total == 2
    assert out.compile_ms >= 0 and out.test_ms >= 0
    assert out.tests[0]['exit_code'] == 0


def test_real_sandbox_assertion_is_wrong_answer_not_runtime_error():
    src = 'int twice(int a) { return a + a; }'
    tests = [{'name': 'bad', 'code': 'assert(twice(2) == 5);'}]
    out = evaluate_code(src, '#include <assert.h>', tests, test_timeout=5)
    assert out.status == 'WRONG_ANSWER', f'{out.status}: {out.failure_reason}'
    assert out.tests[0]['signal'] == 6
    assert out.tests[0]['signal_name'] == 'SIGABRT'
    assert 'Assertion' in out.tests[0]['stderr']
    assert 'assertion failure' in out.failure_reason


def test_real_sandbox_abort_is_runtime_error():
    src = '#include <stdlib.h>\nvoid die(void) { abort(); }'
    out = evaluate_code(src, '', [{'name': 'abort', 'code': 'die();'}], test_timeout=5)
    assert out.status == 'RUNTIME_ERROR', f'{out.status}: {out.failure_reason}'
    assert out.tests[0]['signal'] == 6
    assert 'assertion' not in out.failure_reason.lower()


def test_real_sandbox_sigsegv_is_runtime_error():
    src = 'int deref(void) { volatile int *p = 0; return *p; }'
    out = evaluate_code(src, '', [{'name': 'segv', 'code': 'deref();'}], test_timeout=5)
    assert out.status == 'RUNTIME_ERROR'
    assert out.tests[0]['signal'] == 11
    assert out.tests[0]['signal_name'] == 'SIGSEGV'


def test_real_sandbox_timeout_is_timeout():
    src = 'int spin(void) { for (;;) { } }'
    out = evaluate_code(src, '', [{'name': 'spin', 'code': 'spin();'}], test_timeout=1)
    assert out.status == 'TIMEOUT', f'{out.status}: {out.failure_reason}'
    assert out.tests[0]['timed_out'] is True


def test_real_sandbox_compile_error_keeps_logs():
    out = evaluate_code('int broken(void) { return ;; }', '',
                        [{'name': 'x', 'code': 'broken();'}])
    assert out.status == 'COMPILE_ERROR'
    assert out.compile_success is False
    assert 'error' in out.compile_log.lower()
    assert out.compile_exit_code not in (0, None)


def test_real_sandbox_program_harness_mode():
    src = 'int func0(int x) { return x + 1; }'
    program = '#include <assert.h>\nint main(void){assert(func0(1)==2);return 0;}'
    out = evaluate_code(src, '', [{'name': 'upstream_harness', 'program': program}])
    assert out.status == 'TEST_PASS'
    assert out.harness_mode == 'program'
    assert out.tests_total == 1
    assert '不是第 1 个 assert' in out.harness_note


def test_real_sandbox_cpp_language():
    src = 'int add(int a, int b) { return a + b; }'
    out = evaluate_code(src, '', [{'name': 'cpp', 'code': 'if (add(2,3)!=5) return 1;'}],
                        language='cpp')
    assert out.status == 'TEST_PASS', f'{out.status}: {out.compile_log}'
    assert out.compile_success is True


def test_real_sandbox_logs_special_characters_and_truncation():
    src = ('#include <stdio.h>\nint noisy(void) { '
           'printf("quote=\\"x\\"\\nfile=\\\"a.c\\"\\nunicode=\\xe4\\xb8\\xad\\n"); '
           'for (int i = 0; i < 4000; i++) printf("y"); return 0; }')
    out = evaluate_code(src, '', [{'name': 'noisy', 'code': 'if (noisy() != 1) return 1;'}],
                        log_limit=256)
    assert out.status == 'WRONG_ANSWER'
    test = out.tests[0]
    assert 'quote="x"' in test['stdout']
    assert test['stdout_truncated'] is True
    assert 'truncated' in out.run_log
