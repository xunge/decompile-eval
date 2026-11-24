"""Diagnostics tests for the structured sandbox protocol (no Docker required)."""
import json
import subprocess

import pytest

from app import sandbox
from app.sandbox import (
    EvalOutcome, evaluate_code, outcome_from_events, parse_docker_output,
)


def compile_event(**overrides):
    event = {
        'type': 'compile', 'duration_ms': 12, 'timed_out': False,
        'exit_code': 0, 'signal': None,
        'stdout': '', 'stderr': '',
        'stdout_truncated': False, 'stderr_truncated': False,
        'stdout_bytes': 0, 'stderr_bytes': 0,
    }
    event.update(overrides)
    return event


def _test_event(index=0, **overrides):
    event = {
        'type': 'test', 'index': index, 'duration_ms': 7, 'timed_out': False,
        'exit_code': 0, 'signal': None,
        'stdout': '', 'stderr': '',
        'stdout_truncated': False, 'stderr_truncated': False,
        'stdout_bytes': 0, 'stderr_bytes': 0,
    }
    event.update(overrides)
    return event


def classify(events, total=1, mode='cases'):
    return outcome_from_events(events, total, mode, log_limit=4096)


def test_pass_records_timings():
    out = classify([compile_event(), _test_event(index=0)], total=1)
    assert out.status == 'TEST_PASS'
    assert out.compile_success is True
    assert out.tests_passed == 1 and out.tests_total == 1
    assert out.compile_ms == 12 and out.test_ms == 7
    assert out.tests[0]['exit_code'] == 0 and out.tests[0]['signal'] is None


def test_assert_failure_is_wrong_answer():
    out = classify([compile_event(), _test_event(
        signal=6, exit_code=None,
        stderr="program: /input/program.c:8: main: Assertion `f()==1' failed.\n")])
    assert out.status == 'WRONG_ANSWER'
    assert out.tests_passed == 0
    assert 'assertion' in out.failure_reason.lower()
    assert out.tests[0]['signal_name'] == 'SIGABRT'
    assert out.compile_success is True


def test_abort_without_assert_is_runtime_error():
    out = classify([compile_event(), _test_event(signal=6, exit_code=None, stderr='')])
    assert out.status == 'RUNTIME_ERROR'
    assert out.failure_reason == 'Test 0: SIGABRT (6)'


def test_segv_is_runtime_error():
    out = classify([compile_event(), _test_event(signal=11, exit_code=None)])
    assert out.status == 'RUNTIME_ERROR'
    assert 'SIGSEGV' in out.failure_reason


def test_positive_exit_134_is_not_assumed_to_be_assertion():
    out = classify([compile_event(), _test_event(exit_code=134, signal=None)])
    assert out.status == 'RUNTIME_ERROR'
    assert 'exit code 134' in out.failure_reason


def test_compile_error_keeps_compiler_output():
    out = classify([compile_event(exit_code=1, stderr='error: expected expression')])
    assert out.status == 'COMPILE_ERROR'
    assert out.compile_success is False
    assert 'expected expression' in out.compile_log


def test_compile_timeout_is_distinct():
    out = classify([compile_event(timed_out=True, exit_code=None)])
    assert out.status == 'COMPILE_TIMEOUT'
    assert out.compile_success is False


def test_test_timeout_is_distinct():
    out = classify([compile_event(), _test_event(timed_out=True, exit_code=None, duration_ms=10000)])
    assert out.status == 'TIMEOUT'
    assert 'timed out' in out.failure_reason


def test_missing_testcase_events_are_infra_not_model_failure():
    out = classify([compile_event()], total=2)
    assert out.status == 'INFRA_ERROR'
    assert out.compile_success is None
    assert 'missing testcase' in out.error.lower()


def test_no_events_is_infra_error():
    out = classify([])
    assert out.status == 'INFRA_ERROR'
    assert out.compile_success is None


def test_special_characters_and_newlines_roundtrip():
    stdout = 'line1 "quoted" \\ slash\nline2 \t tab \u4e2d\u6587\n'
    stderr = "it's a 'quote' and a \x01 control byte\n"
    out = classify([compile_event(), _test_event(exit_code=1, stdout=stdout, stderr=stderr)])
    assert out.tests[0]['stdout'] == stdout
    assert out.tests[0]['stderr'] == stderr
    assert 'line2' in out.run_log


def test_truncation_is_flagged_in_payload_and_logs():
    out = classify([compile_event(stdout_truncated=True, stdout='abc'),
                    _test_event(exit_code=1, stderr_truncated=True, stderr='boom')])
    assert out.tests[0]['stderr_truncated'] is True
    assert 'truncated' in out.run_log
    detail = json.loads(out.detail())
    assert detail['tests'][0]['stderr_truncated'] is True


def test_program_harness_note_and_single_case():
    out = classify([compile_event(), _test_event(index=0, exit_code=1)],
                   total=1, mode='program')
    assert out.harness_mode == 'program'
    assert '不是第 1 个 assert' in out.harness_note
    assert out.tests_total == 1


def test_mixed_tests_prefer_crash_over_wrong_answer():
    out = classify([compile_event(),
                    _test_event(0), _test_event(1, exit_code=1),
                    _test_event(2, signal=11)], total=3)
    assert out.status == 'RUNTIME_ERROR'
    assert out.tests_passed == 1
    assert out.tests_total == 3


def test_program_harness_mode_detection():
    assert sandbox.harness_mode([{'program': 'int main(){return 0;}'}]) == 'program'
    assert sandbox.harness_mode([{'code': 'return 0;'}]) == 'cases'


def test_json_event_parser_ignores_noise():
    events, noise = sandbox.parse_events(
        'docker warning\n{"type":"compile","exit_code":0}\nnot json\n')
    assert len(events) == 1 and events[0]['type'] == 'compile'
    assert noise == ['docker warning', 'not json']


def test_docker_permission_error_is_infra(monkeypatch):
    monkeypatch.setattr(subprocess, 'run',
                        lambda *a, **k: (_ for _ in ()).throw(PermissionError('denied')))
    out = evaluate_code('int f(void){return 0;}', '', [{'code': 'f();'}])
    assert out.status == 'INFRA_ERROR'
    assert 'permission' in out.error.lower()
    assert out.compile_success is None


def test_docker_missing_is_infra(monkeypatch):
    monkeypatch.setattr(subprocess, 'run',
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError('docker')))
    out = evaluate_code('int f(void){return 0;}', '', [{'code': 'f();'}])
    assert out.status == 'INFRA_ERROR'
    assert out.compile_success is None


def test_empty_model_output_is_model_empty():
    out = evaluate_code('   ', '', [{'code': 'f();'}])
    assert out.status == 'MODEL_EMPTY'
    assert out.compile_success is False


def test_no_tests_is_infra_error():
    out = evaluate_code('int f(void){return 0;}', '', [])
    assert out.status == 'INFRA_ERROR'


def test_legacy_marker_protocol_still_parses():
    out = parse_docker_output(
        '__DEC_COMPILE__:0\n\n__DEC_LOG_END__\n__DEC_TEST__:0:0\n__DEC_TEST__:1:0\n', 2)
    assert out.status == 'TEST_PASS' and out.tests_passed == 2
