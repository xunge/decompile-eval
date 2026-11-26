from app.sandbox import (extract_c, build_c_file, parse_docker_output,
                         outcome_from_events)


def test_extract_markdown():
    assert extract_c('```c\nint x(void){return 1;}\n```') == 'int x(void){return 1;}'
    assert extract_c('int x(void){return 1;}') == 'int x(void){return 1;}'


def test_build_single_main_with_two_tests():
    code=build_c_file('int add(int x,int y){return x+y;}', '', [
        {'name':'one','code':'if (add(1,2)!=3) return 1;'},
        {'name':'two','code':'if (add(2,2)!=4) return 1;'},
    ])
    assert 'case 0:' in code and 'case 1:' in code
    assert code.count('int main(') == 1


def test_parse_success():
    out=parse_docker_output('__DEC_COMPILE__:0\n\n__DEC_LOG_END__\n__DEC_TEST__:0:0\n__DEC_TEST__:1:0\n',2)
    assert out.status=='TEST_PASS' and out.compile_success is True
    assert out.tests_passed==2 and out.tests_total==2


def test_parse_wrong_answer():
    out=parse_docker_output('__DEC_COMPILE__:0\n\n__DEC_LOG_END__\n__DEC_TEST__:0:1\n__DEC_TEST__:1:0\n',2)
    assert out.status=='WRONG_ANSWER' and out.tests_passed==1


def test_parse_compiler_error():
    out=parse_docker_output('__DEC_COMPILE__:1\nerror: expected expression\n__DEC_LOG_END__\n',1)
    assert out.status=='COMPILE_ERROR' and 'expected expression' in out.compile_log


def test_parse_missing_markers_is_not_false_model_failure():
    out=parse_docker_output('docker: permission denied',2)
    assert out.status=='INFRA_ERROR' and out.compile_success is None


def _source_event(ok=True):
    return {'type': 'source_compile', 'exit_code': 0 if ok else 1,
            'signal': None, 'timed_out': False, 'duration_ms': 8,
            'stdout': '', 'stderr': '' if ok else 'error: expected expression'}


def test_official_rc_counts_candidate_only_even_when_harness_compile_fails():
    events = [
        _source_event(True),
        {'type': 'compile', 'exit_code': 1, 'signal': None, 'timed_out': False,
         'duration_ms': 9, 'stdout': '', 'stderr': 'incompatible pointer type'},
    ]
    out = outcome_from_events(events, 1, 'program')
    assert out.status == 'COMPILE_ERROR'
    assert out.compile_success is False          # full harness failed
    assert out.source_compile_success is True    # official RC passes
    assert out.compile_stage == 'harness'


def test_source_compile_failure_stops_before_harness():
    out = outcome_from_events([_source_event(False)], 1, 'program')
    assert out.status == 'COMPILE_ERROR'
    assert out.compile_success is False
    assert out.source_compile_success is False
    assert out.compile_stage == 'source'
    assert 'expected expression' in out.compile_log


def test_source_compile_pass_and_harness_pass():
    events = [
        _source_event(True),
        {'type': 'compile', 'exit_code': 0, 'signal': None, 'timed_out': False,
         'duration_ms': 9, 'stdout': '', 'stderr': ''},
        {'type': 'test', 'index': 0, 'exit_code': 0, 'signal': None,
         'timed_out': False, 'duration_ms': 1, 'stdout': '', 'stderr': '',
         'stdout_truncated': False, 'stderr_truncated': False},
    ]
    out = outcome_from_events(events, 1, 'cases')
    assert out.status == 'TEST_PASS'
    assert out.compile_success is True and out.source_compile_success is True
    assert out.compile_stage == 'ok'


def test_legacy_events_have_no_source_metric():
    out = outcome_from_events([{'type': 'compile', 'exit_code': 0, 'signal': None,
                                'timed_out': False, 'duration_ms': 1,
                                'stdout': '', 'stderr': ''}], 0, 'program')
    assert out.source_compile_success is None
