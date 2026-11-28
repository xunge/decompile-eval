"""Host fast mode tests (real local gcc; no Docker)."""
import shutil
import pytest

from app.sandbox import EvalOutcome
from app.host_sandbox import evaluate_code_host

pytestmark = pytest.mark.skipif(not shutil.which('gcc'), reason='host gcc not available')


def test_host_compile_and_run():
    src = 'int add(int a,int b){return a+b;}'
    out = evaluate_code_host(src, '', [
        {'name': 'one', 'code': 'if (add(2,3)!=5) return 1;'},
        {'name': 'two', 'code': 'if (add(-2,3)!=1) return 1;'},
    ], test_timeout=5)
    assert out.status == 'TEST_PASS', f'{out.error}; {out.compile_log}; {out.run_log}'
    assert out.compile_success is True and out.tests_passed == 2
    assert out.source_compile_success is True


def test_host_compile_error():
    out = evaluate_code_host('int add(int a,int b){', '', [
        {'name': 'one', 'code': 'if (add(2,3)!=5) return 1;'}], test_timeout=5)
    assert out.status == 'COMPILE_ERROR'
    assert out.compile_success is False


def test_host_test_timeout():
    src = 'int spin(void){while(1){} return 0;}'
    out = evaluate_code_host(src, '', [
        {'name': 'one', 'code': 'if (spin()!=0) return 1;'}],
        test_timeout=1)
    assert out.status == 'TIMEOUT'
    assert out.tests_passed == 0


def test_host_mode_runs_per_case_diagnostics_for_tcp():
    from app.case_extract import DIAG_HEADER, extract_manifest
    harness = '''#include <assert.h>
int main(void){ int x=1; assert(x==0); assert(x==1); assert(x+1==2); return 0; }
'''
    manifest = extract_manifest(harness, 'c')
    assert manifest.status == 'OK' and len(manifest.cases) == 3
    out = evaluate_code_host(
        'int unused(void){return 0;}', '', [{'program': harness}],
        diagnostic={'harness': manifest.rewritten_harness,
                    'header': DIAG_HEADER, 'max_cases': 10},
        test_timeout=5)
    # Official harness fails at the first assertion, but diagnostics still
    # report every independent case rather than treating the harness as one test.
    assert out.status == 'WRONG_ANSWER'
    assert len(out.cases) == 3
    assert out.diagnostic_status in ('OK', 'SCAN_LIMIT')
    assert sum(1 for c in out.cases if c.get('passed')) == 2


def test_host_source_only_rc_survives_harness_compile_failure():
    # GNU strdup is accepted in the source-only stage (gnu11), while the full
    # harness is compiled with strict c11 and may reject the implicit declaration.
    src = 'char *func0(char *s){return strdup(s);}'
    harness = '#include <string.h>\nint main(void){ char *p=func0("x"); return p && p[0] == \'x\' ? 0 : 1; }'
    out = evaluate_code_host(src, '', [{'program': harness}], test_timeout=5)
    assert out.source_compile_success is True
    assert out.compile_success is False
    assert out.compile_stage == 'harness'
