"""Compile and execute candidate C only inside an isolated Docker container.

IMPORTANT: This is a development sandbox, not a multi-tenant security boundary.
The benchmark harness itself must be trusted. Never execute generated C on the host.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import subprocess
import tempfile
import time
from .config import SANDBOX_IMAGE, MAX_SOURCE_CHARS


@dataclass
class EvalOutcome:
    status: str
    compile_success: bool | None
    tests_passed: int
    tests_total: int
    compile_log: str = ''
    run_log: str = ''
    error: str = ''


def extract_c(text: str) -> str:
    """Strip optional Markdown code fences without altering the actual source."""
    text = text.strip()
    match = re.search(r'```(?:c|cpp|c\+\+)?\s*\n(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def build_c_file(candidate: str, dependencies: str, tests: list[dict]) -> str:
    """Compose one executable with a switch-based independent testcase dispatcher."""
    if len(tests) == 1 and 'program' in tests[0]:
        # Official LLM4Binary test is an entire C/C++ harness, commonly with main().
        return dependencies + '\n' + candidate + '\n' + tests[0]['program'] + '\n'
    cases = '\n'.join(
        f'case {i}: {{\n{test["code"]}\nreturn 0;\n}}'
        for i, test in enumerate(tests)
    )
    return (
      '#include <stdlib.h>\n'
      + dependencies + '\n'
      + candidate + '\n'
      + 'int main(int argc, char **argv) {\n'
      + 'if (argc < 2) return 98;\n'
      + 'switch (atoi(argv[1])) {\n'
      + cases + '\n'
      + 'default: return 99;\n}\n}\n'
    )


def parse_docker_output(raw: str, total_tests: int) -> EvalOutcome:
    match = re.search(r'^__DEC_COMPILE__:(\d+)$', raw, re.MULTILINE)
    if not match:
        return EvalOutcome('INFRA_ERROR', None, 0, total_tests, run_log=raw[-4000:],
                           error='Compiler status marker missing (Docker/sandbox may have failed).')
    compile_rc = int(match.group(1))
    logmatch = re.search(r'__DEC_COMPILE__:\d+\n(.*?)\n__DEC_LOG_END__', raw, re.DOTALL)
    compile_log = (logmatch.group(1) if logmatch else '')[-5000:]
    if compile_rc != 0:
        return EvalOutcome('COMPILE_TIMEOUT' if compile_rc == 124 else 'COMPILE_ERROR',
                           False, 0, total_tests, compile_log=compile_log)
    records = re.findall(r'^__DEC_TEST__:(\d+):(\d+)$', raw, re.MULTILINE)
    results = {int(i): int(rc) for i, rc in records}
    if len(results) != total_tests or set(results) != set(range(total_tests)):
        return EvalOutcome('INFRA_ERROR', None, 0, total_tests,
                           compile_log=compile_log, run_log=raw[-4000:],
                           error='Missing test markers; sandbox likely stopped unexpectedly.')
    passed = sum(rc == 0 for rc in results.values())
    if passed == total_tests:
        status = 'TEST_PASS'
    elif 124 in results.values() or 137 in results.values():
        status = 'TIMEOUT'
    elif any(rc >= 128 or rc not in (0, 1) for rc in results.values()):
        status = 'RUNTIME_ERROR'
    else:
        status = 'WRONG_ANSWER'
    return EvalOutcome(status, True, passed, total_tests,
                       compile_log=compile_log,
                       run_log='\n'.join(f'Test {i}: {"PASS" if results[i]==0 else "FAIL (exit "+str(results[i])+")"}'
                                         for i in range(total_tests)))


def evaluate_code(prediction: str, dependencies: str, tests: list[dict], language: str = 'c') -> EvalOutcome:
    """No host compilation fallback is provided; a missing Docker engine is an INFRA_ERROR."""
    if not tests:
        return EvalOutcome('INFRA_ERROR', None, 0, 0,
                           error='No executable testcases are attached to this sample.')
    candidate = extract_c(prediction)
    if not candidate:
        return EvalOutcome('MODEL_EMPTY', False, 0, len(tests), error='Model output was empty.')
    full_source = build_c_file(candidate, dependencies, tests)
    if len(full_source) > MAX_SOURCE_CHARS:
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests),
                           error='Generated source exceeds MVP size limit.')

    with tempfile.TemporaryDirectory(prefix='deceval_') as temp:
        path = Path(temp)
        src = path / ('program.cpp' if language == 'cpp' else 'program.c')
        src.write_text(full_source, encoding='utf-8')
        # Read access for unprivileged uid=65534 inside a read-only bind mount.
        os.chmod(path, 0o755)
        os.chmod(src, 0o644)
        script = r'''
set +e
compiler=gcc
std=-std=c11
source=/input/program.c
if [ "$2" = cpp ]; then compiler=g++; std=-std=c++17; source=/input/program.cpp; fi
timeout -s KILL 45s "$compiler" "$std" -O0 -w "$source" -o /work/program -lm 2>/work/compile.log
c=$?
printf '__DEC_COMPILE__:%s\n' "$c"
head -c 5000 /work/compile.log
printf '\n__DEC_LOG_END__\n'
if [ "$c" -ne 0 ]; then exit 0; fi
i=0
while [ "$i" -lt "$1" ]; do
  timeout -s KILL 10s /work/program "$i" >/dev/null 2>/dev/null
  r=$?
  printf '__DEC_TEST__:%s:%s\n' "$i" "$r"
  i=$((i + 1))
done
'''
        cmd = [
            'docker','run','--rm','--network','none','--read-only',
            '--pids-limit','64','--memory','1g','--cpus','1',
            '--cap-drop','ALL','--security-opt','no-new-privileges',
            '--user','65534:65534','--workdir','/work',
            '--tmpfs','/work:rw,exec,nosuid,nodev,size=192m,mode=1777',
            '--tmpfs','/tmp:rw,nosuid,nodev,size=64m,mode=1777',
            '--mount', f'type=bind,source={path},target=/input,readonly',
            SANDBOX_IMAGE, '/bin/sh','-c', script, 'sandbox', str(len(tests)), language
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=min(600, 60 + 12 * len(tests)), check=False)
        except FileNotFoundError:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker command not found. Install Docker and pull the sandbox image.')
        except subprocess.TimeoutExpired:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker invocation exceeded overall execution timeout.')
        except Exception as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),error=f'Docker error: {exc}')
        raw = (out.stdout or '')[-12000:]
        if out.returncode != 0:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               run_log=raw[-4000:],error=(out.stderr or raw)[-2500:])
        return parse_docker_output(raw,len(tests))
