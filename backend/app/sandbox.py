"""Compile and execute candidate C/C++ only inside an isolated Docker container.

IMPORTANT: this is a development sandbox, not a multi-tenant security boundary.
The benchmark harness itself must be trusted. Never execute generated code on
the host.

The container runs a small Python runner (present in the ``gcc:14`` image) that
compiles the program, executes every testcase separately and reports one JSON
object per step.  JSON avoids fragile log parsing and keeps stdout/stderr,
exit codes, signals, timeouts and durations intact.  Logs are capped per
stream and the truncation flag is kept in the structured payload.
"""
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json
import os
import re
import signal
import subprocess
import tempfile
from .config import (
    SANDBOX_IMAGE, MAX_SOURCE_CHARS, COMPILE_TIMEOUT, TEST_TIMEOUT,
    LOG_LIMIT_BYTES,
)

# Maximum time the docker CLI itself may take.  Kept generous because image
# startup can be slow on a busy machine; the in-container runner enforces the
# real compile/test timeouts.
MAX_CONTAINER_LIFETIME = 3600

ASSERTION_RE = re.compile(
    r"(?is)(assertion\s+[`'\"].*?['\"]?\s*failed|assertion\s+failed|assert\(.*?\)\s+failed)"
)


@dataclass
class EvalOutcome:
    status: str
    compile_success: bool | None
    tests_passed: int
    tests_total: int
    compile_log: str = ''
    run_log: str = ''
    error: str = ''
    compile_ms: int = 0
    test_ms: int = 0
    compile_exit_code: int | None = None
    compile_signal: int | None = None
    compile_timed_out: bool = False
    compile_stdout: str = ''
    compile_stderr: str = ''
    tests: list[dict] = field(default_factory=list)
    harness_mode: str = 'cases'
    harness_note: str = ''
    failure_reason: str = ''

    def detail(self) -> str:
        """Structured payload persisted in results.detail_json."""
        return json.dumps({
            'harness_mode': self.harness_mode,
            'harness_note': self.harness_note,
            'failure_reason': self.failure_reason,
            'compile': {
                'exit_code': self.compile_exit_code,
                'signal': self.compile_signal,
                'timed_out': self.compile_timed_out,
                'duration_ms': self.compile_ms,
                'stdout': self.compile_stdout,
                'stderr': self.compile_stderr,
            },
            'tests': self.tests,
        }, ensure_ascii=False, default=str)


RUNNER_SOURCE = r'''
import json
import os
import subprocess
import sys
import time
import traceback


def emit(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def read_capped(path, limit):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fp:
            raw = fp.read(limit)
    except OSError:
        return "", 0, False
    return raw.decode("utf-8", "replace"), size, size > limit


def run(cmd, timeout, out_path, err_path):
    start = time.perf_counter()
    timed_out = False
    rc = None
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        try:
            rc = subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout,
                                check=False).returncode
        except subprocess.TimeoutExpired:
            timed_out = True
        except OSError as exc:
            return {"spawn_error": str(exc), "duration_ms": round((time.perf_counter() - start) * 1000)}
    return {
        "duration_ms": round((time.perf_counter() - start) * 1000),
        "timed_out": timed_out,
        "returncode": rc,
    }


def event(kind, result, out_path, err_path, limit, index=None):
    out, out_size, out_trunc = read_capped(out_path, limit)
    err, err_size, err_trunc = read_capped(err_path, limit)
    rc = result.get("returncode")
    obj = {
        "type": kind,
        "duration_ms": result.get("duration_ms", 0),
        "timed_out": bool(result.get("timed_out")),
        "exit_code": rc if rc is not None and rc >= 0 else None,
        "signal": -rc if rc is not None and rc < 0 else None,
        "stdout": out,
        "stderr": err,
        "stdout_truncated": out_trunc,
        "stderr_truncated": err_trunc,
        "stdout_bytes": out_size,
        "stderr_bytes": err_size,
    }
    if index is not None:
        obj["index"] = index
    if result.get("spawn_error"):
        obj["spawn_error"] = result["spawn_error"]
    emit(obj)


def main():
    language = sys.argv[1]
    total = int(sys.argv[2])
    compile_timeout = float(sys.argv[3])
    test_timeout = float(sys.argv[4])
    limit = int(sys.argv[5])
    source = "/input/program.cpp" if language == "cpp" else "/input/program.c"
    compiler = "g++" if language == "cpp" else "gcc"
    std = "-std=c++17" if language == "cpp" else "-std=c11"
    compiler_cmd = [compiler, std, "-O0", "-w", source, "-o", "/work/program", "-lm"]

    result = run(compiler_cmd, compile_timeout, "/work/compile.out", "/work/compile.err")
    event("compile", result, "/work/compile.out", "/work/compile.err", limit)
    if result.get("spawn_error") or result.get("timed_out") or result.get("returncode") != 0:
        emit({"type": "done", "tests": 0, "compiled": False})
        return

    for index in range(total):
        result = run(["/work/program", str(index)], test_timeout,
                     "/work/test.out", "/work/test.err")
        event("test", result, "/work/test.out", "/work/test.err", limit, index=index)
    emit({"type": "done", "tests": total, "compiled": True})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
'''


def extract_c(text: str) -> str:
    """Strip optional Markdown code fences without altering the actual source."""
    text = text.strip()
    match = re.search(r'```(?:c|cpp|c\+\+)?\s*\n(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def harness_mode(tests: list[dict]) -> str:
    return 'program' if len(tests) == 1 and 'program' in tests[0] else 'cases'


HARNESS_NOTE_PROGRAM = ('完整 harness：Test 0 是整套测试程序的一次执行，'
                        '不是第 1 个 assert 断言；TCP 按 1 个用例计算。')


def build_c_file(candidate: str, dependencies: str, tests: list[dict]) -> str:
    """Compose one executable with a switch-based independent testcase dispatcher."""
    if harness_mode(tests) == 'program':
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


def _capped(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit]


def _stream(ev: dict, key: str) -> str:
    text = str(ev.get(key) or '')
    if ev.get(key + '_truncated'):
        text += f"\n... [truncated: first {len(text.encode('utf-8'))} bytes kept]"
    return text


def _signal_name(sig) -> str:
    try:
        return signal.Signals(int(sig)).name
    except (ValueError, TypeError):
        return f'SIG{sig}'


def _classify_test(ev: dict) -> tuple[str, str]:
    """Return (classification, human readable reason) for one executed testcase."""
    if ev.get('spawn_error'):
        return 'INFRA_ERROR', f'could not start program: {ev["spawn_error"]}'
    if ev.get('timed_out'):
        ms = int(ev.get('duration_ms') or 0)
        return 'TIMEOUT', f'timed out after {ms} ms'
    logs = (str(ev.get('stdout') or '') + '\n' + str(ev.get('stderr') or ''))
    sig = ev.get('signal')
    if sig is not None:
        sig = int(sig)
        if sig == int(signal.SIGABRT) and ASSERTION_RE.search(logs):
            return 'ASSERT_FAILURE', 'SIGABRT (assertion failure)'
        return 'RUNTIME_ERROR', f'{_signal_name(sig)} ({sig})'
    code = ev.get('exit_code')
    if code == 0:
        return 'PASS', ''
    if code == 1:
        return 'WRONG_ANSWER', 'exit code 1'
    return 'RUNTIME_ERROR', f'exit code {code}'


def _test_record(ev: dict) -> dict:
    classification, reason = _classify_test(ev)
    return {
        'index': int(ev.get('index', 0)),
        'status': classification,
        'reason': reason,
        'exit_code': ev.get('exit_code'),
        'signal': ev.get('signal'),
        'signal_name': _signal_name(ev['signal']) if ev.get('signal') is not None else '',
        'timed_out': bool(ev.get('timed_out')),
        'duration_ms': int(ev.get('duration_ms') or 0),
        'stdout': str(ev.get('stdout') or ''),
        'stderr': str(ev.get('stderr') or ''),
        'stdout_truncated': bool(ev.get('stdout_truncated')),
        'stderr_truncated': bool(ev.get('stderr_truncated')),
        'stdout_bytes': int(ev.get('stdout_bytes') or 0),
        'stderr_bytes': int(ev.get('stderr_bytes') or 0),
    }


def parse_events(raw: str) -> tuple[list[dict], list[str]]:
    """Split runner stdout into JSON events and unrecognised noise lines."""
    events: list[dict] = []
    noise: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('{'):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                noise.append(line)
                continue
            if isinstance(obj, dict) and obj.get('type') in ('compile', 'test', 'done'):
                events.append(obj)
                continue
        noise.append(line)
    return events, noise


def _compile_event(events: list[dict]) -> dict | None:
    for ev in reversed(events):
        if ev.get('type') == 'compile':
            return ev
    return None


def outcome_from_events(events: list[dict], total_tests: int, mode: str,
                        raw_noise: str = '', stderr_text: str = '',
                        log_limit: int = 5000) -> EvalOutcome:
    """Pure classification of structured runner events (also used by unit tests)."""
    compile_ev = _compile_event(events)
    note = HARNESS_NOTE_PROGRAM if mode == 'program' else ''
    if compile_ev is None:
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            run_log=_capped(raw_noise, 4000), error='Sandbox produced no compile event.',
            harness_mode=mode, harness_note=note)
    compile_ms = int(compile_ev.get('duration_ms') or 0)
    compile_exit = compile_ev.get('exit_code')
    compile_signal = compile_ev.get('signal')
    compile_timed_out = bool(compile_ev.get('timed_out'))
    compile_stdout = str(compile_ev.get('stdout') or '')
    compile_stderr = str(compile_ev.get('stderr') or '')
    compile_log = _capped(
        ('--- compiler stdout ---\n' + _stream(compile_ev, 'stdout') + '\n'
         '--- compiler stderr ---\n' + _stream(compile_ev, 'stderr')), 20000)
    if compile_ev.get('spawn_error'):
        return EvalOutcome('INFRA_ERROR', None, 0, total_tests,
                           compile_log=compile_log, error=f'Compiler could not start: {compile_ev["spawn_error"]}',
                           compile_ms=compile_ms, compile_stdout=compile_stdout,
                           compile_stderr=compile_stderr, compile_exit_code=compile_exit,
                           compile_signal=compile_signal, compile_timed_out=compile_timed_out,
                           harness_mode=mode, harness_note=note)
    if compile_timed_out:
        return EvalOutcome('COMPILE_TIMEOUT', False, 0, total_tests,
                           compile_log=compile_log, error='Compilation timed out.',
                           compile_ms=compile_ms, compile_stdout=compile_stdout,
                           compile_stderr=compile_stderr, compile_exit_code=compile_exit,
                           compile_signal=compile_signal, compile_timed_out=compile_timed_out,
                           harness_mode=mode, harness_note=note)
    if compile_ev.get('exit_code') not in (0, None) or compile_ev.get('signal'):
        return EvalOutcome('COMPILE_ERROR', False, 0, total_tests,
                           compile_log=compile_log, error='Compilation failed.',
                           compile_ms=compile_ms, compile_stdout=compile_stdout,
                           compile_stderr=compile_stderr, compile_exit_code=compile_exit,
                           compile_signal=compile_signal, compile_timed_out=compile_timed_out,
                           harness_mode=mode, harness_note=note)

    test_events = {int(ev.get('index', 0)): ev for ev in events if ev.get('type') == 'test'}
    if set(test_events) != set(range(total_tests)):
        return EvalOutcome('INFRA_ERROR', None, 0, total_tests,
                           compile_log=compile_log, run_log=_capped(raw_noise, 4000),
                           error='Missing testcase events; the sandbox probably stopped.',
                           compile_ms=compile_ms, compile_stdout=compile_stdout,
                           compile_stderr=compile_stderr, compile_exit_code=compile_exit,
                           compile_signal=compile_signal, compile_timed_out=compile_timed_out,
                           harness_mode=mode, harness_note=note)
    records = [_test_record(test_events[i]) for i in range(total_tests)]
    passed = sum(r['status'] == 'PASS' for r in records)
    test_ms = sum(r['duration_ms'] for r in records)
    statuses = {r['status'] for r in records}
    failures = [r for r in records if r['status'] != 'PASS']
    if passed == total_tests:
        status = 'TEST_PASS'
    elif 'TIMEOUT' in statuses:
        status = 'TIMEOUT'
    elif 'RUNTIME_ERROR' in statuses or 'INFRA_ERROR' in statuses:
        status = 'RUNTIME_ERROR'
    else:
        status = 'WRONG_ANSWER'
    reason = '; '.join(f"Test {r['index']}: {r['reason']}" for r in failures
                       if r['reason'])[:600]
    run_log = _capped(_format_run_log(records, note, reason), 20000)
    return EvalOutcome(status, True, passed, total_tests,
                       compile_log=compile_log, run_log=run_log, error=reason,
                       compile_ms=compile_ms, test_ms=test_ms,
                       compile_exit_code=compile_exit, compile_signal=compile_signal,
                       compile_timed_out=compile_timed_out,
                       compile_stdout=compile_stdout, compile_stderr=compile_stderr,
                       tests=records, harness_mode=mode, harness_note=note,
                       failure_reason=reason)


def _format_run_log(records: list[dict], note: str, reason: str) -> str:
    lines: list[str] = []
    if note:
        lines.append(note)
    for r in records:
        if r['status'] == 'PASS':
            lines.append(f"Test {r['index']}: PASS ({r['duration_ms']} ms)")
            continue
        sig = f", signal={r['signal_name']}({r['signal']})" if r['signal'] is not None else ''
        lines.append(f"Test {r['index']}: {r['status']} ({r['reason']}; "
                     f"exit={r['exit_code']}{sig}, {r['duration_ms']} ms)")
        if r['stdout']:
            lines.append('--- stdout ---\n' + _capped(r['stdout'], 4000))
        if r['stderr']:
            lines.append('--- stderr ---\n' + _capped(r['stderr'], 4000))
        if r['stdout_truncated'] or r['stderr_truncated']:
            lines.append('--- logs truncated by sandbox limit ---')
    if reason:
        lines.append('Failure reason: ' + reason)
    return '\n'.join(lines)


def parse_docker_output(raw: str, total_tests: int,
                        log_limit: int = 5000) -> EvalOutcome:
    """Backward compatible entry point.

    New sandboxes emit JSON events; the legacy ``__DEC_`` marker format is still
    accepted so old callers/tests keep working.
    """
    events, noise = parse_events(raw)
    if events:
        mode = 'program' if total_tests == 1 else 'cases'
        return outcome_from_events(events, total_tests, mode,
                                   raw_noise='\n'.join(noise), log_limit=log_limit)
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
    status = ('TEST_PASS' if passed == total_tests else
              'TIMEOUT' if 124 in results.values() or 137 in results.values() else
              'RUNTIME_ERROR' if any(rc >= 128 or rc not in (0, 1) for rc in results.values()) else
              'WRONG_ANSWER')
    return EvalOutcome(status, True, passed, total_tests,
                       compile_log=compile_log,
                       run_log='\n'.join(f'Test {i}: {"PASS" if results[i]==0 else "FAIL (exit "+str(results[i])+")"}'
                                         for i in range(total_tests)))


def _docker_command(path: Path, language: str, total_tests: int,
                    compile_timeout: int, test_timeout: int, log_limit: int,
                    docker_bin: str = 'docker') -> list[str]:
    # Resource limits intentionally unchanged from v0.2; the container is
    # networkless, read-only, unprivileged and has its own tmpfs workdir.
    return [
        docker_bin, 'run', '--rm', '--network', 'none', '--read-only',
        '--pids-limit', '64', '--memory', '1g', '--cpus', '1',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--user', '65534:65534', '--workdir', '/work',
        '--tmpfs', '/work:rw,exec,nosuid,nodev,size=192m,mode=1777',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m,mode=1777',
        '--mount', f'type=bind,source={path},target=/input,readonly',
        SANDBOX_IMAGE, 'python3', '/input/runner.py', language, str(total_tests),
        str(compile_timeout), str(test_timeout), str(log_limit),
    ]


def _prepare_input(candidate: str, dependencies: str, tests: list[dict],
                   language: str, temp: str) -> Path:
    path = Path(temp)
    source = build_c_file(candidate, dependencies, tests)
    src = path / ('program.cpp' if language == 'cpp' else 'program.c')
    src.write_text(source, encoding='utf-8')
    (path / 'runner.py').write_text(RUNNER_SOURCE, encoding='utf-8')
    # Read access for unprivileged uid=65534 inside a read-only bind mount.
    os.chmod(path, 0o755)
    os.chmod(src, 0o644)
    os.chmod(path / 'runner.py', 0o644)
    return path


def _guard_source(candidate: str, tests: list[dict]) -> EvalOutcome | None:
    if not tests:
        return EvalOutcome('INFRA_ERROR', None, 0, 0,
                           error='No executable testcases are attached to this sample.')
    if not candidate:
        return EvalOutcome('MODEL_EMPTY', False, 0, len(tests), error='Model output was empty.')
    return None


def evaluate_code(prediction: str, dependencies: str, tests: list[dict],
                  language: str = 'c', *, compile_timeout: int = COMPILE_TIMEOUT,
                  test_timeout: int = TEST_TIMEOUT,
                  log_limit: int = LOG_LIMIT_BYTES,
                  docker_bin: str = 'docker') -> EvalOutcome:
    """Synchronous sandbox entry point (scripts, model checks, tests).

    No host compilation fallback is provided: a missing Docker engine is an
    INFRA_ERROR, never a model failure.
    """
    if not tests:
        return EvalOutcome('INFRA_ERROR', None, 0, 0,
                           error='No executable testcases are attached to this sample.')
    candidate = extract_c(prediction)
    guard = _guard_source(candidate, tests)
    if guard:
        return guard
    full_source = build_c_file(candidate, dependencies, tests)
    if len(full_source) > MAX_SOURCE_CHARS:
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests),
                           error='Generated source exceeds MVP size limit.')
    mode = harness_mode(tests)
    with tempfile.TemporaryDirectory(prefix='deceval_') as temp:
        path = _prepare_input(candidate, dependencies, tests, language, temp)
        cmd = _docker_command(path, language, len(tests), compile_timeout,
                              test_timeout, log_limit, docker_bin)
        overall = min(MAX_CONTAINER_LIFETIME,
                      compile_timeout + test_timeout * len(tests) + 90)
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=overall, check=False)
        except FileNotFoundError:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker command not found. Install Docker and pull the sandbox image.')
        except subprocess.TimeoutExpired:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker invocation exceeded overall execution timeout.')
        except PermissionError as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker permission error: {exc}')
        except Exception as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker error: {exc}')
        return _outcome_from_process(out.returncode, out.stdout, out.stderr,
                                     len(tests), mode, log_limit)


async def evaluate_code_async(prediction: str, dependencies: str, tests: list[dict],
                              language: str = 'c', *,
                              compile_timeout: int = COMPILE_TIMEOUT,
                              test_timeout: int = TEST_TIMEOUT,
                              log_limit: int = LOG_LIMIT_BYTES,
                              docker_bin: str = 'docker') -> EvalOutcome:
    """Async sandbox entry point used by the concurrent pipeline.

    Docker runs as an asyncio subprocess, so the event loop keeps serving
    concurrent inference requests while compilation is in progress.
    """
    if not tests:
        return EvalOutcome('INFRA_ERROR', None, 0, 0,
                           error='No executable testcases are attached to this sample.')
    candidate = extract_c(prediction)
    guard = _guard_source(candidate, tests)
    if guard:
        return guard
    full_source = build_c_file(candidate, dependencies, tests)
    if len(full_source) > MAX_SOURCE_CHARS:
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests),
                           error='Generated source exceeds MVP size limit.')
    mode = harness_mode(tests)
    with tempfile.TemporaryDirectory(prefix='deceval_') as temp:
        path = _prepare_input(candidate, dependencies, tests, language, temp)
        cmd = _docker_command(path, language, len(tests), compile_timeout,
                              test_timeout, log_limit, docker_bin)
        overall = min(MAX_CONTAINER_LIFETIME,
                      compile_timeout + test_timeout * len(tests) + 90)
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker command not found. Install Docker and pull the sandbox image.')
        except PermissionError as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker permission error: {exc}')
        except Exception as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker error: {exc}')
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=overall)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker invocation exceeded overall execution timeout.')
        except asyncio.CancelledError:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass
            raise
        return _outcome_from_process(proc.returncode, stdout, stderr,
                                     len(tests), mode, log_limit)


def _outcome_from_process(returncode: int | None, stdout: bytes | str | None,
                          stderr: bytes | str | None, total_tests: int,
                          mode: str, log_limit: int) -> EvalOutcome:
    raw = stdout.decode('utf-8', 'replace') if isinstance(stdout, bytes) else (stdout or '')
    err = stderr.decode('utf-8', 'replace') if isinstance(stderr, bytes) else (stderr or '')
    events, noise = parse_events(raw)
    if not events:
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            run_log=_capped(raw[-4000:], 4000),
            error=(_capped(err, 2500) or 'Sandbox produced no structured output '
                   f'(docker exit code {returncode}).'),
            harness_mode=mode,
            harness_note=HARNESS_NOTE_PROGRAM if mode == 'program' else '')
    if returncode not in (0, None):
        # The runner itself failed after emitting a partial stream.
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            run_log=_capped(raw[-4000:], 4000),
            error=(_capped(err, 2500) or f'Sandbox runner exited with code {returncode}.'),
            harness_mode=mode,
            harness_note=HARNESS_NOTE_PROGRAM if mode == 'program' else '')
    return outcome_from_events(events, total_tests, mode,
                               raw_noise='\n'.join(noise), stderr_text=err,
                               log_limit=log_limit)
