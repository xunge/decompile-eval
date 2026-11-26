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
    cases: list[dict] = field(default_factory=list)
    diagnostic_status: str = ''
    case_coverage: dict = field(default_factory=dict)
    source_compile_success: bool | None = None
    compile_stage: str = ''

    def detail(self) -> str:
        """Structured payload persisted in results.detail_json."""
        return json.dumps({
            'harness_mode': self.harness_mode,
            'harness_note': self.harness_note,
            'failure_reason': self.failure_reason,
            'compile_stage': self.compile_stage,
            'source_compile_ok': self.source_compile_success,
            'compile': {
                'exit_code': self.compile_exit_code,
                'signal': self.compile_signal,
                'timed_out': self.compile_timed_out,
                'duration_ms': self.compile_ms,
                'stdout': self.compile_stdout,
                'stderr': self.compile_stderr,
            },
            'tests': self.tests,
            'diagnostic_status': self.diagnostic_status,
            'case_count': len(self.cases),
            'case_coverage': self.case_coverage,
        }, ensure_ascii=False, default=str)


RUNNER_SOURCE = r"""
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


def run(cmd, timeout, out_path, err_path, env=None):
    start = time.perf_counter()
    timed_out = False
    rc = None
    with open(out_path, "wb") as out, open(err_path, "wb") as err:
        try:
            rc = subprocess.run(cmd, stdout=out, stderr=err, timeout=timeout,
                                env=env, check=False).returncode
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


def read_records(path, limit=16384):
    try:
        with open(path, "rb") as fp:
            data = fp.read()
    except OSError:
        return []
    fields = data.split(b"\0")
    records = []
    i = 0
    while i + 11 < len(fields):
        if fields[i + 11] != b"END":
            i += 1
            continue

        def dec(value):
            return value.decode("utf-8", "replace")[:limit]

        try:
            record = {
                "index": int(fields[i]),
                "static_id": int(fields[i + 1]),
                "line": int(fields[i + 2]),
                "passed": fields[i + 3] == b"1",
                "actual_ok": fields[i + 4] == b"1",
                "expected_ok": fields[i + 5] == b"1",
                "expression": dec(fields[i + 6]),
                "lhs": dec(fields[i + 7]),
                "rhs": dec(fields[i + 8]),
                "actual": dec(fields[i + 9]),
                "expected": dec(fields[i + 10]),
            }
        except ValueError:
            i += 12
            continue
        records.append(record)
        i += 12
    return records


def run_diag_cases(program, max_cases, test_timeout, log_limit):
    records_path = "/work/dec_cases.bin"
    failures = []
    k = 1
    status = "OK"
    while k <= max_cases:
        try:
            os.remove(records_path)
        except FileNotFoundError:
            pass
        env = dict(os.environ)
        env["DEC_DIAG_STOP"] = str(k)
        env["DEC_DIAG_SKIP"] = ",".join(str(item) for item in failures)
        env["DEC_DIAG_PATH"] = records_path
        result = run([program], test_timeout, "/work/diag.out", "/work/diag.err", env=env)
        duration = result.get("duration_ms", 0)
        timed_out = bool(result.get("timed_out"))
        rc = result.get("returncode")
        records = read_records(records_path)
        prior = [r for r in records if not r["passed"] and r["index"] < k
                 and r["index"] not in failures]
        if prior:
            failures = sorted(set(failures + [min(r["index"] for r in prior)]))
            continue
        record = next((r for r in records if r["index"] == k), None)
        if record is not None:
            out, out_size, out_trunc = read_capped("/work/diag.out", log_limit)
            err, err_size, err_trunc = read_capped("/work/diag.err", log_limit)
            emit({
                "type": "case",
                "index": k,
                "static_id": record["static_id"],
                "line": record["line"],
                "expression": record["expression"],
                "lhs": record["lhs"],
                "rhs": record["rhs"],
                "actual": record["actual"],
                "expected": record["expected"],
                "actual_ok": record["actual_ok"],
                "expected_ok": record["expected_ok"],
                "passed": record["passed"],
                "after_prior_failure": bool(failures) and k > min(failures),
                "duration_ms": duration,
                "exit_code": rc if rc is not None and rc >= 0 else None,
                "signal": -rc if rc is not None and rc < 0 else None,
                "timed_out": timed_out,
                "stdout": out,
                "stderr": err,
                "stdout_truncated": out_trunc,
                "stderr_truncated": err_trunc,
                "stdout_bytes": out_size,
                "stderr_bytes": err_size,
            })
            if not record["passed"]:
                failures = sorted(set(failures + [k]))
            k += 1
            continue
        if timed_out:
            status = "TIMEOUT"
            break
        if rc not in (0, 3):
            status = "CRASH"
            break
        if k == 1:
            status = "NO_CASES_EXECUTED"
        break
    else:
        status = "SCAN_LIMIT"
    emit({"type": "diag_done", "status": status, "cases": k - 1,
          "max_cases": max_cases, "failures": failures})


def main():
    language = sys.argv[1]
    total = int(sys.argv[2])
    compile_timeout = float(sys.argv[3])
    test_timeout = float(sys.argv[4])
    limit = int(sys.argv[5])
    diagnostic = sys.argv[6] == "1" if len(sys.argv) > 6 else False
    max_cases = int(sys.argv[7]) if len(sys.argv) > 7 else 0
    source = "/input/program.cpp" if language == "cpp" else "/input/program.c"
    compiler = "g++" if language == "cpp" else "gcc"
    std = "-std=c++17" if language == "cpp" else "-std=c11"
    compiler_cmd = [compiler, std, "-O0", "-w", source, "-o", "/work/program", "-lm"]

    # Official LLM4Decompile compile-rate: compile the generated function
    # alone (include headers + candidate), not candidate + test harness.
    # The paper-era GCC accepted implicit declarations and pointer-conversion
    # warnings; GCC 14 promotes them to errors.  gnu11 + explicit downgrades
    # keep the compile-rate comparison as close to the paper as possible.
    only_source = "/input/program_only.cpp" if language == "cpp" else "/input/program_only.c"
    only_std = "-std=gnu++17" if language == "cpp" else "-std=gnu11"
    compat = [] if language == "cpp" else [
        "-D_GNU_SOURCE",
        "-Wno-error=implicit-function-declaration",
        "-Wno-error=incompatible-pointer-types",
        "-Wno-error=int-conversion",
    ]
    source_cmd = [compiler, only_std, *compat, "-O0", "-S", only_source,
                  "-o", "/work/program_only.s", "-lm"]
    source_result = run(source_cmd, compile_timeout,
                        "/work/source_compile.out", "/work/source_compile.err")
    event("source_compile", source_result, "/work/source_compile.out",
          "/work/source_compile.err", limit)
    source_compiled = not (source_result.get("spawn_error")
                           or source_result.get("timed_out")
                           or source_result.get("returncode") != 0)
    if not source_compiled:
        emit({"type": "done", "tests": 0, "compiled": False,
              "source_compiled": False, "diag": diagnostic})
        return

    result = run(compiler_cmd, compile_timeout, "/work/compile.out", "/work/compile.err")
    event("compile", result, "/work/compile.out", "/work/compile.err", limit)
    compiled = not (result.get("spawn_error") or result.get("timed_out")
                    or result.get("returncode") != 0)
    if not compiled:
        emit({"type": "done", "tests": 0, "compiled": False,
              "source_compiled": True, "diag": diagnostic})
        return

    for index in range(total):
        result = run(["/work/program", str(index)], test_timeout,
                     "/work/test.out", "/work/test.err")
        event("test", result, "/work/test.out", "/work/test.err", limit, index=index)

    if diagnostic and max_cases > 0:
        diag_std = "-std=gnu++17" if language == "cpp" else "-std=gnu11"
        diag_source = "/input/program_diag.cpp" if language == "cpp" else "/input/program_diag.c"
        diag_cmd = [compiler, diag_std, "-O0", "-w", diag_source,
                    "-o", "/work/program_diag", "-lm"]
        diag_result = run(diag_cmd, compile_timeout,
                          "/work/diag_compile.out", "/work/diag_compile.err")
        event("diag_compile", diag_result, "/work/diag_compile.out",
              "/work/diag_compile.err", limit)
        if not (diag_result.get("spawn_error") or diag_result.get("timed_out")
                or diag_result.get("returncode") != 0):
            run_diag_cases("/work/program_diag", max_cases, test_timeout, limit)
        else:
            emit({"type": "diag_done", "status": "DIAG_COMPILE_ERROR",
                  "cases": 0, "max_cases": max_cases, "failures": []})

    emit({"type": "done", "tests": total, "compiled": True, "diag": diagnostic})


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
"""


def extract_c(text: str) -> str:
    """Strip optional Markdown code fences without altering the actual source."""
    text = text.strip()
    match = re.search(r'```(?:c|cpp|c\+\+)?\s*\n(.*?)```', text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def harness_mode(tests: list[dict]) -> str:
    return 'program' if len(tests) == 1 and 'program' in tests[0] else 'cases'


HARNESS_NOTE_PROGRAM = ('完整 harness：Test 0 是整套测试程序的一次执行，'
                        '不是第 1 个 assert 断言。官方 RE 以该次执行为准；'
                        'TCP 由独立诊断用例单独计算（不可提取时为 N/A）。')


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
            if isinstance(obj, dict) and obj.get('type') in (
                    'compile', 'source_compile', 'test', 'done', 'case',
                    'diag_compile', 'diag_done'):
                events.append(obj)
                continue
        noise.append(line)
    return events, noise


def _compile_event(events: list[dict]) -> dict | None:
    for ev in reversed(events):
        if ev.get('type') == 'compile':
            return ev
    return None


def _source_compile_event(events: list[dict]) -> dict | None:
    for ev in reversed(events):
        if ev.get('type') == 'source_compile':
            return ev
    return None


def _compile_fields(ev: dict) -> dict:
    return {
        'duration_ms': int(ev.get('duration_ms') or 0),
        'exit_code': ev.get('exit_code'),
        'signal': ev.get('signal'),
        'timed_out': bool(ev.get('timed_out')),
        'stdout': str(ev.get('stdout') or ''),
        'stderr': str(ev.get('stderr') or ''),
        'spawn_error': ev.get('spawn_error'),
    }


def _compiler_log(ev: dict) -> str:
    return _capped(
        ('--- compiler stdout ---\n' + _stream(ev, 'stdout') + '\n'
         '--- compiler stderr ---\n' + _stream(ev, 'stderr')), 20000)


def outcome_from_events(events: list[dict], total_tests: int, mode: str,
                        raw_noise: str = '', stderr_text: str = '',
                        log_limit: int = 5000) -> EvalOutcome:
    """Pure classification of structured runner events (also used by unit tests).

    RC follows the official LLM4Decompile protocol: a candidate is counted as
    compiled when ``c_include + c_func_decompile`` compiles by itself.  The
    combined candidate+harness compile is tracked separately as
    ``compile_success`` and gates RE/TCP.
    """
    note = HARNESS_NOTE_PROGRAM if mode == 'program' else ''
    source_ev = _source_compile_event(events)
    source_ok = None
    source_log = ''

    # Stage 1, official RC: candidate function alone.
    if source_ev is not None:
        src = _compile_fields(source_ev)
        source_log = _compiler_log(source_ev)
        if src['spawn_error']:
            return EvalOutcome(
                'INFRA_ERROR', None, 0, total_tests,
                compile_log=source_log,
                error=f'Compiler could not start (source stage): {src["spawn_error"]}',
                compile_ms=src['duration_ms'], compile_stdout=src['stdout'],
                compile_stderr=src['stderr'], compile_exit_code=src['exit_code'],
                compile_signal=src['signal'], compile_timed_out=src['timed_out'],
                harness_mode=mode, harness_note=note,
                source_compile_success=None, compile_stage='source')
        if src['timed_out']:
            return EvalOutcome(
                'COMPILE_TIMEOUT', False, 0, total_tests,
                compile_log=source_log,
                error='Candidate-only compilation timed out.',
                compile_ms=src['duration_ms'], compile_stdout=src['stdout'],
                compile_stderr=src['stderr'], compile_exit_code=src['exit_code'],
                compile_signal=src['signal'], compile_timed_out=True,
                harness_mode=mode, harness_note=note,
                source_compile_success=False, compile_stage='source')
        if src['exit_code'] not in (0, None) or src['signal']:
            return EvalOutcome(
                'COMPILE_ERROR', False, 0, total_tests,
                compile_log=source_log,
                error='Candidate-only compilation failed.',
                compile_ms=src['duration_ms'], compile_stdout=src['stdout'],
                compile_stderr=src['stderr'], compile_exit_code=src['exit_code'],
                compile_signal=src['signal'], compile_timed_out=False,
                harness_mode=mode, harness_note=note,
                source_compile_success=False, compile_stage='source')
        source_ok = True

    # Stage 2, full harness compile + execution for RE/TCP.
    compile_ev = _compile_event(events)
    if compile_ev is None:
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            run_log=_capped(raw_noise, 4000),
            error='Sandbox produced no harness compile event.',
            harness_mode=mode, harness_note=note,
            source_compile_success=source_ok, compile_stage='harness')
    full = _compile_fields(compile_ev)
    compile_log = _compiler_log(compile_ev)
    stage = 'harness' if source_ev is not None else ''
    if full['spawn_error']:
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            compile_log=compile_log,
            error=f'Compiler could not start (harness stage): {full["spawn_error"]}',
            compile_ms=full['duration_ms'], compile_stdout=full['stdout'],
            compile_stderr=full['stderr'], compile_exit_code=full['exit_code'],
            compile_signal=full['signal'], compile_timed_out=full['timed_out'],
            harness_mode=mode, harness_note=note,
            source_compile_success=source_ok, compile_stage=stage)
    if full['timed_out']:
        return EvalOutcome(
            'COMPILE_TIMEOUT', False, 0, total_tests,
            compile_log=compile_log, error='Harness compilation timed out.',
            compile_ms=full['duration_ms'], compile_stdout=full['stdout'],
            compile_stderr=full['stderr'], compile_exit_code=full['exit_code'],
            compile_signal=full['signal'], compile_timed_out=True,
            harness_mode=mode, harness_note=note,
            source_compile_success=source_ok, compile_stage=stage)
    if full['exit_code'] not in (0, None) or full['signal']:
        return EvalOutcome(
            'COMPILE_ERROR', False, 0, total_tests,
            compile_log=compile_log, error='Harness compilation failed.',
            compile_ms=full['duration_ms'], compile_stdout=full['stdout'],
            compile_stderr=full['stderr'], compile_exit_code=full['exit_code'],
            compile_signal=full['signal'], compile_timed_out=False,
            harness_mode=mode, harness_note=note,
            source_compile_success=source_ok, compile_stage=stage)

    test_events = {int(ev.get('index', 0)): ev for ev in events if ev.get('type') == 'test'}
    if set(test_events) != set(range(total_tests)):
        return EvalOutcome(
            'INFRA_ERROR', None, 0, total_tests,
            compile_log=compile_log, run_log=_capped(raw_noise, 4000),
            error='Missing testcase events; the sandbox probably stopped.',
            compile_ms=full['duration_ms'], compile_stdout=full['stdout'],
            compile_stderr=full['stderr'], compile_exit_code=full['exit_code'],
            compile_signal=full['signal'], compile_timed_out=full['timed_out'],
            harness_mode=mode, harness_note=note,
            source_compile_success=source_ok,
            compile_stage='ok' if source_ev is not None else '')
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
    case_events = [ev for ev in events if ev.get('type') == 'case']
    diag_done = next((ev for ev in reversed(events) if ev.get('type') == 'diag_done'), None)
    coverage = {}
    if diag_done:
        coverage = {'executed': int(diag_done.get('cases') or 0),
                    'max_cases': int(diag_done.get('max_cases') or 0),
                    'failures': diag_done.get('failures') if isinstance(diag_done.get('failures'), list) else []}
    return EvalOutcome(
        status, True, passed, total_tests,
        compile_log=compile_log, run_log=run_log, error=reason,
        compile_ms=full['duration_ms'], test_ms=test_ms,
        compile_exit_code=full['exit_code'], compile_signal=full['signal'],
        compile_timed_out=full['timed_out'],
        compile_stdout=full['stdout'], compile_stderr=full['stderr'],
        tests=records, harness_mode=mode, harness_note=note,
        failure_reason=reason, cases=case_events,
        diagnostic_status=(diag_done or {}).get('status', ''),
        case_coverage=coverage,
        source_compile_success=source_ok,
        compile_stage='ok' if source_ev is not None else '')


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
                    docker_bin: str = 'docker', diagnostic: dict | None = None,
                    max_cases: int = 0) -> list[str]:
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
        '1' if diagnostic else '0', str(max_cases),
    ]


def _prepare_input(candidate: str, dependencies: str, tests: list[dict],
                   language: str, temp: str, diagnostic: dict | None = None) -> Path:
    path = Path(temp)
    source = build_c_file(candidate, dependencies, tests)
    src = path / ('program.cpp' if language == 'cpp' else 'program.c')
    src.write_text(source, encoding='utf-8')
    only_src = path / ('program_only.cpp' if language == 'cpp' else 'program_only.c')
    # Official c_include combines headers from the reference function and the
    # test harness.  Add harness includes to the candidate-only source too.
    harness_includes = []
    for test in tests:
        for text in (test.get('program') or '', test.get('code') or ''):
            for line in text.splitlines():
                if '#include' in line and line.strip() not in harness_includes:
                    harness_includes.append(line.strip())
    only_source = dependencies + '\n' + '\n'.join(harness_includes) + '\n' + candidate + '\n'
    only_src.write_text(only_source, encoding='utf-8')
    (path / 'runner.py').write_text(RUNNER_SOURCE, encoding='utf-8')
    if diagnostic and diagnostic.get('harness'):
        harness = diagnostic['harness']
        diag_source = dependencies + '\n' + candidate + '\n' + harness + '\n'
        diag = path / ('program_diag.cpp' if language == 'cpp' else 'program_diag.c')
        diag.write_text(diag_source, encoding='utf-8')
        header = diagnostic.get('header') or ''
        if header:
            (path / 'dec_diag.h').write_text(header, encoding='utf-8')
            os.chmod(path / 'dec_diag.h', 0o644)
        os.chmod(diag, 0o644)
    # Read access for unprivileged uid=65534 inside a read-only bind mount.
    os.chmod(path, 0o755)
    os.chmod(src, 0o644)
    os.chmod(only_src, 0o644)
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
                  docker_bin: str = 'docker',
                  diagnostic: dict | None = None) -> EvalOutcome:
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
    tmp_root = os.getenv('DEC_EVAL_TMP_DIR') or os.getenv('TMPDIR') or None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='deceval_', dir=tmp_root) as temp:
        path = _prepare_input(candidate, dependencies, tests, language, temp,
                              diagnostic=diagnostic)
        max_cases = int((diagnostic or {}).get('max_cases') or 0)
        cmd = _docker_command(path, language, len(tests), compile_timeout,
                              test_timeout, log_limit, docker_bin,
                              diagnostic=diagnostic, max_cases=max_cases)
        overall = min(MAX_CONTAINER_LIFETIME,
                      compile_timeout + test_timeout * (len(tests) + max_cases) + 120)
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
                                     len(tests), mode, log_limit, bool(diagnostic))


async def evaluate_code_async(prediction: str, dependencies: str, tests: list[dict],
                              language: str = 'c', *,
                              compile_timeout: int = COMPILE_TIMEOUT,
                              test_timeout: int = TEST_TIMEOUT,
                              log_limit: int = LOG_LIMIT_BYTES,
                              docker_bin: str = 'docker',
                              diagnostic: dict | None = None) -> EvalOutcome:
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
    tmp_root = os.getenv('DEC_EVAL_TMP_DIR') or os.getenv('TMPDIR') or None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='deceval_', dir=tmp_root) as temp:
        path = _prepare_input(candidate, dependencies, tests, language, temp,
                              diagnostic=diagnostic)
        max_cases = int((diagnostic or {}).get('max_cases') or 0)
        cmd = _docker_command(path, language, len(tests), compile_timeout,
                              test_timeout, log_limit, docker_bin,
                              diagnostic=diagnostic, max_cases=max_cases)
        overall = min(MAX_CONTAINER_LIFETIME,
                      compile_timeout + test_timeout * (len(tests) + max_cases) + 120)
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
                                     len(tests), mode, log_limit, bool(diagnostic))


def _outcome_from_process(returncode: int | None, stdout: bytes | str | None,
                          stderr: bytes | str | None, total_tests: int,
                          mode: str, log_limit: int,
                          diagnostic_requested: bool = False) -> EvalOutcome:
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
    outcome = outcome_from_events(events, total_tests, mode,
                                   raw_noise='\n'.join(noise), stderr_text=err,
                                   log_limit=log_limit)
    if diagnostic_requested and not outcome.diagnostic_status:
        if outcome.status in ('COMPILE_ERROR', 'COMPILE_TIMEOUT'):
            outcome.diagnostic_status = 'NOT_RUN_COMPILE_ERROR'
        elif not outcome.cases:
            outcome.diagnostic_status = 'NO_CASES_EXECUTED'
    return outcome
