"""Optional host-gcc fast evaluation path.

This is intentionally NOT a security boundary.  It exists for trusted local
experiments where Docker startup dominates runtime.  The default execution mode
remains Docker; host mode must be enabled by configuration and explicitly
acknowledged for each run.
"""
from __future__ import annotations

import os
import resource
import shutil
from functools import lru_cache
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from .config import (
    COMPILE_TIMEOUT, HOST_CC, HOST_CXX, LOG_LIMIT_BYTES, MAX_SOURCE_CHARS,
    TEST_TIMEOUT,
)
from .sandbox import (EvalOutcome, build_c_file, extract_c, harness_mode,
                      outcome_from_events)


@lru_cache(maxsize=8)
def _compiler_version(executable: str) -> dict:
    path = shutil.which(executable) if not os.path.isabs(executable) else executable
    if not path or not os.path.isfile(path) or not os.access(path, os.X_OK):
        return {'path': path or executable, 'version': '', 'available': False,
                'error': f'compiler not found: {executable}'}
    try:
        proc = subprocess.run([path, '--version'], capture_output=True, text=True,
                              timeout=5, check=False)
        first = (proc.stdout or proc.stderr or '').strip().splitlines()
        if proc.returncode != 0 or not first:
            return {'path': path, 'version': '', 'available': False,
                    'error': f'compiler --version failed: exit {proc.returncode}'}
        return {'path': path, 'version': first[0][:300], 'available': True, 'error': ''}
    except Exception as exc:
        return {'path': path, 'version': '', 'available': False,
                'error': f'{type(exc).__name__}: {exc}'}


def host_toolchain() -> dict:
    cc = _compiler_version(HOST_CC)
    cxx = _compiler_version(HOST_CXX)
    return {
        'available': bool(cc['available'] and cxx['available']),
        'engine': 'host',
        'cc': {'name': HOST_CC, **cc},
        'cxx': {'name': HOST_CXX, **cxx},
        'cc_version': cc['version'],
        'cxx_version': cxx['version'],
    }


def _limits(test_timeout: int):
    def apply():
        try:
            resource.setrlimit(resource.RLIMIT_CPU,
                               (max(1, int(test_timeout) + 1),
                                max(2, int(test_timeout) + 2)))
            resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024,) * 2)
            resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))
        except Exception:
            pass
    return apply


def _run(argv: list[str], cwd: str, timeout: float, extra_env: dict | None = None) -> dict:
    started = time.perf_counter()
    env = {
        'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
        'HOME': cwd,
        'TMPDIR': cwd,
        'LANG': 'C',
        'LC_ALL': 'C',
    }
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    try:
        proc = subprocess.Popen(
            argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True, shell=False,
            preexec_fn=_limits(timeout))
    except Exception as exc:
        return {'spawn_error': f'{type(exc).__name__}: {exc}', 'timed_out': False,
                'returncode': None, 'stdout': '', 'stderr': '',
                'duration_ms': int((time.perf_counter() - started) * 1000)}
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try: proc.kill()
            except OSError: pass
        out, err = proc.communicate()
    stdout = (out or b'').decode('utf-8', 'replace')
    stderr = (err or b'').decode('utf-8', 'replace')
    exit_code = None if proc.returncode is None or proc.returncode < 0 else proc.returncode
    return {
        'returncode': proc.returncode,
        'exit_code': exit_code,
        'signal': -proc.returncode if proc.returncode is not None and proc.returncode < 0 else None,
        'stdout': stdout[:max(20000, LOG_LIMIT_BYTES)],
        'stderr': stderr[:max(20000, LOG_LIMIT_BYTES)],
        'stdout_truncated': len(stdout) > max(20000, LOG_LIMIT_BYTES),
        'stderr_truncated': len(stderr) > max(20000, LOG_LIMIT_BYTES),
        'timed_out': timed_out,
        'duration_ms': int((time.perf_counter() - started) * 1000),
    }


def _read_records(path: str, limit: int = 16384) -> list[dict]:
    try:
        with open(path, 'rb') as fp:
            data = fp.read()
    except OSError:
        return []
    fields = data.split(b'\0')
    records = []
    i = 0

    def decode(value):
        return value.decode('utf-8', 'replace')[:limit]

    while i + 11 < len(fields):
        if fields[i + 11] != b'END':
            i += 1
            continue
        try:
            records.append({
                'index': int(fields[i]), 'static_id': int(fields[i + 1]),
                'line': int(fields[i + 2]), 'passed': fields[i + 3] == b'1',
                'actual_ok': fields[i + 4] == b'1',
                'expected_ok': fields[i + 5] == b'1',
                'expression': decode(fields[i + 6]),
                'lhs': decode(fields[i + 7]), 'rhs': decode(fields[i + 8]),
                'actual': decode(fields[i + 9]),
                'expected': decode(fields[i + 10]),
            })
        except ValueError:
            i += 12
            continue
        i += 12
    return records


def _run_diagnostics(diag_source: str, header: str, max_cases: int,
                     temp: str, language: str, test_timeout: int) -> list[dict]:
    diag_path = Path(temp) / ('program_diag.cpp' if language == 'cpp' else 'program_diag.c')
    header_path = Path(temp) / 'dec_diag.h'
    # The rewritten harness is generated for Docker's /input mount.
    diag_source = diag_source.replace('/input/dec_diag.h', 'dec_diag.h')
    diag_path.write_text(diag_source, encoding='utf-8')
    header_path.write_text(header, encoding='utf-8')
    compiler = HOST_CXX if language == 'cpp' else HOST_CC
    std = '-std=gnu++17' if language == 'cpp' else '-std=gnu11'
    exe = Path(temp) / 'program_diag'
    compile_result = _run([compiler, std, '-O0', '-w', str(diag_path),
                           '-o', str(exe), '-lm'], temp, max(10, test_timeout))
    events = [{'type': 'diag_compile', **compile_result}]
    if (compile_result.get('spawn_error') or compile_result.get('timed_out')
            or compile_result.get('returncode') != 0):
        events.append({'type': 'diag_done', 'status': 'DIAG_COMPILE_ERROR',
                       'cases': 0, 'max_cases': max_cases, 'failures': []})
        return events
    records_path = Path(temp) / 'dec_cases.bin'
    failures: list[int] = []
    k = 1
    status = 'OK'
    while k <= max_cases:
        try: records_path.unlink()
        except OSError: pass
        result = _run([str(exe)], temp, test_timeout, extra_env={
            'DEC_DIAG_STOP': k,
            'DEC_DIAG_SKIP': ','.join(str(x) for x in failures),
            'DEC_DIAG_PATH': str(records_path),
        })
        records = _read_records(str(records_path))
        prior = [r for r in records if not r['passed'] and r['index'] < k
                 and r['index'] not in failures]
        if prior:
            failures = sorted(set(failures + [min(r['index'] for r in prior)]))
            continue
        record = next((r for r in records if r['index'] == k), None)
        if record is not None:
            events.append({
                'type': 'case', 'index': k, 'static_id': record['static_id'],
                'line': record['line'], 'expression': record['expression'],
                'lhs': record['lhs'], 'rhs': record['rhs'],
                'actual': record['actual'], 'expected': record['expected'],
                'actual_ok': record['actual_ok'], 'expected_ok': record['expected_ok'],
                'passed': record['passed'],
                'after_prior_failure': bool(failures) and k > min(failures),
                'duration_ms': result.get('duration_ms', 0),
                'exit_code': result.get('exit_code'), 'signal': result.get('signal'),
                'timed_out': result.get('timed_out', False),
                'stdout': result.get('stdout', ''), 'stderr': result.get('stderr', ''),
            })
            if not record['passed']:
                failures = sorted(set(failures + [k]))
            k += 1
            continue
        if result.get('timed_out'):
            status = 'TIMEOUT'; break
        rc = result.get('returncode')
        if rc not in (0, 3):
            status = 'CRASH'; break
        if k == 1:
            status = 'NO_CASES_EXECUTED'
        break
    else:
        status = 'SCAN_LIMIT'
    events.append({'type': 'diag_done', 'status': status, 'cases': k - 1,
                   'max_cases': max_cases, 'failures': failures})
    return events


def evaluate_code_host(prediction: str, dependencies: str, tests: list[dict],
                       language: str = 'c', *,
                       compile_timeout: int = COMPILE_TIMEOUT,
                       test_timeout: int = TEST_TIMEOUT,
                       log_limit: int = LOG_LIMIT_BYTES,
                       diagnostic: dict | None = None) -> EvalOutcome:
    candidate = extract_c(prediction)
    if not candidate:
        return EvalOutcome('MODEL_EMPTY', False, 0, len(tests),
                           error='Model output was empty.')
    if not tests:
        return EvalOutcome('INFRA_ERROR', None, 0, 0,
                           error='No executable testcases are attached to this sample.')
    source = build_c_file(candidate, dependencies, tests)
    if len(source) > MAX_SOURCE_CHARS:
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests),
                           error='Generated source exceeds MVP size limit.')
    mode = harness_mode(tests)
    tmp_root = os.getenv('DEC_EVAL_TMP_DIR') or os.getenv('TMPDIR') or None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    compiler = HOST_CXX if language == 'cpp' else HOST_CC
    std = '-std=c++17' if language == 'cpp' else '-std=c11'
    with tempfile.TemporaryDirectory(prefix='deceval_host_', dir=tmp_root) as temp:
        src = Path(temp) / ('program.cpp' if language == 'cpp' else 'program.c')
        exe = Path(temp) / 'program'
        src.write_text(source, encoding='utf-8')
        events: list[dict] = []

        # Stage 1: official RC = candidate-only compile (include headers from
        # the harness, but not the harness code).  Keep the same GNU11/GCC14
        # compatibility profile as the Docker source stage.
        harness_includes = []
        for test in tests:
            for text in (test.get('program') or '', test.get('code') or ''):
                for line in text.splitlines():
                    if '#include' in line and line.strip() not in harness_includes:
                        harness_includes.append(line.strip())
        only_source = (dependencies + '\n' + '\n'.join(harness_includes)
                       + '\n' + candidate + '\n')
        only_src = Path(temp) / ('program_only.cpp' if language == 'cpp' else 'program_only.c')
        only_src.write_text(only_source, encoding='utf-8')
        only_std = '-std=gnu++17' if language == 'cpp' else '-std=gnu11'
        compat = [] if language == 'cpp' else [
            '-D_GNU_SOURCE',
            '-Wno-error=implicit-function-declaration',
            '-Wno-error=incompatible-pointer-types',
            '-Wno-error=int-conversion',
        ]
        source_result = _run(
            [compiler, only_std, *compat, '-O0', '-S', str(only_src),
             '-o', str(Path(temp) / 'program_only.s'), '-lm'],
            temp, compile_timeout)
        source_event = {'type': 'source_compile', **source_result,
                        'duration_ms': source_result.get('duration_ms', 0)}
        events.append(source_event)
        if (source_result.get('spawn_error') or source_result.get('timed_out')
                or source_result.get('returncode') != 0):
            return outcome_from_events(events, len(tests), mode,
                                       log_limit=log_limit)

        # Stage 2: full harness compile + execution for RE/TCP.
        compile_result = _run(
            [compiler, std, '-O0', '-w', str(src), '-o', str(exe), '-lm'],
            temp, compile_timeout)
        compile_event = {'type': 'compile', **compile_result,
                         'duration_ms': compile_result.get('duration_ms', 0)}
        # Strip keys that the event parser does not expect but keep the useful ones.
        events.append(compile_event)
        if (compile_result.get('spawn_error') or compile_result.get('timed_out')
                or compile_result.get('returncode') != 0):
            return outcome_from_events(events, len(tests), mode,
                                       log_limit=log_limit)
        for index in range(len(tests)):
            result = _run([str(exe), str(index)], temp, test_timeout)
            events.append({'type': 'test', 'index': index, **result})
        if diagnostic and diagnostic.get('harness') and int(diagnostic.get('max_cases') or 0) > 0:
            diag_source = dependencies + '\n' + candidate + '\n' + diagnostic['harness'] + '\n'
            events.extend(_run_diagnostics(diag_source, diagnostic.get('header') or '',
                                           int(diagnostic['max_cases']), temp,
                                           language, test_timeout))
        events.append({'type': 'done', 'tests': len(tests)})
        return outcome_from_events(events, len(tests), mode,
                                   log_limit=log_limit)


async def evaluate_code_host_async(prediction: str, dependencies: str,
                                   tests: list[dict], **kwargs) -> EvalOutcome:
    import asyncio
    return await asyncio.to_thread(
        evaluate_code_host, prediction, dependencies, tests, **kwargs)
