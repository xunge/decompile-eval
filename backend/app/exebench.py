"""ExeBench I/O-wrapper execution adapter.

ExeBench functions are executed by compiling a C++ wrapper (provided in the
dataset) together with the candidate/reference C source and a vendored copy of
nlohmann/json + clib/synthesizer.  Each test writes input.json, runs
``program input.json output.json`` and reads output.json.  Comparison follows
the official ``diff_io`` semantics.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from .config import (
    MAX_SOURCE_CHARS, SANDBOX_IMAGE, COMPILE_TIMEOUT, TEST_TIMEOUT,
    LOG_LIMIT_BYTES,
)
from .sandbox import EvalOutcome, extract_c

_DEPS = Path(__file__).with_name('exebench_deps')
_RUNNER = Path(__file__).with_name('exebench_runner.py')


def sample_signature(sample: dict) -> str:
    h = hashlib.sha256()
    for key in ('reference_source', 'dependencies', 'test_wrapper',
                'tests_json', 'real_iospec', 'language', 'compiler', 'optimization'):
        h.update(str(sample.get(key) or '').encode('utf-8'))
        h.update(b'\0')
    return h.hexdigest()


def normalize_exebench_io(value: Any) -> Any:
    """Decode legacy {'var': [...], 'value': [...]} I/O records."""
    if isinstance(value, dict) and 'var' in value and 'value' in value:
        keys, vals = value['var'], value['value']
        if isinstance(keys, list) and isinstance(vals, list) and len(keys) == len(vals):
            return {str(k): _decode(v) for k, v in zip(keys, vals)}
    if isinstance(value, list) and all(isinstance(x, dict) and 'var' in x for x in value):
        return {str(x['var']): _decode(x.get('value')) for x in value}
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def normalize_tests(tests: Any) -> list[dict]:
    """Normalize unified ExeBench tests and legacy testcases."""
    if not isinstance(tests, list):
        return []
    out = []
    for i, item in enumerate(tests):
        if not isinstance(item, dict):
            continue
        raw_input = item.get('input', item.get('args'))
        if raw_input is None and 'input_json' in item:
            raw_input = item['input_json']
        expected = item.get('expected_output')
        if expected is None and 'output' in item:
            expected = item['output']
        if expected is None and 'expected' in item:
            expected = item['expected']
        if raw_input is None or expected is None:
            continue
        out.append({
            'name': str(item.get('name') or f'test_{i}')[:100],
            'input': normalize_exebench_io(raw_input),
            'expected_output': normalize_exebench_io(expected),
            'dummy_funcs': str(item.get('dummy_funcs') or ''),
            'dummy_funcs_seed': item.get('dummy_funcs_seed'),
            'case_index': i,
        })
    return out


def rewrite_wrapper(wrapper: str) -> str:
    """Replace the dataset's temporary candidate include with candidate.c."""
    pattern = re.compile(r'#include\s+["\'][^"\']*(?:pytmpfile|/tmp/)[^"\']*\.c["\']')
    replaced, count = pattern.subn('#include "candidate.c"', wrapper, count=1)
    if count:
        return replaced
    if re.search(r'#include\s+["\']candidate\.c["\']', wrapper):
        return wrapper
    return 'extern "C" {\n#include "candidate.c"\n}\n' + wrapper


def diff_io(observed: Any, expected: Any) -> bool:
    """Exact official ExeBench diff_io implementation."""
    if type(observed) is not type(expected):
        return False
    if isinstance(observed, list):
        if len(observed) != len(expected):
            return False
        return all(diff_io(a, b) for a, b in zip(observed, expected))
    if isinstance(observed, dict):
        for key in observed:
            if key not in expected or not diff_io(observed[key], expected[key]):
                return False
        return True
    if isinstance(observed, float):
        return math.isclose(observed, expected)
    return observed == expected


def diff_detail(observed: Any, expected: Any, path: str = '$') -> str:
    if type(observed) is not type(expected):
        return f'{path}: type {type(observed).__name__} != {type(expected).__name__}'
    if isinstance(observed, list):
        if len(observed) != len(expected):
            return f'{path}: length {len(observed)} != {len(expected)}'
        for i, (a, b) in enumerate(zip(observed, expected)):
            if not diff_io(a, b):
                return diff_detail(a, b, f'{path}[{i}]')
        return ''
    if isinstance(observed, dict):
        for key in observed:
            if key not in expected:
                return f'{path}.{key}: unexpected key'
            if not diff_io(observed[key], expected[key]):
                return diff_detail(observed[key], expected[key], f'{path}.{key}')
        return ''
    if isinstance(observed, float):
        return '' if math.isclose(observed, expected) else f'{path}: {observed!r} != {expected!r}'
    return '' if observed == expected else f'{path}: {observed!r} != {expected!r}'


def _group_tests(source: str, tests: list[dict], prefix: str = '') -> list[dict]:
    groups: dict[tuple[str, str], dict] = {}
    for test in tests:
        key = (str(test.get('dummy_funcs') or ''), json.dumps(
            test.get('dummy_funcs_seed'), ensure_ascii=False, sort_keys=True))
        group = groups.setdefault(key, {'dummy_funcs': key[0], 'tests': []})
        group['tests'].append(test)
    variants = []
    for i, group in enumerate(groups.values()):
        variants.append({
            'group': i,
            'source': prefix + group['dummy_funcs'] + '\n' + source + '\n',
            'tests': [{'index': t.get('case_index', j),
                       'input': t['input']} for j, t in enumerate(group['tests'])],
        })
    return variants


def _prepare_input(source: str, wrapper: str, tests: list[dict],
                   temp: str, compile_timeout: int, test_timeout: int,
                   prefix: str = '') -> Path:
    root = Path(temp)
    inp = root / 'input'
    inp.mkdir(parents=True)
    shutil.copytree(_DEPS, inp / 'include')
    shutil.copy2(_DEPS / 'clib' / 'synthesizer.c', inp / 'synthesizer.c')
    shutil.copy2(_RUNNER, inp / 'exebench_runner.py')
    (inp / 'wrapper.cpp').write_text(rewrite_wrapper(wrapper), encoding='utf-8')
    variants = _group_tests(source, tests, prefix)
    for variant in variants:
        variant['compile_timeout'] = compile_timeout
        variant['test_timeout'] = test_timeout
    (inp / 'variants.json').write_text(json.dumps(variants, ensure_ascii=False),
                                       encoding='utf-8')
    for path in [inp, inp / 'include', inp / 'synthesizer.c',
                 inp / 'exebench_runner.py', inp / 'wrapper.cpp', inp / 'variants.json']:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)
    return root


def _docker_command(inp: Path, docker_bin: str) -> list[str]:
    return [
        docker_bin, 'run', '--rm', '--network', 'none', '--read-only',
        '--pids-limit', '64', '--memory', '2g', '--cpus', '1',
        '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
        '--env', 'PYTHONDONTWRITEBYTECODE=1',
        '--user', '65534:65534', '--workdir', '/work',
        '--tmpfs', '/work:rw,exec,nosuid,nodev,size=512m,mode=1777',
        '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m,mode=1777',
        '--mount', f'type=bind,source={inp},target=/input,readonly',
        SANDBOX_IMAGE, 'python3', '/input/exebench_runner.py', '/input', '/work',
    ]


def _parse_events(raw: str) -> tuple[list[dict], str]:
    events = []
    noise = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except ValueError:
            noise.append(line)
            continue
        if isinstance(item, dict) and item.get('type') in ('compile', 'test', 'done'):
            events.append(item)
        else:
            noise.append(line)
    return events, '\n'.join(noise[-100:])


def _classify(events: list[dict], tests: list[dict], source: str,
              prefix: str = '') -> EvalOutcome:
    compiles = {int(e['group']): e for e in events if e.get('type') == 'compile'}
    test_events = {(int(e.get('group', -1)), int(e.get('index', -1))): e
                   for e in events if e.get('type') == 'test'}
    groups = _group_tests(source, tests, prefix)
    records = []
    compiled_all = bool(groups)
    link_error = False
    compile_logs = []
    for group in groups:
        gi = group['group']
        ev = compiles.get(gi)
        ok = bool(ev and not ev.get('spawn_error') and not ev.get('timed_out')
                  and ev.get('returncode') == 0)
        compiled_all = compiled_all and ok
        if ev:
            compile_logs.append(
                f'--- group {gi} stdout ---\n{ev.get("stdout", "")}\n'
                f'--- group {gi} stderr ---\n{ev.get("stderr", "")}')
            if not ok and any(mark in (ev.get('stderr') or '') for mark in
                              ('undefined reference', 'ld:', 'collect2:')):
                link_error = True
        for test in group['tests']:
            index = int(test['index'])
            ev = test_events.get((gi, index))
            expected = tests[index]['expected_output'] if index < len(tests) else None
            if not ok:
                records.append(_record(index, 'LINK_ERROR' if link_error else 'COMPILE_ERROR',
                                       tests[index].get('input') if index < len(tests) else None,
                                       expected, None, '', '', None, 0,
                                       'wrapper compilation failed'))
                continue
            if ev is None:
                records.append(_record(index, 'INFRA_ERROR',
                                       tests[index].get('input') if index < len(tests) else None,
                                       expected, None, '', '', None, 0,
                                       'missing wrapper test event'))
                continue
            actual = ev.get('actual')
            if ev.get('timed_out'):
                status, msg = 'TIMEOUT', 'wrapper execution timed out'
            elif ev.get('returncode') != 0:
                status, msg = 'RUNTIME_ERROR', f'exit {ev.get("returncode")}'
            elif ev.get('output_error'):
                status, msg = 'INVALID_OUTPUT', ev['output_error']
            elif not diff_io(actual, expected):
                status, msg = 'WRONG_ANSWER', diff_detail(actual, expected)
            else:
                status, msg = 'PASS', ''
            records.append(_record(index, status,
                                   tests[index].get('input') if index < len(tests) else None,
                                   expected, actual,
                                   ev.get('stdout', ''), ev.get('stderr', ''),
                                   ev.get('returncode'), ev.get('duration_ms', 0),
                                   msg))
    passed = sum(r['status'] == 'PASS' for r in records)
    if not compiled_all:
        status = 'LINK_ERROR' if link_error else 'COMPILE_ERROR'
    elif passed == len(records):
        status = 'TEST_PASS'
    elif any(r['status'] == 'TIMEOUT' for r in records):
        status = 'TIMEOUT'
    elif any(r['status'] == 'RUNTIME_ERROR' for r in records):
        status = 'RUNTIME_ERROR'
    elif any(r['status'] == 'INVALID_OUTPUT' for r in records):
        status = 'INVALID_OUTPUT'
    else:
        status = 'WRONG_ANSWER'
    reason = '; '.join(f'{r["name"]}: {r["error_message"]}' for r in records
                       if r['error_message'])[:800]
    return EvalOutcome(
        status, compiled_all, passed, len(records),
        compile_log='\n'.join(compile_logs)[:20000],
        run_log='\n'.join(f'{r["name"]}: {r["status"]} {r["error_message"]}' for r in records)[:20000],
        error=reason, compile_ms=sum(int(e.get('duration_ms') or 0) for e in compiles.values()),
        test_ms=sum(int(e.get('duration_ms') or 0) for e in test_events.values()),
        tests=records, harness_mode='exebench', failure_reason=reason,
        source_compile_success=compiled_all, compile_stage='exebench')


def _record(index: int, status: str, input_value: Any, expected: Any, actual: Any,
            stdout: str, stderr: str, exit_code: int | None, duration_ms: int,
            error: str) -> dict:
    return {
        'index': index, 'name': f'test_{index}', 'status': status,
        'passed': status == 'PASS', 'input': input_value,
        'expected_output': expected,
        'actual_output': actual, 'stdout': stdout or '', 'stderr': stderr or '',
        'exit_code': exit_code, 'duration_ms': int(duration_ms or 0),
        'error_message': error or None,
    }


def evaluate_exebench(candidate: str, reference_source: str, dependencies: str,
                      wrapper: str, tests: list[dict], *, function_signature: str = '',
                      language: str = 'c',
                      compile_timeout: int = COMPILE_TIMEOUT,
                      test_timeout: int = TEST_TIMEOUT,
                      docker_bin: str = 'docker',
                      log_limit: int = LOG_LIMIT_BYTES) -> EvalOutcome:
    candidate = extract_c(candidate or '')
    if not candidate:
        return EvalOutcome('MODEL_EMPTY', False, 0, len(tests),
                           error='Model output was empty.', harness_mode='exebench')
    if not tests:
        return EvalOutcome('INVALID_BENCHMARK', None, 0, 0,
                           error='ExeBench sample has no I/O tests.')
    if not wrapper:
        return EvalOutcome('INVALID_BENCHMARK', None, 0, len(tests),
                           error='ExeBench sample has no wrapper.')
    prefix = dependencies + '\n'
    source = candidate + '\n'
    if len(prefix + source) > MAX_SOURCE_CHARS:
        return EvalOutcome('COMPILE_ERROR', False, 0, len(tests),
                           error='Generated source exceeds size limit.')
    tmp_root = os.getenv('DEC_EVAL_TMP_DIR') or os.getenv('TMPDIR') or None
    if tmp_root:
        os.makedirs(tmp_root, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='decexebench_', dir=tmp_root) as temp:
        root = _prepare_input(source, wrapper, tests, temp,
                              compile_timeout, test_timeout, prefix=prefix)
        cmd = _docker_command(root / 'input', docker_bin)
        overall = min(3600, compile_timeout * max(1, len(_group_tests(source, tests)))
                      + test_timeout * len(tests) + 120)
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=overall, check=False)
        except FileNotFoundError:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='Docker command not found.')
        except subprocess.TimeoutExpired:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error='ExeBench Docker invocation timed out.')
        except PermissionError as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker permission error: {exc}')
        except Exception as exc:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'Docker error: {type(exc).__name__}: {exc}')
        events, noise = _parse_events(proc.stdout.decode('utf-8', 'replace'))
        if not events:
            return EvalOutcome('INFRA_ERROR', None, 0, len(tests),
                               error=f'No ExeBench events from sandbox: {noise[:1000]}',
                               compile_log=proc.stderr.decode('utf-8', 'replace')[-20000:])
        return _classify(events, tests, source, prefix)


async def evaluate_exebench_async(candidate: str, reference_source: str,
                                  dependencies: str, wrapper: str,
                                  tests: list[dict], **kwargs) -> EvalOutcome:
    return await asyncio.to_thread(
        evaluate_exebench, candidate, reference_source, dependencies,
        wrapper, tests, **kwargs)
