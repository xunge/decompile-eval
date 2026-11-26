#!/usr/bin/env python3
"""Docker-side ExeBench wrapper runner (runs offline, no network)."""
import json
import os
import subprocess
import sys
import time

INPUT = sys.argv[1]
WORK = sys.argv[2]
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)


def emit(obj):
    print(json.dumps(obj, ensure_ascii=False, default=str), flush=True)


def run(cmd, timeout, **kwargs):
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              **kwargs)
        return {
            'returncode': proc.returncode,
            'stdout': (proc.stdout or '')[:20000],
            'stderr': (proc.stderr or '')[:20000],
            'timed_out': False,
            'duration_ms': int((time.perf_counter() - start) * 1000),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            'returncode': None,
            'stdout': (exc.stdout or '')[:20000] if isinstance(exc.stdout, str) else '',
            'stderr': (exc.stderr or '')[:20000] if isinstance(exc.stderr, str) else '',
            'timed_out': True,
            'duration_ms': int((time.perf_counter() - start) * 1000),
        }
    except Exception as exc:
        return {'returncode': None, 'stdout': '', 'stderr': str(exc)[:2000],
                'timed_out': False, 'duration_ms': int((time.perf_counter() - start) * 1000),
                'spawn_error': f'{type(exc).__name__}: {exc}'}


variants = json.load(open(os.path.join(INPUT, 'variants.json'), encoding='utf-8'))
wrapper = open(os.path.join(INPUT, 'wrapper.cpp'), encoding='utf-8').read()
open('wrapper.cpp', 'w', encoding='utf-8').write(wrapper)

for variant in variants:
    group = variant['group']
    open('candidate.c', 'w', encoding='utf-8').write(variant['source'])
    compile_cmd = [
        'g++', '-fpermissive', '-O0', '-w',
        '-I', os.path.join(INPUT, 'include'),
        '-I', os.path.join(INPUT, 'include', 'clib'),
        'wrapper.cpp', os.path.join(INPUT, 'synthesizer.c'),
        '-o', 'program', '-lm',
    ]
    compiled = run(compile_cmd, float(variant['compile_timeout']))
    emit({'type': 'compile', 'group': group, **compiled})
    if compiled.get('spawn_error') or compiled.get('timed_out') or compiled.get('returncode') != 0:
        continue
    for test in variant['tests']:
        index = int(test['index'])
        in_path = f'input_{index}.json'
        out_path = f'output_{index}.json'
        try:
            os.remove(out_path)
        except OSError:
            pass
        with open(in_path, 'w', encoding='utf-8') as fp:
            json.dump(test.get('input', {}), fp, ensure_ascii=False)
        result = run(['./program', in_path, out_path], float(variant['test_timeout']))
        actual = None
        output_error = ''
        try:
            with open(out_path, encoding='utf-8') as fp:
                actual = json.load(fp)
        except Exception as exc:
            output_error = f'{type(exc).__name__}: {exc}'
        emit({'type': 'test', 'group': group, 'index': index, 'actual': actual,
              'output_error': output_error, **result})

emit({'type': 'done', 'groups': len(variants)})
