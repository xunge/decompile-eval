#!/usr/bin/env python3
"""Run reference C/C++ functions + original tests through the same Docker sandbox.

Reference validation must happen before quoting RC/RE on a newly converted
dataset: if the dataset's own reference source cannot pass its harness, the
failure is a dataset/environment problem, not a model failure.

The full structured result (classification, exit codes, signals, timings,
stdout/stderr) is written to a JSON report next to the dataset unless --out is
given.  Exit status is non-zero when at least one reference sample fails.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / 'backend'
sys.path.insert(0, str(BACKEND))
from app.sandbox import evaluate_code  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset', help='Normalized JSONL')
    p.add_argument('--limit', type=int, default=10, help='0 means all samples')
    p.add_argument('--out', default='', help='JSON report path (default: data/reports/...)')
    p.add_argument('--compile-timeout', type=int, default=60)
    p.add_argument('--test-timeout', type=int, default=10)
    args = p.parse_args()

    source = Path(args.dataset)
    out_path = Path(args.out) if args.out else (
        Path('data/reports') / f'{source.stem}_reference.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    records = []
    for index, line in enumerate(source.open(encoding='utf-8'), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        result = evaluate_code(
            row['reference_source'], row.get('dependencies', ''), row['tests'],
            language=row.get('language', 'c'),
            compile_timeout=args.compile_timeout, test_timeout=args.test_timeout)
        counts[result.status] += 1
        records.append({
            'sample_id': row.get('sample_id'),
            'status': result.status,
            'failure_reason': result.failure_reason,
            'compile_success': result.compile_success,
            'tests_passed': result.tests_passed,
            'tests_total': result.tests_total,
            'compile_ms': result.compile_ms,
            'test_ms': result.test_ms,
            'compile_log': result.compile_log,
            'run_log': result.run_log,
            'detail': json.loads(result.detail()),
        })
        reason = result.failure_reason or result.error or ''
        print(f"{row.get('sample_id')}: {result.status} {reason[:200]}")
        if args.limit and sum(counts.values()) >= args.limit:
            break

    report = {
        'dataset': str(source.resolve()),
        'limit': args.limit,
        'summary': dict(counts),
        'samples': records,
    }
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Summary:', dict(counts))
    print('Report:', out_path)
    if counts['TEST_PASS'] == 0:
        raise SystemExit('No reference harness passed: inspect upstream test semantics '
                         'before treating model failures as real.')


if __name__ == '__main__':
    main()
