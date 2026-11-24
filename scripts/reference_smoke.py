#!/usr/bin/env python3
"""Run reference C functions + original tests through the same Docker sandbox.

This must be run before quoting RC/RE on newly converted datasets. Without
oracle validation, harness failures could be counted as model failures.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / 'backend'
sys.path.insert(0, str(BACKEND))
from app.sandbox import evaluate_code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',help='Normalized JSONL')
    p.add_argument('--limit',type=int,default=10)
    args=p.parse_args()
    counts=Counter()
    for index,line in enumerate(Path(args.dataset).open(encoding='utf-8'),1):
        if not line.strip(): continue
        row=json.loads(line)
        result=evaluate_code(row['reference_source'], row.get('dependencies',''), row['tests'],language=row.get('language','c'))
        counts[result.status]+=1
        print(f"{row['sample_id']}: {result.status} {result.error or result.compile_log[:180]}")
        if args.limit and sum(counts.values())>=args.limit: break
    print('Summary:',dict(counts))
    if counts['TEST_PASS']==0:
        raise SystemExit('No reference harness passed: inspect upstream test semantics before real evaluation.')


if __name__=='__main__':
    main()
