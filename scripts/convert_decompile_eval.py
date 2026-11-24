#!/usr/bin/env python3
"""Convert public LLM4Binary/decompile-eval HumanEval/MBPP Arrow rows to DecompileEval JSONL.

The upstream 'test' column is a full C/C++ test-harness program. It is placed
in `tests[0].program` and compiled after the candidate function; NEVER use it
in the model prompt. Oracle-check before publishing metrics.
"""
import argparse
import json
from pathlib import Path


def asm_text(raw):
    text = str(raw or '').strip()
    before = '# This is the assembly code:'
    after = '# What is the source code?'
    if before in text and after in text:
        text = text.split(before, 1)[1].split(after, 1)[0].strip()
    return text


def get_opt(x):
    opt = str(x or 'O0').strip().upper().lstrip('-')
    if opt not in ('O0', 'O1', 'O2', 'O3', 'OS', 'OFAST'):
        return None
    return 'Os' if opt == 'OS' else 'Ofast' if opt == 'OFAST' else opt


def convert_row(row, subset, pos, include_cpp=False):
    lang = str(row.get('language') or 'c').strip().lower()
    if lang in ('c++', 'cc', 'cxx'):
        lang = 'cpp'
    if lang not in ('c', 'cpp') or (lang == 'cpp' and not include_cpp):
        return None, 'unsupported_language'
    opt = get_opt(row.get('opt'))
    asm = asm_text(row.get('asm'))
    src = str(row.get('func') or '')
    test = str(row.get('test') or '').strip()
    deps = str(row.get('func_dep') or '')
    if not opt or not asm or not src or not test:
        return None, 'missing_opt_asm_source_or_harness'
    idx = row.get('index', pos)
    # Index should represent the same function across optimization variants.
    sample_id = f"{subset}_{idx}_{opt}"
    item = {
        'sample_id': sample_id,
        'problem_id': f'{subset}_{idx}',
        'compiler': 'unspecified', 'optimization': opt,
        'architecture': 'x86_64', 'language': lang,
        'assembly': asm, 'reference_source': src,
        'dependencies': deps,
        'tests': [{'name':'upstream_harness', 'program':test}],
    }
    return item, None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/llm4binary-decompile-eval')
    parser.add_argument('--subset', choices=['humaneval','mbpp'], required=True)
    parser.add_argument('--out', default=None)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--include-cpp', action='store_true')
    parser.add_argument('--opt', choices=['all','O0','O1','O2','O3'], default='all')
    args=parser.parse_args()
    filename=Path(args.raw)/args.subset/'data-00000-of-00001.arrow'
    if not filename.is_file():
        parser.error(f'Missing {filename}; run scripts/download_benchmarks.py first')
    output=Path(args.out) if args.out else Path('data/import') / (f'{args.subset}_{args.opt}.jsonl' if args.opt != 'all' else f'{args.subset}.jsonl')
    output.parent.mkdir(parents=True,exist_ok=True)
    from datasets import Dataset
    dataset=Dataset.from_file(str(filename))
    kept=0; skipped={}; seen=set()
    with output.open('w',encoding='utf-8') as writer:
        for pos,row in enumerate(dataset):
            obj,reason=convert_row(row,args.subset,pos,args.include_cpp)
            if args.opt != 'all' and obj is not None and obj['optimization'] != args.opt:
                continue
            if reason:
                skipped[reason]=skipped.get(reason,0)+1
                continue
            if obj['sample_id'] in seen:
                raise ValueError(f'Duplicate sample_id: {obj["sample_id"]}')
            seen.add(obj['sample_id'])
            writer.write(json.dumps(obj,ensure_ascii=False)+'\n')
            kept+=1
            if args.max_samples and kept>=args.max_samples: break
    print(f'{args.subset}: saved {kept} rows -> {output} ({output.stat().st_size/1024/1024:.1f} MiB)')
    print('skipped:',skipped)
    print('IMPORTANT: this is a format conversion, not an oracle-test validation. Perform reference smoke test.')


if __name__=='__main__':
    main()
