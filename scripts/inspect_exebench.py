#!/usr/bin/env python3
"""Inspect official ExeBench test_real archive without extracting arbitrary tar members.

This only inspects schema/targets; it does not turn ExeBench into a validated
RE benchmark. ExeBench contains IO pairs and C++ wrappers requiring a separate
adapter/oracle validation stage.
"""
import argparse
import json
import tarfile
from pathlib import Path
import zstandard as zstd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', default='data/raw/jordiae-exebench-v1.01/test_real.tar.gz')
    p.add_argument('--limit', type=int, default=3)
    args=p.parse_args()
    z=zstd.ZstdDecompressor()
    examples=0
    with tarfile.open(args.archive,'r:gz') as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith('.jsonl.zst'): continue
            f=archive.extractfile(member)
            if f is None: continue
            with z.stream_reader(f) as reader:
                import io
                for line in io.TextIOWrapper(reader,encoding='utf-8'):
                    item=json.loads(line)
                    obj=item.get('text',item)
                    target_asm=obj.get('asm') or {}
                    print('---',examples+1,'---')
                    print('fname:',obj.get('fname'))
                    print('asm targets:',list(target_asm.keys()) if isinstance(target_asm,dict) else [v.get('target') for v in target_asm[:10]])
                    print('real_deps bytes:',len(obj.get('real_deps') or ''))
                    print('real_exe_wrapper bytes:',len(obj.get('real_exe_wrapper') or ''))
                    print('real_io_pairs:',len(obj.get('real_io_pairs') or []))
                    examples+=1
                    if examples >= args.limit: return
    print('No records found')


if __name__=='__main__': main()
