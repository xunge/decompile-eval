#!/usr/bin/env python3
"""Convert HumanEval/MBPP or ExeBench test_real into DecompileEval-shaped JSONL.

HumanEval/MBPP: executable `tests[0].program` (upstream harness).
ExeBench: staging records containing `real_exe_wrapper` + independent I/O pairs.

Compiler handling:
  --compiler all (default) keeps the original dataset assembly unless --regenerate.
  --compiler gcc/clang selects genuine compiler-specific assembly. For HumanEval
  and MBPP it recompiles from source (original Arrow has no compiler column).
  For ExeBench GCC uses archived targets, while Clang recompiles from source.
  --regenerate recompiles both with GCC/Clang when --compiler=all.
  For GCC use --gcc /path/to/gcc-11 (verified against 11.4.0 by default).
  Linked ELF + AT&T + address-free output approximates original benchmark style;
  exact equality is NOT guaranteed without the original build system and flags.
  Recompiled output is NOT the official dataset assembly and is labeled.
  IMPORTANT: these are NOT directly executable by the HumanEval/MBPP sandbox;
  an ExeBench-specific wrapper runner + reference oracle validation is needed
  before reporting RC/RE/TCP. ExeBench output intentionally has no fake `tests`.
"""
import argparse
from collections import Counter
from difflib import SequenceMatcher
import hashlib
from functools import lru_cache
import io
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


OPTS = ('O0', 'O1', 'O2', 'O3', 'Os', 'Ofast')


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
    return {'OS': 'Os', 'OFAST': 'Ofast'}.get(opt, opt)


def convert_row(row, subset, pos, include_cpp=False):
    """Existing HumanEval/MBPP format; kept backward compatible."""
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
    item = {
        'sample_id': f'{subset}_{idx}_{opt}',
        'problem_id': f'{subset}_{idx}',
        'compiler': 'unspecified',
        'optimization': opt,
        'architecture': 'x86_64',
        'language': lang,
        'assembly': asm,
        'reference_source': src,
        'dependencies': deps,
        'tests': [{'name': 'upstream_harness', 'program': test}],
    }
    return item, None


def iter_exebench_rows(archive_path):
    """Stream ExeBench TAR.GZ -> JSONL.ZST -> original raw records."""
    try:
        import zstandard as zstd
    except ImportError as exc:
        raise SystemExit('ExeBench requires zstandard: python -m pip install zstandard') from exc

    with tarfile.open(archive_path, mode='r:gz') as tar:
        members = sorted((m for m in tar if m.isfile() and m.name.endswith('.jsonl.zst')),
                         key=lambda m: m.name)
        if not members:
            raise ValueError('No *.jsonl.zst found in ExeBench archive')
        for member in members:
            f = tar.extractfile(member)
            if f is None:
                continue
            with zstd.ZstdDecompressor().stream_reader(f) as stream:
                with io.TextIOWrapper(stream, encoding='utf-8') as text:
                    for lineno, line in enumerate(text, 1):
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        # Official archive stores each function under "text".
                        row = record.get('text', record) if isinstance(record, dict) else record
                        if isinstance(row, dict):
                            yield row, member.name, lineno


def iter_asm_variants(raw):
    """Handle official raw asm map AND transformed HF-style array."""
    if isinstance(raw, dict):
        for target, data in raw.items():
            if isinstance(data, dict):
                code = data.get('func_asm') or data.get('code') or ''
            else:
                code = data or ''
            if isinstance(code, dict):
                code = code.get('func_asm') or code.get('code') or ''
            yield str(target), asm_text(code)
    elif isinstance(raw, list):
        for data in raw:
            if not isinstance(data, dict):
                continue
            code = data.get('code') or data.get('func_asm') or ''
            if isinstance(code, dict):
                code = code.get('func_asm') or code.get('code') or ''
            yield str(data.get('target') or ''), asm_text(code)


def target_opt(target):
    match = re.search(r'(?i)(?:^|[_-])(o0|o1|o2|o3|os|ofast)(?=$|[_-])', target)
    return get_opt(match.group(1)) if match else None


def target_compiler(target):
    match = re.search(r'(?i)(?:^|[_-])(gcc|clang)(?=$|[_-])', target)
    return match.group(1).lower() if match else 'unspecified'


def target_arch(target):
    value = target.lower()
    if 'x86_64' in value or 'amd64' in value:
        return 'x86_64'
    if 'aarch64' in value or 'arm64' in value:
        return 'aarch64'
    if re.search(r'(?:^|[_-])x86(?=$|[_-])', value):
        return 'x86'  # Do not assume that label means x86-64.
    if 'mips' in value:
        return 'mips'
    if 'arm' in value:
        return 'arm'
    return 'unspecified'


def normalize_io(value):
    """Preserve typed raw ExeBench inputs/outputs; support HF flattened form."""
    if isinstance(value, dict) and 'var' in value and 'value' in value:
        keys, vals = value['var'], value['value']
        if isinstance(keys, list) and isinstance(vals, list) and len(keys) == len(vals):
            return {str(k): decode_json(v) for k, v in zip(keys, vals)}
    if isinstance(value, list) and all(isinstance(x, dict) and 'var' in x for x in value):
        return {str(x['var']): decode_json(x.get('value')) for x in value}
    return value if value is not None else {}


def decode_json(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except (ValueError, TypeError):
            return x
    return x


def exebench_record(row, pos, target, code, member, lineno):
    opt = target_opt(target)
    if not opt:
        return None, 'unknown_optimization'
    if not code.strip():
        return None, 'empty_assembly'
    source = row.get('func_def') or ''
    wrapper = row.get('real_exe_wrapper') or ''
    if not source:
        return None, 'missing_reference_source'
    if not wrapper:
        return None, 'missing_real_exe_wrapper'

    raw_pairs = row.get('real_io_pairs') or []
    cases = []
    for i, pair in enumerate(raw_pairs):
        if not isinstance(pair, dict):
            continue
        cases.append({
            'case_id': i,
            'input': normalize_io(pair.get('input')),
            'expected_output': normalize_io(pair.get('output')),
            'dummy_funcs': pair.get('dummy_funcs') or '',
            'dummy_funcs_seed': pair.get('dummy_funcs_seed'),
        })
    clean_target = re.sub(r'[^A-Za-z0-9_-]+', '_', target)[:60]
    target_hash = hashlib.sha256(target.encode('utf-8')).hexdigest()[:8]
    pid = f'exebench_test_real_{pos:06d}'
    obj = {
        'sample_id': f'{pid}_{clean_target}_{target_hash}',
        'problem_id': pid,
        'dataset': 'ExeBench-v1.01-test_real',
        'compiler': target_compiler(target),
        'optimization': opt,
        'architecture': target_arch(target),
        'language': 'c',
        'assembly': code,
        'reference_source': source,
        'dependencies': row.get('real_deps') or '',
        'function_name': row.get('fname') or '',
        'function_signature': row.get('func_head_types') or row.get('func_head') or '',
        'assembly_target': target,
        'real_exe_wrapper': wrapper,
        'real_iospec': row.get('real_iospec'),
        'testcases': cases,
        'testcase_count': len(cases),
        'test_protocol': 'exebench_real_io_wrapper',
        'oracle_validated': False,
        'source_path': row.get('path') or '',
        'source_ref': row.get('ref') or '',
        'origin': {'archive_member': member, 'line': lineno},
    }
    return obj, None


# ---------------------------------------------------------------------------
# Compiler-aware assembly regeneration, benchmark-style linked ELF disassembly
# ---------------------------------------------------------------------------

FUNC_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z_0-9:]*)\s*\([^;{}]*\)\s*"
    r"(?:__attribute__\s*\(\([^{}]*\)\)\s*)?\{",
    re.DOTALL,
)
OBJDUMP_HEADER = re.compile(r'^\s*[0-9a-fA-F]+\s+<([^>]+)>:$')
OBJDUMP_INSN = re.compile(r'^\s*[0-9a-fA-F]+:\s*(.*?)\s*$')
OBJDUMP_RELOC = re.compile(r'^\s*[0-9a-fA-F]+:\s*(R_[A-Za-z0-9_]+)\s+(.+?)\s*$')
SPLIT_SUFFIX = r'(?:part|cold|isra|constprop|llvm|clone)(?:\.[\w.]+)?'


def source_function_name(source, supplied=''):
    if supplied and str(supplied).strip():
        return str(supplied).strip()
    for match in FUNC_PATTERN.finditer(source):
        name = match.group('name')
        if name not in {'if', 'for', 'while', 'switch', 'catch'}:
            return name
    return ''


def compiler_path(family, args, lang='c'):
    override = (args.gxx if family == 'gcc' else args.clangxx) if lang == 'cpp' else (args.gcc if family == 'gcc' else args.clang)
    exe = shutil.which(override) if not Path(override).is_file() else override
    if not exe:
        raise SystemExit(f'Compiler {override!r} not found. Install a suitable compiler '
                         f'or pass --{family} /absolute/path/to/compiler.')
    return str(Path(exe).resolve())


@lru_cache(maxsize=8)
def compiler_version(exe):
    run = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=10)
    return (run.stdout or run.stderr).splitlines()[0] if run.returncode == 0 else 'unknown'


@lru_cache(maxsize=8)
def gcc_numeric_version(exe):
    run = subprocess.run([exe, '-dumpfullversion', '-dumpversion'],
                         capture_output=True, text=True, timeout=10)
    if run.returncode != 0 or not run.stdout.strip():
        raise ValueError(f'Could not obtain GCC version from {exe}: {run.stderr[-300:]}')
    return run.stdout.strip().splitlines()[0].strip()


def verify_compiler(family, args, lang='c'):
    exe = compiler_path(family, args, lang=lang)
    if family == 'gcc' and args.require_gcc_version:
        actual = gcc_numeric_version(exe)
        if actual != args.require_gcc_version and not args.allow_version_mismatch:
            raise SystemExit(
                f'GCC version mismatch: requested {args.require_gcc_version}, '
                f'but {exe} is {actual}. Install GCC {args.require_gcc_version} and set '
                f'--gcc /path/to/gcc, or explicitly use --allow-version-mismatch '
                f'for a non-reproduction experiment.')
    return exe


def target_arch_native():
    machine = platform.machine().lower()
    return {'amd64': 'x86_64', 'x86_64': 'x86_64', 'arm64': 'aarch64',
            'aarch64': 'aarch64'}.get(machine, machine)


@lru_cache(maxsize=4096)
def demangle(symbol):
    tool = shutil.which('c++filt')
    if not tool:
        return symbol
    result = subprocess.run([tool, symbol], capture_output=True, text=True, timeout=5)
    return result.stdout.strip() if result.returncode == 0 else symbol


def symbol_match(name, expected):
    if not name or not expected:
        return False
    left = name.split('(')[0].strip()
    right = expected.split('(')[0].strip()
    return (left == right or left.split('::')[-1] == right.split('::')[-1])


def extract_function_asm(disassembly, expected):
    """Extract one symbol, never silently return an unrelated helper function."""
    functions = []
    current = None
    for line in disassembly.splitlines():
        m = OBJDUMP_HEADER.match(line)
        if m:
            if current:
                functions.append(current)
            current = [m.group(1), []]
        elif current and line.strip():
            current[1].append(line)
    if current:
        functions.append(current)
    if not functions:
        raise ValueError('No function symbols found in binary')

    matched = [f for f in functions if symbol_match(demangle(f[0]), expected)]
    if len(matched) != 1:
        if not expected and len(functions) == 1:
            matched = functions
        else:
            raise ValueError(f'Unable to uniquely locate function {expected!r}: '
                             f'found {len(matched)} matches; functions={[x[0] for x in functions][:15]}')
    symbol, lines = matched[0]
    if not lines:
        raise ValueError(f'No disassembled instructions for {symbol}')
    return symbol + ':\n' + '\n'.join(lines)


def normalize_asm_format(asm, *, style, binary_type, architecture):
    """Match original decompile-eval's address-free AT&T objdump presentation.

    Object relocations must not be dropped blindly: unresolved call targets are
    rewritten to symbolic names and other relocations are kept as comments.
    """
    if style == 'raw':
        return asm, {'relocation_count': sum(bool(OBJDUMP_RELOC.match(x)) for x in asm.splitlines())}
    if style != 'benchmark':
        raise ValueError(f'Unknown asm style: {style}')
    out = []
    reloc_count = 0
    for lineno, row in enumerate(asm.splitlines()):
        if lineno == 0 and row.endswith(':'):
            out.append(row.strip())
            continue
        reloc = OBJDUMP_RELOC.match(row)
        if reloc:
            reloc_count += 1
            reloc_type, reloc_target = reloc.groups()
            # Use the linker-provided symbol when available. An unlinked .o
            # can otherwise print misleading 'call <func+...>' placeholders.
            if out and reloc_type.endswith(('PLT32', 'PC32')) and re.match(
                    r'^\s*(?:callq?|jmpq?)\s+', out[-1]):
                opcode = out[-1].strip().split()[0]
                symbol = re.sub(r'(?i)[+-]0x[0-9a-f]+$', '', reloc_target.strip())
                if symbol and not symbol.startswith('.'):
                    out[-1] = f'{opcode:<6} {symbol}'
                    continue
            # Keep unresolved data relocations as comments: never fabricate a
            # resolved address or pretend this is identical to linked ELF.
            if out:
                out[-1] += f'  # {reloc_type} {reloc_target.strip()}'
            continue
        m = OBJDUMP_INSN.match(row)
        if not m:
            continue  # drop blank lines / objdump headers within the function
        inst = m.group(1).strip()
        if not inst:
            continue
        # objdump adds address-resolution hints, which the official sample
        # format does not contain. The operand (including <symbol>) is retained.
        inst = re.sub(r'\s+#\s+[0-9a-fA-Fx]+\s+<[^>]+>\s*$', '', inst)
        if architecture in ('x86_64', 'x86'):
            mnem = re.match(r'^([A-Za-z.][\w.]*)\b(.*)$', inst)
            if mnem:
                op, operands = mnem.groups()
                if architecture == 'x86_64' and op in ('call', 'ret'):
                    op += 'q'
                # Keep direct 'jmp' untouched; only indirect 64-bit jumps use jmpq.
                if architecture == 'x86_64' and op == 'jmp' and operands.lstrip().startswith('*'):
                    op = 'jmpq'
                inst = f'{op:<6}{operands.strip() and " " + operands.strip() or ""}'
        out.append(inst.rstrip())
    if len(out) < 2:
        raise ValueError('No disassembled instructions after normalization')
    return '\n'.join(out), {'relocation_count': reloc_count}


def split_function_references(asm, name):
    if not name:
        return []
    pattern = rf'<({re.escape(name)}\.{SPLIT_SUFFIX})(?:\+0x[0-9a-f]+)?>'
    return sorted(set(re.findall(pattern, asm)))


def compilation_flags(opt, args):
    flags = [f'-{opt}']
    if args.pie:
        flags += ['-fPIE']
    else:
        flags += ['-fno-pie']
    protector = {
        'strong': '-fstack-protector-strong',
        'basic': '-fstack-protector',
        'all': '-fstack-protector-all',
        'none': '-fno-stack-protector',
        'default': None,
    }[args.stack_protector]
    if protector:
        flags.append(protector)
    if args.cf_protection != 'default' and target_arch_native() in ('x86_64', 'x86'):
        flags.append('-fcf-protection=' + args.cf_protection)
    if args.march:
        flags.append('-march=' + args.march)
    flags += args.extra_compile_flag
    return flags


def _run_checked(cmd, *, timeout, label):
    run = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if run.returncode != 0:
        detail = (run.stderr or run.stdout).strip()[-2400:]
        raise ValueError(f'{label} failed (exit {run.returncode}): {detail}')
    return run


def compile_function_assembly(src, deps, supplied_name, lang, family, opt, args):
    """Compile one function; optionally link a full ELF before disassembly.

    The ELF is never executed. Linking removes unresolved PLT calls/data
    relocations and yields the same class of objdump input as the original
    HumanEval benchmark, though exact equality cannot be guaranteed.
    """
    exe = verify_compiler(family, args, lang=lang)
    if not src.strip():
        raise ValueError('Empty source code')
    source = str(deps or '') + '\n' + str(src or '') + '\n'
    expected = source_function_name(src, supplied_name)
    suffix = '.cc' if lang == 'cpp' else '.c'
    flags = compilation_flags(opt, args)
    with tempfile.TemporaryDirectory(prefix='decomp-asm-') as tmp:
        path = Path(tmp) / ('candidate' + suffix)
        obj = Path(tmp) / 'candidate.o'
        path.write_text(source, encoding='utf-8')
        _run_checked([exe, *flags, '-c', str(path), '-o', str(obj)],
                     timeout=args.compile_timeout, label=f'{family} -{opt} compile')
        binary = obj
        if args.binary_type == 'executable':
            link_inputs = [str(obj)]
            # The benchmark function typically has no main(); provide a
            # separate empty entry point without changing the target unit.
            if expected != 'main':
                stub = Path(tmp) / ('benchmark_entry' + suffix)
                stub.write_text('int main(void) { return 0; }\n', encoding='utf-8')
                entry_obj = Path(tmp) / 'entry.o'
                _run_checked([exe, *flags, '-c', str(stub), '-o', str(entry_obj)],
                             timeout=args.compile_timeout, label='entry compilation')
                link_inputs.append(str(entry_obj))
            binary = Path(tmp) / 'benchmark.elf'
            pie_flags = ['-pie'] if args.pie else ['-no-pie']
            libraries = [f'-l{lib}' for lib in args.link_lib]
            link_cmd = [exe, *pie_flags, *link_inputs, *libraries, '-o', str(binary)]
            _run_checked(link_cmd, timeout=args.compile_timeout, label=f'{family} link')
        # -r is only needed on .o; it produces relocation rows that are not
        # meaningful inside a linked ELF's instruction stream.
        dis_opts = ['-dr' if args.binary_type == 'object' else '-d',
                    '-M', 'att', '--no-show-raw-insn', str(binary)]
        dis = _run_checked([args.objdump, *dis_opts], timeout=20, label='objdump')
        asm_raw = extract_function_asm(dis.stdout, expected)
        asm, fmt_meta = normalize_asm_format(
            asm_raw, style=args.asm_style, binary_type=args.binary_type,
            architecture=target_arch_native())
        split_refs = split_function_references(asm, expected)
    meta = {
        'compiler': family,
        'compiler_version': compiler_version(exe),
        'compiler_numeric_version': gcc_numeric_version(exe) if family == 'gcc' else None,
        'compiler_executable': exe,
        'assembly_origin': f'recompiled_{args.binary_type}_objdump',
        'assembly_syntax': 'att',
        'assembly_style': args.asm_style,
        'binary_type': args.binary_type,
        'compile_flags': flags,
        'link_flags': (['-pie'] if args.pie else ['-no-pie']) +
                      [f'-l{x}' for x in args.link_lib] if args.binary_type == 'executable' else [],
        'architecture': target_arch_native(),
        'source_function_name': expected,
        'split_function_references': split_refs,
        'function_may_be_incomplete': bool(split_refs),
        **fmt_meta,
    }
    return asm, meta


def asm_mnemonics(text):
    ops = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.endswith(':') or OBJDUMP_RELOC.match(line):
            continue
        line = re.sub(r'^[0-9a-fA-F]+:\s*', '', line)
        token = line.split(None, 1)[0]
        if re.fullmatch(r'[A-Za-z][\w.]*', token):
            ops.append({'callq': 'call', 'retq': 'ret', 'jmpq': 'jmp'}.get(token, token))
    return ops


def comparison_to_original(original, generated):
    left, right = asm_mnemonics(original), asm_mnemonics(generated)
    return {
        'exact_assembly_match': asm_text(original) == asm_text(generated),
        'original_instruction_count': len(left),
        'generated_instruction_count': len(right),
        'mnemonic_similarity': round(SequenceMatcher(None, left, right,
                                                        autojunk=False).ratio(), 6),
        'original_length': len(original),
        'generated_length': len(generated),
    }


def write_failure(writer, sample_id, compiler, opt, error):
    if writer is not None:
        writer.write(json.dumps({'sample_id': sample_id, 'compiler': compiler,
                                 'optimization': opt, 'error': str(error)}, ensure_ascii=False) + '\n')


def convert_arrow(args, output):
    filename = Path(args.raw) / args.subset / 'data-00000-of-00001.arrow'
    if not filename.is_file():
        raise SystemExit(f'Missing {filename}; run scripts/download_benchmarks.py first')
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit('HumanEval/MBPP requires: python -m pip install datasets pyarrow') from exc
    dataset = Dataset.from_file(str(filename))
    # Original Arrow does not expose compiler provenance. Never just relabel it.
    regenerate = args.regenerate or args.compiler != 'all'
    if regenerate:
        families = ('gcc', 'clang') if args.compiler == 'all' else (args.compiler,)
        for family in families:
            verify_compiler(family, args)
        if not shutil.which('objdump'):
            raise SystemExit('objdump is required for recompilation (install binutils)')
    else:
        families = ()
        print('NOTE: original HumanEval/MBPP asm has no compiler field: compiler=unspecified. '
              'Use --compiler gcc or clang to recompile it.')

    kept, skipped, seen = 0, Counter(), set()
    failure_file = output.with_suffix('.failures.jsonl') if regenerate else None
    from contextlib import nullcontext
    failures = failure_file.open('w', encoding='utf-8') if failure_file else nullcontext(None)
    with output.open('w', encoding='utf-8') as writer, failures as error_writer:
        for pos, row in enumerate(dataset):
            obj, reason = convert_row(row, args.subset, pos, args.include_cpp)
            if reason:
                skipped[reason] += 1
                continue
            if args.opt != 'all' and obj['optimization'] != args.opt:
                continue
            if regenerate:
                candidates = []
                for family in families:
                    key = f'{obj["sample_id"]}_{family}'
                    try:
                        asm, meta = compile_function_assembly(
                            obj['reference_source'], obj['dependencies'],
                            row.get('func_name') or '', obj['language'],
                            family, obj['optimization'], args)
                    except (ValueError, subprocess.TimeoutExpired) as exc:
                        skipped[f'{family}_compile_or_extract_failed'] += 1
                        write_failure(error_writer, key, family, obj['optimization'], exc)
                        continue
                    record = dict(obj)
                    record.update(meta)
                    record['sample_id'] = f'{args.subset}_{row.get("index", pos)}_{family}_{obj["optimization"]}'
                    record['assembly'] = asm
                    record['reference_assembly'] = obj['assembly']
                    record['reference_comparison'] = comparison_to_original(
                        obj['assembly'], asm)
                    candidates.append(record)
            else:
                candidates = [obj]
            for record in candidates:
                if record['sample_id'] in seen:
                    raise ValueError(f'Duplicate sample_id: {record["sample_id"]}')
                seen.add(record['sample_id'])
                writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                kept += 1
                if args.max_samples and kept >= args.max_samples:
                    break
            if args.max_samples and kept >= args.max_samples:
                break
    print(f'{args.subset}: saved {kept} -> {output} ({output.stat().st_size/1048576:.2f} MiB)')
    print('Skipped:', dict(skipped))
    if regenerate:
        print(f'Recompilation failure details: {failure_file}')
        print('IMPORTANT: regenerated assembly is not the original public benchmark input.')
    print('NOTE: validate reference code before reporting metrics.')


def build_exebench_regenerated(row, pos, compiler, opt, member, lineno, args):
    source = str(row.get('func_def') or '')
    deps = str(row.get('real_deps') or '')
    asm, meta = compile_function_assembly(
        source, deps, row.get('fname') or '', 'c', compiler, opt, args)
    target = f'recompiled_{compiler}_{meta["architecture"]}_{opt}'
    obj, reason = exebench_record(row, pos, target, asm, member, lineno)
    if reason:
        raise ValueError(reason)
    obj.update(meta)
    obj['assembly_target'] = target
    original_targets = list(iter_asm_variants(row.get('asm')))
    obj['original_asm_targets'] = [target for target, _ in original_targets]
    # Select a real original target, not just any optimization variant.
    matches = [(t, a) for t, a in original_targets
               if target_opt(t) == opt and target_compiler(t) == compiler
               and target_arch(t) == meta['architecture']]
    if len(matches) == 1:
        obj['reference_assembly_target'] = matches[0][0]
        obj['reference_comparison'] = comparison_to_original(matches[0][1], asm)
    return obj


def convert_exebench(args, output):
    archive = Path(args.archive)
    if not archive.is_file():
        raise SystemExit(f'Missing ExeBench archive: {archive}; download test_real.tar.gz first')
    # Official raw ExeBench is GCC. Clang needs recompilation. --regenerate
    # also allows controlled GCC-vs-Clang experiments on the same host.
    regenerate = args.regenerate or args.compiler == 'clang'
    if regenerate:
        families = ('gcc', 'clang') if args.compiler == 'all' else (args.compiler,)
        for family in families:
            verify_compiler(family, args)
        if not shutil.which('objdump'):
            raise SystemExit('objdump is required for recompilation (install binutils)')
    else:
        families = ()
    kept, skipped, targets, seen = 0, Counter(), Counter(), set()
    failure_file = output.with_suffix('.failures.jsonl') if regenerate else None
    from contextlib import nullcontext
    failures = failure_file.open('w', encoding='utf-8') if failure_file else nullcontext(None)
    with output.open('w', encoding='utf-8') as writer, failures as error_writer:
        for pos, (row, member, lineno) in enumerate(iter_exebench_rows(archive)):
            skipped['source_functions_seen'] += 1
            if regenerate:
                opts = (('O0', 'O1', 'O2', 'O3') if args.opt == 'all' else (args.opt,))
                candidates = []
                for family in families:
                    for opt in opts:
                        key = f'exebench_test_real_{pos:06d}_{family}_{opt}'
                        try:
                            obj = build_exebench_regenerated(row, pos, family, opt, member, lineno, args)
                        except (ValueError, subprocess.TimeoutExpired) as exc:
                            skipped[f'{family}_compile_or_extract_failed'] += 1
                            write_failure(error_writer, key, family, opt, exc)
                            continue
                        candidates.append(obj)
            else:
                candidates = []
                for target, code in iter_asm_variants(row.get('asm')):
                    targets[target] += 1
                    if args.compiler != 'all' and target_compiler(target) != args.compiler:
                        continue
                    if args.opt != 'all' and target_opt(target) != args.opt:
                        continue
                    obj, reason = exebench_record(row, pos, target, code, member, lineno)
                    if reason:
                        skipped[reason] += 1
                        continue
                    obj['assembly_origin'] = 'official_exebench'
                    candidates.append(obj)
            for obj in candidates:
                if args.target_contains and args.target_contains.lower() not in obj['assembly_target'].lower():
                    continue
                if len(obj['testcases']) < args.min_io:
                    skipped['too_few_io_pairs'] += 1
                    continue
                if obj['sample_id'] in seen:
                    raise ValueError(f'Duplicate sample_id: {obj["sample_id"]}')
                seen.add(obj['sample_id'])
                writer.write(json.dumps(obj, ensure_ascii=False) + '\n')
                kept += 1
                if args.max_samples and kept >= args.max_samples:
                    break
            if args.max_samples and kept >= args.max_samples:
                break
    print(f'exebench: saved {kept} assembly variants -> {output} ({output.stat().st_size/1048576:.2f} MiB)')
    print('Skipped:', dict(skipped))
    if regenerate:
        print(f'Recompilation failure details: {failure_file}')
        print('IMPORTANT: regenerated assembly is NOT official ExeBench assembly.')
    else:
        print('Available native targets:', targets.most_common(12))
    print('IMPORTANT: ExeBench output is STAGING ONLY, not directly executable by')
    print('HumanEval/MBPP tests[].program runner; oracle validation is required.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--raw', default='data/raw/llm4binary-decompile-eval', help='HumanEval/MBPP Arrow root')
    p.add_argument('--subset', choices=['humaneval', 'mbpp', 'exebench'], required=True)
    p.add_argument('--archive', default='data/raw/jordiae-exebench-v1.01/test_real.tar.gz',
                   help='ExeBench test_real archive')
    p.add_argument('--out', help='Output JSONL path; ExeBench defaults to data/staging/')
    p.add_argument('--max-samples', type=int, default=0, help='0 = all; counts compiled variants')
    p.add_argument('--include-cpp', action='store_true', help='HumanEval/MBPP only')
    p.add_argument('--opt', choices=['all', *OPTS], default='all')
    p.add_argument('--compiler', choices=['all', 'gcc', 'clang'], default='all',
                   help='Explicit gcc/clang regenerates asm for Arrow; clang regenerates for ExeBench')
    p.add_argument('--regenerate', action='store_true',
                   help='Recompile source. With --compiler all, generate GCC AND Clang asm')
    p.add_argument('--gcc', default='gcc-11', help='GCC compiler executable (default: gcc-11)')
    p.add_argument('--clang', default='clang', help='Clang compiler executable')
    p.add_argument('--gxx', default='g++-11', help='G++ driver for optional C++ samples')
    p.add_argument('--clangxx', default='clang++', help='Clang++ driver for optional C++ samples')
    p.add_argument('--require-gcc-version', default='11.4.0',
                   help='Fail unless GCC reports exactly this version; empty disables check')
    p.add_argument('--allow-version-mismatch', action='store_true',
                   help='Explicitly allow a different GCC version (not benchmark reproduction)')
    p.add_argument('--binary-type', choices=['executable', 'object'], default='executable',
                   help='Link ELF before objdump (recommended), or inspect unresolved object')
    p.add_argument('--asm-style', choices=['benchmark', 'raw'], default='benchmark',
                   help='Address-free AT&T benchmark style (default) or raw objdump text')
    p.add_argument('--stack-protector',
                   choices=['strong', 'basic', 'all', 'none', 'default'], default='strong')
    pie_group = p.add_mutually_exclusive_group()
    pie_group.add_argument('--pie', action='store_true', dest='pie', default=True)
    pie_group.add_argument('--no-pie', action='store_false', dest='pie')
    p.add_argument('--cf-protection', choices=['branch', 'none', 'default'],
                   default='branch', help='Use endbr64 on x86 where supported')
    p.add_argument('--march', default='', help='Optional CPU architecture flag, e.g. x86-64')
    p.add_argument('--objdump', default='objdump', help='GNU objdump executable')
    p.add_argument('--link-lib', action='append', default=None,
                   help='Link extra library, repeatable; default -lm (e.g. --link-lib pthread)')
    p.add_argument('--extra-compile-flag', action='append', default=[],
                   help='Additional compiler option; repeatable, for controlled experiments')
    p.add_argument('--compile-timeout', type=int, default=30, help='Recompile timeout (seconds per sample)')
    p.add_argument('--target-contains', default='', help='ExeBench assembly target name filter')
    p.add_argument('--min-io', type=int, default=1, help='ExeBench minimum I/O cases')
    args = p.parse_args()
    if args.link_lib is None:
        args.link_lib = ['m']
    for lib in args.link_lib:
        if not re.fullmatch(r'[A-Za-z0-9_+.\-]+', lib):
            p.error(f'Invalid library name: {lib!r}')
    if args.max_samples < 0 or args.min_io < 0 or args.compile_timeout <= 0:
        p.error('--max-samples, --min-io must be nonnegative; --compile-timeout > 0')
    if args.out:
        output = Path(args.out)
    elif args.subset == 'exebench':
        suffix = f'_{args.compiler}' if args.compiler != 'all' or args.regenerate else ''
        output = Path('data/staging') / f'exebench{suffix}_{args.opt}.jsonl'
    else:
        suffix = f'_{args.compiler}' if args.compiler != 'all' or args.regenerate else ''
        output = Path('data/import') / f'{args.subset}{suffix}_{args.opt}.jsonl'
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.subset == 'exebench':
        convert_exebench(args, output)
    else:
        convert_arrow(args, output)


if __name__ == '__main__':
    main()
