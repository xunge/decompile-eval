"""Extract and instrument individual test cases from a C/C++ harness.

The official harness stays untouched for the authoritative RE verdict.  For
diagnostics we use tree-sitter (a real C/C++ parser, not a regex) to locate
``assert(...)`` calls, then rewrite each call into a small diagnostic block
that:

* evaluates each operand exactly once,
* records input/expected/actual values for common scalar/string types,
* stops after the requested runtime assertion index,
* can continue past a previously failed assertion in a fresh process so that
  later independent cases are not silently dropped.

Dynamic assertions (loops/conditionals) are supported at runtime: the helper
increments a runtime index, so one static assert inside a loop can yield several
distinct cases.  Anything we cannot instrument safely is reported as
``UNSUPPORTED_CASE_EXTRACTION`` instead of fabricated data.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass, field
from typing import Any

import tree_sitter_c
import tree_sitter_cpp
from tree_sitter import Language, Parser

PARSER_VERSION = 'tree-sitter-c/cpp@0.27;rewrite=1'
METRIC_VERSION = 'tcp_micro_macro_v1'

_COMPARISON_OPERATORS = {'==', '!=', '<', '<=', '>', '>='}
_LOGICAL_OPERATORS = {'&&', '||'}
_RISK_NODES = {'assignment_expression', 'update_expression', 'comma_expression'}
_LOOP_NODES = {'for_statement', 'while_statement', 'do_statement',
               'for_range_loop'}
_CONDITIONAL_NODES = {'if_statement', 'conditional_expression', 'switch_statement'}


def _language(language: str):
    if language == 'cpp':
        return Language(tree_sitter_cpp.language())
    return Language(tree_sitter_c.language())


_PARSERS = {
    'c': Parser(_language('c')),
    'cpp': Parser(_language('cpp')),
}
_PARSER_LOCK = threading.Lock()


@dataclass
class ExtractedCase:
    static_id: int
    line: int
    column: int
    end_line: int
    end_column: int
    start_byte: int
    end_byte: int
    expression: str
    comparison_op: str = ''
    lhs: str = ''
    rhs: str = ''
    comparison_rule: str = 'expression'
    dynamic: bool = False
    conditional: bool = False
    semantics_risk: bool = False
    function: str = ''
    source_text: str = ''

    def as_manifest(self) -> dict:
        return {
            'static_id': self.static_id,
            'line': self.line,
            'column': self.column,
            'end_line': self.end_line,
            'end_column': self.end_column,
            'span': [self.start_byte, self.end_byte],
            'expression': self.expression,
            'comparison_operator': self.comparison_op,
            'lhs': self.lhs,
            'rhs': self.rhs,
            'comparison_rule': self.comparison_rule,
            'dynamic': self.dynamic,
            'conditional': self.conditional,
            'semantics_risk': self.semantics_risk,
            'function': self.function,
        }


@dataclass
class Manifest:
    status: str
    language: str
    parser_version: str = PARSER_VERSION
    cases: list[ExtractedCase] = field(default_factory=list)
    rewritten_harness: str = ''
    reason: str = ''
    harness_hash: str = ''
    parse_errors: int = 0

    @property
    def ok(self) -> bool:
        return self.status == 'OK'

    def as_dict(self) -> dict:
        return {
            'status': self.status,
            'language': self.language,
            'parser_version': self.parser_version,
            'reason': self.reason,
            'harness_hash': self.harness_hash,
            'static_case_count': len(self.cases),
            'dynamic_case_count': sum(1 for c in self.cases if c.dynamic),
            'conditional_case_count': sum(1 for c in self.cases if c.conditional),
            'semantics_risk_count': sum(1 for c in self.cases if c.semantics_risk),
            'parse_errors': self.parse_errors,
            'cases': [c.as_manifest() for c in self.cases],
            'rewritten_harness': self.rewritten_harness,
        }


def _text(source: bytes, node) -> str:
    return source[node.start_byte:node.end_byte].decode('utf-8', 'replace')


def _function_name(source: bytes, node) -> str:
    current = node.parent
    while current is not None:
        if current.type == 'function_definition':
            declarator = current.child_by_field_name('declarator')
            while declarator is not None and declarator.type not in (
                    'identifier', 'function_declarator', 'pointer_declarator'):
                declarator = declarator.child_by_field_name('declarator')
            if declarator is not None:
                return _text(source, declarator)
            return ''
        current = current.parent
    return ''


def _has_risk(node) -> bool:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in _RISK_NODES:
            return True
        stack.extend(current.children)
    return False


def _context_flags(node) -> tuple[bool, bool]:
    dynamic = conditional = False
    current = node.parent
    while current is not None:
        if current.type in _LOOP_NODES:
            dynamic = True
        if current.type in _CONDITIONAL_NODES:
            conditional = True
            dynamic = True
        current = current.parent
    return dynamic, conditional


def _c_string(text: str) -> str:
    # JSON string escapes are valid C/C++ string literal escapes for the
    # characters that matter here (\", \\, \n, \t, \uXXXX).
    return json.dumps(text, ensure_ascii=True)


def _collect_asserts(source: bytes, root) -> list:
    found = []
    stack = [(root, 0)]
    while stack:
        node, order = stack.pop()
        if node.type == 'call_expression':
            function = node.child_by_field_name('function')
            if function is not None:
                name = source[function.start_byte:function.end_byte].decode('utf-8', 'replace')
                if name == 'assert':
                    arguments = node.child_by_field_name('arguments')
                    expr = None
                    if arguments is not None:
                        named = arguments.named_children
                        if len(named) == 1:
                            expr = named[0]
                    if expr is not None:
                        found.append((node, expr))
        for child in reversed(node.children):
            stack.append((child, order + 1))
    found.sort(key=lambda item: item[0].start_byte)
    return found


def _replacement(case: ExtractedCase, language: str) -> str:
    sid = case.static_id
    line = case.line
    expr_c = _c_string(case.expression)
    lhs_c = _c_string(case.lhs or case.expression)
    rhs_c = _c_string(case.rhs or '1')
    if language == 'cpp':
        if case.comparison_op in _COMPARISON_OPERATORS:
            return (
                '{ auto _dc_l = (' + case.lhs + '); auto _dc_r = (' + case.rhs + '); '
                'int _dc_ok = (_dc_l ' + case.comparison_op + ' _dc_r); '
                'int _dc_i = dec_emit_bool(' + str(sid) + ',' + str(line) + ',_dc_ok,"'
                + _c_string(case.expression)[1:-1] + '"); '
                'dec_control(_dc_i,_dc_ok); }'
            )
        return (
            '{ auto _dc_v = (' + case.expression + '); int _dc_ok = !!(_dc_v); '
            'int _dc_i = dec_emit_bool(' + str(sid) + ',' + str(line) + ',_dc_ok,"'
            + _c_string(case.expression)[1:-1] + '"); '
            'dec_control(_dc_i,_dc_ok); }'
        )
    if case.comparison_op in _COMPARISON_OPERATORS:
        return (
            '{ __auto_type _dc_l = (' + case.lhs + '); '
            '__auto_type _dc_r = (' + case.rhs + '); '
            'char _dc_ab[512]; char _dc_eb[512]; '
            'int _dc_ao = DEC_FMT(_dc_l,_dc_ab,sizeof(_dc_ab)); '
            'int _dc_eo = DEC_FMT(_dc_r,_dc_eb,sizeof(_dc_eb)); '
            'int _dc_ok = (_dc_l ' + case.comparison_op + ' _dc_r); '
            'int _dc_i = dec_emit(' + str(sid) + ',' + str(line) + ',_dc_ok,'
            + expr_c + ',' + lhs_c + ',' + rhs_c + ',_dc_ab,_dc_eb,_dc_ao,_dc_eo); '
            'dec_control(_dc_i,_dc_ok); }'
        )
    return (
        '{ __auto_type _dc_v = (' + case.expression + '); '
        'char _dc_ab[512]; '
        'int _dc_ao = DEC_FMT(_dc_v,_dc_ab,sizeof(_dc_ab)); '
        'int _dc_ok = !!(_dc_v); '
        'int _dc_i = dec_emit(' + str(sid) + ',' + str(line) + ',_dc_ok,'
        + expr_c + ',' + expr_c + ',"1",_dc_ab,"1",_dc_ao,1); '
        'dec_control(_dc_i,_dc_ok); }'
    )


def extract_manifest(harness: str, language: str = 'c') -> Manifest:
    """Parse a complete harness and build a diagnostic manifest + rewrite."""
    language = 'cpp' if language == 'cpp' else 'c'
    digest = hashlib.sha256((language + '\0' + harness).encode('utf-8')).hexdigest()
    if not harness.strip():
        return Manifest('UNSUPPORTED_CASE_EXTRACTION', language, reason='empty harness',
                        harness_hash=digest)
    if re.search(r'^[ \t]*#[ \t]*define[ \t]+NDEBUG\b', harness, re.MULTILINE):
        return Manifest('UNSUPPORTED_CASE_EXTRACTION', language,
                        reason='harness defines NDEBUG; assert instrumentation would '
                               'not match official compile semantics',
                        harness_hash=digest)
    try:
        source = harness.encode('utf-8')
        # tree-sitter Parser objects are not thread-safe; extraction is cached
        # anyway, so serialising this cheap parse is fine.
        with _PARSER_LOCK:
            tree = _PARSERS[language].parse(source)
    except Exception as exc:  # pragma: no cover - defensive
        return Manifest('PARSE_ERROR', language, reason=f'{type(exc).__name__}: {exc}',
                        harness_hash=digest)
    if tree.root_node.has_error:
        return Manifest('PARSE_ERROR', language,
                        reason='tree-sitter reported syntax errors in the harness',
                        harness_hash=digest,
                        parse_errors=tree.root_node.error_count
                        if hasattr(tree.root_node, 'error_count') else 1)
    asserts = _collect_asserts(source, tree.root_node)
    if not asserts:
        return Manifest('NO_ASSERT_CASES', language,
                        reason='no assert() call was found in the harness',
                        harness_hash=digest)
    cases: list[ExtractedCase] = []
    for index, (node, expr_node) in enumerate(asserts):
        dynamic, conditional = _context_flags(node)
        expression = _text(source, expr_node)
        comparison_op = ''
        lhs = rhs = ''
        rule = 'boolean'
        if expr_node.type == 'binary_expression':
            operator_node = expr_node.child_by_field_name('operator')
            operator = _text(source, operator_node) if operator_node is not None else ''
            if operator in _COMPARISON_OPERATORS:
                comparison_op = operator
                left = expr_node.child_by_field_name('left')
                right = expr_node.child_by_field_name('right')
                lhs = _text(source, left) if left is not None else ''
                rhs = _text(source, right) if right is not None else ''
                rule = 'operator'
            elif operator in _LOGICAL_OPERATORS:
                rule = 'logical_expression'
        cases.append(ExtractedCase(
            static_id=index,
            line=node.start_point[0] + 1,
            column=node.start_point[1] + 1,
            end_line=node.end_point[0] + 1,
            end_column=node.end_point[1] + 1,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            expression=expression,
            comparison_op=comparison_op,
            lhs=lhs,
            rhs=rhs,
            comparison_rule=rule,
            dynamic=dynamic,
            conditional=conditional,
            semantics_risk=_has_risk(expr_node),
            function=_function_name(source, node),
            source_text=_text(source, node),
        ))
    # Rewrite from the end so byte offsets stay valid.
    rewritten = source
    for case in sorted(cases, key=lambda c: c.start_byte, reverse=True):
        replacement = _replacement(case, language).encode('utf-8')
        rewritten = rewritten[:case.start_byte] + replacement + rewritten[case.end_byte:]
    include = b'#include "/input/dec_diag.h"\n'
    rewritten = include + rewritten
    return Manifest('OK', language, cases=cases,
                    rewritten_harness=rewritten.decode('utf-8'),
                    harness_hash=digest)


def manifest_from_dict(data: dict) -> dict:
    """Validate/return a cached manifest dictionary (schema version check)."""
    if not isinstance(data, dict) or data.get('parser_version') != PARSER_VERSION:
        return {}
    return data


DIAG_HEADER = r'''
#ifndef DEC_DIAG_H
#define DEC_DIAG_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>
#include <stddef.h>

static int dec_diag_index = 0;
static int dec_diag_stop = -1;
static int dec_diag_ready = 0;
static int dec_diag_skip[512];
static int dec_diag_skip_count = 0;
static const char *dec_diag_path = "/work/dec_cases.bin";

static void dec_diag_init(void) {
    if (dec_diag_ready) return;
    dec_diag_ready = 1;
    const char *stop = getenv("DEC_DIAG_STOP");
    dec_diag_stop = stop ? atoi(stop) : -1;
    const char *skip = getenv("DEC_DIAG_SKIP");
    while (skip && *skip && dec_diag_skip_count < 512) {
        dec_diag_skip[dec_diag_skip_count++] = atoi(skip);
        while (*skip && *skip != ',') skip++;
        if (*skip == ',') skip++;
    }
    const char *path = getenv("DEC_DIAG_PATH");
    if (path && *path) dec_diag_path = path;
}

static int dec_diag_is_skipped(int idx) {
    int i;
    for (i = 0; i < dec_diag_skip_count; i++) {
        if (dec_diag_skip[i] == idx) return 1;
    }
    return 0;
}

static void dec_diag_field(int fd, const char *text) {
    size_t n = text ? strlen(text) : 0;
    if (n) { ssize_t ignored = write(fd, text, n); (void)ignored; }
    ssize_t ignored2 = write(fd, "\0", 1); (void)ignored2;
}

static int dec_emit(int static_id, int line, int ok,
                    const char *expr, const char *lhs, const char *rhs,
                    const char *actual, const char *expected,
                    int actual_ok, int expected_ok) {
    int idx;
    int fd;
    char b1[32], b2[32], b3[32], b4[8], b5[8];
    dec_diag_init();
    idx = ++dec_diag_index;
    snprintf(b1, sizeof b1, "%d", idx);
    snprintf(b2, sizeof b2, "%d", static_id);
    snprintf(b3, sizeof b3, "%d", line);
    snprintf(b4, sizeof b4, "%d", ok ? 1 : 0);
    snprintf(b5, sizeof b5, "%d", actual_ok ? 1 : 0);
    fd = open(dec_diag_path, O_WRONLY | O_CREAT | O_APPEND, 0600);
    if (fd >= 0) {
        char b6[8];
        snprintf(b6, sizeof b6, "%d", expected_ok ? 1 : 0);
        dec_diag_field(fd, b1);
        dec_diag_field(fd, b2);
        dec_diag_field(fd, b3);
        dec_diag_field(fd, b4);
        dec_diag_field(fd, b5);
        dec_diag_field(fd, b6);
        dec_diag_field(fd, expr);
        dec_diag_field(fd, lhs);
        dec_diag_field(fd, rhs);
        dec_diag_field(fd, actual);
        dec_diag_field(fd, expected);
        dec_diag_field(fd, "END");
        close(fd);
    }
    return idx;
}

static void dec_control(int idx, int ok) {
    dec_diag_init();
    if (!ok && !dec_diag_is_skipped(idx)) _exit(3);
    if (dec_diag_stop > 0 && idx >= dec_diag_stop) _exit(0);
}

static int dec_emit_bool(int static_id, int line, int ok, const char *expr) {
    /* C++ diagnostics keep the assertion verdict but do not pretend to have
       typed operand values. */
    return dec_emit(static_id, line, ok, expr, expr, "1", "", "", 0, 0);
}

#ifndef __cplusplus
static int dec_fmt_schar(const signed char *p, char *b, size_t n) { snprintf(b, n, "%d", (int)*p); return 1; }
static int dec_fmt_uchar(const unsigned char *p, char *b, size_t n) { snprintf(b, n, "%u", (unsigned)*p); return 1; }
static int dec_fmt_char(const char *p, char *b, size_t n) { snprintf(b, n, "%d", (int)*p); return 1; }
static int dec_fmt_short(const short *p, char *b, size_t n) { snprintf(b, n, "%d", (int)*p); return 1; }
static int dec_fmt_ushort(const unsigned short *p, char *b, size_t n) { snprintf(b, n, "%u", (unsigned)*p); return 1; }
static int dec_fmt_int(const int *p, char *b, size_t n) { snprintf(b, n, "%d", *p); return 1; }
static int dec_fmt_uint(const unsigned int *p, char *b, size_t n) { snprintf(b, n, "%u", *p); return 1; }
static int dec_fmt_long(const long *p, char *b, size_t n) { snprintf(b, n, "%ld", *p); return 1; }
static int dec_fmt_ulong(const unsigned long *p, char *b, size_t n) { snprintf(b, n, "%lu", *p); return 1; }
static int dec_fmt_ll(const long long *p, char *b, size_t n) { snprintf(b, n, "%lld", *p); return 1; }
static int dec_fmt_ull(const unsigned long long *p, char *b, size_t n) { snprintf(b, n, "%llu", *p); return 1; }
static int dec_fmt_bool(const _Bool *p, char *b, size_t n) { snprintf(b, n, "%d", *p ? 1 : 0); return 1; }
static int dec_fmt_float(const float *p, char *b, size_t n) { snprintf(b, n, "%.9g", (double)*p); return 1; }
static int dec_fmt_double(const double *p, char *b, size_t n) { snprintf(b, n, "%.17g", *p); return 1; }
static int dec_fmt_ldouble(const long double *p, char *b, size_t n) { snprintf(b, n, "%.21Lg", *p); return 1; }
static int dec_fmt_cstr(char *const *p, char *b, size_t n) { snprintf(b, n, "%s", *p ? *p : "(null)"); return 1; }
static int dec_fmt_cstr_const(const char *const *p, char *b, size_t n) { snprintf(b, n, "%s", *p ? *p : "(null)"); return 1; }
static int dec_fmt_unknown(const void *p, char *b, size_t n) { (void)p; b[0] = 0; (void)n; return 0; }

#define DEC_FMT(v, b, n) _Generic(&(v), \
    signed char *: dec_fmt_schar, const signed char *: dec_fmt_schar, \
    unsigned char *: dec_fmt_uchar, const unsigned char *: dec_fmt_uchar, \
    char *: dec_fmt_char, const char *: dec_fmt_char, \
    short *: dec_fmt_short, const short *: dec_fmt_short, \
    unsigned short *: dec_fmt_ushort, const unsigned short *: dec_fmt_ushort, \
    int *: dec_fmt_int, const int *: dec_fmt_int, \
    unsigned int *: dec_fmt_uint, const unsigned int *: dec_fmt_uint, \
    long *: dec_fmt_long, const long *: dec_fmt_long, \
    unsigned long *: dec_fmt_ulong, const unsigned long *: dec_fmt_ulong, \
    long long *: dec_fmt_ll, const long long *: dec_fmt_ll, \
    unsigned long long *: dec_fmt_ull, const unsigned long long *: dec_fmt_ull, \
    _Bool *: dec_fmt_bool, const _Bool *: dec_fmt_bool, \
    float *: dec_fmt_float, const float *: dec_fmt_float, \
    double *: dec_fmt_double, const double *: dec_fmt_double, \
    long double *: dec_fmt_ldouble, const long double *: dec_fmt_ldouble, \
    char **: dec_fmt_cstr, const char **: dec_fmt_cstr_const, \
    default: dec_fmt_unknown)(&(v), b, n)
#endif

#endif
'''
