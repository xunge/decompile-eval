"""Tree-sitter harness extraction/rewrite tests (no Docker)."""
from app.case_extract import PARSER_VERSION, extract_manifest


def test_sequential_asserts_and_comparisons():
    harness = '''#include <assert.h>
#include <math.h>
int main(){
    int a = 3;
    assert(f(a) == 1);              /* trailing comment */
    assert(fabs(g(1.5) - 1.5) < 1e-6);
    assert(strcmp(name, "abc") == 0);
    return 0;
}
'''
    manifest = extract_manifest(harness, 'c')
    assert manifest.status == 'OK'
    assert len(manifest.cases) == 3
    first, second, third = manifest.cases
    assert first.comparison_op == '==' and first.lhs == 'f(a)' and first.rhs == '1'
    assert first.line == 5 and first.column == 5
    assert second.comparison_op == '<' and second.rhs == '1e-6'
    assert third.lhs == 'strcmp(name, "abc")' and third.rhs == '0'
    assert 'dec_emit(' in manifest.rewritten_harness
    assert 'assert(f(a)' not in manifest.rewritten_harness


def test_loop_assert_is_marked_dynamic():
    harness = '#include <assert.h>\nint main(){ for(int i=0;i<3;i++){ assert(g(i)==i); } return 0; }\n'
    manifest = extract_manifest(harness, 'c')
    assert manifest.status == 'OK'
    assert len(manifest.cases) == 1
    assert manifest.cases[0].dynamic is True


def test_assert_inside_conditional_is_dynamic():
    harness = '#include <assert.h>\nint main(){ if (flag) assert(f()==1); return 0; }\n'
    case = extract_manifest(harness, 'c').cases[0]
    assert case.dynamic is True and case.conditional is True


def test_strings_and_comments_do_not_create_fake_cases():
    harness = '''#include <assert.h>
int main(){
    printf("assert(fake)");
    /* assert(also_fake); */
    assert(real(1)==1);
    return 0;
}
'''
    manifest = extract_manifest(harness, 'c')
    assert len(manifest.cases) == 1
    assert manifest.cases[0].expression == 'real(1)==1'


def test_multiline_assert_span_and_expression():
    harness = '''#include <assert.h>
int main(){
    assert(
        f(1) ==
        1
    );
    return 0;
}
'''
    case = extract_manifest(harness, 'c').cases[0]
    assert case.line == 3
    assert case.end_line == 6
    assert 'f(1)' in case.lhs and case.rhs.strip() == '1'


def test_no_asserts_is_reported_not_fabricated():
    manifest = extract_manifest('#include <stdio.h>\nint main(){return 0;}\n', 'c')
    assert manifest.status == 'NO_ASSERT_CASES'
    assert manifest.cases == []
    assert manifest.rewritten_harness == ''


def test_parse_error_is_reported():
    manifest = extract_manifest('int main({ assert(1); }', 'c')
    assert manifest.status == 'PARSE_ERROR'
    assert manifest.cases == []


def test_cpp_manifest_marks_values_unsupported_but_keeps_cases():
    harness = '#include <cassert>\nint main(){ assert(f(2)==2); return 0; }\n'
    manifest = extract_manifest(harness, 'cpp')
    assert manifest.status == 'OK'
    assert manifest.language == 'cpp'
    assert len(manifest.cases) == 1
    assert 'dec_emit_bool' in manifest.rewritten_harness


def test_ndebug_harness_is_not_diagnosed():
    manifest = extract_manifest('#define NDEBUG\n#include <assert.h>\nint main(){assert(1);return 0;}\n', 'c')
    assert manifest.status == 'UNSUPPORTED_CASE_EXTRACTION'


def test_parser_version_present():
    assert 'tree-sitter' in PARSER_VERSION
