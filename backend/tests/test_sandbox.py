from app.sandbox import extract_c, build_c_file, parse_docker_output


def test_extract_markdown():
    assert extract_c('```c\nint x(void){return 1;}\n```') == 'int x(void){return 1;}'
    assert extract_c('int x(void){return 1;}') == 'int x(void){return 1;}'


def test_build_single_main_with_two_tests():
    code=build_c_file('int add(int x,int y){return x+y;}', '', [
        {'name':'one','code':'if (add(1,2)!=3) return 1;'},
        {'name':'two','code':'if (add(2,2)!=4) return 1;'},
    ])
    assert 'case 0:' in code and 'case 1:' in code
    assert code.count('int main(') == 1


def test_parse_success():
    out=parse_docker_output('__DEC_COMPILE__:0\n\n__DEC_LOG_END__\n__DEC_TEST__:0:0\n__DEC_TEST__:1:0\n',2)
    assert out.status=='TEST_PASS' and out.compile_success is True
    assert out.tests_passed==2 and out.tests_total==2


def test_parse_wrong_answer():
    out=parse_docker_output('__DEC_COMPILE__:0\n\n__DEC_LOG_END__\n__DEC_TEST__:0:1\n__DEC_TEST__:1:0\n',2)
    assert out.status=='WRONG_ANSWER' and out.tests_passed==1


def test_parse_compiler_error():
    out=parse_docker_output('__DEC_COMPILE__:1\nerror: expected expression\n__DEC_LOG_END__\n',1)
    assert out.status=='COMPILE_ERROR' and 'expected expression' in out.compile_log


def test_parse_missing_markers_is_not_false_model_failure():
    out=parse_docker_output('docker: permission denied',2)
    assert out.status=='INFRA_ERROR' and out.compile_success is None
