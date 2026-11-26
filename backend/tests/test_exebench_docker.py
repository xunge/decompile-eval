"""Opt-in real Docker tests for the ExeBench wrapper adapter."""
import os
import pytest

from app.exebench import evaluate_exebench

pytestmark = pytest.mark.skipif(os.getenv('RUN_DOCKER_TESTS') != '1',
                                reason='Set RUN_DOCKER_TESTS=1 to test real Docker')

WRAPPER = r'''
#include <nlohmann/json.hpp>
#include <fstream>
extern "C" {
#include "candidate.c"
}
int main(int argc, char** argv) {
  nlohmann::json input;
  std::ifstream(argv[1]) >> input;
  nlohmann::json output;
  output["value"] = func(input["x"]);
  std::ofstream(argv[2]) << output;
  return 0;
}
'''

BAD_WRAPPER = r'''
#include <nlohmann/json.hpp>
#include <fstream>
extern "C" {
#include "candidate.c"
}
int main(int argc, char** argv) {
  std::ofstream(argv[2]) << "{not-json";
  return 0;
}
'''


def _tests():
    return [
        {'name': 'test_0', 'input': {'x': 1}, 'expected_output': {'value': 2},
         'dummy_funcs': 'int helper(int x){return x+1;}',
         'dummy_funcs_seed': 1, 'case_index': 0},
        {'name': 'test_1', 'input': {'x': 2}, 'expected_output': {'value': 4},
         'dummy_funcs': 'int helper(int x){return x*2;}',
         'dummy_funcs_seed': 2, 'case_index': 1},
    ]


def test_dummy_funcs_are_compiled_per_test_group():
    source = 'int helper(int);\nint func(int x){return helper(x);}'
    out = evaluate_exebench(source, source, '', WRAPPER, _tests(),
                            compile_timeout=30, test_timeout=10)
    assert out.status == 'TEST_PASS', f'{out.error}; {out.compile_log}'
    assert out.tests_passed == 2 and out.tests_total == 2
    assert out.tests[0]['actual_output'] == {'value': 2}
    assert out.tests[1]['actual_output'] == {'value': 4}


def test_wrapper_invalid_json_is_not_pass():
    source = 'int func(int x){return x;}'
    tests = [{'name': 'test_0', 'input': {'x': 1}, 'expected_output': {'value': 1},
              'dummy_funcs': '', 'dummy_funcs_seed': None, 'case_index': 0}]
    out = evaluate_exebench(source, source, '', BAD_WRAPPER, tests,
                            compile_timeout=30, test_timeout=10)
    assert out.status == 'INVALID_OUTPUT'
    assert out.tests[0]['status'] == 'INVALID_OUTPUT'
    assert out.tests[0]['actual_output'] is None


def test_reference_oracle_detects_wrong_expected_output():
    source = 'int func(int x){return x;}'
    tests = [{'name': 'test_0', 'input': {'x': 1}, 'expected_output': {'value': 2},
              'dummy_funcs': '', 'dummy_funcs_seed': None, 'case_index': 0}]
    out = evaluate_exebench(source, source, '', WRAPPER, tests,
                            compile_timeout=30, test_timeout=10)
    assert out.status == 'WRONG_ANSWER'
    assert out.tests[0]['actual_output'] == {'value': 1}
