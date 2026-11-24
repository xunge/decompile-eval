"""Small illustrative fixtures. They are NOT HumanEval or published benchmark results."""
EXAMPLES = [
  {
    'sample_id':'demo_sum_O0', 'problem_id':'sum_two', 'optimization':'O0', 'compiler':'gcc',
    'architecture':'x86_64','language':'c',
    'assembly':'sum_two:\n  push rbp\n  mov rbp, rsp\n  mov DWORD PTR [rbp-4], edi\n  mov DWORD PTR [rbp-8], esi\n  mov edx, DWORD PTR [rbp-4]\n  mov eax, DWORD PTR [rbp-8]\n  add eax, edx\n  pop rbp\n  ret',
    'reference_source':'int sum_two(int a, int b) { return a + b; }',
    'demo_prediction':'int sum_two(int a, int b) { return a + b; }',
    'tests':[{'name':'正数','code':'if (sum_two(2, 3) != 5) return 1;'}, {'name':'负数','code':'if (sum_two(-5, 2) != -3) return 1;'}],
  },
  {
    'sample_id':'demo_max_O2', 'problem_id':'max_two', 'optimization':'O2', 'compiler':'gcc',
    'architecture':'x86_64','language':'c',
    'assembly':'max_two:\n  cmp edi, esi\n  mov eax, esi\n  cmovge eax, edi\n  ret',
    'reference_source':'int max_two(int a, int b) { return a >= b ? a : b; }',
    'demo_prediction':'int max_two(int a, int b) { return a >= b ? a : b; }',
    'tests':[{'name':'左大','code':'if (max_two(9, 2) != 9) return 1;'}, {'name':'右大','code':'if (max_two(-2, 4) != 4) return 1;'}],
  },
  {
    'sample_id':'demo_even_O2', 'problem_id':'is_even', 'optimization':'O2', 'compiler':'gcc',
    'architecture':'x86_64','language':'c',
    'assembly':'is_even:\n  mov eax, edi\n  not eax\n  and eax, 1\n  ret',
    'reference_source':'int is_even(int n) { return (n & 1) == 0; }',
    'demo_prediction':'int is_even(int n) { return (n & 1) != 0; }',
    'tests':[{'name':'偶数','code':'if (is_even(4) != 1) return 1;'}, {'name':'奇数','code':'if (is_even(7) != 0) return 1;'}],
  },
  {
    'sample_id':'demo_square_O0', 'problem_id':'square', 'optimization':'O0', 'compiler':'gcc',
    'architecture':'x86_64','language':'c',
    'assembly':'square:\n  push rbp\n  mov rbp, rsp\n  mov DWORD PTR [rbp-4], edi\n  mov eax, DWORD PTR [rbp-4]\n  imul eax, eax\n  pop rbp\n  ret',
    'reference_source':'int square(int n) { return n * n; }',
    'demo_prediction':'int square(int n) { return n * ; }',
    'tests':[{'name':'平方','code':'if (square(5) != 25) return 1;'}],
  }
]
