import json
from app.sandbox import build_c_file, extract_c
from app.main import validate_sample
from app.worker import generate_prediction


def test_full_benchmark_program():
    sample={
        'sample_id':'humaneval_0_O0', 'assembly':'func0:\n ret',
        'reference_source':'int func0(int x){return x+1;}',
        'tests':[{'name':'upstream_harness',
                  'program':'#include <assert.h>\nint main(void){assert(func0(1)==2);return 0;}'}]
    }
    validated=validate_sample(sample,1)
    tests=json.loads(validated['tests_json'])
    program=build_c_file(sample['reference_source'],'',tests)
    assert program.count('int main')==1
    assert 'assert(func0(1)==2)' in program


def test_completion_api_payload(monkeypatch):
    captured={}
    class FakeResponse:
        def raise_for_status(self): pass
        def json(self): return {'choices':[{'text':'int func0(void){return 1;}'}]}
    class FakeClient:
        def __init__(self,**kw): pass
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def post(self,url,headers,json):
            captured.update({'url':url,'headers':headers,'payload':json})
            return FakeResponse()
    monkeypatch.setattr('app.worker.httpx.Client',FakeClient)
    model={ 'kind':'openai','api_mode':'completion','api_key_env':'',
            'base_url':'http://127.0.0.1:8001/v1',
            'api_model':'llm4decompile-1.3b-v1.5','system_prompt':'' }
    result=generate_prediction(model, {'assembly':'func0:\n ret'},
        {'prompt_template':'# This is the assembly code:\n{assembly}\n# What is the source code?\n',
         'temperature':0,'max_tokens':512})
    assert result.startswith('int func0')
    assert captured['url'].endswith('/v1/completions')
    assert captured['payload']['prompt'].startswith('# This is the assembly code:')
    assert 'messages' not in captured['payload']
