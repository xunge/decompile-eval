from typing import Literal
from pydantic import BaseModel, Field, field_validator

DEFAULT_PROMPT = '# This is the assembly code:\n{assembly}\n# What is the source code?\n'

class ModelCreate(BaseModel):
    name: str = Field(min_length=1,max_length=100)
    kind: Literal['openai'] = 'openai'
    base_url: str = Field(min_length=7,max_length=500)
    api_model: str = Field(min_length=1,max_length=200)
    api_mode: Literal['chat','completion'] = 'chat'
    api_key_env: str = Field(default='',max_length=100)
    system_prompt: str = Field(default='You are an expert binary decompiler. Output only valid C source code.',max_length=1000)

    @field_validator('api_key_env')
    @classmethod
    def valid_env(cls, v):
        if v and not (v.replace('_','').isalnum() and (v[0].isalpha() or v[0]=='_')):
            raise ValueError('api_key_env must be an environment variable name')
        return v

class RunCreate(BaseModel):
    dataset_id: int = Field(gt=0)
    model_id: int = Field(gt=0)
    optimizations: list[str] = Field(default_factory=lambda:['O0','O1','O2','O3'])
    sample_limit: int = Field(default=0,ge=0,le=12000)
    prompt_template: str = Field(default=DEFAULT_PROMPT,min_length=10,max_length=4000)
    temperature: float = Field(default=0,ge=0,le=2)
    max_tokens: int = Field(default=1024,ge=128,le=8192)

    @field_validator('optimizations')
    @classmethod
    def check_opts(cls, values):
        allowed={'O0','O1','O2','O3','Os','Ofast'}
        if not values or not set(values).issubset(allowed):
            raise ValueError('Select at least one valid optimization level')
        return list(dict.fromkeys(values))
