from typing import Literal
from pydantic import BaseModel, Field, field_validator
from .config import MAX_COMPILE_CONCURRENCY, MAX_INFERENCE_CONCURRENCY

DEFAULT_PROMPT = '# This is the assembly code:\n{assembly}\n# What is the source code?\n'
OPTIMIZATIONS = ['O0', 'O1', 'O2', 'O3', 'Os', 'Ofast']


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


class _OptimizationMixin(BaseModel):
    optimizations: list[str] = Field(default_factory=lambda:['O0','O1','O2','O3'])
    sample_limit: int = Field(default=0,ge=0,le=100000)

    @field_validator('optimizations')
    @classmethod
    def check_opts(cls, values):
        if not values or not set(values).issubset(set(OPTIMIZATIONS)):
            raise ValueError('Select at least one valid optimization level')
        return list(dict.fromkeys(values))


class RunCreate(_OptimizationMixin):
    dataset_id: int = Field(gt=0)
    model_id: int = Field(gt=0)
    prompt_template: str = Field(default=DEFAULT_PROMPT,min_length=10,max_length=4000)
    temperature: float = Field(default=0,ge=0,le=2)
    max_tokens: int = Field(default=1024,ge=128,le=8192)
    # None means "use the server default" so environment configuration stays
    # authoritative unless the client explicitly overrides it.
    inference_concurrency: int | None = Field(default=None, ge=0, le=MAX_INFERENCE_CONCURRENCY)
    compile_concurrency: int | None = Field(default=None, ge=1, le=MAX_COMPILE_CONCURRENCY)
    execution_mode: Literal['docker', 'host'] = 'docker'
    host_execution_ack: bool = False


class ReferenceCheckCreate(_OptimizationMixin):
    compile_concurrency: int | None = Field(default=None, ge=1, le=MAX_COMPILE_CONCURRENCY)


class BatchRunCreate(RunCreate):
    """Offline generation through vLLM ``LLM.generate``."""

    model_path: str | None = Field(default=None, max_length=1000)
    batch_size: int | None = Field(default=None, ge=1, le=4096)


class DeploymentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    served_name: str | None = Field(default=None, max_length=100)
    model_path: str = Field(min_length=1, max_length=1000)
    gpu_ids: str = Field(default='0', max_length=100)
    port: int = Field(ge=1024, le=65535)
    dtype: Literal['auto', 'half', 'float16', 'bfloat16', 'float32'] = 'auto'
    max_model_len: int = Field(default=4096, ge=128, le=1_000_000)
    max_num_seqs: int = Field(default=64, ge=1, le=4096)
    max_num_batched_tokens: int = Field(default=8192, ge=1, le=1_000_000)
    gpu_memory_utilization: float = Field(default=0.7, ge=0.05, le=0.99)
    deterministic: bool = True
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    disable_prefix_caching: bool = True
    enforce_eager: bool = False
    api_mode: Literal['completion', 'chat'] = 'completion'


class ExternalDeploymentCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    base_url: str = Field(min_length=7, max_length=500)
    served_name: str = Field(default='', max_length=100)
    api_mode: Literal['completion', 'chat'] = 'completion'


class ServerDatasetImport(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    version: str = Field(default='v1', max_length=40)
    description: str = Field(default='', max_length=1000)
    path: str = Field(min_length=1, max_length=2000)
