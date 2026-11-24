# DecompileEval v0.4 — 大模型反编译并发评测平台

> Vue 3 + FastAPI + SQLite + Docker + vLLM（OpenAI 兼容 Completion API）。
> **v0.4 的核心变化：动态槽位调度（默认 128 在途推理请求）、真实 Tokenizer 上下文预算与 INPUT_TOO_LONG、可选 vLLM 离线批量适配器、Token/s 与 vLLM /metrics 遥测。**
> v0.3 已实现的异步推理/编译流水线、结构化日志与参考源码基线继续保持。
> 演示数据仅用于验证流程，不作为公开基准成绩。真实模型复现步骤见 [REAL_BENCHMARKS.md](REAL_BENCHMARKS.md)。

## 一、已实现功能

- **数据集管理**：导入统一 JSONL，查看样本与 O0/O1/O2/O3；支持 C/C++。
- **模型管理**：OpenAI 兼容接口（vLLM Completion / Chat），API Key 仅从后端环境变量读取。
- **离线评测**：导入已有 `sample_id` + `prediction`，不重新推理。
- **参考源码基线**：用**完全相同的编译器、测试程序、Docker 环境**执行数据集自带的 `reference_source`，结果保存为独立任务；失败表示数据集/环境问题，而不是模型错误。
- **动态并发调度**：默认最多 **128 个在途推理请求**（可配置，测试过 16/32/64/128）；任意请求完成立即补位，不等待同批其他请求，也不等待 Docker。
- **异步并发评测**：推理（`httpx.AsyncClient` 连接池）与 Docker 编译/测试（有界编译 worker + 有界队列）并行；推理完成即落盘并进入编译队列。
- **离线批量适配器**：可选 `OfflineBatchAdapter`，通过本地 `vllm.LLM.generate(prompts, sampling_params)` 批量生成；每个样本仍是独立 Prompt，不拼接函数；现有 `OnlineAdapter` 负责实时流水线。
- **上下文安全**：使用 vLLM 真实 `/tokenize` 统计输入 Token，按 4096 上下文动态收缩输出预算（不改写 Prompt/温度），超限记录 `INPUT_TOO_LONG`，绝不自动截断汇编。
- **性能遥测**：请求数、Token/s、samples/s、GPU 推理耗时、编译/测试耗时、沙箱总耗时、任务总耗时；可选读取 vLLM `/metrics`（运行/等待请求、KV 缓存）。
- **持久化与恢复**：逐样本结果即时写 SQLite；支持取消、重试、断点续跑；已完成样本不会重复评测。
- **诊断日志**：容器内 Python runner 输出结构化 JSON；保留编译/测试 stdout、stderr、退出码、信号、超时、耗时和截断标记。
- **错误分类**：`TEST_PASS / WRONG_ANSWER / RUNTIME_ERROR / TIMEOUT / COMPILE_ERROR / COMPILE_TIMEOUT / MODEL_EMPTY`，基础设施问题单独记为 `INFRA_ERROR`，模型接口失败记为 `INFERENCE_ERROR`。
- **指标与可视化**：RC / RE / TCP、按优化等级统计、模型对比、推理/编译双进度、吞吐量、逐样本三栏与日志。
- **REST API**：FastAPI Swagger `http://127.0.0.1:8010/docs`。

当前定位仍是**单用户、单机、本地开发**：单进程内运行、一次评估一个任务、无鉴权，不要暴露公网。

## 二、流水线架构

v0.2 是严格串行：`推理 → 编译 → 测试 → 下一个样本`，GPU 在 Docker 编译期间空闲。v0.4 改为：

```
样本列表
   │
   ▼
推理生产者（最多 inference_concurrency=128 个在途协程，httpx.AsyncClient 连接池复用；
            任一请求完成立即从待推理队列补位，不等待同批其他请求）
   │  每个样本生成完成 → 立即 INSERT results(status=GENERATED, model_output, tokens)
   ▼
有界编译队列（maxsize = MAX_PENDING_COMPILE，满时反压推理）
   │
   ▼
编译 worker（compile_concurrency 个协程，asyncio 子进程调 docker，互不阻塞事件循环）
   │  每个样本完成 → 立即 UPDATE results 为终态 + 日志/信号/耗时
   ▼
SQLite（WAL + busy_timeout + 进程内写锁）
```

- 每个样本使用独立的临时目录和独立 Docker 容器，互不共享文件名。
- 一个任务的推理、编译、持久化全部结束后才判定任务完成；单个样本异常不会卡住队列。
- 本版本一次只执行一个任务（单机 MVP），但单个任务内部完全流水化。
- 进度计数器保存在内存，逐样本结果持久化在 `results`；服务重启后由数据库重算进度并允许 Retry 续跑。

### 任务/样本状态

| 类型 | 状态 |
|---|---|
| 任务 | `queued → running → completed / completed_with_errors / cancelled / interrupted` |
| 样本中间态 | `GENERATED`（已生成待编译，仅存在于运行中/取消后） |
| 样本终态 | `TEST_PASS, WRONG_ANSWER, RUNTIME_ERROR, TIMEOUT, COMPILE_ERROR, COMPILE_TIMEOUT, MODEL_EMPTY` |
| 排除统计 | `INFRA_ERROR`（Docker/权限/环境）、`INFERENCE_ERROR`（模型接口） |

`GENERATED`/`COMPILING` 不计入 RC/RE/TCP 分母；旧数据库没有这些状态，指标口径与 v0.2 一致。

## 三、Ubuntu 24.04 运行

### 0. 依赖

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip nodejs npm docker.io
sudo systemctl enable --now docker
docker pull gcc:14
docker run --rm gcc:14 gcc --version
```

运行 API 的用户需要 Docker 权限。**不要用 `sudo docker` 或 `chmod 666 /var/run/docker.sock` 绕过权限**；正确做法是 rootless Docker，或仅在可信开发机上加入 docker 组（等同近似 root 权限）。

### 1. 后端

```bash
cd decompile-eval-mvp/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

检查：

```bash
curl http://127.0.0.1:8010/api/health
curl http://127.0.0.1:8010/api/openapi.json | head
```

### 2. 前端

```bash
cd decompile-eval-mvp/frontend
npm install
npm run dev          # http://127.0.0.1:5173
```

### 3. 第一份评测（无需模型）

「新建评测任务」→ 数据集选「演示数据集（4个函数）」→ 模型选「演示固定输出（非真实LLM）」→ 创建。Docker 不可用时状态为 `INFRA_ERROR`，不会产生虚假百分比。

### 4. 接入 LLM4Decompile-1.3B-v1.5

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/llm4decompile-1.3b-v1.5 \
  --served-model-name llm4decompile-1.3b-v1.5 \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.70
```

模型管理页面注册：Base URL `http://127.0.0.1:8001/v1`、Model ID `llm4decompile-1.3b-v1.5`、接口类型 **Completion**、Key 环境变量留空。Prompt 必须保持：

```text
# This is the assembly code:
{assembly}
# What is the source code?
```

## 四、配置项

所有参数都可通过环境变量覆盖（在启动后端前设置），任务创建时可在前端覆盖两个并发数。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `INFERENCE_CONCURRENCY` | 128 | 单任务在途推理请求上限（1–512） |
| `COMPILE_CONCURRENCY` | 4 | 单任务 Docker 编译/测试并发上限（1–64） |
| `MAX_PENDING_COMPILE` | 128 | 已生成但等待编译的样本上限（背压） |
| `INFERENCE_TIMEOUT` | 180 | 单次模型请求超时（秒） |
| `INFERENCE_RETRIES` | 2 | 429/5xx/连接错误的重试次数 |
| `COMPILE_TIMEOUT` | 60 | 编译超时（秒） |
| `TEST_TIMEOUT` | 10 | 单个测试用例运行超时（秒） |
| `MAX_MODEL_LEN` | 4096 | 模型上下文；输入+输出超过即 `INPUT_TOO_LONG` |
| `TOKENIZER_TIMEOUT` | 10 | vLLM `/tokenize` 超时 |
| `DEC_EVAL_BATCH_MODEL_PATH` | 空 | 离线批量模式的本地模型路径 |
| `DEC_EVAL_BATCH_SIZE` | 32 | 离线批量每轮样本数 |
| `DEC_EVAL_VLLM_METRICS` | `1` | 是否读取 vLLM `/metrics`（0 关闭） |
| `DEC_EVAL_LOG_LIMIT_BYTES` | 32768 | 每个 stdout/stderr 流保留上限，超限标记 truncated |
| `DEC_EVAL_SANDBOX_IMAGE` | `gcc:14` | 沙箱镜像 |
| `DEC_EVAL_DATA_DIR` | `./storage` | SQLite 与离线预测目录 |

`POST /api/runs` 可携带 `inference_concurrency`、`compile_concurrency`；服务端按 512/32 上限裁剪。
服务端默认值只是上限：调度器不会等批，任意请求结束即补位，因此 128 在途不会造成一次性创建全部样本的请求。

### 离线批量生成（可选）

`POST /api/runs/batch` 使用 `OfflineBatchAdapter`：

- 需要后端 Python 环境安装 `vllm`（否则返回 503）；模型路径通过 `model_path` 或 `DEC_EVAL_BATCH_MODEL_PATH` 指定。
- 每批把 **独立 Prompt 列表** 传给 `LLM.generate(prompts, SamplingParams)`，按列表位置映射回 `sample_id`，绝不拼接多个函数。
- 仍会做真实 Tokenizer 预算检查；超长样本单独记录 `INPUT_TOO_LONG`。
- 该模式是同步 GPU 调用，放在线程中执行；取消只在批次边界生效（见已知限制）。

## 五、数据集格式

```json
{
  "sample_id": "example_add_O0",
  "problem_id": "add",
  "optimization": "O0",
  "compiler": "gcc",
  "architecture": "x86_64",
  "language": "c",
  "assembly": "add:\n  mov eax, edi\n  add eax, esi\n  ret",
  "reference_source": "int add(int a, int b) { return a + b; }",
  "dependencies": "",
  "tests": [
    {"name":"positive", "code":"if (add(2, 3) != 5) return 1;"}
  ]
}
```

- 模型唯一输入是 `assembly`；`reference_source`、`dependencies`、`tests` 绝不进入 Prompt。
- `tests[].code` 会被放入合成 `main()` 的 switch 分支，每个用例独立进程、独立计时。
- `tests[].program` 表示上游完整 harness（自带 `main`），整个 harness 只执行一次；此时 `tests_total = 1`，**TCP 等于 RE，不是断言级 TCP**，前端会明确提示“Test 0 是整套程序，不是第 1 个断言”。
- `dependencies` 只允许来自已审核的可信 Benchmark。

## 六、指标定义

对状态不属于 `{INFRA_ERROR, INFERENCE_ERROR, GENERATED, COMPILING}` 的有效样本：

- **RC** = 编译成功样本数 / 有效样本数
- **RE** = 全部测试通过样本数 / 有效样本数
- **TCP** = 通过用例总数 / 有效样本用例总数（用例加权；完整 harness 只有 1 个用例）

编译失败、空输出、`INPUT_TOO_LONG` 属于有效样本失败并计入分母（不能通过删除超长样本来提高 RE）；Docker 启动失败、权限错误、模型接口失败不计入分母并单独计数。RE 只代表给定测试集全部通过，不代表严格语义等价。不同公开 Benchmark 的指标口径可能不同，报告时必须固定数据集、测试用例、评测器版本和统计定义。

## 七、错误诊断与 exit 134

容器内 `runner.py` 逐事件输出 JSON（编译 1 条 + 每个测试 1 条），包含 `exit_code`、`signal`、`timed_out`、`duration_ms`、`stdout`、`stderr`、`*_truncated`。退出码与信号来自 `subprocess` 的 `returncode`，因此 **exit(134) 与 SIGABRT 可区分**。

| 现象 | 分类 |
|---|---|
| 编译失败 | `COMPILE_ERROR`（保留编译 stdout/stderr） |
| 编译超时 | `COMPILE_TIMEOUT` |
| 测试超时 | `TIMEOUT` |
| `SIGABRT` + stderr 中明确 `Assertion ... failed` | `WRONG_ANSWER`（断言失败） |
| `SIGABRT`（abort/double free/堆损坏/无断言信息） | `RUNTIME_ERROR` |
| `SIGSEGV`/`SIGBUS`/`SIGFPE` 等 | `RUNTIME_ERROR` |
| 非 0 且非 1 的普通退出码（含 `exit(134)`） | `RUNTIME_ERROR` |
| 非 0/1 且无信号、返回 1 | `WRONG_ANSWER` |
| Docker 不存在/权限/容器异常 | `INFRA_ERROR`（排除） |
| 模型空输出 | `MODEL_EMPTY`（有效失败） |
| 输入 Token + 1 超过 `max_model_len` | `INPUT_TOO_LONG`（有效失败，不截断、不排除） |
| 模型 HTTP/超时重试后仍失败 | `INFERENCE_ERROR`（排除） |

修改 `backend/app/sandbox.py` 时不要放宽上述规则；Docker 资源限制（`--network none`、只读根、1g 内存、1 CPU、64 PIDs、非特权用户、tmpfs）保持不变。

## 八、主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/overview` | 计数概览 |
| GET/POST | `/api/datasets`, `/api/datasets/import` | 数据集 |
| GET | `/api/datasets/{id}/samples` | 样本列表 |
| POST | `/api/datasets/{id}/reference-check` | 创建参考源码基线任务（body: `optimizations`, `sample_limit`, `compile_concurrency`） |
| GET/POST | `/api/models`, `/api/models/{id}/test` | 模型 |
| POST | `/api/runs` | 创建在线评测（可选并发数） |
| POST | `/api/runs/batch` | 创建 vLLM 离线批量评测（`model_path`, `batch_size`） |
| POST | `/api/runs/offline` | 离线预测评测 |
| GET | `/api/runs`, `/api/runs/{id}` | 任务列表 / 详情（含 progress、metrics） |
| GET | `/api/runs/{id}/progress` | 轻量实时进度（前端 1s 轮询） |
| POST | `/api/runs/{id}/cancel`, `/api/runs/{id}/retry` | 取消 / 重试 |
| DELETE | `/api/runs/{id}`, `/api/runs` | 删除单个已结束任务 / 批量清理 |
| GET | `/api/results/{id}` | 样本详情（含 extracted_source、tests_detail） |
| GET | `/api/compare?run_ids=...` | 同数据集任务对比 |

`/api/models` 不返回内部参考模型；参考基线任务在任务列表中以「参考源码基线（内部）」显示。

## 九、进度与前端

任务详情实时显示（`/api/runs/{id}/progress`，1 秒刷新）：

- 总样本、等待推理、正在推理、已生成代码、等待编译、正在编译、已完成评测、失败样本
- 推理进度 `inference_done / total` 与编译测试进度 `evaluated / total` 分开显示
- 推理并发、编译并发、在途请求、推理请求数、推理吞吐（样本/s）、评测吞吐（样本/s）、Token/s、总 Token/s、输入超长、总耗时
- 可获取时显示 vLLM `/metrics`：GPU 运行/等待请求数、KV 缓存占用
- 参考源码基线任务会显示“失败代表数据集/环境问题”的提示

逐样本弹窗：输入汇编、实际 Prompt、生成参数（请求/生效 max_tokens、tokenizer 来源、temperature）、模型原始输出、提取后的 C 代码、参考源码、编译日志、测试 stdout/stderr、退出码、信号、错误分类、推理/编译/测试/沙箱/总耗时、Token 用量、完整 harness 提示。

## 十、数据库兼容与迁移

启动时自动做**增量迁移**（`ALTER TABLE ... ADD COLUMN`），不删除历史数据：

- `runs.started_at`
- `results.inference_ms, compile_ms, test_ms, sandbox_ms, prompt_tokens, completion_tokens, prompt_text, generation_params_json, detail_json`
- 新增 `idx_results_run_status`

旧任务照常查看；旧结果没有结构化 `detail_json` 时前端回退到 `compile_log/run_log`。删除任务仍会级联删除结果并清理离线预测文件。

## 十一、测试

```bash
cd decompile-eval-mvp/backend
source .venv/bin/activate
python -m pytest -q
# 77 passed, 9 skipped（Docker 用例默认跳过）
```

注意：必须用 `python -m pytest`（让 `backend/` 进入 `sys.path`）；直接 `pytest` 可能报 `No module named 'app'`。

真实 Docker 集成测试（可选）：

```bash
RUN_DOCKER_TESTS=1 python -m pytest -q tests/test_docker_integration.py
# 9 passed，覆盖：正常编译、assert失败(SIGABRT)、abort、SIGSEGV、超时、
# 编译错误、完整 harness、C++、特殊字符与日志截断
```

若后端运行在私有 PID/mount namespace 中，Docker 守护进程可能看不到它的 `/tmp`，此时把 `TMPDIR` 指向工作区内目录：

```bash
mkdir -p storage/.tmp-docker
TMPDIR=$PWD/storage/.tmp-docker RUN_DOCKER_TESTS=1 python -m pytest -q tests/test_docker_integration.py
```

测试分层：

- `tests/test_sandbox_errors.py`：结构化事件解析、assert/abort/SIGSEGV/超时/编译错误/Docker 权限、特殊字符、截断、legacy marker 兼容。
- `tests/test_pipeline.py`：16/128 并发上限、槽位即时补位、推理/编译真实重叠、乱序完成、推理失败不阻塞、编译失败不阻塞、取消停止派发、断点续跑不重复、重试不覆盖终态、INPUT_TOO_LONG 计入指标、Token/耗时遥测、SQLite 并发写、429 退避、progress 接口、参考源码任务、离线批量模式。
- `tests/test_adapters.py`：真实 Tokenizer 预算、超长判定、Prompt 不拼接、离线批量按位置映射、vLLM 指标解析、缓存。
- `tests/test_api.py`：原有 API、删除、迁移幂等、transient 指标排除、参考模型保护。
- `tests/test_api.py::test_metrics_excludes_infra_errors` 等：指标口径保持不变。
- `tests/test_docker_integration.py`：真实 Docker E2E（opt-in）。

## 十二、性能与实测结果

测试环境：LLM4Decompile-1.3B-v1.5，vLLM `http://127.0.0.1:8001/v1`，HumanEval-Decompile O0 前 128 条，`temperature=0`、`max_tokens=512`、输入 Prompt 完全一致；Token 预取不计入计时。

| 在途并发 | 请求数 | 失败 | 输入超长 | 总耗时 | samples/s | completion Token/s | 总 Token/s | p95 延迟 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 0 | 0 | 9.11s | 14.06 | 2355 | 15709 | 2.24s |
| 32 | 128 | 0 | 0 | 6.72s | 19.06 | 3215 | 21326 | 2.84s |
| 64 | 128 | 0 | 0 | 5.72s | 22.37 | 3716 | 24974 | 3.69s |
| 128 | 128 | 0 | 0 | 5.26s | 24.32 | 4000 | 27107 | 4.84s |

128 并发相对 16 并发：samples/s **1.73×**，completion Token/s **1.70×**，无失败、无超长样本。延迟随并发上升是 vLLM 排队造成的，属预期。

真实全流水线（同一模型 + 真实 Docker、`inference_concurrency=128`、`compile_concurrency=4`、16 条 O0）：

- 同一 16 样本、同一模型/参数：`inference_concurrency=16` 用时 **42.6s**，`=128` 用时 **22.5s**（1.9×）；
  两次 RC/RE/TCP **完全一致**（93.75% / 81.25% / 81.25%），状态分布相同，证明并发调度不改变评测语义；
- 平均 GPU 推理 0.59s/sample；Docker 沙箱总耗时累计 137s（每容器约 8.6s），4 worker 并行后成为瓶颈；
  容器内 gcc 编译本身仅累计 1.0s，说明 `sandbox_ms` 与 `compile_ms` 必须分开记录；
- 推理与编译重叠，GPU 不再等待 Docker 启动。

复现方法：

```bash
# 1) 纯推理吞吐（16/32/64/128）
source backend/.venv/bin/activate
python scripts/bench_inference.py --jsonl data/import/humaneval.jsonl \
  --opt O0 --limit 128 --concurrency 16,32,64,128 --max-tokens 512

# 2) 完整流水线：同一数据集/模型/Prompt，修改 inference_concurrency 后对比
curl -s -X POST http://127.0.0.1:8010/api/runs -H 'Content-Type: application/json' \
  -d '{"dataset_id":2,"model_id":2,"optimizations":["O0"],"sample_limit":16,
       "inference_concurrency":128,"compile_concurrency":4}'
curl -s http://127.0.0.1:8010/api/runs/1/progress   # token_rate/eval_rate/elapsed
```

## 十三、已知限制

1. 单机、单进程、一次一个任务；跨机/多任务并发需 Redis/RQ 等（本版不强求）。
2. 离线批量模式要求后端 Python 环境安装 `vllm`，且 `LLM.generate` 是同步 GPU 调用；批次执行中不能被取消，只在批次边界检查取消/背压。
3. 若 `/tokenize` 暂时不可用，会用保守估算并在 `generation_params.tokenizer=estimated` 标记；正常 vLLM 部署使用真实 Tokenizer。
4. 无鉴权、可注册任意模型 endpoint；不得公网暴露，公开部署需 SSRF/配额/速率限制。
5. 进度计数在内存中；重启后从数据库重算，当前进行中的计时会归零重来。
6. 前端用 1s 轮询而非 SSE；高频轮询已限制为轻量接口，vLLM `/metrics` 抓取有 2s 缓存。
7. 完整 harness 的 `tests_total=1`，TCP 不能冒充断言级统计。
8. 仅函数级 C/C++；不支持多文件链接、数据结构完全恢复或语义等价证明。
9. 数据集 SHA/模型版本/Prompt 哈希尚未自动写入实验快照（README 仍要求人工记录）。
10. 参考源码基线会把选中样本全部重跑一遍，数据集很大时耗时明显，可先限制 `sample_limit`。
11. `compile_ms` 是容器内 gcc 耗时；Docker 启动开销记录在 `sandbox_ms`，两者与 `inference_ms` 共同构成 `latency_ms`。

## 十四、项目结构

```text
backend/
  app/
    config.py     环境配置（并发、超时、Token 预算、日志上限）
    adapters.py   OnlineAdapter / OfflineBatchAdapter / 真实 Tokenizer / vLLM metrics
    db.py         SQLite 模型 + 增量迁移
    schemas.py    API 参数与 Prompt 模板
    sandbox.py    结构化 Docker 编译/测试、信号与错误分类（同步 + 异步）
    worker.py     异步生产者-消费者流水线、进度、取消/续跑、重试
    main.py       REST API、指标、删除、参考源码检查
    seed.py       演示样本
  tests/
    test_api.py               API/删除/迁移/指标
    test_sandbox_errors.py    日志与错误分类
    test_pipeline.py          并发流水线
    test_adapters.py           Adapter / Tokenizer / 离线批量 / metrics
    test_docker_integration.py 真实 Docker（opt-in）
    test_real_bench_adapters.py
    test_sandbox.py
frontend/src/App.vue            Vue 3 工作台（进度/日志/参考基线）
scripts/                        下载、转换、参考冒烟、推理吞吐基准（bench_inference.py）
data/import, data/raw           可重建的数据（被 .gitignore 忽略）
storage/                        SQLite 与离线预测（被忽略）
```
