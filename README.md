# DecompileEval v0.6 — 大模型反编译并发评测平台

> Vue 3 + FastAPI + SQLite + Docker + vLLM（OpenAI 兼容 Completion API）。
> **v0.5 的核心变化：真实 TCP_micro/TCP_macro（不再把完整 harness 当成 1 个用例）、tree-sitter 逐用例诊断、官方 RE 与诊断用例分离、INFERENCE_CONCURRENCY=0 AUTO 持续投递、COMPILE_CONCURRENCY=64 默认与资源保护。v0.6 增加官方 RC 两阶段编译、vLLM 部署/GPU 监控、Edit Similarity 和 ExeBench 官方 wrapper 评测。**
> v0.4 的 Tokenizer 预算、离线批量适配器、Token/s 与 vLLM /metrics 遥测继续保持。
> v0.6 新增本地 vLLM 部署管理（启动/日志/停止/重启/恢复）与 nvidia-ml-py GPU 实时监控。
> 演示数据仅用于验证流程，不作为公开基准成绩。真实模型复现步骤见 [REAL_BENCHMARKS.md](REAL_BENCHMARKS.md)。

## 一、已实现功能

- **数据集管理**：导入统一 JSONL（默认不限制文件字节数、最多 100000 条样本）；文件来源可选「本地上传」或「服务器文件」，共用同一套校验和流式导入；查看样本与 O0/O1/O2/O3，并删除无评测任务的数据集；支持 C/C++。
- **模型管理**：OpenAI 兼容接口（vLLM Completion / Chat），API Key 仅从后端环境变量读取。
- **离线评测**：导入已有 `sample_id` + `prediction`，不重新推理。
- **参考源码基线**：用**完全相同的编译器、测试程序、Docker 环境**执行数据集自带的 `reference_source`，结果保存为独立任务；失败表示数据集/环境问题，而不是模型错误。
- **动态并发调度**：默认最多 **128 个在途推理请求**（可配置，测试过 16/32/64/128）；任意请求完成立即补位，不等待同批其他请求，也不等待 Docker。
- **异步并发评测**：推理（`httpx.AsyncClient` 连接池）与 Docker 编译/测试（有界编译 worker + 有界队列）并行；推理完成即落盘并进入编译队列。
- **离线批量适配器**：可选 `OfflineBatchAdapter`，通过本地 `vllm.LLM.generate(prompts, sampling_params)` 批量生成；每个样本仍是独立 Prompt，不拼接函数；现有 `OnlineAdapter` 负责实时流水线。
- **上下文安全**：使用 vLLM 真实 `/tokenize` 统计输入 Token，按 4096 上下文动态收缩输出预算（不改写 Prompt/温度），超限记录 `INPUT_TOO_LONG`，绝不自动截断汇编。
- **性能遥测**：请求数、Token/s、samples/s、GPU 推理耗时、编译/测试耗时、沙箱总耗时、任务总耗时；可选读取 vLLM `/metrics`（运行/等待请求、KV 缓存）。
- **真实 TCP**：tree-sitter 解析官方 C/C++ harness 的每个 `assert`，动态循环断言按运行时实例展开；首个断言失败后仍独立诊断后续用例；官方完整 harness 仍是 RE 唯一裁决。
- **逐用例结果**：输入/期望/实际、比较表达式、退出码/信号、stdout/stderr、耗时、来源与诊断状态全部入库；无法可靠序列化时值为 null 并标注，不编造。
- **AUTO 推理**：`INFERENCE_CONCURRENCY=0` 时使用有界动态水位持续投递，任一请求完成立即补位，不等待整批；连接池与待编译队列均有资源上限。
- **宿主机快速模式（可选）**：显式开启后跳过 Docker，直接使用本机 gcc/g++，适合可信本地快速迭代；默认仍使用 Docker。
- **编译资源保护**：默认 64，按 `/proc/meminfo` 可用内存与 CPU 自动收缩实际值，前端显示请求值与实际值。
- **持久化与恢复**：逐样本结果即时写 SQLite；支持取消、重试、断点续跑；已完成样本不会重复评测。
- **诊断日志**：容器内 Python runner 输出结构化 JSON；保留编译/测试 stdout、stderr、退出码、信号、超时、耗时和截断标记。
- **错误分类**：`TEST_PASS / WRONG_ANSWER / RUNTIME_ERROR / TIMEOUT / COMPILE_ERROR / COMPILE_TIMEOUT / MODEL_EMPTY`，基础设施问题单独记为 `INFRA_ERROR`，模型接口失败记为 `INFERENCE_ERROR`。
- **指标与可视化**：RC / RE / TCP / Edit Similarity、按优化等级统计、模型对比、推理/编译双进度、吞吐量、逐样本三栏与日志。
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
| `INFERENCE_CONCURRENCY` | 0 | 0=AUTO 自适应持续投递；>0 为固定在途上限（1–512） |
| `AUTO_INITIAL_INFLIGHT` | 32 | AUTO 初始水位 |
| `AUTO_MAX_INFLIGHT` | 256 | AUTO 水位资源上限 |
| `AUTO_STEP` | 16 | 有排队工作时每次完成增加的水位 |
| `COMPILE_CONCURRENCY` | 64 | Docker 编译/测试并发请求值（1–128） |
| `COMPILE_MEMORY_FRACTION` | 0.6 | 可用于编译容器的内存比例 |
| `COMPILE_MEMORY_PER_WORKER_MB` | 1024 | 每个容器内存预算（实际并发 = min(请求, 内存/CPU)) |
| `MAX_PENDING_COMPILE` | 128 | 已生成但等待编译的样本上限（背压） |
| `MAX_DIAGNOSTIC_CASES` | 200 | 单个 harness 动态用例扫描上限 |
| `DEC_EVAL_DIAGNOSTICS` | 1 | 是否启用逐用例诊断（0 关闭） |
| `INFERENCE_TIMEOUT` | 180 | 单次模型请求超时（秒） |
| `INFERENCE_RETRIES` | 2 | 429/5xx/连接错误的重试次数 |
| `COMPILE_TIMEOUT` | 60 | 编译超时（秒） |
| `TEST_TIMEOUT` | 10 | 单个测试用例运行超时（秒） |
| `MAX_MODEL_LEN` | 4096 | 模型上下文；输入+输出超过即 `INPUT_TOO_LONG` |
| `DEC_EVAL_MAX_DATASET_UPLOAD_BYTES` | 0 | 数据集上传字节上限；0=不限制（仍最多 100000 条样本） |
| `DEC_EVAL_SERVER_IMPORT_DIRS` | `data/import:data/staging` | 服务器端可选 JSONL 目录，`:` 分隔；不在目录内的路径一律拒绝 |
| `DEC_EVAL_ALLOW_HOST_EXECUTION` | 1 | 是否允许宿主机快速编译模式；每次任务仍需显式勾选风险确认；生产环境建议设为 0 |
| `DEC_EVAL_HOST_CC` / `DEC_EVAL_HOST_CXX` | gcc / g++ | 宿主机快速模式编译器 |
| `TOKENIZER_TIMEOUT` | 10 | vLLM `/tokenize` 超时 |
| `DEC_EVAL_BATCH_MODEL_PATH` | 空 | 离线批量模式的本地模型路径 |
| `DEC_EVAL_BATCH_SIZE` | 32 | 离线批量每轮样本数 |
| `DEC_EVAL_VLLM_METRICS` | `1` | 是否读取 vLLM `/metrics`（0 关闭） |
| `DEC_EVAL_VLLM_BIN` | `/home/jiang/venvs/vllm/bin/vllm` | 受信任的 vLLM 可执行文件 |
| `DEC_EVAL_ALLOW_DEPLOY` | `1` | 是否允许部署管理 |
| `DEC_EVAL_ADMIN_TOKEN` | 空 | 设置后部署写操作需 `X-Admin-Token` |
| `DEC_EVAL_DEPLOY_MAX_RUNNING` | 4 | 最大活跃部署数 |
| `DEC_EVAL_GPU_MONITOR` | `1` | 启用 GPU/vLLM 监控线程 |
| `DEC_EVAL_GPU_INTERVAL` | 1.5 | GPU 采样间隔秒 |
| `DEC_EVAL_GPU_HISTORY` | 1800 | 历史保留秒数 |
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

## 六、指标定义：官方 RE 与真实 TCP

### 原 TCP=RE 的根因

旧导入器把官方 HumanEval/MBPP 的完整 `test` 保存为
`tests=[{"name":"upstream_harness","program":...}]`。旧 `build_c_file` 只编译/运行一次该程序，
`tests_total=1`，所以 TCP 与 RE 退化成同一个值。实际上一个 harness 往往包含 3–8 个 `assert`，
甚至循环中的动态断言；一次 harness 执行不能代表一个逻辑测试用例。

v0.5 保留官方 harness 作为 **RE 唯一裁决**，另建诊断层为每个动态断言实例生成真实用例。

### 定义

对符合 Benchmark 协议的样本集合：

- **RC** = 候选函数单独编译成功数 / 官方协议分母（与 LLM4Decompile 官方 `gcc -S onlyfunc` 口径一致）
- **RE** = 官方完整 harness 编译成功且全部通过的函数数 / 官方协议分母（诊断层不放宽）
- **RC_harness** = 候选函数 + 官方 harness 整体编译成功数 / 分母；仅用于诊断，不代替官方 RC
- **TCP_micro** = 所有可诊断函数通过的原子用例数 / 所有可诊断函数的有效原子用例数
- **TCP_macro** = mean(每个可诊断函数的通过用例数 / 该函数有效用例数)
- 默认 TCP = TCP_micro；界面同时显示 micro 与 macro，并标注分母。
- **Edit Similarity (ES)** = `1 - Levenshtein(target, prediction) / max(len(target), len(prediction))`；
  - 与官方 `cal_edit_sim.py` 一致：逐行 `strip()`、去掉空行后用 `\n` 连接，按字符编辑距离计算；
  - `target` 为参考 `reference_source`，`prediction` 为去除 Markdown fence 后的模型输出；
  - 编译失败/运行失败的样本仍计算 ES（它不依赖执行），`INPUT_TOO_LONG`/`INFERENCE_ERROR` 无输出记为 N/A；
  - 运行指标为所有有 ES 样本的算术平均，界面显示百分比和样本数。

### 两阶段编译（与官方一致）

- `source` 阶段：`gcc -S` 编译 `include + 候选函数`，不链接测试 harness；**官方 RC 只看这一阶段**；
- `harness` 阶段：候选 + 完整 harness 编译并执行；**RE 只看这一阶段 + 运行通过**；
- 当前沙箱使用 `gcc:14`；source 阶段使用 `-std=gnu11 -D_GNU_SOURCE`，并把 GCC 14 升级为 error 的
  `implicit-function-declaration` / `incompatible-pointer-types` / `int-conversion` 显式降回 warning，
  以兼容论文时期 GCC 11/12 的编译行为；完整 harness 阶段仍保留严格配置，避免把不可运行的代码算作 RE。
- 这样 `humaneval_660` 的 `strdup` 候选函数在 source 阶段通过（RC=1），但严格 harness 阶段可能失败（RE=0），与官方两阶段语义一致。

示例：函数 A 有 5 个 case 通过 3 个：RC=1、RE=0、A 的 TCP=3/5。
两个函数 A(2/2)、B(3/5)：micro=5/7=71.43%，macro=(1.0+0.6)/2=80%。
micro 与 macro 相等可以是真实结果，不强制制造差异。

### 分母、覆盖与失败分类

- 官方 RE 分母：所有非 `INFRA_ERROR/INFERENCE_ERROR/GENERATED/COMPILING` 的样本。
- 细粒度 TCP 分母：状态为 `OK`/`DISCRETE` 且执行过 case 的函数；未执行/不可诊断样本不进入 TCP，但单独报告覆盖率。
- `cases_declared`：manifest 声明的静态 case 数（或 discrete tests 数）。
- `cases_total`：真正执行的 case 数（PASS+FAIL）。
- `uncovered_cases = declared - total`；`case_coverage = diagnosed_samples / evaluated_samples`。
- 失败类型分布来自结果状态（COMPILE_ERROR/TIMEOUT/WRONG_ANSWER/RUNTIME_ERROR/...）。
- 编译阶段分为 `source`（候选函数单独编译，决定官方 RC）与 `harness`（候选+测试整体编译，决定 RE）：
  - `source` 失败：官方 RC=0、RE=0；
  - `source` 成功但 `harness` 失败：官方 RC=1、RE=0（例如模型签名与测试调用类型不匹配）；
  - 已知 case 记为 `NOT_RUN_COMPILE_ERROR`，不伪造输出，不进入 TCP 分子/分母。
- 环境故障：`INFRA_ERROR` 排除；模型接口失败：`INFERENCE_ERROR` 排除。
- harness 定义不清晰/无法可靠拆分：`UNSUPPORTED_CASE_EXTRACTION` / `INVALID_HARNESS` / `NO_ASSERT_CASES`，TCP=N/A。
- `metric_version=tcp_micro_macro_v1`，`rc_metric_version=official_source_compile_v1`；数据集 SHA-256 与版本写入 run config。旧结果没有 `source_compile_ok`，仍显示旧的整体编译 RC，并标注“旧口径”，不会被静默改写；重新评测后使用官方 RC。

## 七、逐用例诊断（tree-sitter）

1. 导入时/首次评测时解析完整 harness；manifest 缓存在 `case_manifests`，键为 `(sample_id, harness_hash, parser_version)`，不重复解析。
2. 使用 **tree-sitter-c/cpp** 定位每个 `assert` 调用的源码区间，记录行/列/字节 span、完整表达式、顶层比较运算符与左右操作数。
3. 对每个 assert 生成诊断块：
   - 左右操作数各求值**一次**（禁止为了打印二次调用）；
   - C 诊断用 `_Generic` 对常见标量/字符串类型做类型安全序列化；结构体/枚举/复杂类型走 `default`，值记为 null 并标注 `VALUE_UNSUPPORTED`；
   - C++ 诊断只保留 PASS/FAIL 与表达式，值不做伪造。
4. 每个 case 在**同一容器内**重新启动官方程序实现隔离：`DEC_DIAG_STOP=k` 表示执行到第 k 个运行时断言后退出；已失败断言通过 `DEC_DIAG_SKIP` 被“求值但不中止”，从而继续诊断后续独立 case。失败后的 case 标记 `AFTER_PRIOR_FAILURE`（状态可能受中止语义影响）。
5. 动态断言按运行时索引展开：循环内一个静态 assert 产生 `a0#0`、`a0#1`… 等稳定 case id；静态 case 用 `a0`。
6. 官方 harness 与诊断 harness 分别编译、分别执行；官方结果决定 RE；诊断只提供 case 级 TCP 与展示。
7. 诊断编译失败/无断言/语法错误/NDEBUG 等情况显式标记，不做字符串猜测，不虚构用例。

**覆盖与边界**：支持连续 assert、循环/条件动态 assert、多行表达式、字符串/注释、数字/负数/浮点容差、`strcmp`/`fabs`、复合 bool。结构体/数组/指针返回值等无法类型安全序列化时，pass/fail 仍可得，实际/期望值为 null 并展示原表达式和 stderr。诊断重写改变了代码布局，但不改变官方 harness 的 RE 判定，也不删除/弱化任何断言。

## 八、错误诊断与 exit 134

容器内 `runner.py` 逐事件输出 JSON（候选源码编译 1 条 + 完整 harness 编译 1 条 + 每个测试 1 条），包含 `exit_code`、`signal`、`timed_out`、`duration_ms`、`stdout`、`stderr`、`*_truncated`。退出码与信号来自 `subprocess` 的 `returncode`，因此 **exit(134) 与 SIGABRT 可区分**。

| 现象 | 分类 |
|---|---|
| 编译失败 | `COMPILE_ERROR`（`compile_stage=source` 或 `harness`，保留编译 stdout/stderr） |
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

## 九、主要 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查 |
| GET | `/api/overview` | 计数概览 |
| GET/POST | `/api/datasets`, `/api/datasets/import` | 数据集 |
| GET | `/api/datasets/{id}/samples` | 样本列表 |
| GET | `/api/datasets/server-files` | 列出允许目录中的服务器端 `.jsonl` 文件 |
| POST | `/api/datasets/import-server` | 从服务器允许目录导入 JSONL（body: `name/version/description/path`） |
| DELETE | `/api/datasets/{id}` | 删除数据集及其样本（已有评测任务时 409，需先删除任务） |
| POST | `/api/datasets/{id}/reference-check` | 创建参考源码基线任务（body: `optimizations`, `sample_limit`, `compile_concurrency`） |
| POST | `/api/datasets/{id}/exebench-oracle?sample_limit=20` | 批量验证 ExeBench 参考源码 / wrapper，写入 Oracle 状态 |
| GET/POST | `/api/models`, `/api/models/{id}/test` | 模型 |
| DELETE | `/api/models/{id}` | 删除无评测任务引用的模型；活跃部署关联模型需先停止/取消部署 |
| POST | `/api/runs` | 创建在线评测（可选并发数） |
| POST | `/api/runs/batch` | 创建 vLLM 离线批量评测（`model_path`, `batch_size`） |
| POST | `/api/runs/offline` | 离线预测评测 |
| GET | `/api/runs`, `/api/runs/{id}` | 任务列表 / 详情（含 progress、metrics） |
| GET | `/api/runs/{id}/progress` | 轻量实时进度（前端 1s 轮询） |
| POST | `/api/runs/{id}/cancel`, `/api/runs/{id}/retry` | 取消 / 重试 |
| DELETE | `/api/runs/{id}`, `/api/runs` | 删除单个已结束任务 / 批量清理 |
| GET | `/api/results/{id}` | 样本详情（含 extracted_source、tests_detail、生成参数/时间戳） |
| GET | `/api/runs/{run_id}/results/{result_id}/test-cases` | 逐用例列表；`status=pass/fail/unsupported`、`page`、`page_size`、`export=json/csv` |
| GET | `/api/compare?run_ids=...` | 同数据集任务对比 |

`/api/models` 不返回内部参考模型；参考基线任务在任务列表中以「参考源码基线（内部）」显示。

## 十、进度与前端

任务详情实时显示（`/api/runs/{id}/progress`，1 秒刷新）：

- 总样本、等待推理、正在推理、已生成代码、等待编译、正在编译、已完成评测、失败样本
- 推理进度 `inference_done / total` 与编译测试进度 `evaluated / total` 分开显示
- 推理并发（0 显示 AUTO 与当前水位/上限）、编译并发（请求值 vs 资源保护后的实际值）、在途请求、推理请求数、推理吞吐（样本/s）、评测吞吐（样本/s）、Token/s、总 Token/s、输入超长、总耗时
- 可获取时显示 vLLM `/metrics`：GPU 运行/等待请求数、KV 缓存占用
- 参考源码基线任务会显示“失败代表数据集/环境问题”的提示

逐样本弹窗：输入汇编、实际 Prompt、生成参数（请求/生效 max_tokens、tokenizer 来源、temperature）、模型原始输出、提取后的 C 代码、参考源码、编译日志、测试 stdout/stderr、退出码、信号、错误分类、推理/编译/测试/沙箱/总耗时、Token 用量、完整 harness 提示。

诊断面板：
- 官方 harness 状态与诊断状态并列展示；不一致时红色警告，RE 以官方为准；
- 全部用例默认展示（不默认只看失败），支持全部/通过/失败/无法诊断筛选；
- 表格显示 case id、输入、expected、actual、PASS/FAIL、错误类型、退出码/信号、stdout/stderr；
- 分页 + JSON/CSV 导出；
- 旧结果/case 声明为 0 时显示“旧结果不可用/需重新诊断评测”。

## 十一、数据库兼容与迁移

启动时自动做**增量迁移**（`ALTER TABLE ... ADD COLUMN`），不删除历史数据：

- `runs.started_at`
- `results.inference_ms, compile_ms, test_ms, sandbox_ms, prompt_tokens, completion_tokens, prompt_text, generation_params_json, detail_json`
- `results.cases_declared, cases_total, cases_passed, test_case_status, metric_version`
- `results.source_compile_ok, compile_stage`（官方 RC 的两阶段编译结果）
- `results.edit_similarity`（官方 Edit Similarity，0–1；历史结果启动时从已存 model_output/reference_source 回填，不改状态）
- `results.submitted_at, generated_at, first_token_at, compile_started_at, compile_finished_at`
- 新表 `case_manifests`（manifest 缓存）与 `test_case_results`（逐用例结果，`run_id+sample_id+case_id` 唯一）
- `datasets.source_hash`、`runs.metric_version`
- 新增 `idx_results_run_status`

旧任务照常查看；旧结果没有结构化 `detail_json` 时前端回退到 `compile_log/run_log`。删除任务仍会级联删除结果并清理离线预测文件。

## 十二、测试

```bash
cd decompile-eval-mvp/backend
source .venv/bin/activate
python -m pytest -q
# 168 passed, 18 skipped（Docker 用例默认跳过）
```

注意：必须用 `python -m pytest`（让 `backend/` 进入 `sys.path`）；直接 `pytest` 可能报 `No module named 'app'`。

真实 Docker 集成测试（可选）：

```bash
RUN_DOCKER_TESTS=1 python -m pytest -q tests/test_docker_integration.py
# 18 passed，覆盖：正常编译、assert失败(SIGABRT)、abort、SIGSEGV、超时、源码/整体编译两阶段、
# 编译错误、完整 harness、C++、逐用例失败隔离、动态循环、单次求值、
# 字符串/浮点/布尔值、特殊字符与日志截断
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
- `tests/test_case_extract.py`：tree-sitter assert/span/循环/多行/字符串注释、NDEBUG、C++、解析失败。
- `tests/test_cases.py`：TCP micro/macro、官方 RE 不被诊断覆盖、失败/后续通过、动态 case id、编译失败 NOT_RUN、API 分页/筛选/导出/一致性警告。
- `tests/test_api.py::test_metrics_excludes_infra_errors` 等：指标口径保持不变。
- `tests/test_docker_integration.py`：真实 Docker E2E + 诊断用例（opt-in）。

## 十三、性能与实测结果

测试环境：LLM4Decompile-1.3B-v1.5，vLLM `http://127.0.0.1:8001/v1`，HumanEval-Decompile O0 前 128 条，`temperature=0`、`max_tokens=512`、输入 Prompt 完全一致；Token 预取不计入计时。

| 在途并发 | 请求数 | 失败 | 输入超长 | 总耗时 | samples/s | completion Token/s | 总 Token/s | p95 延迟 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 128 | 0 | 0 | 8.83s | 14.50 | 2400 | 16171 | 2.06s |
| 32 | 128 | 0 | 0 | 6.60s | 19.38 | 3194 | 21609 | 2.55s |
| 64 | 128 | 0 | 0 | 5.71s | 22.43 | 3694 | 25000 | 3.66s |
| 128 | 128 | 0 | 0 | 5.21s | 24.59 | 4023 | 27385 | 4.79s |

128 并发相对 16 并发：samples/s **1.70×**，completion Token/s **1.68×**，无失败、无超长样本。延迟随并发上升是 vLLM 排队造成的，属预期。

真实全流水线（真实 vLLM + 真实 Docker，8 条 HumanEval O0，`inference_concurrency=0` AUTO、`compile_concurrency=4`）：

- 总耗时 18.6s（冷启动）/ 5.0s（镜像与 vLLM 已热）；8/8 完成；RC=87.5%（旧整体编译口径）、RE=75.0%、TCP_micro=93.1%、TCP_macro=90.48%；
- 诊断 case 27/29（声明 33，覆盖率 87.5%，其余为编译失败样本的 NOT_RUN 用例）；
- vLLM 侧最高 running=7、waiting=0；约 1s 内 8 条推理全部完成，随后 GPU 空闲等待 Docker（沙箱约 8–9s/样本，4 worker）。
  这说明 **瓶颈是 Docker 启动/编译吞吐，而不是请求调度批次**；AUTO 已在推理阶段把可用样本全部送满。
  但当前 8 条样本太少，AUTO 水位保持 32；真正有 128+ 待处理样本时水位才会增长。

真实 HumanEval 混合示例（`humaneval_656_O0` 参考实现通过，`humaneval_664_O0` 故意返回 0.0f）：

| 指标 | 值 | 分母 |
|---|---:|---|
| RC | 100% | 2 个函数 |
| RE | 50% | 2 个函数 |
| TCP_micro | 70% | 7/10 个执行 case |
| TCP_macro | 50% | mean(1.0, 0.0) |
| Edit Similarity | 运行指标 | 所有有输出的样本平均 |
| 诊断覆盖率 | 100% | 2/2 函数，10/10 case |

示例 case（单位：原始值）：

```
humaneval_656_O0 a0 PASS  func0(a,6,0.3) == 1               actual=1 expected=1
humaneval_656_O0 a1 PASS  func0(a,6,0.05) == 0              actual=0 expected=0
...
humaneval_664_O0 a0 FAIL  func0(3.5f) == 0.5f               actual=0 expected=0.5f
humaneval_664_O0 a1 FAIL  fabs(func0(1.33f)-0.33f) < 1e-4   actual=0.3300000131 expected=1e-4
humaneval_664_O0 a2 FAIL  fabs(func0(123.456f)-0.456f)<1e-4 actual=0.4560000002 expected=1e-4
```

a1/a2 标记 `AFTER_PRIOR_FAILURE`：官方 harness 会在 a0 失败时中止，诊断层为独立 case 重新起进程并跳过已失败断言，因此不省略后续用例；该标记提示状态可能受中止语义影响。官方 RE 仍为 0。

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

## 十四、模型部署与 GPU 监控

### 功能

- 网页选择本地模型路径、GPU、端口、dtype、`max-model-len`、`max-num-seqs`、`max-num-batched-tokens`、`gpu-memory-utilization`、接口类型，一键启动独立 vLLM 进程。
- 固定使用 `/home/jiang/venvs/vllm/bin/vllm`（可用 `DEC_EVAL_VLLM_BIN` 覆盖），参数以 argv 列表传递，**`shell=False`**。
- 确定性部署参数：`deterministic`（设置 `VLLM_BATCH_INVARIANT=1`）、`seed`、`disable_prefix_caching`、`enforce_eager`；用于让同一请求在不同并发下尽量得到相同输出。
- 部署列表展示状态、PID、端点、GPU/端口、确定性参数、启动日志、最新请求数/Token 吞吐；支持停止（保留配置可重启）、重启，以及 **取消部署**（停止托管进程并移除部署记录；外部服务只取消监控，不停止外部进程）。
- 部署成功自动写入现有 `models` 表（同一 `deployment_id` 更新，不重复建模型），评测页可直接选择。
- 配置持久化到 `deployments`；后端重启后通过 PID + `/proc/<pid>/cmdline` 校验恢复管理，不误杀 PID 复用进程。
- 外部 vLLM 服务可注册和监控；`managed=0`，stop/restart 返回 409，绝不停止外部进程。
- GPU 监控基于 **nvidia-ml-py**：利用率、显存、温度、功率、GPU 进程；后端每 1.5s 采样，前端每 2s 刷新，SQLite 保存 1800s 历史，支持利用率/显存/温度/功率曲线。
- vLLM `/metrics`：running/waiting 请求、输入/输出 Token/s（计数器差分），同时写入历史。
- 部署权限：`DEC_EVAL_ALLOW_DEPLOY=0` 可完全关闭；设置 `DEC_EVAL_ADMIN_TOKEN` 后写操作必须带 `X-Admin-Token`。前端有管理令牌输入框（sessionStorage）。

### 校验与安全

- 模型路径必须是绝对路径、存在、可读、包含 `config.json`；
- 端口 1024–65535，拒绝本栈保留端口（8001/8010/5173/3080 等）和已占用端口；
- GPU id 必须为数字且不超过 NVML 检测数量；启用时检查每卡可用显存；多卡自动加 `--tensor-parallel-size`；
- dtype/接口类型枚举、所有数值范围校验；最多 `DEC_EVAL_DEPLOY_MAX_RUNNING` 个活跃部署；
- 停止前再次校验 PID cmdline（vllm + serve + 端口 + 模型路径），不匹配则只标记停止、不发信号；进程使用独立 session，支持 SIGTERM→SIGKILL；
- 日志写入 `storage/deployments/deployment_<id>.log`，API 只读该受控路径。

### API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/deployments` | 已部署模型、状态、最新指标 |
| POST | `/api/deployments` | 启动托管部署（需管理员） |
| GET | `/api/deployments/{id}` | 详情 |
| POST | `/api/deployments/{id}/stop` | 停止托管进程（需管理员） |
| POST | `/api/deployments/{id}/restart` | 按持久化配置重启（需管理员） |
| DELETE | `/api/deployments/{id}` | 取消部署：停止托管进程并移除配置；外部服务仅取消监控（需管理员） |
| GET | `/api/deployments/{id}/logs?tail=200` | 启动日志尾部 |
| GET | `/api/deployments/{id}/metrics?seconds=300` | vLLM 指标历史 |
| POST | `/api/deployments/external` | 注册外部服务（需管理员） |
| GET | `/api/gpus?history_seconds=300` | GPU 当前值 + 历史曲线 |

启动示例：

```bash
curl -X POST http://127.0.0.1:8010/api/deployments \
  -H 'Content-Type: application/json' -H 'X-Admin-Token: change-me' \
  -d '{"name":"LLM4Decompile","served_name":"llm4decompile-1.3b-v1.5",
       "model_path":"/home/jiang/share/model/llm4decompile-1.3b-v1.5",
       "gpu_ids":"0","port":8002,"dtype":"bfloat16","max_model_len":4096,
       "max_num_seqs":64,"max_num_batched_tokens":8192,
       "gpu_memory_utilization":0.7,"api_mode":"completion"}'
```

### 让 vLLM 输出可复现

部署表单新增：

- **确定性输出（deterministic）**：启动子进程时设置 `VLLM_BATCH_INVARIANT=1`；
- **随机种子 seed**：传给 vLLM `--seed`；
- **禁用 prefix caching**：传 `--no-enable-prefix-caching`；
- **enforce eager**：传 `--enforce-eager`（更稳定，但更慢）。

这些参数会持久化并参与重启恢复。实际评测还应使用：

```text
temperature=0
top_p=1 / top_k=1
固定模型部署参数（max-num-seqs / max-num-batched-tokens）
```

如果使用采样（temperature>0），还需要在请求参数中显式传入相同 seed。批量不变模式不是零成本，通常会降低吞吐，启用后应重新做性能基线。

环境变量：`DEC_EVAL_VLLM_BIN`、`DEC_EVAL_ALLOW_DEPLOY`、`DEC_EVAL_ADMIN_TOKEN`、`DEC_EVAL_DEPLOY_START_TIMEOUT`、`DEC_EVAL_DEPLOY_STOP_TIMEOUT`、`DEC_EVAL_DEPLOY_MAX_RUNNING`、`DEC_EVAL_GPU_MONITOR`、`DEC_EVAL_GPU_INTERVAL`、`DEC_EVAL_GPU_HISTORY`。

## 十五、ExeBench 支持

ExeBench 使用官方 wrapper 协议，而不是 HumanEval 的 `tests[].program`：

1. 将 `real_deps + 候选函数 + 该测试的 dummy_funcs` 写入 `candidate.c`；
2. 将官方 `real_exe_wrapper` 中的临时 `#include` 替换为 `candidate.c`；
3. 在 Docker `gcc:14` 中使用 `g++ -fpermissive` 编译 wrapper、vendored `clib/synthesizer.c`、`nlohmann/json.hpp`；
4. 每条测试写 `input.json`，执行 `program input.json output.json`；
5. 用官方 `diff_io` 语义比较输出：类型严格、嵌套 list/dict、浮点 `math.isclose`；
6. 不同 `(dummy_funcs, dummy_funcs_seed)` 分组编译，相同条件只编译一次。

本地固定的官方依赖在 `backend/app/exebench_deps/`（nlohmann/json 3.11.3、ExeBench synthesizer），运行时不联网下载。

### 转换

```bash
python scripts/convert_decompile_eval.py --subset exebench \
  --target-contains real_gcc_x86_O0 --max-samples 20 \
  --out data/import/exebench_gcc_o0_20.jsonl
```

输出默认写入 `data/import/`，可直接通过数据集页面上传。样例记录包含 `test_protocol=exebench_io`、`test_wrapper`、`tests[].input/expected_output`、`dummy_funcs`、`dummy_funcs_seed`、`real_iospec`、函数签名及原始 GCC/Clang 标记；不会生成虚假 `program`。

GCC O2 / Clang O0 可显式重新编译并标记：

```bash
python scripts/convert_decompile_eval.py --subset exebench --compiler gcc \
  --regenerate --opt O2 --max-samples 20
python scripts/convert_decompile_eval.py --subset exebench --compiler clang \
  --regenerate --opt O0 --max-samples 20
```

### Oracle

- 评测每个 ExeBench 样本前，先用 `reference_source` 编译 wrapper 并执行全部 I/O；
- Oracle 结果写入 `samples.oracle_validated/test_count/passed_count/failed_count/error/signature`；
- Oracle 失败标记 `ORACLE_ERROR` 并从 RC/RE/TCP 分母排除，不归责于模型；
- 数据集页面显示 Oracle 已通过/待验证/失败数量，并可批量触发 `exebench-oracle`。

### 状态与指标

逐测试状态：`PASS`、`WRONG_ANSWER`、`COMPILE_ERROR`、`LINK_ERROR`、`RUNTIME_ERROR`、`TIMEOUT`、`INVALID_OUTPUT`、`ORACLE_ERROR`、`INVALID_BENCHMARK`、`INFRA_ERROR`。

- RC：候选 wrapper 编译+链接成功；
- RE：全部测试 PASS；
- TCP micro/macro：使用每条 wrapper I/O 测试的 PASS/FAIL；
- 编译失败/链接失败记模型失败，测试通过数 0；
- Oracle 无效、数据错误、Docker 基础设施错误不计入模型失败（单独报告）；
- 同一函数不同编译器/优化级别使用不同 `sample_id`，共享 `problem_id`。

### 已实际验证

- 20 条官方 ExeBench `real_gcc_x86_O0`：Oracle 20/20 通过；逐样本真实 wrapper 执行 20/20 通过，200/200 tests PASS；
- 5 条 GCC O2 重新编译样本：Oracle 5/5；
- 5 条 Clang O0 重新编译样本：Oracle 5/5；
- Docker 集成测试覆盖 dummy_funcs 分组、非法 JSON 输出、错误 expected output。

## 十六、宿主机快速模式（可选）

Docker 仍是默认执行方式。为避开每个样本启动容器的固定开销，可在新建评测时选择：

```text
编译执行环境：宿主机 gcc/g++ 快速（无隔离）
```

同时必须勾选风险确认。服务端开关默认开启（可用环境变量关闭）：

```bash
export DEC_EVAL_ALLOW_HOST_EXECUTION=1   # 显式开启；设为 0 可禁用
# 可选：自定义编译器
export DEC_EVAL_HOST_CC=gcc
export DEC_EVAL_HOST_CXX=g++
```

使用宿主机模式时：

- 编译和执行都由后端进程直接调用本机 `gcc` / `g++`；
- 宿主机模式也会执行 tree-sitter 逐用例诊断（官方 harness + `program_diag`），因此 TCP micro/macro 正常显示；诊断编译/运行同样在宿主机完成。
- 宿主机 RC 现在与 Docker 的官方口径一致：候选函数单独编译成功才算 RC；完整 harness 失败仍记 RE=0。
- 前端在宿主机模式选择区显示 `gcc --version` / `g++ --version`，并调用 `GET /api/host-toolchain`；run config 会保存版本快照，任务详情可查看当时使用的 GCC 版本；
- 不启动 Docker；
- 仍然保留编译/测试超时、进程组清理、CPU/文件大小/文件描述符 rlimit；
- **没有 Docker 的 network none、只读根文件系统、非 root、PID 命名空间等隔离**，只适合可信本地环境；
- ExeBench 需要官方 C++ wrapper 和固定依赖，当前不支持宿主机快速模式，接口会返回 422，仍使用 Docker；
- 宿主机模式先执行候选源码单独编译（`gcc -S` + gnu11 兼容 flags）得到官方 RC，再执行完整 harness；历史宿主机任务如果没有 source 字段，仍显示旧口径。
- 宿主机模式的 TCP 来自逐断言诊断用例；如果 harness 不可拆分，仍显示 TCP=N/A。

实测演示数据集 4 个样本，宿主机编译约 40ms/样本、测试约 7ms/样本，`sandbox_ms` 约 50ms/样本；Docker 路径通常需要数秒/样本。该模式只用于开发期快速迭代，不应作为公网/多租户的评测后端。

## 十七、已知限制

1. 单机、单进程、一次一个任务；跨机/多任务并发需 Redis/RQ 等（本版不强求）。
2. 离线批量模式要求后端 Python 环境安装 `vllm`，且 `LLM.generate` 是同步 GPU 调用；批次执行中不能被取消，只在批次边界检查取消/背压。
3. 若 `/tokenize` 暂时不可用，会用保守估算并在 `generation_params.tokenizer=estimated` 标记；正常 vLLM 部署使用真实 Tokenizer。
4. 无鉴权、可注册任意模型 endpoint；不得公网暴露，公开部署需 SSRF/配额/速率限制。
5. 进度计数在内存中；重启后从数据库重算，当前进行中的计时会归零重来。
6. 前端用 1s 轮询而非 SSE；高频轮询已限制为轻量接口，vLLM `/metrics` 抓取有 2s 缓存。
7. 逐用例诊断依赖 tree-sitter 与 assert 重写：宏内 assert、跨翻译单元、NDEBUG、依赖运行时全局状态或未定义行为的 harness 可能无法可靠拆分，此时明确标记 UNSUPPORTED，TCP=N/A；官方 RE 不受影响。
8. 失败断言后续 case 使用 `AFTER_PRIOR_FAILURE` 诊断；跳过中止可能改变依赖中止语义的状态，界面会提示，但官方 RE 仍是唯一裁决。
9. 非流式 HTTP 无法观察首 token 时间：`first_token_at` 留空并在界面标注 N/A；提交/生成完成/编译开始/结束时间已持久化。
10. 结构体/枚举/指针返回值等无法类型安全序列化时，实际/期望值为 null；不会编造数值。
11. 仅函数级 C/C++；不支持多文件链接、数据结构完全恢复或语义等价证明。
12. 数据集 SHA-256 已写入 `datasets.source_hash` 与 run config；模型 commit/权重哈希、Prompt 哈希仍需人工记录或后续补实验快照。
13. 参考源码基线会把选中样本全部重跑一遍，数据集很大时耗时明显，可先限制 `sample_limit`。
14. `compile_ms` 是容器内 gcc 耗时；Docker 启动开销记录在 `sandbox_ms`，两者与 `inference_ms` 共同构成 `latency_ms`。
15. GPU 监控依赖 NVIDIA 驱动与 nvidia-ml-py；无驱动/容器内无 `/dev/nvidia*` 时 `/api/gpus` 返回 available=false，部署功能仍可用但无 GPU 曲线（本项目的自动化测试用 fake NVML，真实 vLLM 启动需在有 GPU 的主机后端执行）。
16. 部署管理默认只允许可信本地管理员；跨机器/多租户部署需要额外鉴权、资源配额与进程隔离。

## 十八、项目结构

```text
backend/
  app/
    config.py     环境配置（AUTO/编译资源、超时、Token 预算、诊断）
    adapters.py   OnlineAdapter / OfflineBatchAdapter / 真实 Tokenizer / vLLM metrics
    case_extract.py tree-sitter 断言提取、manifest、诊断源码重写、C 诊断头
    db.py         SQLite 模型 + 增量迁移
    schemas.py    API 参数与 Prompt 模板
    sandbox.py    结构化 Docker 编译/测试、信号与错误分类（同步 + 异步）
    worker.py     异步生产者-消费者流水线、进度、取消/续跑、重试
    main.py       REST API、指标、删除、参考源码检查、部署/GPU 路由
    deployments.py vLLM 部署校验、启动、恢复、停止、外部注册
    gpu.py        nvidia-ml-py 采样、vLLM /metrics、历史曲线
    edit_sim.py   官方 Edit Similarity
    exebench.py   ExeBench wrapper 执行/Oracle/diff_io
    exebench_runner.py Docker 内 ExeBench wrapper 运行器
    exebench_deps/ 固定版本 nlohmann/json 与 synthesizer
    seed.py       演示样本
  tests/
    test_api.py               API/删除/迁移/指标
    test_sandbox_errors.py    日志与错误分类
    test_pipeline.py          并发流水线
    test_adapters.py           Adapter / Tokenizer / 离线批量 / metrics
    test_case_extract.py       断言解析/重写/动态/边界
    test_cases.py              TCP micro/macro、case 合并、API 导出
    test_deployments.py        部署校验/启动/停止/恢复/权限/外部服务
    test_deployment_process.py 真实进程生命周期（fake vLLM，无 GPU）
    test_gpu.py                NVML/GPU 历史/vLLM 指标 mock 测试
    test_exebench.py           ExeBench 导入/diff_io/状态/分组
    test_exebench_docker.py    ExeBench 真实 wrapper Oracle 测试（opt-in）
    test_docker_integration.py 真实 Docker + 诊断用例（opt-in）
    test_real_bench_adapters.py
    test_sandbox.py
frontend/src/App.vue            Vue 3 工作台（进度/日志/参考基线）
scripts/                        下载、转换、参考冒烟、推理吞吐基准（bench_inference.py）
data/import, data/raw           可重建的数据（被 .gitignore 忽略）
storage/                        SQLite 与离线预测（被忽略）
```
