# DecompileEval MVP — 大模型反编译自动化评测平台

> **v0.2 增量版本**：已增加 LLM4Decompile v1.5 Text Completions 模式、官方完整 harness 的编译测试、HumanEval/MBPP 数据集下载及转换脚本。**真实模型运行步骤详见 [REAL_BENCHMARKS.md](REAL_BENCHMARKS.md)**。本文下方的 v0.1 基础介绍仍可参考。

> **版本：v0.1 初稿 / 本科毕设原型。** Vue 3 + FastAPI + SQLite + Docker 沙箱。
>
> 内置「演示固定输出（非真实 LLM）」用于验证从创建任务到评测报告的整个流程。所有演示指标均不属于公开基准结果。

## 一、目前已经实现

- **数据集管理**：导入统一 JSONL 格式的数据集，查看样本及 O0/O1/O2/O3。
- **模型管理**：接入 OpenAI 兼容 Chat Completions API（本地 vLLM 或第三方模型），测试连接。API Key 仅从后端环境变量读取。
- **离线评测**：导入已有模型的 `sample_id` 和 `prediction`，不重新推理即可开展编译/测试。
- **评测任务**：异步单机任务队列，样本级持久化，查询进度、停止、异常重试/断点恢复。
- **功能验证**：将候选 C 代码与可信测试代码组合，在 Docker 容器中编译和运行；分类记录编译失败、执行失败、超时和沙箱错误。
- **指标与可视化**：RC、RE、TCP、优化等级统计、模型对比、样本三栏视图、编译/运行日志。
- **REST API**：FastAPI Swagger 文档 `http://127.0.0.1:8010/docs`。

**与长期目标的区别：** 当前 MVP 为**单用户、本地开发运行**，采用进程内单 Worker 队列（而不是 Redis/RQ）；尚未实现注册/登录、分布式并发、Excel/CSV 报告导出、原始 HumanEval/ExeBench 格式自动转换、Ghidra 集成，也没有实现严格的安全沙箱认证。建议本科生后续按需增加。

## 二、Ubuntu 24.04 运行命令

### 0. 安装前置环境

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip nodejs npm docker.io
sudo systemctl enable --now docker
```

需要 **Node.js 18+**（推荐 20/22）和 Python 3.10+。发行版自带的 Node 版本如果太旧，需要自行更新 Node。

**Docker 权限：** 运行 API 的用户必须有使用 Docker 的权限，否则任务会标为 `INFRA_ERROR`。对于个人、可信的开发工作站，可以按 Docker 官方建议配置 rootless Docker；使用 docker 组虽然方便，但**等同赋予近似 root 权限**，不要在多用户共享服务器上随意授权。

若是你自己管理的隔离开发机，也可运行：

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

然后拉取 **评测容器镜像**：

```bash
docker pull gcc:14
docker run --rm gcc:14 gcc --version
```

### 1. 启动后端（终端 A）

在项目解压目录运行：

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
curl http://127.0.0.1:8010/api/datasets
```

### 2. 启动前端（终端 B）

```bash
cd decompile-eval-mvp/frontend
npm install
npm run dev
```

浏览器访问：**http://127.0.0.1:5173**

> 若通过 SSH 使用远程服务器，推荐 `ssh -L 5173:127.0.0.1:5173 -L 8010:127.0.0.1:8010 user@host`，再在本地浏览器打开 127.0.0.1:5173。不要未经身份认证把本系统暴露到公网。

### 3. 运行第一份完整评测（无模型 API Key）

1. 进入「新建评测任务」；
2. 数据集选择 **演示数据集（4个函数）**；
3. 模型选择 **演示固定输出（非真实LLM）**；
4. 保留 O0/O1/O2/O3，样本数量上限为 0；
5. 点击 **创建并执行评测**；
6. 前往「评测任务」，查看 RC、RE、TCP、每个样本的源码和错误日志。

此流程 **真正使用 Docker 中的 GCC 编译和执行**，但使用的 C 预测代码来自预置 fixture，不调用 LLM。若没有安装 Docker / 无权限 / 镜像未拉取，状态显示 `INFRA_ERROR`，不会产生虚假的评测百分比。

### 4. 导入离线模型结果

文件示例：`examples/demo_predictions.jsonl`，每行格式：

```json
{"sample_id":"demo_sum_O0","prediction":"int sum_two(int a, int b) { return a+b; }"}
```

网页「新建评测任务」右侧选择内置数据集并上传示例文件即可；初稿要求样本 ID 全部覆盖，无需调用模型 API。

### 5. 接入已有 vLLM 反编译模型

例如在另一个终端启动模型（根据实际 GPU、模型路径配置；需自行安装 vLLM）：

```bash
vllm serve /path/to/your-decompile-model \
  --served-model-name decompiler-a \
  --host 127.0.0.1 --port 8001
```

然后到「模型管理」注册：

- 模型名称：`Decompiler-A`
- API Base URL：`http://127.0.0.1:8001/v1`
- 服务端模型 ID：`decompiler-a`
- API Key 对应环境变量：本地无鉴权服务留空

远程 API 示例：在**启动后端之前**设置 `export OPENAI_API_KEY='...'`，填写 `api_key_env = OPENAI_API_KEY`，不要输入真实密钥到前端表单。API Key 不会写入 SQLite。

## 三、导入数据集格式

`examples/demo_dataset.jsonl` 是可直接导入的示例数据。每行一个 JSON 对象：

```json
{
  "sample_id": "example_add_O0",
  "problem_id": "add",
  "optimization": "O0",
  "compiler": "gcc",
  "architecture": "x86_64",
  "language": "c",
  "assembly": "add:\n  mov eax, edi\n  add eax, esi\n  ret",
  "reference_source": "int add(int a, int b) { return a+b; }",
  "dependencies": "",
  "tests": [
    {"name":"basic", "code":"if (add(2, 3) != 5) return 1;"}
  ]
}
```

- `assembly` 是 LLM 的唯一主要输入（Prompt 中使用 `{assembly}`）；`reference_source`、`dependencies` 和 `tests` **绝对不能泄漏给模型**。
- `tests` 里的 `code` 是 `main()` 的一个分支内执行的可信 C 测试语句，失败时 `return 1`，成功时执行结束返回 0。
- `dependencies` 为可信测试环境中的头文件/函数/全局数据依赖，也可能包含辅助实现；**只能从已审核的可信 Benchmark 导入**。
- 第一版仅支持 C。C++、多个源文件或复杂项目的测试 harness 需要扩展专用数据集 Adapter。
- 公开的 HumanEval-Decompile、MBPP、ExeBench **不能保证原始文件开箱即用**：需先按照此格式规范化，正确构造可执行测试 harness。不要直接把上游 `test` 字段未经检查地替换进 `tests`。

## 四、指标与错误处理

对**具有完整测试用例且执行环境有效**的样本：

- **RC**：编译成功的样本数 / 有效评测样本数。
- **RE**：编译成功、且全部测试通过的样本数 / 有效评测样本数。
- **TCP**：通过的测试用例总数 / 有效样本测试用例总数（测试用例加权版本）。

`INFRA_ERROR` 与 `INFERENCE_ERROR` 不计入分母，并单独计数。编译错误、模型空输出等属于有效样本失败，计入分母。单样本 `TEST_PASS` 的含义是**给定测试集全部通过**，不代表所有输入下严格语义等价。

注意：不同公开 Benchmark 对 RC/RE/TCP 可能有不同统计口径。发表论文时必须固定规范化数据集、测试用例、评测器版本和统计定义，并把平台指标与原论文的官方指标分别标注。

## 五、项目目录

```text
backend/
  app/
    main.py       REST API、数据导入、统计、任务接口
    db.py         SQLite 模型和演示数据初始化
    schemas.py    API 参数和 Prompt 模板
    worker.py     异步推理、任务处理、断点恢复
    sandbox.py    Docker C 编译/执行隔离和错误解析
    seed.py       4 个演示样本（非公开 Benchmark）
  tests/          自动化测试
  requirements.txt
frontend/
  index.html
  vite.config.js
  package.json
  src/App.vue     多页面 Vue 评测工作台
  src/style.css   界面样式
examples/
  demo_dataset.jsonl
  demo_predictions.jsonl
```

数据库会自动保存在 `storage/decompile_eval.sqlite3`，上传的离线预测文件保存在 `storage/offline/`。删除 `storage/` 会丢失当前任务数据，请在开发前备份。

## 六、自动化测试命令

```bash
cd decompile-eval-mvp/backend
source .venv/bin/activate
pytest -q
```

当前含 11 项后端测试，覆盖：数据集导入、任务写库、指标统计、Markdown 代码提取、编译/测试状态解析。测试用 mock 替代真实 Docker 调用；**不是**真实 Docker 沙箱的端到端测试。

可选：在安装 Docker 并拉取 gcc:14 后，运行**真实沙箱集成测试**：

```bash
RUN_DOCKER_TESTS=1 pytest -q tests/test_docker_integration.py
```

前端生产构建检查：

```bash
cd decompile-eval-mvp/frontend
npm run build
```

## 七、已知限制、工程选择与安全要求

1. **单机任务队列：** 使用进程内单 Worker；只支持一台后端、`uvicorn` 单进程。正式上线或需要跨机器任务时，迁移至 Redis+RQ，分离 API/推理/评测 Worker。
2. **执行隔离：** 在 Docker 中禁用网络、设置只读根文件系统、内存、CPU、进程数和时间限制，并使用非特权用户。但 Docker 不等于针对恶意提交的强安全隔离。当前适合可信、单用户离线科研环境。
3. **无鉴权：** 不得直接暴露公网；API 允许注册任意 endpoint，公开部署还需防 SSRF、权限、配额和速率限制。
4. **缓存/复现：** 当前预测、指标、原始响应及日志主要存储 SQLite。尚缺评测器镜像 digest、Prompt 哈希、数据集 SHA256 和模型版本的完整实验快照。
5. **结果导出：** 本版前端提供查看及模型对比；CSV/Excel/论文表格导出还未做。
6. **评分范围：** 当前仅测试函数级 C 代码；不存在跨多文件链接、数据结构全恢复或安全语义等价证明。
7. **前端安装：** 首次 `npm install` 需要可访问 npm registry 的网络。

## 八、升级路线

**v0.2（本科生优先）：** Redis+RQ、CSV 导出、实验快照与重试细化、OpenAI API 超时重试与并发限速、前端 Monaco 代码 Diff。

**v0.3（科研扩展）：** HumanEval/MBPP/ExeBench 数据集适配器、GCC/Clang/O0-O3 对比、更多可执行测试协议、对比实验导出。

**v1.0（研究平台）：** 分布式执行 Worker、按模型自动并发与资源分配、IR/SIR 等额外输入表示及安全相关语义验证。
