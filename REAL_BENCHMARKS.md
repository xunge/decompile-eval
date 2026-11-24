# DecompileEval v0.2 — LLM4Decompile-1.3B-v1.5 实测指导

> **v0.5 说明**：官方 harness 仍是 RE 唯一裁决；新增 tree-sitter 逐用例诊断得到 TCP_micro/TCP_macro，完整 harness 不再被当作 1 个测试用例。推理默认 `INFERENCE_CONCURRENCY=0` AUTO 持续投递，编译默认 64 并受资源保护。真实 Tokenizer、`INPUT_TOO_LONG`、离线批量适配器、`/api/runs/{id}/progress` 与参考源码基线继续保持。下文步骤仍然适用。

**说明**：这是基于 v0.1 的增量版本。支持 vLLM Text Completions 和完整官方测试 harness。HumanEval/MBPP 的数据转换已实现，但仍必须对原始参考代码进行 oracle 测试；ExeBench 官方原始数据只提供下载和检查，**未**宣称与早期论文的 2621 样本一致，也未实现其可执行 wrapper 适配。

## 0. Python 依赖

下载、转换工具依赖：

```bash
cd decompile-eval-mvp
python3 -m venv .bench-env
source .bench-env/bin/activate
python -m pip install -U pip
python -m pip install 'huggingface_hub>=0.28' 'datasets>=3,<5' 'zstandard>=0.23'
```

若已在后端虚拟环境中，直接安装上述三个库即可。

建议国内网络使用清华 PyPI 源，例如 `-i https://pypi.tuna.tsinghua.edu.cn/simple`；若无法访问 Hugging Face，可使用你信任的、可用的 `HF_ENDPOINT` 镜像地址，但应确认其实际可访问性及数据来源。

## 1. 下载三个评测集

```bash
python scripts/download_benchmarks.py --out data/raw
```

默认仅下载如下文件（均来自官方 HF）：

- HumanEval: LLM4Binary/decompile-eval `humaneval/data-00000-of-00001.arrow`（约 17 MB）
- MBPP: 同仓库 `mbpp/data-00000-of-00001.arrow`（约 72.6 MB）
- ExeBench v1.01: jordiae/exebench `test_real.tar.gz`（约 9.24 MB）

只下载一个：

```bash
python scripts/download_benchmarks.py --datasets humaneval
python scripts/download_benchmarks.py --datasets mbpp
python scripts/download_benchmarks.py --datasets exebench
```

## 2. 转换 HumanEval 和 MBPP

先只转换 O0（容易小规模测试），然后再使用 all：

```bash
python scripts/convert_decompile_eval.py --subset humaneval --opt O0 --max-samples 20
python scripts/convert_decompile_eval.py --subset humaneval --opt all
python scripts/convert_decompile_eval.py --subset mbpp --opt O0 --max-samples 20
python scripts/convert_decompile_eval.py --subset mbpp --opt all
```

输出为 `data/import/humaneval_O0.jsonl`、`humaneval.jsonl`、`mbpp_O0.jsonl`、`mbpp.jsonl`。

**关键区别**：上游的 `test` 是带 `main` 的完整 C/C++ 测试程序（或等价完整 harness），不可放到旧 MVP 的 `tests[].code` 主函数分支内；v0.2 新增 `tests[].program`，会把 `func_dep + candidate + test` 组成完整编译单元。默认跳过 C++，因为 v1.5 面向 C；若要包含 C++ 可以传 `--include-cpp`，但不建议用它报告该模型的正式结果。

只转换格式并不意味着数据可直接获得可信 RE。请先检查参考源码与测试程序是否可在沙箱中跑通：

```bash
python scripts/reference_smoke.py data/import/humaneval_O0.jsonl --limit 10
python scripts/reference_smoke.py data/import/mbpp_O0.jsonl --limit 10
```

如果参考代码失败，先排查头文件、测试 harness、编译参数、原始数据是否满足当前沙箱，不能把这些失败当作模型错误。

## 3. 启动本地模型

若你已有 vLLM 环境，无需重复安装：

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  /home/jiang/share/model/llm4decompile-1.3b-v1.5 \
  --served-model-name llm4decompile-1.3b-v1.5 \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70
```

检验：

```bash
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llm4decompile-1.3b-v1.5","prompt":"# This is the assembly code:\nfunc0:\n  lea (%rdi,%rsi,1),%eax\n  ret\n# What is the source code?\n","temperature":0,"max_tokens":512}'
```

**不要将 v1.5 作为 Chat Completion 模型使用**；该模型训练时采用完整文本 Prompt，参考官方文档。

## 4. Web 页面使用

以 v0.2 后端/前端启动系统。模型管理页面添加模型：

| 项 | 值 |
| --- | --- |
| 名称 | LLM4Decompile-1.3B-v1.5 |
| Base URL | http://127.0.0.1:8001/v1 |
| Model ID | llm4decompile-1.3b-v1.5 |
| 接口类型 | Completion |
| API Key 环境变量 | 留空 |

在数据集页面上传 `data/import/humaneval_O0.jsonl` 并命名为 HumanEval-O0-SMOKE；创建评测任务选新模型、O0、20 条样本，先做联调。随后导入全部 HumanEval 或 MBPP 并测试 O0～O3。

**输入 Prompt（必须统一版本）：**

```text
# This is the assembly code:
{assembly}
# What is the source code?
```

`max_tokens` 初测 1024，之后依据样本长度和截断统计调整；温度设置 0。v1.5 官方模型标称最大上下文 4096 token，输入+输出超限样本必须单独标记，不能把截断悄悄当作模型语义错误。

## 5. ExeBench 阶段

下载后的 `data/raw/jordiae-exebench-v1.01/test_real.tar.gz` 是原始 ExeBench **v1.01**，可以通过：

```bash
python scripts/inspect_exebench.py --limit 3
```

查看 `asm` 包含哪些编译器/架构/优化目标及原始 `real_io_pairs`、`real_exe_wrapper` 字段。

**暂不能直接导入 Web 计算 RE。** 它和 LLM4Decompile 论文所用 2621 个 ExeBench 函数不是一致的样本集合。需选定 GCC x86-64 O0～O3 的 asm 项、组织 real_deps、将执行 wrapper 和 IO 对比改写成可重复的完整测试 harness，并首先验证参考函数通过。完成前可只进行结构检查或单独记录模型输出，不应把未经验证的“测试失败”纳入正式 RE。

## 6. 复现实验的校验点

1. 记录模型具体路径和 commit/sha256，以及完整 Prompt、token 上限、温度。
2. 数据集分组以文件中 compiler/optimization/architecture 为准，不推断缺失字段。
3. 先通过 oracle reference smoke，再跑 20 样本、O0 全量，最后 O0～O3。
4. 与官方报告比对之前，确认测试集合、数据版本、GCC/G++ 依赖、优化与超时规则一致。
5. 真实模型输出本身可能出现 `#include`、markdown、多个函数或截断，需查看编译日志；RC/RE 不等于绝对语义等价。

原始链接：
- 模型：https://huggingface.co/LLM4Binary/llm4decompile-1.3b-v1.5
- HumanEval/MBPP：https://huggingface.co/datasets/LLM4Binary/decompile-eval
- ExeBench v1.01：https://huggingface.co/datasets/jordiae/exebench
- 官方评测：https://github.com/albertan017/LLM4Decompile

## 7. 当前 v0.2 评测注意事项

- 数据集格式里 `compiler` 字段并未由该 Hugging Face 数据源直接提供，因此转换器写 `unspecified`；不要在未核实源数据之前把其结果报告为 GCC 与 Clang 横向对比。
- 上游 `test` 作为一个完整 `program` harness 编译执行：**RC/RE 可以按整套断言判定，但 TCP 在当前平台只计“整套 harness 是否通过”（每样本测试总数为 1），不能直接声称是原始独立断言级 TCP。** 真正的 TCP 需要把测试拆成独立可执行的断言或由 Benchmark 提供原始逐用例结果。
- 对 `test` 中包含多个源码、额外依赖或框架特性的样本，v0.2 仍可能无法复现官方执行环境。仅当参考源码自身通过相同 harness 时，才能将样本纳入可信 RE 统计。
- 参考数据和模型结果均需原始索引与样本哈希追踪；v0.2 尚无官方论文级复现认证。
