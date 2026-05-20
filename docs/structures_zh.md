# Mini-SGLang 系统架构

## 系统架构

Mini-SGLang 被设计为一个分布式系统，用于高效处理大语言模型（LLM）推理。它由多个独立的进程组成，各司其职。

### 核心组件

- **API Server**: 用户入口。提供兼容 OpenAI 的 API（如 `/v1/chat/completions`），接收提示词并返回生成文本。
- **Tokenizer Worker**: 将输入文本转换为模型可理解的数字（token）。
- **Detokenizer Worker**: 将模型生成的数字（token）转换回人类可读的文本。
- **Scheduler Worker**: 核心工作进程。在多 GPU 场景下，每个 GPU 对应一个 Scheduler Worker（称为 **TP Rank**）。负责管理该 GPU 的计算和资源分配。

### 数据流

组件间通过 **ZeroMQ (ZMQ)** 进行控制消息通信，GPU 间的张量数据传输则使用 **NCCL**（通过 `torch.distributed`）。

![Process overview diagram](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期:**

1. **用户** 发送请求到 **API Server**。
2. **API Server** 将请求转发给 **Tokenizer**。
3. **Tokenizer** 将文本转为 token，发送给 **Scheduler（Rank 0）**。
4. **Scheduler（Rank 0）** 将请求广播给所有其他 Scheduler（多 GPU 情况下）。
5. **所有 Scheduler** 调度请求并触发各自本地的 **Engine** 计算下一个 token。
6. **Scheduler（Rank 0）** 收集输出 token 并发送给 **Detokenizer**。
7. **Detokenizer** 将 token 转为文本，发回 **API Server**。
8. **API Server** 将结果流式返回给 **用户**。

## 代码组织（`minisgl` 包）

源码位于 `python/minisgl`，各模块说明如下：

- `minisgl.core`: 定义核心数据结构——`Req`（请求状态）、`Batch`（批次状态）、`Context`（推理全局状态）以及 `SamplingParams`（用户提供的采样参数）。
- `minisgl.distributed`: 提供张量并行中 all-reduce 和 all-gather 的接口，以及保存 TP 信息的 `DistributedInfo` 数据类。
- `minisgl.layers`: 实现构建 LLM 的基础组件（含 TP 支持），包括线性层、LayerNorm、Embedding、RoPE 等。它们共享 `minisgl.layers.base` 中定义的公共基类。
- `minisgl.models`: 实现 LLM 模型，包括 Llama 和 Qwen3。同时定义了从 HuggingFace 加载权重和权重分片的工具。
- `minisgl.attention`: 提供注意力后端的接口，实现了 `flashattention` 和 `flashinfer` 后端。由 `AttentionLayer` 调用，使用 `Context` 中存储的元数据。
- `minisgl.kvcache`: 提供 KV Cache 池和 KV Cache 管理器的接口，实现了 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`: 通用工具集，包括日志配置和 zmq 封装。
- `minisgl.engine`: 实现 `Engine` 类，即单个进程上的 TP Worker。管理模型、Context、KVCache、注意力后端以及 CUDA Graph 回放。
- `minisgl.message`: 定义 api_server、tokenizer、detokenizer 和 scheduler 之间（通过 zmq）交换的消息类型。所有消息类型均支持自动序列化与反序列化。
- `minisgl.scheduler`: 实现 `Scheduler` 类，运行在每个 TP Worker 进程上，管理对应的 `Engine`。Rank 0 的 scheduler 接收 tokenizer 的消息，与其他 TP Worker 上的 scheduler 通信，并向 detokenizer 发送消息。
- `minisgl.server`: 定义 CLI 参数和 `launch_server`（启动 Mini-SGLang 的所有子进程）。同时在 `minisgl.server.api_server` 中实现 FastAPI 前端服务器，提供 `/v1/chat/completions` 等端点。
- `minisgl.tokenizer`: 实现 `tokenize_worker` 函数，处理 tokenization 和 detokenization 请求。
- `minisgl.llm`: 提供 `LLM` 类作为 Python 接口，方便地与 Mini-SGLang 系统交互。
- `minisgl.kernel`: 实现自定义 CUDA 内核，借助 `tvm-ffi` 提供 Python 绑定和 JIT 接口。
- `minisgl.benchmark`: 基准测试工具。
