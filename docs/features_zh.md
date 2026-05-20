# Mini-SGLang 功能详解

## 在线服务

Mini-SGLang 支持通过兼容 OpenAI API 的服务器进行在线服务。它提供标准的 `/v1/chat/completions` 端点，可与现有工具和客户端无缝集成。运行 `python -m minisgl --help` 可查看详细的命令行参数和配置选项。

## 交互式 Shell 模式

为方便演示和测试，提供了交互式 Shell 模式。用户可以直接输入提示词，LLM 会实时生成回复。Shell 自动缓存聊天历史以保持上下文。使用 `/reset` 命令可清空对话历史并开始新会话。

示例:

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

## 分布式服务

为在多 GPU 上扩展性能，Mini-SGLang 支持张量并行（Tensor Parallelism, TP）。通过 `--tp n` 参数指定并行度，其中 `n` 是 GPU 数量。

## 支持的模型

目前支持以下 Dense 模型架构:

- [`Llama-3`](https://huggingface.co/collections/meta-llama/llama-31) 系列
- [`Qwen-3`](https://huggingface.co/collections/Qwen/qwen3) 系列（含 MoE）
- [`Qwen-2.5`](https://huggingface.co/collections/Qwen/qwen25) 系列

## Chunked Prefill（分块预填充）

Chunked Prefill 是 [Sarathi-Serve](https://arxiv.org/abs/2403.02310) 提出的技术，默认开启。该功能将长提示词在 prefill 阶段拆分为较小的块，显著降低峰值内存使用，防止长上下文服务中的 OOM 错误。可通过 `--max-prefill-length n` 配置块大小。注意，将 `n` 设置得过小（如 128）可能严重影响性能。

## Page Size（页大小）

可通过 `--page-size` 参数指定系统的页大小。

## Attention Backends（注意力后端）

Mini-SGLang 集成了高性能注意力内核:

- [`FlashAttention`](https://github.com/Dao-AILab/flash-attention)（`fa`）
- [`FlashInfer`](https://github.com/flashinfer-ai/flashinfer)（`fi`）
- [`TensorRT-LLM fmha`](https://github.com/NVIDIA/TensorRT-LLM)（`trtllm`）

支持对 prefill 和 decode 阶段使用不同的后端以最大化效率。例如，在 NVIDIA Hopper GPU 上默认 prefill 用 `FlashAttention 3`、decode 用 `FlashInfer`。

使用 `--attn` 参数指定后端。若提供两个值（如 `--attn fa,fi`），第一个指定 prefill 后端，第二个指定 decode 后端。注意某些注意力后端可能覆盖用户提供的页大小（如 `trtllm` 仅支持页大小 16、32、64）。

## CUDA Graph

为减少 decode 阶段的 CPU 启动开销，Mini-SGLang 支持 CUDA Graph 的捕获与回放，默认开启。可通过 `--cuda-graph-max-bs n` 设置捕获的最大 batch size。将 `n` 设为 `0` 可禁用此功能。

## Radix Cache（基数树缓存）

借鉴 [SGLang](https://github.com/sgl-project/sglang.git) 的原始设计，Mini-SGLang 实现了 Radix Cache 来管理 KV 缓存。这允许跨请求复用共享前缀的 KV 缓存，减少冗余计算。默认开启，可通过 `--cache naive` 切换为朴素的缓存管理策略。

![radix](https://lmsys.org/images/blog/sglang/radix_attn.jpg)
*Radix Attention 示意图，源自 [LMSYS Blog](https://lmsys.org/blog/2024-01-17-sglang/)。*

## Overlap Scheduling（重叠调度）

为进一步减少 CPU 开销，Mini-SGLang 采用了 [NanoFlow](https://arxiv.org/abs/2408.12757) 提出的 Overlap Scheduling。该技术将 CPU 调度开销与 GPU 计算重叠执行，提升系统整体吞吐量。

![overlap](https://lmsys.org/images/blog/sglang_v0_4/scheduler.jpg)
*Overlap Scheduling 示意图，源自 [LMSYS Blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/)。*
