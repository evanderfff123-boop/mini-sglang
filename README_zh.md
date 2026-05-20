<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

一个**轻量级、高性能**的大语言模型推理框架。

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的精简实现，旨在揭开现代 LLM 服务系统的复杂性。整个代码库仅 **~5,000 行 Python**，既能作为实用的推理引擎，也是研究者和开发者学习推理系统的透明参考。

## ✨ 核心特性

- **高性能**: 通过多项高级优化达到业界领先的吞吐量和延迟表现。
- **轻量且易读**: 代码库干净、模块化，全部带有类型标注，易于理解和修改。
- **高级优化**:
  - **Radix Cache**: 复用跨请求共享前缀的 KV 缓存。
  - **Chunked Prefill**: 降低长上下文服务的峰值内存使用。
  - **Overlap Scheduling**: 将 CPU 调度开销隐藏在 GPU 计算中。
  - **Tensor Parallelism**: 跨多 GPU 扩展推理能力。
  - **优化内核**: 集成 **FlashAttention** 和 **FlashInfer** 实现极致性能。
  - ...

## 🚀 快速开始

> **⚠️ 平台支持**: Mini-SGLang 目前仅支持 **Linux**（x86_64 和 aarch64）。由于依赖 Linux 特定的 CUDA 内核（`sgl-kernel`、`flashinfer`），Windows 和 macOS 不受支持。建议在 Windows 上使用 [WSL2](https://learn.microsoft.com/zh-cn/windows/wsl/install) 或使用 Docker 实现跨平台兼容。

### 1. 环境配置

我们推荐使用 `uv` 进行快速可靠的安装（`uv` 与 `conda` 不冲突）。

```bash
# 创建虚拟环境（推荐 Python 3.10+）
uv venv --python=3.12
source .venv/bin/activate
```

**前置条件**: Mini-SGLang 依赖 JIT 编译的 CUDA 内核。请确保已安装 **NVIDIA CUDA Toolkit** 且版本与驱动匹配。可通过 `nvidia-smi` 查看驱动的 CUDA 能力。

### 2. 安装

从源码直接安装：

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 在 Windows（WSL2）上安装</b></summary>

由于 Mini-SGLang 依赖 Linux 特定组件，Windows 用户应使用 WSL2：

1. **安装 WSL2**（如尚未安装）：
   ```powershell
   # 在 PowerShell 中（以管理员身份运行）
   wsl --install
   ```

2. **在 WSL2 上安装 CUDA**：
   - 参考 [NVIDIA WSL2 CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保 Windows GPU 驱动支持 WSL2

3. **在 WSL2 中安装 Mini-SGLang**：
   ```bash
   # 在 WSL2 终端中执行
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从 Windows 访问**: 服务器启动后可在 Windows 浏览器和应用中通过 `http://localhost:8000` 访问。

</details>

<details>
<summary><b>🐳 使用 Docker 运行</b></summary>

**前置条件**:
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **构建 Docker 镜像**：
   ```bash
   docker build -t minisgl .
   ```

2. **启动服务**：
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **交互式 Shell 模式**：
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **使用 Docker Volume 持久化缓存**（推荐，加速后续启动）：
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. 在线服务

一行命令启动兼容 OpenAI API 的服务器。

```bash
# 在单 GPU 上部署 Qwen/Qwen3-0.6B
python -m minisgl --model "Qwen/Qwen3-0.6B"

# 在 4 块 GPU 上使用 Tensor Parallelism 部署，端口 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

服务器启动后，可用 `curl` 或任何兼容 OpenAI 的客户端发送请求。

### 4. 交互式 Shell

添加 `--shell` 参数直接在终端与模型对话。

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

使用 `/reset` 可清空聊天历史。

## Benchmark

### 离线推理

详见 [bench.py](./benchmark/offline/bench.py)。设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 可进行 overlap scheduling 消融实验。

测试配置:

- 硬件: 1xH200 GPU
- 模型: Qwen3-0.6B, Qwen3-14B
- 请求总数: 256 条序列
- 输入长度: 100-1024 tokens 随机采样
- 输出长度: 100-1024 tokens 随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理

详见 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置:

- 硬件: 4xH200 GPU，NVLink 互联
- 模型: Qwen3-32B
- 数据集: [Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，回放前 1000 个请求

启动命令:

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **注意**: 如遇 HuggingFace 下载模型网络问题，可使用 `--model-source modelscope` 从 ModelScope 下载：
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 延伸阅读

- **[功能详解](./docs/features_zh.md)**: 探索所有可用功能和命令行参数。
- **[系统架构](./docs/structures_zh.md)**: 深入了解 Mini-SGLang 的设计与数据流。
