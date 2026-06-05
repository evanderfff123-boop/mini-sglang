from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, NamedTuple, Tuple

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers import set_rope_device
from minisgl.models import create_model, load_weight
from minisgl.moe import create_moe_backend
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

from .config import EngineConfig
from .graph import GraphRunner, get_free_memory, mem_GB
from .sample import BatchSamplingArgs, Sampler

logger = init_logger(__name__)


class ForwardOutput(NamedTuple):
    """前向传播的输出结果，包含下一步的token和同步事件。"""
    next_tokens_gpu: torch.Tensor
    next_tokens_cpu: torch.Tensor
    copy_done_event: torch.cuda.Event

# 定义推理引擎核心类，用于执行模型推理相关的高性能计算任务
class Engine:
    """推理引擎核心类，负责模型加载、KV cache管理、前向传播和采样。"""

    # 初始化方法，接收一个 EngineConfig 配置对象
    def __init__(self, config: EngineConfig):
        """初始化引擎：设置设备、分布式通信、模型、KV cache、采样器和CUDA图捕获。"""
        # 断言确保在此之前 PyTorch 的 CUDA 运行时尚未初始化，防止多进程进程组初始化产生冲突
        assert not torch.cuda.is_initialized()
        # 初始化和注册张量并行（Tensor Parallelism）的 Rank 与总大小信息到全局
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # 调用辅助函数微调或修正配置项中的具体参数
        _adjust_config(config)

        # --- 设备与CUDA流初始化 ---
        # 根据当前进程所处的 TP rank 号，构造对应的 GPU 设备对象
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        # 将当前的 PyTorch CUDA 上下文绑定到指定的 GPU 设备上
        torch.cuda.set_device(self.device)
        # 设定 PyTorch 随机数生成器的种子，保障后续随机抽样等逻辑的一致性与可复现性
        torch.manual_seed(42)
        # 创建一个独立的 CUDA 流（CUDA Stream），用于后续的前向传播异步执行
        self.stream = torch.cuda.Stream()
        # 将当前线程的默认工作 CUDA 流切换为刚才创建的自定义流
        torch.cuda.set_stream(self.stream)
        # 保存并记录执行推理和模型权重所使用的数据精度（如 FP16 或 BF16）
        self.dtype = config.dtype
        # 实例化全局上下文管理对象，设定单页的大小
        self.ctx = Context(config.page_size)
        # 将当前创建的上下文设置为本轮执行的全局上下文
        set_global_ctx(self.ctx)

        # --- 分布式通信初始化 ---
        # 初始化张量并行（TP）在分布式环境中的 CPU 通信进程组（通常采用 gloo 后端实现基础握手与内存对齐）
        self.tp_cpu_group = self._init_communication(config)
        # 同步各 rank 进程并获取当前可用（空闲）的物理显存容量大小
        init_free_memory = self._sync_get_memory()[1]
        # 仅让主 rank（Rank 0）在日志中打印模型加载前的空闲显存（转换为 GB 单位）
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # --- 模型初始化（先创建meta设备上的骨架，再加载权重）---
        # 将旋转位置编码（RoPE）对应的元操作绑定到当前的物理 GPU 设备上
        set_rope_device(self.device)
        # 在 meta（元）设备以及指定的数据精度下运行，以避免在声明参数时开辟实际物理内存
        with torch.device("meta"), torch_dtype(config.dtype):
            # 根据模型架构配置，创建未加载权重的模型骨架
            self.model = create_model(config.model_config)
        # 从本地加载模型权重状态字典，并将其拷贝填充到构建好的模型实例中（此时实际占用显存）
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # --- KV cache池初始化 ---
        # 结合配置及除去模型加载后剩下的可用显存，计算当前环境可容纳的最大 KV 缓存物理页数
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        # 根据计算得出的总页数和单页容量，换算出能够存储的最大 Token 总数
        num_tokens = self.num_pages * config.page_size
        # 创建 KV Cache 物理内存池，将其绑定至全局上下文以及本类属性中
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            # 传入模型参数配置以获得每层维度和 Attention 头数量
            model_config=config.model_config,
            # 总分配页数在基础页数上加 1，该额外空间用作 dummy page（占位或异常回退页）
            num_pages=self.num_pages + 1,
            # 指定单页存放 Token 的数量
            page_size=config.page_size,
            # 指定分配该显存池的物理目标 GPU 设备
            device=self.device,
            # 指定显存池的数据存储精度
            dtype=self.dtype,
        )

        # --- page table（页表）初始化 ---
        # 注意：1. 保证128字节对齐（便于GPU高效访问）；2. 存储的是原始位置而非页号
        # 将推理支持的最大序列长度，限制在用户配置值和物理缓存实际总承载量的最小值内
        self.max_seq_len = min(config.max_seq_len, num_tokens)  # 取配置值和KV cache容量中的较小值
        # 将最大序列长度向上舍入至 32 的倍数，以便满足 GPU 内存指令的对齐边界
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        # 在显存上创建二维页表张量，初始化为 0。多开辟 1 行空间供占位请求（dummy request）使用
        self.ctx.page_table = self.page_table = torch.zeros(  # +1 用于dummy request（占位请求）
            # 页表尺寸设计为：(最大运行并发数 + 1, 向上对齐的最大序列长度)
            (config.max_running_req + 1, aligned_max_seq_len),
            # 页表中存储物理位置偏移索引，采用 32 位有符号整数
            dtype=torch.int32,
            # 将页表分配至当前 GPU 设备上
            device=self.device,
        )

        # --- Attention和MoE后端初始化 ---
        # 根据底层注意力计算架构及模型配置，初始化 Attention 执行后端（如 FlashAttention 等算子库）
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        # 检查模型是否属于混合专家（MoE）架构
        if config.model_config.is_moe:
            # 如果是 MoE 模型，初始化指定的专家路由及计算后端，并注册至全局上下文
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # --- 采样器初始化 ---
        # 实例化采样器（Sampler）对象，用于接收 logits，并按温度、核采样或 TopK 采样逻辑输出 Token
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        # 再次执行同步并检测系统在完成所有对象初始化后的剩余空闲显存大小
        post_free_memory = self._sync_get_memory()[0]
        # 仅由 Rank 0 进程打印出最终初始化完毕后的系统可用空闲显存（以 GB 为单位）
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # --- CUDA图捕获初始化 ---
        # 实例化一个专为 CUDA Graph 静态执行准备的哑请求（dummy request）实例
        self.dummy_req = Req(
            # 在 CPU 上分配一个只包含单 Token（例如 [0]）的张量作为占位输入
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            # 使用最大槽位索引作为该哑请求在页表中的定位行索引
            table_idx=config.max_running_req,
            # 设定当前已缓存 Token 长度为 0
            cached_len=0,
            # 设定本次解码的目标输出 Token 长度为 1
            output_len=1,
            # 设置该请求的唯一标识（UID）为 -1，表示其仅为系统内部占位符
            uid=-1,
            # 将采样参数设置为空
            sampling_params=None,  # type: ignore
            # 将缓存控制句柄设置为空
            cache_handle=None,  # type: ignore
        )
        # 将页表中哑请求所在的行全部填充为物理 Token 池的虚拟上限位置（即 dummy page 的起始绝对偏移）
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)
        # 创建 CUDA 图（CUDA Graphs）管理器实例，用以录制并快速回放特定 batch size 的 GPU 执行指令
        self.graph_runner = GraphRunner(
            # 传入用于捕获与执行的自定义 CUDA 流
            stream=self.stream,
            # 传入绑定的 GPU 设备
            device=self.device,
            # 传入已加载参数的模型实例
            model=self.model,
            # 传入绑定的注意力计算后端
            attn_backend=self.attn_backend,
            # 传入捕获 CUDA Graph 所依据的 Batch Size 列表
            cuda_graph_bs=config.cuda_graph_bs,
            # 传入 CUDA Graph 机制所能支持的最大 Batch Size 上限
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            # 传入初始可用显存以确保图捕获能申请到足量的静态工作缓存（Workspace）
            free_memory=init_free_memory,
            # 传入向上对齐后的最大序列长度
            max_seq_len=aligned_max_seq_len,
            # 传入模型的总词表大小
            vocab_size=config.model_config.vocab_size,
            # 传入刚刚初始化的哑请求对象，用于录制过程中的输入对齐与静态页表占位
            dummy_req=self.dummy_req,
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化分布式通信：根据配置选择gloo+pynccl或纯nccl后端。"""
        if config.tp_info.size == 1 or config.use_pynccl:
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.group.WORLD
            assert tp_cpu_group is not None
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            assert tp_cpu_group is not None
        return tp_cpu_group

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载模型权重：使用dummy权重或从指定路径加载真实权重。"""
        if config.use_dummy_weight:
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """根据可用显存和模型内存占用，计算KV cache的page数量。"""
        new_free_memory = self._sync_get_memory()[1]
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        num_pages = config.num_page_override
        if num_pages is None:
            model_memory = old_free_memory - new_free_memory
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            num_pages = available_memory // cache_per_page

        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        num_tokens = num_pages * config.page_size
        real_kv_size = num_pages * cache_per_page
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        return num_pages

    def _sync_get_memory(self) -> Tuple[int, int]:
        """获取所有TP rank上的最小和最大可用显存，并检查是否均衡。"""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        free_memory = get_free_memory(self.device)
        # 用正负数编码min和max，通过一次all_reduce获取全局最小/最大值
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())
        max_free_memory = -int(free_mem_tensor[1].item())
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")

        return min_free_memory, max_free_memory

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """对一个 batch 执行前向传播，并返回下一步预测的 token（分别保存在 GPU 和 CPU 上）。
        
        参数:
            batch (Batch): 当前待处理的批数据，包含请求信息和状态。
            args (BatchSamplingArgs): 采样参数（如 Temperature, Top-P 等）。
            
        返回:
            ForwardOutput: 包含 GPU 上的 token、CPU 上的 token 以及用于同步的 CUDA 事件。
        """
        # 验证当前活跃的 CUDA 流是否为该实例专用的工作流(self.stream)
        # 确保所有 CUDA 算子都在指定的流中串行执行，避免多流并发导致数据竞争
        assert torch.cuda.current_stream() == self.stream
        # 使用上下文管理器准备前向传播所需的环境（例如设置位置编码、KV Cache 寻址元数据等）
        with self.ctx.forward_batch(batch):
            # 判断当前 batch 的形状和状态是否满足使用 CUDA Graph 的条件
            # CUDA Graph 能够将静态拓扑的 GPU 算子打包，消除 CPU 提交任务的开销，常用于 Decode 阶段
            if self.graph_runner.can_use_cuda_graph(batch):
                logits = self.graph_runner.replay(batch)  # 使用CUDA图加速decode阶段
            else:
                logits = self.model.forward() # 常规前向传播（如 Prefill 阶段，或形状动态变化时）

        for req in batch.reqs:
            req.complete_one()  # 将每个请求已生成的 token 计数器加 1

        # 1. 提取有效 batch 大小内的 Logits 并进行采样，获取预测的 Token ID
        # 2. 将 Token ID 的数据类型统一转换为 int32，保持在 GPU 上
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 将生成的 Token 从 GPU 异步拷贝到 CPU
        # non_blocking=True 允许 CPU 线程不等待拷贝完成而直接向下执行，实现 CPU-GPU 流水线并行
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 创建一个 CUDA 事件，用于后续跟踪异步拷贝任务的完成状态
        copy_done_event = torch.cuda.Event()
        # 在当前 CUDA 工作流中记录（Record）该事件
        # 当 GPU 执行完 record 之前的所有任务（包括采样和数据拷贝）时，该事件会被标记为已完成
        copy_done_event.record(self.stream)
        # 返回打包好的前向传播结果。
        # 外部调度器（运行在 CPU 上）可以在必要时通过 copy_done_event.synchronize() 
        # 或 event.query() 来安全地读取 next_tokens_cpu，从而避免提前访问导致数据未准备就绪
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    def shutdown(self) -> None:
        """释放引擎资源：销毁CUDA图和分布式进程组。"""
        self.graph_runner.destroy_cuda_graphs()
        torch.distributed.destroy_process_group()
        destroy_distributed()


def _align_up_32(num: int) -> int:
    """将数值向上对齐到32的倍数（用于page table的列对齐）。"""
    return (num + 31) // 32 * 32


def _adjust_config(config: EngineConfig):
    """根据硬件能力和模型特性自动调整配置参数。"""
    def override(attr: str, value: Any):  # 危险操作，需谨慎使用
        object.__setattr__(config, attr, value)

    if config.attention_backend == "auto":
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        override("attention_backend", backend)
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        override("page_size", 64)
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    if config.model_config.is_moe and config.moe_backend == "auto":
        override("moe_backend", "fused")
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
