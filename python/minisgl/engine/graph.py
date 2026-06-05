from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from minisgl.core import Batch, Req, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import init_logger
from tqdm import tqdm

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    """CUDA图捕获所需的缓冲区，存储input_ids、positions等输入输出tensor。"""
    input_ids: torch.Tensor  # 输入token ID
    out_loc: torch.Tensor  # 输出位置索引
    positions: torch.Tensor  # 位置编码位置
    logits: torch.Tensor  # 模型输出logits

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        """初始化指定batch size的CUDA图捕获缓冲区。"""
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        """将缓冲区的tensor切片设置到batch中，使模型前向时使用缓冲区内存。"""
        _slice = slice(batch.padded_size)
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]

    def copy_from(self, batch: Batch) -> None:
        """将batch中的数据拷贝到缓冲区（在replay前准备输入数据）。"""
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    """根据可用显存决定需要捕获CUDA图的batch size列表。"""
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


def mem_GB(size: int) -> str:
    """将字节数转换为GiB单位的可读字符串。"""
    return f"{size / (1024**3):.2f} GiB"


def get_free_memory(device: torch.device) -> int:
    """获取指定GPU设备的当前可用显存（字节）。"""
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    """CUDA 图（CUDA Graph）捕获和执行器，用于管理和调度多个常用 Batch Size 的 CUDA 图。
    通过预先录制 GPU 算子执行序列，消除 CPU 提交任务的开销，极大地加速大模型在 Decode（生成）阶段的前向传播。
    """

    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        """初始化 GraphRunner：确定支持的 Batch Size 列表并触发 CUDA 图的捕获。
        参数:
            stream: 用于捕获和重放 CUDA 图的专用工作流。
            device: 运行设备（GPU）。
            model: 模型实例（例如 Transformer）。
            attn_backend: 核心 Attention 后端（负责管理 KV Cache 寻址等）。
            cuda_graph_bs: 显式指定的 CUDA 图 Batch Size 列表。
            cuda_graph_max_bs: 支持的最大 Batch Size。
            free_memory: 当前可用的显存大小（用于自动决定捕获哪些 Batch Size）。
            max_seq_len: 支持的最大序列长度。
            vocab_size: 词表大小（用于计算输出 Logits 占用的显存）。
            dummy_req: 填充用的虚拟请求对象（用于拼凑 Batch）。
        """
        # 1. 自动或手动确定需要录制的 Batch Size 列表（例如 [1, 2, 4, 8, 16]）
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend  # attention后端，用于图和replay的准备
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0  # 录制的最大 Batch Size
        self.graph_bs_list = sorted(cuda_graph_bs)  # 升序排列的 Batch Size 列表
        self.dummy_req = dummy_req  # 用于 Padding 的虚拟请求
        self.stream = stream
        self.device = device
        # 2. 执行核心的 CUDA 图捕获流程
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        """捕获（录制）多个 Batch Size 的 CUDA 图。
        
        基本原理：
        CUDA 图要求输入/输出张量的【内存地址（指针）是固定不变的】。
        因此，我们必须预先分配好一个“静态缓冲区（Static Buffer）”，在录制和重放时都读写这个固定的缓冲区。
        """
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        # 初始化 Attention 后端的图捕获状态（如分配固定的 KV Cache 索引缓冲区）
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        # 清理显存碎片，确保捕获阶段有连续且干净的显存空间
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # 分配静态输入/输出缓冲区（Buffer），用于在重放时存放输入数据和输出的 Logits
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        
        # 创建进度条（通常只在 Tensor Parallel 的主进程/主卡上显示，避免打印混乱）
        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # 非主rank不显示进度条
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                pool = graph.pool()  # 复用CUDA图的内存池以减少总显存占用
            self.graph_map[bs] = graph

        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        """判断当前batch是否可以使用CUDA图加速（仅decode阶段且batch size在支持范围内）。"""
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        """回放CUDA图：将输入拷贝到缓冲区后执行预录制的CUDA图。"""
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]  # 根据padded size选择对应batch size的图
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        """将batch填充到CUDA图支持的最近batch size（使用dummy req填充）。"""
        padded_size = (  # 选择第一个大于等于batch size的可用图batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # 注意：必须在释放NCCL资源前调用，否则可能导致程序挂起
    def destroy_cuda_graphs(self) -> None:
        """销毁CUDA图并触发垃圾回收。"""
        del self.graph_map
        gc.collect()
