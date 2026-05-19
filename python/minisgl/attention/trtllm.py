from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.core import Batch, get_global_ctx

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class TRTLLMCaptureData(BaseCaptureData):
    """TensorRT-LLM 后端 CUDA Graph 捕获数据"""
    pass


@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    """TensorRT-LLM 注意力计算所需的元数据"""
    cu_seqlens_k: torch.Tensor  # K 侧累积序列长度（GPU）
    cu_seqlens_q: torch.Tensor  # Q 侧累积序列长度（GPU）
    cache_seqlens: torch.Tensor # 缓存中每个序列的长度（GPU）
    max_seqlen_k: int           # K 侧最大序列长度
    max_seqlen_q: int           # Q 侧最大序列长度

    page_table: torch.Tensor    # 页表（GPU）

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """获取每个序列最后一个 token 的索引"""
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TensorRTLLMBackend(BaseAttnBackend):
    """基于 TensorRT-LLM (FlashInfer TRTLLM API) 的注意力计算后端"""

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config  # 模型配置
        self.kvcache = ctx.kv_cache  # 全局 KV 缓存
        self.page_size = ctx.page_size  # 页面大小
        self.capture: TRTLLMCaptureData | None = None  # CUDA Graph 捕获数据
        self.max_graph_bs = 0  # 最大 CUDA Graph batch size
        self.capture_bs: List[int] = []  # 需要捕获的 batch size 列表
        self.scale = config.head_dim**-0.5  # softmax 缩放因子
        self.workspace_buffer = torch.empty(  # 工作空间缓冲区（128MB）
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行 TensorRT-LLM 注意力计算前向传播"""
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        metadata = batch.attn_metadata
        assert isinstance(metadata, TRTLLMMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 将当前 KV 存入缓存
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))  # 获取层缓存

        if batch.is_prefill:
            # prefill 阶段：使用 trtllm_batch_context_with_kv_cache
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,  # 工作空间
                block_tables=metadata.page_table,        # 块表（页表）
                seq_lens=metadata.cache_seqlens,          # 序列长度
                max_q_len=metadata.max_seqlen_q,          # Q 最大长度
                max_kv_len=metadata.max_seqlen_k,         # KV 最大长度
                bmm1_scale=self.scale,                    # 第一个 bmm 的缩放
                bmm2_scale=1.0,                           # 第二个 bmm 的缩放
                cum_seq_lens_q=metadata.cu_seqlens_q,     # Q 累积长度
                cum_seq_lens_kv=metadata.cu_seqlens_k,    # KV 累积长度
                kv_layout="NHD",                          # KV 缓存布局
                batch_size=batch.size,                    # batch 大小
                out_dtype=q.dtype,                        # 输出数据类型
            )
        else:
            # decode 阶段：使用 trtllm_batch_decode_with_kv_cache
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_seq_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )

    def prepare_metadata(self, batch: Batch) -> None:
        """为当前 batch 准备 TensorRT-LLM 所需的元数据"""
        reqs = batch.padded_reqs

        padded_size = len(reqs)  # padding 后的 batch 大小
        seqlens_q = [req.extend_len for req in reqs]  # 每个请求的扩展长度
        seqlens_k = [req.device_len for req in reqs]  # 每个请求在设备上的总长度
        cached_lens = [req.cached_len for req in reqs]  # 每个请求已缓存的前缀长度
        max_seqlen_k = max(seqlens_k)  # K 侧最大长度
        max_seqlen_q = max(seqlens_q)  # Q 侧最大长度
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)  # 异步复制到 GPU
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        if max_seqlen_q == 1:
            # decode 阶段：每个扩展长度为 1
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill 无缓存命中
            cu_seqlens_q = cu_seqlens_k
        else:  # 普通 extend prefill 有部分缓存命中
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table
        new_page_table = torch.stack(  # 注意：全局页表以 page_size=1 为单位，需要根据实际 page_size 切片
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")  # 转换为实际 page 索引
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的缓冲区"""
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = TRTLLMCaptureData.create(
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        """准备捕获 CUDA Graph：创建元数据并绑定到预分配缓冲区"""
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = TRTLLMMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        """准备重放 CUDA Graph：将当前元数据复制到已捕获的缓冲区中"""
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, TRTLLMMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q 对 decode 阶段始终为 [0, 1, 2, ..., bs]
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
