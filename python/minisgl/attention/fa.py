from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.utils import is_sm100_supported

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass
class FACaptureData(BaseCaptureData):
    """FlashAttention 后端 CUDA Graph 捕获数据"""
    pass


@dataclass
class FAMetadata(BaseAttnMetadata):
    """FlashAttention 注意力计算所需的元数据"""
    cu_seqlens_k: torch.Tensor  # K 侧累积序列长度（GPU）
    cu_seqlens_q: torch.Tensor  # Q 侧累积序列长度（GPU）
    cache_seqlens: torch.Tensor # 缓存中每个序列的长度（GPU）
    max_seqlen_k: int           # K 侧最大序列长度
    max_seqlen_q: int           # Q 侧最大序列长度

    page_table: torch.Tensor    # 页表（GPU）

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """获取每个序列最后一个 token 的索引"""
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    """基于 FlashAttention (sgl-kernel) 的注意力计算后端"""

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()
        self.config = config  # 模型配置
        self.kvcache = ctx.kv_cache  # 全局 KV 缓存
        self.page_size = ctx.page_size  # 页面大小
        self.capture: FACaptureData | None = None  # CUDA Graph 捕获数据
        self.max_graph_bs = 0  # 最大 CUDA Graph batch size
        self.capture_bs: List[int] = []  # 需要捕获的 batch size 列表
        self.scale = config.head_dim**-0.5  # softmax 缩放因子（1/sqrt(d)）
        self.version = 4 if is_sm100_supported() else 3  # FA 版本：Blackwell 使用 FA4，否则 FA3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行 FlashAttention 注意力计算前向传播"""
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 将当前 KV 存入缓存
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),  # 该层的 K 缓存
            v_cache=self.kvcache.v_cache(layer_id),  # 该层的 V 缓存
            page_table=metadata.page_table,          # 页表
            cache_seqlens=metadata.cache_seqlens,     # 缓存序列长度
            cu_seqlens_q=metadata.cu_seqlens_q,       # Q 累积长度
            cu_seqlens_k=metadata.cu_seqlens_k,       # K 累积长度
            max_seqlen_q=metadata.max_seqlen_q,       # Q 最大长度
            softmax_scale=self.scale,                 # softmax 缩放
            version=self.version,                     # FA 版本
        )

    def prepare_metadata(self, batch: Batch) -> None:
        """为当前 batch 准备 FlashAttention 所需的元数据"""
        reqs = batch.padded_reqs

        padded_size = len(reqs)  # padding 后的 batch 大小
        seqlens_q = [req.extend_len for req in reqs]  # 每个请求的扩展长度
        seqlens_k = [req.device_len for req in reqs]  # 每个请求在设备上的总长度
        cached_lens = [req.cached_len for req in reqs]  # 每个请求已缓存的前缀长度
        max_seqlen_k = max(seqlens_k)  # K 侧最大长度
        max_seqlen_q = max(seqlens_q)  # Q 侧最大长度
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)  # CPU 上创建序列长度
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)  # 异步复制到 GPU
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)  # K 累积长度
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)  # 异步复制到 GPU

        if max_seqlen_q == 1:
            # decode 阶段：每个请求扩展长度为 1
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill 无缓存命中
            cu_seqlens_q = cu_seqlens_k  # Q 累积长度等于 K 累积长度
        else:  # 普通 extend prefill 有部分缓存命中
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        page_table = get_global_ctx().page_table  # 全局页表
        new_page_table = torch.stack(  # 注意：全局页表以 page_size=1 为单位，需要根据实际 page_size 切片
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        if self.page_size > 1:
            new_page_table.div_(self.page_size, rounding_mode="floor")  # 将页号转换为实际 page 索引
        batch.attn_metadata = FAMetadata(
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
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)  # 排序以确保确定性

    def prepare_for_capture(self, batch: Batch) -> None:
        """准备捕获 CUDA Graph：创建元数据并绑定到预分配缓冲区"""
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,  # 最大序列长度为页表列数 * page_size
            max_seqlen_q=1,  # decode 阶段 Q 长度固定为 1
            page_table=capture.page_table[:bs, :],
        )
        batch.attn_metadata = metadata

    def prepare_for_replay(self, batch: Batch) -> None:
        """准备重放 CUDA Graph：将当前元数据复制到已捕获的缓冲区中"""
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FAMetadata)
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q 对 decode 阶段始终为 [0, 1, 2, ..., bs]（即无操作）
        table_len = metadata.page_table.size(1)
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)  # 复制 K 累积长度
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)  # 复制缓存长度
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)  # 复制页表


def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 表示无限上下文窗口
    softcap: float = 0.0,  # 0.0 表示不启用 softcap
    num_splits: int = 0,  # 可调整以优化速度
    pack_gqa: bool | None = None,  # 可调整以优化速度
    causal: bool = True,
) -> torch.Tensor:
    """调用 sgl-kernel 的 flash_attn_with_kvcache 执行分页 KV 缓存的 FlashAttention"""
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: 在 Blackwell 上支持 FA4
    )
