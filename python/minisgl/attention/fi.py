from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.env import ENV
from minisgl.utils import div_even, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    """计算大于等于 n 的最小 2 的幂"""
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    """FlashInfer 后端 CUDA Graph 捕获数据"""
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens  # 用于 last_page_len 的占位张量

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table  # 页表索引（FlashInfer 将其视为 indices）


@dataclass
class FIMetadata(BaseAttnMetadata):
    """FlashInfer 注意力计算所需的元数据"""
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # CPU 上的 Query 累积序列长度
    cu_seqlens_k_cpu:   torch.Tensor  # CPU 上的 Key 累积序列长度
    cu_seqlens_q_gpu:   torch.Tensor  # GPU 上的 Query 累积序列长度
    indices:            torch.Tensor  # GPU 上的页表索引（用于分页 KV 缓存）
    last_page_len_cpu:  torch.Tensor  # CPU 上每页最后一个 page 的实际 token 数
    num_qo_heads:       int           # Query/Output 头数
    num_kv_heads:       int           # Key/Value 头数
    head_dim:           int           # 每个头的维度
    page_size:          Literal[1]    # 页大小（当前仅支持 1）
    pos_encoding_mode:  str           # 位置编码模式
    seq_lens_cpu:       torch.Tensor  # CPU 上的每个序列长度
    dtype:              torch.dtype   # 数据类型
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper  # FlashInfer wrapper
    initialized:        bool = False  # 是否已初始化（已完成 plan 调用）
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        # 验证各张量所在的设备符合预期
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        """获取每个 batch 中最后一个 token 的索引（用于取最后位置的 logits）"""
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1  # cu_seqlans 相邻差值为最后一个位置


class FlashInferBackend(BaseAttnBackend):
    """基于 FlashInfer 库的注意力计算后端"""

    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config  # 模型配置
        self.kvcache = get_global_ctx().kv_cache  # 全局 KV 缓存
        self.device = self.kvcache.device  # 设备
        self.float_workspace_buffer = torch.empty(  # FlashInfer 需要的 float workspace 缓冲区（128MB）
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(  # prefill 阶段的 FlashInfer wrapper
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 较慢，使用 fa2
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(  # decode 阶段的 FlashInfer wrapper
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )

        # 注意：将 prefill wrapper 的 int workspace 共享给 decode wrapper
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # 初始化一些数据成员
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)  # 本设备上的 QO 头数
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)  # 本设备上的 KV 头数

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)  # 缓存的全 1 CPU 张量
        # 用于 CUDA Graph
        self.capture_bs: List[int] = []  # 需要捕获的 batch size 列表
        self.max_graph_bs = 0  # 最大 CUDA Graph batch size
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}  # 不同 batch size 的 Graph wrapper
        self.capture: FICaptureData | None = None  # 捕获数据
        self.last_event = torch.cuda.Event()  # 用于同步的 CUDA event
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        """初始化 FlashInfer 的 plan（仅一次），为 prefill/decode 准备调度参数"""
        if metadata.initialized:
            return  # 已初始化，跳过

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        metadata.initialized = True
        # FlashInfer plan 会重用宿主端的固定内存缓冲区并启动异步 H2D 拷贝
        # 在下一次 plan 修改宿主缓冲区前，需要等待前一次完成
        self.last_event.synchronize()
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # decode 模式的 plan
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,  # 异步执行
            )
        else:
            # prefill 模式的 plan
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,  # 因果注意力
            )
        self.last_event.record()  # 记录当前 plan 完成的事件

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """获取指定长度的全 1 CPU 张量（用于 last_page_len 占位）"""
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]  # 从已有缓存中截取
        # 扩展到下一个 2 的幂，减少重新分配次数
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        return self.cached_ones_cpu[:bs]

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        """执行 FlashInfer 注意力计算前向传播"""
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # 将 page 维度视为 1（因为 page_size=1）
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)  # 确保 plan 已初始化
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)  # 将当前 KV 存入缓存
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))  # 获取层缓存
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))  # 展平以适应 FlashInfer 格式
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)  # 执行注意力计算

    def prepare_metadata(self, batch: Batch) -> None:
        """为当前 batch 准备 FlashInfer 所需的元数据"""
        reqs = batch.padded_reqs

        padded_size = len(reqs)  # padding 后的 batch 大小
        seqlens_q = [req.extend_len for req in reqs]  # 每个请求的扩展（新增）长度
        seqlens_k = [req.device_len for req in reqs]  # 每个请求在设备上的总长度
        cached_lens = [req.cached_len for req in reqs]  # 每个请求已缓存的前缀长度
        max_seqlen_q = max(seqlens_q)
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)  # CPU 上的 KV 序列长度
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)  # K 侧累积长度
        if max_seqlen_q == 1:  # decode 阶段：每个请求扩展长度都为 1
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)  # Q 侧为均匀递增
        elif all(l == 0 for l in cached_lens):  # prefill，无缓存命中
            cu_seqlens_q_cpu = cu_seqlens_k_cpu  # Q 侧与 K 侧相同（全部需要计算）
        else:  # 普通 extend prefill，部分缓存命中
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table  # 全局页表
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),  # 异步复制到 GPU
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),  # 拼接页表索引
            last_page_len_cpu=self._get_ones_cpu(padded_size),  # page_size=1，每页的 last_page_len 为 1
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,  # 根据阶段选择 wrapper
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获所需的缓冲区"""
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        # 创建捕获数据缓冲区
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # 使用 1D 作为不规则索引
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)  # 排序以确保确定性

    @cached_property
    def use_tensor_cores(self) -> bool:
        """判断是否使用 Tensor Cores（GQA >= 4 或环境变量覆盖）"""
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value  # 环境变量覆盖
        GQA = self.config.num_qo_heads // self.config.num_kv_heads  # GQA 分组比
        return GQA >= 4  # GQA >= 4 时使用 Tensor Cores 更高效

    def prepare_for_capture(self, batch: Batch) -> None:
        """准备捕获 CUDA Graph：为当前 batch size 创建 Graph wrapper"""
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],  # 使用预分配缓冲区
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer  # 共享 int workspace
        self.prepare_metadata(batch)  # 准备元数据
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        metadata.wrapper = self.graph_wrappers[bs]  # 替换为 Graph wrapper
        self._initialize_metadata_once(metadata)  # 初始化 plan

    def prepare_for_replay(self, batch: Batch) -> None:
        """准备重放 CUDA Graph：复用已捕获的 Graph wrapper"""
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.wrapper = self.graph_wrappers[bs]  # 使用已捕获的 Graph wrapper
        self._initialize_metadata_once(metadata)  # 初始化 plan
