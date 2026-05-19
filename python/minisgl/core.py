from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal

import torch

if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


@dataclass
class SamplingParams:
    """采样参数配置"""
    temperature: float = 0.0  # 采样温度，0 表示 greedy
    top_k: int = -1           # top-k 采样（-1 表示不限制）
    top_p: float = 1.0        # top-p (nucleus) 采样
    ignore_eos: bool = False  # 是否忽略结束符
    max_tokens: int = 1024    # 最大生成 token 数

    @property
    def is_greedy(self) -> bool:
        """判断是否为贪婪解码（温度 <= 0 或 top_k == 1 且 top_p == 1.0）"""
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)
class Req:
    """单个请求的数据结构，包含输入序列和状态信息"""
    input_ids: torch.Tensor   # CPU 上的 token id 序列
    table_idx: int            # 在页表中的索引
    cached_len: int           # 已缓存的前缀长度
    output_len: int           # 已生成的输出 token 数
    uid: int                  # 请求的唯一标识符
    sampling_params: SamplingParams  # 采样参数
    cache_handle: BaseCacheHandle    # 缓存句柄

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)  # 当前在设备上的 token 总数
        self.max_device_len = len(self.input_ids) + self.output_len  # 设备上最大 token 数
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    @property
    def remain_len(self) -> int:
        """还能生成的 token 数"""
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        """需要计算（还未缓存）的 token 数"""
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        """完成一个 decode 步骤：将当前 token 标记为已缓存，设备长度 +1"""
        self.cached_len = self.device_len  # 当前 token 变为已缓存
        self.device_len += 1               # 新增一个 token（下一个要生成的）

    def append_host(self, next_token: torch.Tensor) -> None:
        """将新生成的 token 追加到宿主端的 input_ids 末尾"""
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        """是否还能继续解码（剩余长度 > 0）"""
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """批次数据，包含多个请求及调度器设置的信息"""
    reqs: List[Req]          # 当前批次的请求列表
    phase: Literal["prefill", "decode"]  # 当前阶段：prefill 或 decode
    # 以下字段应由调度器设置
    input_ids: torch.Tensor = field(init=False)   # 当前批次所有输入的 token ids
    positions: torch.Tensor = field(init=False)   # 每个 token 的位置编码
    out_loc: torch.Tensor = field(init=False)     # 输出位置（存储 KV 的位置索引）
    padded_reqs: List[Req] = field(init=False)    # padding 后的请求列表（长度对齐）
    # 以下字段应由注意力后端设置
    attn_metadata: BaseAttnMetadata = field(init=False)  # 注意力计算元数据

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"  # 是否为 prefill 阶段

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"  # 是否为 decode 阶段

    @property
    def size(self) -> int:
        return len(self.reqs)  # 实际 batch 大小

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)  # padding 后的 batch 大小


@dataclass
class Context:
    """全局上下文，保存推理过程中的所有共享状态"""
    page_size: int  # 页大小（每页包含的 token 数）
    # 注意：此表始终以 page_size = 1 为单位
    page_table: torch.Tensor = field(init=False)  # 全局页表
    attn_backend: BaseAttnBackend = field(init=False)  # 注意力计算后端
    moe_backend: BaseMoeBackend = field(init=False)    # MoE 计算后端
    kv_cache: BaseKVCachePool = field(init=False)      # KV 缓存池
    _batch: Batch | None = field(default=None, init=False)  # 当前活跃批次

    @property
    def batch(self) -> Batch:
        assert self._batch is not None, "No active batch in context"
        return self._batch  # 获取当前活跃批次

    @contextmanager
    def forward_batch(self, batch: Batch):
        """上下文管理器，在 forward 传播期间设置活跃批次"""
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch  # 设置当前批次
            yield
        finally:
            self._batch = None  # 清空当前批次


_GLOBAL_CTX: Context | None = None  # 全局单例上下文


def set_global_ctx(ctx: Context):
    """设置全局上下文（仅可设置一次）"""
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    """获取全局上下文"""
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
