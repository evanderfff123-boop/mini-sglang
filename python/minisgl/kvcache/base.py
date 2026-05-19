from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """KV缓存池的基类，定义了KV缓存的接口"""

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...  # 获取指定层的K缓存

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...  # 获取指定层的V缓存

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...  # 将KV存入指定位置

    @property
    @abstractmethod
    def device(self) -> torch.device: ...  # 缓存所在的设备

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...  # 缓存的数据类型

    @property
    @abstractmethod
    def num_layers(self) -> int: ...  # 缓存的层数


@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    """缓存句柄的基类，记录已缓存的前缀长度"""
    cached_len: int  # 已缓存的token长度

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...  # 获取匹配到的缓存索引


class SizeInfo(NamedTuple):
    """缓存大小信息，记录可驱逐和受保护的大小"""
    evictable_size: int  # 可被驱逐的缓存大小
    protected_size: int  # 被保护（不可驱逐）的缓存大小

    @property
    def total_size(self) -> int:
        return self.evictable_size + self.protected_size  # 总缓存大小 = 可驱逐 + 保护


class InsertResult(NamedTuple):
    """插入缓存操作的结果"""
    cached_len: int  # 插入前已在缓存中的长度（应被释放）
    handle: BaseCacheHandle  # 已插入前缀的缓存句柄


class MatchResult(NamedTuple):
    """前缀匹配操作的结果"""
    cuda_handle: BaseCacheHandle  # 匹配到的缓存句柄


class BasePrefixCache(ABC):
    """前缀缓存管理器的基类，负责前缀匹配、插入和驱逐"""

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        锁定或解锁缓存句柄。
        此操作不会修改缓存，只会改变大小信息。
        句柄被锁定时，其对应的缓存不会被驱逐。
        在使用 match_prefix 返回的张量之前，必须先锁定句柄，否则可能被 evict 回收。

        Args:
            handle: 要锁定或解锁的缓存句柄
            unlock: 是否为解锁操作，默认为 False
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        匹配前缀，返回缓存中已匹配的前缀索引。
        此操作不会修改缓存。
        只有在句柄被锁定后，返回的索引才能安全使用。

        Args:
            input_ids: 输入的 token id 序列，形状为 (seq_len,)
        Returns:
            MatchResult: 包含缓存句柄的匹配结果
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        将新的前缀插入缓存。
        此操作会修改缓存。
        Args:
            input_ids: 要插入的 token id 序列，形状为 (seq_len,)
            indices: 存储新前缀的位置索引，形状为 (seq_len,)

        Returns:
            InsertResult: 插入操作的结果
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        从缓存中驱逐一些前缀以释放空间。
        此操作会修改缓存。
        注意：evict(0) 总是安全的，不执行任何操作。
        实际驱逐的大小可能大于请求的大小。
        Args:
            size: 请求驱逐的大小

        Returns:
            torch.Tensor: 被驱逐的索引，形状为 (evict_size,)
        Raises:
            RuntimeError: 如果请求的大小大于可驱逐的大小
        """

    @abstractmethod
    def reset(self) -> None:
        """重置缓存管理器和底层缓存"""

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """获取缓存的大小信息"""

    @abstractmethod
    def check_integrity(self) -> None:
        """检查缓存完整性，如果缓存损坏则抛出异常"""
