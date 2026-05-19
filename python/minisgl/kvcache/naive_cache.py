import torch

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    """朴素缓存句柄，不缓存任何前缀（每次都从头计算）"""
    empty_tensor: torch.Tensor  # 应被 NaivePrefixCache 设置的空张量

    def __init__(self):
        super().__init__(cached_len=0)  # 缓存长度始终为0

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor  # 返回空张量，表示无缓存命中


class NaivePrefixCache(BasePrefixCache):
    """朴素前缀缓存，不做任何缓存优化，每次都重新计算"""

    def __init__(self, device: torch.device):
        self.device = device  # 设备信息
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)  # 空张量
        NaiveCacheHandle.empty_tensor = self.empty_tensor  # 共享空张量到句柄类
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass  # 朴素缓存无需锁定操作

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())  # 总是返回缓存长度为0的结果

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())  # 插入操作不做任何实际缓存

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:
            return self.empty_tensor  # evict(0) 是安全的
        raise NotImplementedError("NaiveCacheManager does not support eviction.")  # 朴素缓存不支持驱逐

    def reset(self) -> None:
        pass  # 朴素缓存无需重置

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)  # 朴素缓存没有缓存空间

    def check_integrity(self) -> None:
        pass  # 朴素缓存无需完整性检查
