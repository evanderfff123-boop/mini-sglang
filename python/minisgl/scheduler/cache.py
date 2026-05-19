from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Req
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
from minisgl.utils import div_ceil

if TYPE_CHECKING:
    from .utils import PendingReq


class CacheManager:
    """管理显存中的 KV Cache 页面分配、释放和前缀缓存。"""

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        device = page_table.device
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size  # 空闲页面槽位（页对齐）
        self.prefix_cache = create_prefix_cache(device=device, type=type)  # 前缀缓存（支持前缀匹配与逐出）
        self.device = device  # 当前设备
        self.num_pages = num_pages  # 总页面数
        self.page_table = page_table  # 页表引用
        self.page_size = page_size  # 每页 token 数

    def match_req(self, req: PendingReq) -> MatchResult:
        """在缓存中匹配请求的前缀，返回匹配结果。"""
        input_len = req.input_len
        assert input_len > 0, "Input length must be greater than 0."
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        """当前可用 token 数（空闲页 + 可逐出页）。"""
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        """锁定一个缓存句柄，防止被逐出。"""
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        """解锁一个缓存句柄，允许被逐出。"""
        self.prefix_cache.lock_handle(handle, unlock=True)

    # --- 页分配 ---
    def allocate_paged(self, reqs: List[Req]) -> None:
        """为一批请求分配新的物理页面，并写入页表。"""
        needed_pages = 0
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)  # 需要新分配的起始页
            last_page = div_ceil(req.device_len, self.page_size)  # 需要分配到的末尾页
            if last_page > first_page:
                needed_pages += last_page - first_page
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            allocated = self._page_to_token(self._allocate(needed_pages))
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

    # --- 缓存请求（prefill 结束后处理） ---
    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        insert_ids = req.input_ids[: req.cached_len]  # 需要插入缓存的前缀 token
        page_indices = self.page_table[req.table_idx, : req.cached_len]  # 对应的页表索引
        old_handle = req.cache_handle  # 旧的缓存句柄
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)  # 插入前缀并获取新句柄
        # unlock until all operations on handle is done
        self.unlock(old_handle)  # 解锁旧句柄
        # this part is already in the prefix cache, free it
        self._free(page_indices[old_handle.cached_len: cached_len])  # 释放已被其他人缓存的页面
        if finished:  # this tail part should be freed
            self._free(page_indices[new_handle.cached_len:])  # 请求结束，释放尾部页面
        else:  # keep the tail part, update the handle
            req.cache_handle = new_handle  # 更新为新的缓存句柄
            self.lock(new_handle)  # 锁定新句柄

    # --- 完整性检查 ---
    def check_integrity(self) -> None:
        """检查空闲页 + 缓存页是否等于总页数，确保没有内存泄漏。"""
        self.prefix_cache.check_integrity()
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            assert torch.all(self.free_slots % self.page_size == 0)  # 确保空闲槽位页对齐

    # --- 延迟释放上下文管理器 ---
    @contextmanager
    def lazy_free_region(self):
        """在上下文内延迟回收页面（批量合并后再释放，减少碎片）。"""
        def lazy_free(indices: torch.Tensor) -> None:
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []
        try:
            self._free = lazy_free  # 替换 _free 为延迟版本
            yield
        finally:
            del self._free  # 恢复正常的 _free
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)  # 一次性合并空闲槽位

    # --- 物理页面分配（核心） ---
    def _allocate(self, needed_pages: int) -> torch.Tensor:
        """从空闲列表中分配指定数量的页面，不够则触发逐出。"""
        if needed_pages > (free_pages := len(self.free_slots)):
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)  # 逐出旧页面
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        allocated = self.free_slots[:needed_pages]
        self.free_slots = self.free_slots[needed_pages:]  # 从空闲列表中移除已分配的
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        """释放指定页面，归还到空闲列表。"""
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    # --- 页面索引转 token 索引 ---
    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        """将页号（page index）转换为页内所有 token 的连续索引。"""
        if self.page_size == 1:
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    """将分配的物理页面地址写入页表对应的位置。"""
    needed_tokens = len(allocated)
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        first_pos, last_pos = first_page * page_size, last_page * page_size
        length = last_pos - first_pos
        table_idx_host[offset: offset + length].fill_(table_idx)
        torch.arange(first_pos, last_pos, out=positions_host[offset: offset + length])
        offset += length
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    offsets = positions_host.to(page_table.device, non_blocking=True)
    page_table[table_idxs, offsets] = allocated
