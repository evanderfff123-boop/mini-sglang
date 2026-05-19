from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

import torch
from minisgl.core import Batch, Req
from minisgl.utils import init_logger

from .utils import PendingReq

if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

logger = init_logger(__name__)


class ChunkedReq(Req):
    """分块请求：当 prefill 输入太长时，分成多个 chunk 处理，不参与 decode。"""

    def append_host(self, next_token: torch.Tensor) -> None:
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager


@dataclass
class PrefillAdder:
    """负责将待处理的请求（PendingReq）逐个添加到 prefill batch 中，并管理资源分配。"""

    token_budget: int  # 当前 batch 剩余的 token 预算
    reserved_size: int  # 为正在 decode 的请求预留的 token 数
    cache_manager: CacheManager  # 缓存管理器
    table_manager: TableManager  # 行表管理器

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        """尝试为一个请求分配缓存句柄和行表槽位。"""
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        handle = self.cache_manager.match_req(req).cuda_handle  # 前缀匹配，得到缓存句柄
        cached_len = handle.cached_len  # 缓存中已匹配的长度
        # TODO: better estimate policy
        extend_len = req.input_len - cached_len  # 需要 prefill 的 token 数
        estimated_len = extend_len + req.output_len  # 预估总需求

        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None  # 显存不够
        self.cache_manager.lock(handle)  # 锁定句柄
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return self.cache_manager.unlock(handle)  # 双重检查，避免竞争

        table_idx = self.table_manager.allocate()  # 分配行表槽位
        if cached_len > 0:  # NOTE: set the cached part
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            page_entry.copy_(handle.get_matched_indices())

        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        """将请求的一部分（可能分块）构造为 Req 对象。"""
        remain_len = pending_req.input_len - cached_len  # 剩余需要 prefill 的长度
        chunk_size = min(self.token_budget, remain_len)  # 本轮可处理的 chunk 大小
        is_chunked = chunk_size < remain_len  # 是否被分块
        CLS = ChunkedReq if is_chunked else Req  # 分块则使用 ChunkedReq
        self.token_budget -= chunk_size  # 扣减预算
        self.reserved_size += remain_len + pending_req.output_len  # 预留剩余所需资源
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        _slice = slice(cached_len, cached_len + chunk_size)
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        """尝试将一个待处理请求加入本轮 prefill batch。"""
        if self.token_budget <= 0:
            return None  # 预算已用完

        if chunked_req := pending_req.chunked_req:  # 该请求已有上一轮缓存的 chunk 信息
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):  # 首次加入，分配资源
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        return None


@dataclass
class PrefillManager:
    """管理 prefill 阶段的请求队列和 batch 调度。"""

    cache_manager: CacheManager  # 缓存管理器
    table_manager: TableManager  # 行表管理器
    decode_manager: DecodeManager  # decode 管理器（用于获取 inflight token 数）
    pending_list: List[PendingReq] = field(default_factory=list)  # 等待调度的请求列表

    def add_one_req(self, req: UserMsg) -> None:
        """将用户请求加入待处理队列。"""
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    # --- Prefill 调度 ---
    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """从待处理队列中构造下一批 prefill batch。"""
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,  # 为正在 decode 的请求预留资源
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            if req := adder.try_add_one(pending_req):  # 尝试加入请求
                pending_req.chunked_req = None  # 清除旧 chunk 标记
                if isinstance(req, ChunkedReq):
                    pending_req.chunked_req = req  # 保存新 chunk 供后续使用
                    chunked_list.append(pending_req)
                reqs.append(req)
            else:
                break  # We cannot add more requests
        if len(reqs) == 0:
            return None
        # 更新待处理队列：先放 chunked 请求（优先处理），再放剩余的未处理请求
        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        """中止一个待处理的请求，返回其已分配的 chunked_req（如果有）。"""
        for i, req in enumerate(self.pending_list):
            if req.uid == uid:
                self.pending_list.pop(i)
                return req.chunked_req
        return None

    @property
    def runnable(self) -> bool:
        """是否有待处理的 prefill 请求。"""
        return len(self.pending_list) > 0
