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


# 定义 PrefillManager 类，用于管理预填充（Prefill）阶段的请求挂起队列、显存测算与批次打包
@dataclass
class PrefillManager:
    """管理 prefill 阶段的请求队列和 batch 调度。"""

    cache_manager: CacheManager  # 缓存管理器
    table_manager: TableManager  # 行表管理器
    decode_manager: DecodeManager  # decode 管理器（用于获取 inflight token 数）
    pending_list: List[PendingReq] = field(default_factory=list)  # 等待调度的请求列表

    # 定义 add_one_req 方法，用于将网络中新传入的用户推理任务，包装后加入本地调度等待队列中
    def add_one_req(self, req: UserMsg) -> None:
        """将用户请求加入待处理队列。"""
        # 将请求封装成 PendingReq 数据结构，并追加在等待列表末尾
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    # --- Prefill 调度 ---
    # 定义核心调度逻辑方法 schedule_next_batch，返回 Batch 对象或 None
    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        """从待处理队列中构造下一批 prefill batch。"""
        # 如果当前没有在排队的等待请求，则直接返回 None，跳过本轮 Prefill 调度
        if len(self.pending_list) == 0:
            return None

        # 考虑到当前依然在后台处于在线持续解码（In-flight Decode）中的任务开销，计算资源预留偏移
        # estimated offset due to in-flight decode
        # 实例化批次拼装累加器（PrefillAdder），用来评估并精确裁定能装入本批次的请求数量
        adder = PrefillAdder(
            # 传入单次前向推理中允许的最大预填充 Token 增量预算数
            token_budget=prefill_budget,
            # 传入为当前解码中在线请求锁定的 K-V Cache 显存 Token 数，以预留资源，防止爆显存
            reserved_size=self.decode_manager.inflight_tokens,
            # 传入缓存管理器引用，用于判定 K-V 物理显存页面的可用情况
            cache_manager=self.cache_manager,
            # 传入表管理器引用，用以判定物理行表槽位的可用数量
            table_manager=self.table_manager,
        )
        # 初始化列表，保存本批次拼装成功的就绪请求实例
        reqs: List[Req] = []
        # 初始化列表，保存本轮调度中因为单次处理不完而被进行分块（Chunked）拆分的等待请求状态
        chunked_list: List[PendingReq] = []
        # 按排队顺序，遍历当前的挂起等待队列
        for pending_req in self.pending_list:
            # 尝试调用累加器的 try_add_one 接口，评估并尝试把当前挂起请求装进本批次中
            if req := adder.try_add_one(pending_req):
                # 如果成功加入本批次，清除该挂起请求中可能遗留的过时 chunk 标识
                pending_req.chunked_req = None
                # 判断累加器根据预算处理后，返回的是否是一个处于分块状态下的子请求
                if isinstance(req, ChunkedReq):
                    # 如果由于输入过长被自动分块，保存这一段新生成的 ChunkedReq，以便下次调度继续读取
                    pending_req.chunked_req = req
                    # 将这个仍有剩余 Token 待处理的挂起请求对象追加到分块暂存队列中
                    chunked_list.append(pending_req)
                # 将这个经审核符合开销、可用于本次推理前向计算的请求加入批次的请求列表
                reqs.append(req)
            # 如果累加器返回 None，表示已达到最大显存物理限制、表容量或单批次的 Token 预算额度
            else:
                # 终止遍历，本轮批次构建无法再接纳更多请求
                break  # We cannot add more requests
        # 如果遍历完后，发现连一个能够执行的请求都没有被筛选出来
        if len(reqs) == 0:
            # 返回 None
            return None
        # 重新打包调度等待队列：必须优先把本轮计算尚未完全结束的分块子请求排到头部，然后再追加剩下的未曾调度的请求
        self.pending_list = chunked_list + self.pending_list[len(reqs):]
        # 返回装载完毕的 Batch 实例，并将推理阶段标记为 "prefill"
        return Batch(reqs=reqs, phase="prefill")

    # 定义 abort_req 方法，用于按 UID 在挂起等待队列中提前检索并移除指定的取消请求
    def abort_req(self, uid: int) -> Req | None:
        """中止一个待处理的请求，返回其已分配的 chunked_req（如果有）。"""
        # 带索引遍历当前的挂起等待队列
        for i, req in enumerate(self.pending_list):
            # 如果匹配到了要强行中止和取消的请求 UID
            if req.uid == uid:
                # 将该请求从挂起队列中弹出删除
                self.pending_list.pop(i)
                # 返回该请求在等待队列中已经被部分分配的物理 chunked_req 对象（如果有的话，供后续外层安全释放其页表）
                return req.chunked_req
        # 如果未找到任何匹配该 UID 的请求，返回 None
        return None

    # 声明只读属性修饰器
    @property
    # 定义 runnable 属性，向外界报告当前管理器内是否仍有待处理的预填充任务
    def runnable(self) -> bool:
        """是否有待处理的 prefill 请求。"""
        # 如果挂起队列长度大于 0，说明存在待处理任务，返回 True，否则返回 False
        return len(self.pending_list) > 0
