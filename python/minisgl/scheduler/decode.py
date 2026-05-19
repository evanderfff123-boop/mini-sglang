from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


@dataclass
class DecodeManager:
    """管理 Decode 阶段的请求（逐个 token 自回归生成）。"""

    page_size: int  # 每页包含的 token 数
    running_reqs: Set[Req] = field(default_factory=set)  # 当前正在 decode 的请求集合

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """筛选出所有可继续 decode 的请求，合并新传入的请求。"""
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        """从运行集合中移除指定请求。"""
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        """根据 uid 中止请求，返回被移除的 Req 或 None。"""
        for req in self.running_reqs:
            if req.uid == uid:
                self.running_reqs.remove(req)
                return req
        return None

    @property
    def inflight_tokens(self) -> int:
        """估算正在 decode 的请求预计还要消耗的 token 总数（用于 prefill 阶段的资源预留）。"""
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 每个请求预留 1 页的余量
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        """组织下一批 decode batch（按 uid 排序）。"""
        if not self.runnable:
            return None
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """是否有请求正在 decode。"""
        return len(self.running_reqs) > 0
