from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Set

from minisgl.core import Batch, Req


# 使用 Python 标准库的数据类修饰器装饰此类，简化其属性声明与字段映射
@dataclass
# 定义 DecodeManager 类，用来维护自回归（Autoregressive）逐词解码生成阶段的活跃请求集合与状态
class DecodeManager:
    """管理 Decode 阶段的请求（逐个 token 自回归生成）。"""

    # 声明 page_size 字段，代表每个物理 K-V 显存页分配可容纳的最大 Token 数量
    page_size: int
    # 声明 running_reqs 字段并使用集合工厂默认初始化，维护目前正处理自回归推理的活跃请求集合
    running_reqs: Set[Req] = field(default_factory=set)

    # 定义 filter_reqs 方法，用以将上一轮新 Prefill 完毕的请求与正在进行的解码请求合并，并筛选出其中有效的运行项
    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        """筛选出所有可继续 decode 的请求，合并新传入的请求。"""
        # 将当前的运行集合 union（合并）新传入的可迭代请求，并仅保留依然满足 .can_decode 条件的活跃请求，将筛选结果重写回运行集合
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    # 定义 remove_req 方法，用以主动将某一已经彻底推理结束的请求从解码运行队列中擦除
    def remove_req(self, req: Req) -> None:
        """从运行集合中移除指定请求。"""
        # 使用丢弃（discard）接口，将指定请求对象自活跃集合中清除（即使对象不存在也不会抛出 KeyError 异常）
        self.running_reqs.discard(req)

    # 定义 abort_req 方法，用于按请求 ID 中断或取消当前正在处于解码阶段的前向任务
    def abort_req(self, uid: int) -> Req | None:
        """根据 uid 中止请求，返回被移除的 Req 或 None。"""
        # 遍历当前正在运行的解码任务集合
        for req in self.running_reqs:
            # 如果匹配到了与指定取消 UID 相同的请求
            if req.uid == uid:
                # 将此请求自当前的运行集合中移除
                self.running_reqs.remove(req)
                # 返回这个被强行中止的请求实例，供外层对其分配的物理 K-V Cache 页表做安全退还
                return req
        # 如果集合内不曾存在该任务，返回 None
        return None

    @property
    # 定义 inflight_tokens 属性方法，估算当前正在运行的所有解码请求，在最终结束生成前可能需要的最大潜在 Token 数量（用于新 Prefill 评估预留）
    def inflight_tokens(self) -> int:
        """估算正在 decode 的请求预计还要消耗的 token 总数（用于 prefill 阶段的资源预留）。"""
                # 为了防止非对齐分配造成溢出，悲观地为每一个处于自回归中的请求，在理论剩余长度基础上，多加预留“约一整页”的空间作为安全缓冲边界
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)
        # 汇总所有解码中请求的理论待生成长度，再加上计算出的预留缓冲空间，反馈最终的安全预测 Token 上限
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    # 定义 schedule_next_batch 方法，用于将处于解码状态中的活跃请求，按照顺序打包打包成下一批次的 Decode 批次
    # 检查是否还有任何可以继续执行计算的请求，若集合为空则返回 None
    def schedule_next_batch(self) -> Batch | None:
        """组织下一批 decode batch（按 uid 排序）。"""
        if not self.runnable:
            return None
        # 按照唯一请求 UID 将正在运行的请求从小到大强制排序（确保张量并行等分布式环境中，各进程调度状态始终保持镜像一致），最后生成 "decode" 阶段的 Batch 对象并返回
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        """是否有请求正在 decode。"""
        # 如果活跃解码请求集合的实际长度大于 0，则返回 True，否则返回 False
        return len(self.running_reqs) > 0
