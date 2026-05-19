from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    """张量并行（TP）的拓扑信息：当前 rank 和总 world_size"""

    rank: int  # 当前进程的 rank 编号
    size: int  # 总进程数（world size）

    def __post_init__(self):
        assert 0 <= self.rank < self.size  # rank 必须在 [0, size) 范围内

    def is_primary(self) -> bool:
        """判断当前 rank 是否为主节点（rank 0）"""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None  # 全局唯一的 TP 信息，初始化后不可更改


def set_tp_info(rank: int, size: int) -> None:
    """设置全局 TP 信息（只能在启动时调用一次）"""
    global _TP_INFO
    if _TP_INFO is not None:
        raise RuntimeError("TP info has been set")  # 防止重复初始化
    _TP_INFO = DistributedInfo(rank, size)


def get_tp_info() -> DistributedInfo:
    """获取全局 TP 信息，未设置时抛出异常"""
    if _TP_INFO is None:
        raise RuntimeError("TP info has not been set")
    return _TP_INFO


def try_get_tp_info() -> DistributedInfo | None:
    """尝试获取全局 TP 信息，未设置时返回 None（不抛异常）"""
    return _TP_INFO


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
