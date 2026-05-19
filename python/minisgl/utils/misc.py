from __future__ import annotations


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """装饰器：确保函数只在作为主脚本运行时才执行"""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None  # 丢弃函数
        else:
            return lambda f: f  # 返回原函数
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None  # 自动调用函数并丢弃返回值
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """整数的整除运算。若 allow_replicate=True 且 b > a，允许 KV head 复制场景下返回 1。"""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1  # 当 a 不能被 b 整除但 b 能被 a 整除时，表示需要复制 KV head
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """整数除法向上取整"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """将 a 向上对齐到 b 的整数倍"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """将 a 向下对齐到 b 的整数倍"""
    return (a // b) * b


class Unset:
    """表示"未设置"的哨兵类"""
    pass


UNSET = Unset()  # 全局"未设置"哨兵实例
