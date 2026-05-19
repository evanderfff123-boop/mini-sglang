from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_radix_module() -> Module:
    """加载预编译的基数树比较模块（C++ 实现，无需 CUDA）"""
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """快速比较两个 1 维 int CPU 张量，返回公共前缀长度（用于基数树节点匹配）"""
    return _load_radix_module().fast_compare_key(x, y)
