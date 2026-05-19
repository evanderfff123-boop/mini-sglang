from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import load_aot

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


@functools.cache
def _load_test_tensor_module() -> Module:
    """加载测试用的张量操作模块（C++ 实现）"""
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """测试用的张量比较函数，比较两个张量并返回结果"""
    return _load_test_tensor_module().test(x, y)
