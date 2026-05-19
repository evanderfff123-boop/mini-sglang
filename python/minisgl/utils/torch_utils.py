from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    """上下文管理器：临时更改 PyTorch 默认张量 dtype"""
    import torch  # 实际使用时才导入

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)  # 恢复原始 dtype


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """装饰器：使用 NVTX range 标注函数，用于 NVIDIA 性能分析"""
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))  # 插入层 ID
            with nvtx.range(display_name):  # NVTX range 标注
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator
