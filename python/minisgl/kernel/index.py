from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Tuple

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,   # 每行的字节大小
    *,
    num_splits: int = 1,  # 将输出切分为几个部分并行处理
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """JIT 编译索引 kernel，按 element_size 和 num_splits 缓存"""
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],            # CUDA kernel 源文件
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],  # 导出 launch 函数
    )


def indexing(
    weights: torch.Tensor,  # 权重张量（如 embedding / lm_head），形状 (vocab_size, hidden_dim)
    indices: torch.Tensor,  # 要查询的索引，形状 (batch_size,)
    *,
    output: torch.Tensor | None = None,  # 预分配的输出张量，若为 None 则自动创建
    vocab_range: Tuple[int, int] | None = None,  # 词汇范围 (start, length)，用于局部张量并行
) -> torch.Tensor:
    """执行索引查找操作（如 embedding lookup 或 lm_head 的 gather），使用 JIT CUDA kernel 加速"""
    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])  # 自动创建输出张量

    element_size = weights.shape[1] * weights.element_size()  # 每行的总字节数
    # 根据行大小选择 num_splits 以优化并行度
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)  # 获取编译好的 kernel
    module.launch(weights, indices, output, vocab_range)  # 启动 CUDA kernel
    return output
