from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_store_module(
    element_size: int,   # 每个 token 的字节大小（hidden_dim * element_size）
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """JIT 编译 KV 缓存存储的 CUDA kernel，按 element_size 缓存编译结果"""
    args = make_cpp_args(element_size, *config)
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],            # CUDA kernel 源文件
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],  # 导出 launch 函数
    )


def store_cache(
    k_cache: torch.Tensor,  # K 缓存张量
    v_cache: torch.Tensor,  # V 缓存张量
    indices: torch.Tensor,  # 输出位置索引
    k: torch.Tensor,        # 输入的 K
    v: torch.Tensor,        # 输入的 V
) -> None:
    """将 KV 张量按索引存储到缓存中（使用 JIT 编译的 CUDA kernel）"""
    num_tokens = k_cache.shape[0]  # 缓存的 token 总数
    k_cache = k_cache.view(num_tokens, -1)  # 展平为 2D（总 token 数，每 token 元素数）
    v_cache = v_cache.view(num_tokens, -1)
    element_size = k_cache.shape[1] * k_cache.element_size()  # 每个 token 的字节数
    module = _jit_store_module(element_size)  # 获取（或编译）对应 element_size 的 kernel
    module.launch(k_cache, v_cache, indices, k, v)  # 启动 CUDA kernel
