from __future__ import annotations

import functools
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """获取当前 CUDA 设备的计算能力（缓存结果以避免重复调用）"""
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        return None  # 没有 CUDA 设备时返回 None
    return torch.cuda.get_device_capability()  # 返回 (major, minor)


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """检查当前 CUDA 架构是否支持指定的计算能力版本"""
    arch = _get_torch_cuda_version()
    if arch is None:
        return False
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    """检查是否支持 SM 9.0（Hopper 架构特性）"""
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    """检查是否支持 SM 10.0（Blackwell 架构特性）"""
    return is_arch_supported(10, 0)
