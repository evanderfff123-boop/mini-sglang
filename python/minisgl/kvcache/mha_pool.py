from __future__ import annotations

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """多头注意力 KV 缓存池，实际存储 KV 缓存的张量"""

    def __init__(
        self,
        num_kv_heads: int,       # KV 头的数量
        num_layers: int,         # 模型层数
        head_dim: int,           # 每个头的维度
        num_pages: int,          # 页面数量（分页管理）
        page_size: int,          # 每页包含的 token 数
        dtype: torch.dtype,      # 张量数据类型
        device: torch.device,    # 存储设备
    ) -> None:
        tp_info = get_tp_info()  # 获取 tensor parallel 信息
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)  # 本设备上的 KV 头数
        self._kv_buffer = torch.empty(  # 主 KV 缓存张量，形状：(2, num_layers, num_pages, page_size, local_kv_heads, head_dim)
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers  # 缓存层数
        self._k_buffer = self._kv_buffer[0]  # K 缓存视图
        self._v_buffer = self._kv_buffer[1]  # V 缓存视图
        self._device = device  # 设备信息
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)  # 展平后的存储形状

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]  # 返回指定层的 K 缓存

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]  # 返回指定层的 V 缓存

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """将 KV 张量存储到指定层的缓存中"""
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),  # 展平后的 K 缓存
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),  # 展平后的 V 缓存
            indices=out_loc,  # 输出位置索引
            k=k,  # 输入的 K
            v=v,  # 输入的 V
        )

    @property
    def device(self) -> torch.device:
        return self._device  # 缓存所在的设备

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype  # 缓存的数据类型

    @property
    def num_layers(self) -> int:
        return self._num_layers  # 缓存的层数
