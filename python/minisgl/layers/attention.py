from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import StateLessOP
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    """注意力层：对 QKV 进行分割、可选的 QK 归一化、RoPE 编码，然后调用后端注意力计算"""

    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        assert num_qo_heads % num_kv_heads == 0  # num_qo_heads 必须能被 num_kv_heads 整除（GQA 约束）
        self.layer_id = layer_id  # 当前层的 ID 编号
        self.head_dim = head_dim  # 每个注意力头的维度
        tp_size = get_tp_info().size
        # 按 TP 并行度切分 Q 头数
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        # 按 TP 并行度切分 KV 头数，允许复制（当不能均匀切分时）
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        # 当前 TP rank 上的 Q 和 KV 的总维度
        self.qo_attn_dim = self.num_qo_heads * head_dim
        self.kv_attn_dim = self.num_kv_heads * head_dim
        # 创建 RoPE（旋转位置编码）对象
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        self.q_norm = q_norm  # 可选的 Q 归一化层（如 QK-Norm）
        self.k_norm = k_norm  # 可选的 K 归一化层（如 QK-Norm）

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """将合并的 QKV 分割，应用 RoPE，调用后端做注意力计算"""
        ctx = get_global_ctx()
        # 将拼接的 QKV 沿最后一维拆分为 Q、K、V
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        # 对 Q 和 K 做 QK-Norm（如果配置了）
        if self.q_norm is not None:
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 对 Q 和 K 施加 RoPE 旋转位置编码
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        # 调用后端注意力实现（如 flash-attention）
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        return o.view(-1, self.qo_attn_dim)
