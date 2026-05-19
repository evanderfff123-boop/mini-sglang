from __future__ import annotations

from typing import TYPE_CHECKING

from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
from minisgl.models import ModelConfig
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    """门控 MLP（SwiGLU/GeGLU）：gate_proj + up_proj 融合 + down_proj。"""

    def __init__(self, config: ModelConfig):
        # 融合的 gate 和 up 投影（列并行，在张量并行中沿 hidden 维度切分）
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        # 激活函数映射表，根据配置选择 SiLU 或 GELU
        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn  # 激活函数（silu_and_mul 或 gelu_and_mul）
        self.down_proj = LinearRowParallel(  # 下投影（行并行，还原 hidden_size）
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：gate_up 融合投影 -> 逐元素激活与门控相乘 -> down 投影。"""
        gate_up = self.gate_up_proj.forward(x)
        del x  # 及时释放输入张量，降低峰值内存
        y = self.act_fn(gate_up)
        del gate_up  # 及时释放中间结果，降低峰值内存
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    """混合专家（MoE）MLP：Router + 多个 FFN 专家。"""

    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(  # MoE 层：根据 router 权重选择 top-k 专家
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        self.gate = LinearReplicated(  # 路由门控：每个副本持有完整的权重
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向传播：展平 tokens -> 路由门控 -> MoE 专家计算 -> 恢复形状。"""
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)  # 将输入展平为 [总 token 数, hidden_dim]
        router_logits = self.gate.forward(hidden_states)  # 路由门控输出各专家的分数
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)  # 恢复原始形状
        return final_hidden_states


class RopeAttn(BaseOP):
    """带 RoPE 旋转位置编码的多头注意力层。"""

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        head_dim = config.head_dim
        # 融合的 QKV 投影（列并行），支持 GQA（分组查询注意力）
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        self.has_qk_norm = has_qk_norm  # 是否对 Q 和 K 进行 LayerNorm
        if has_qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)  # Q 的 LayerNorm
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)  # K 的 LayerNorm
        else:
            self.q_norm = None
            self.k_norm = None
        self.attn = AttentionLayer(  # 核心注意力计算层（含 RoPE 和因果掩码）
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        self.o_proj = LinearOProj(  # 输出投影（行并行），将多头结果合并回 hidden_size
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：QKV 投影 -> 注意力计算 -> 输出投影。"""
        qkv = self.qkv_proj.forward(x)
        del x  # 及时释放输入张量，降低峰值内存
        o = self.attn.forward(qkv)
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
