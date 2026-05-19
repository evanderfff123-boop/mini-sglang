from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    """旋转位置编码（RoPE）的配置参数。"""

    head_dim: int  # 注意力头的维度
    rotary_dim: int  # RoPE 应用的维度数
    max_position: int  # 最大位置编码长度
    base: float  # RoPE 的 base 值（theta）
    scaling: Dict[str, Any] | None  # RoPE 缩放策略（如 NTK-aware 缩放）


@dataclass(frozen=True)
class ModelConfig:
    """模型整体配置，从 HuggingFace PretrainedConfig 中提取。"""

    num_layers: int  # Transformer 层数
    num_qo_heads: int  # Query 注意力头数
    num_kv_heads: int  # Key/Value 注意力头数（GQA 场景下可能小于 num_qo_heads）
    head_dim: int  # 每个注意力头的维度
    hidden_size: int  # 隐藏层维度
    vocab_size: int  # 词表大小
    intermediate_size: int  # FFN 中间层维度
    rms_norm_eps: float  # RMSNorm 的 epsilon，防止除零
    rotary_config: RotaryConfig  # RoPE 旋转位置编码配置
    hidden_act: str  # FFN 激活函数类型（如 silu, gelu）
    tie_word_embeddings: bool  # 是否共享输入输出词嵌入权重
    num_experts: int  # MoE 专家总数（非 MoE 模型为 0）
    num_experts_per_tok: int  # 每个 token 激活的专家数（top-k）
    moe_intermediate_size: int  # MoE 专家 FFN 的中间维度
    norm_topk_prob: bool  # 是否对 top-k 概率做归一化
    model_type: str  # 模型类型标识（如 llama, qwen2）
    architectures: list[str]  # HuggingFace 架构名称列表

    @property
    def is_moe(self) -> bool:
        """是否为 MoE（混合专家）模型。"""
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        """从 HuggingFace 的 PretrainedConfig 构建 ModelConfig。"""
        # 处理嵌套配置：如果顶层有 text_config，则提取其中的模型配置
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            # 从顶层配置继承特定字段（若子配置中缺失）
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        # 提取关键配置参数，对缺失字段使用默认值
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta 是直接属性；Mistral：它位于 rope_scaling 字典内
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )
