from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import MoEMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    """Qwen3 MoE 解码器层：LayerNorm -> 自注意力（含 QK Norm） -> LayerNorm -> MoE MLP。"""

    def __init__(self, config: ModelConfig, layer_id: int):
        # Qwen3 MoE 注意力层：启用 QK LayerNorm，无注意力偏置
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.mlp = Qwen3MLP(config)  # MoE（混合专家）MLP 层
        self.input_layernorm = RMSNormFused(  # 注意力前的 RMSNorm（融合 residual）
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(  # MLP 前的 RMSNorm（融合 residual）
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id  # 缓存 layer_id 供 NVTX 注释使用

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：Pre-Norm -> 自注意力 -> Pre-Norm -> MoE MLP，返回 (输出, 残差)。"""
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    """Qwen3 MoE Transformer 主干网络：词嵌入 -> 多层解码器 -> 最终 RMSNorm。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入（支持张量并行切分）
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(  # 所有 Transformer 解码器层的列表
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(  # 最终输出归一化层（融合 residual）
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向传播：词嵌入 -> 逐层解码 -> 归一化，返回 hidden states。"""
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3MoeForCausalLM(BaseLLMModel):
    """Qwen3 MoE 因果语言模型：主干网络 + LM Head（输出 logits）。"""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)  # Transformer 主干网络
        self.lm_head = ParallelLMHead(  # 语言模型头，将 hidden states 映射到词表 logits
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        """前向传播：从全局上下文中获取 input_ids，输出 logits。"""
        output = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3MoeForCausalLM"]
