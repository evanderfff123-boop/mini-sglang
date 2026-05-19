from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    """词表并行嵌入层：词表按 TP 切分到各 rank，前向时做 all-reduce"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size  # TP 并行度
        self.num_embeddings = num_embeddings  # 全局词表大小
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)  # 当前 rank 分到的词表大小
        # 计算当前 rank 在词表中的范围 [start, end)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)  # (起始索引, 本地词数量)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)  # 当前 rank 的嵌入权重分片
        self._comm = DistributedCommunicator()  # 分布式通信器

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：从权重中按索引查找嵌入向量，多 rank 时做 all-reduce"""
        from minisgl.kernel import indexing

        # 使用自定义索引 kernel 做嵌入查找，单 rank 时不需要 vocab_range 校验
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        # 多 rank 时对结果做 all-reduce 求和
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    """并行的语言模型预测头（LM Head），支持权重绑定（tie embeddings）"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None  # 偏置参数（可选）
        self.tied_embedding = tied_embedding  # 绑定的嵌入层（权重共享）
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """加载状态字典：权重绑定时跳过 lm_head 的 weight 和 bias"""
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # 绑定了嵌入层时，从 state_dict 中弹出 lm_head 的权重和偏置（不需要单独加载）
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        """获取状态字典：权重绑定时返回空（参数由嵌入层管理）"""
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：对隐藏状态做线性变换得到 logits，prefill 时只取最后一个 token 的位置"""
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            # prefill 阶段只需最后一个 token 的 logits（用于预测下一个 token）
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        # 如果绑定了嵌入层，使用嵌入层的权重来做投影
        module = self.tied_embedding or self
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        # 多 rank 时做 all-gather 收集所有 rank 的 logits 分片
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # 对 batch 大于 1 的情况，重排 all-gather 的结果
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]
