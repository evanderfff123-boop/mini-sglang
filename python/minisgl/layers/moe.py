import torch
from minisgl.core import get_global_ctx
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    """混合专家（MoE）层：通过路由选择 top-k 专家进行计算"""

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        self.num_experts = num_experts  # 专家总数
        self.top_k = top_k  # 每个 token 选 top-k 个专家
        self.hidden_size = hidden_size  # 隐藏层维度
        self.intermediate_size = intermediate_size  # FFN 中间维度
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize  # 是否对路由权重重新归一化
        self.activation = activation  # 激活函数类型（silu 或 gelu）
        self.apply_router_weight_on_input = apply_router_weight_on_input  # 是否在输入上应用路由权重
        # 每个专家的中间维度按 TP 切分
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        # gate_up_proj: [num_experts, 2 * inter_dim, hidden_size]
        # 包含 gate 和 up 两个投影的合并权重
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        # down_proj: [num_experts, hidden_size, inter_dim]
        # 专家的输出投影权重
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """前向传播：调用 MoE 后端计算专家的前馈网络，多 rank 时做 all-reduce"""
        ctx = get_global_ctx()
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
