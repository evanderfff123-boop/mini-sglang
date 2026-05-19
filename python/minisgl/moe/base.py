from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    """MoE 后端的抽象基类，定义混合专家层的前向接口"""

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """
        MoE 层前向传播。

        Args:
            hidden_states: 输入隐藏状态 [M, K]
            w1: 第一个专家权重 [E, N, K]
            w2: 第二个专家权重 [E, K, N]（或类似形状）
            gating_output: 门控网络输出 [M, E]
            topk: 每个 token 选择的专家数
            renormalize: 是否对 topk 权重重新归一化
            activation: 激活函数类型（如 "silu", "gelu"）
            apply_router_weight_on_input: 是否在输入上应用路由权重

        Returns:
            输出隐藏状态 [M, K]
        """
        ...
