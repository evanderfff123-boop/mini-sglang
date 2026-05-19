from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    """RMS 层归一化：y = x * rms(x)^(-1) * weight"""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm

        self.eps = eps  # 防止除零的小常数
        self.weight = torch.empty(size)  # 可学习的缩放参数
        self.rmsnorm = rmsnorm  # flashinfer 的 RMSNorm 实现

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入 x 做 RMS 归一化，返回归一化后的张量"""
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        """对输入 x 做 RMS 归一化，结果原地写回 x（节省显存）"""
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    """带残差连接的融合 RMS 层归一化"""

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm  # 普通 rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm  # 融合的残差加法 + rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：如果提供了残差则做融合归一化，否则单独做 rmsnorm
        返回 (归一化结果, 残差)
        """
        if residual is None:
            return self.rmsnorm(x, self.weight, self.eps), x
        # 融合操作：x += residual, 然后对 x 做 rmsnorm
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
