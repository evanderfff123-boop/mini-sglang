from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    """所有大语言模型（LLM）的抽象基类，定义了统一的 forward 接口。"""

    @abstractmethod
    def forward(self) -> torch.Tensor:
        """前向传播，输出 logits 张量。"""
        ...
