from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    """后端消息的基类（tokenizer -> scheduler 的消息）"""

    def encoder(self) -> Dict:
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """批量后端消息"""

    data: List[BaseBackendMsg]  # 消息列表


@dataclass
class ExitMsg(BaseBackendMsg):
    """退出消息：通知 scheduler 进程结束"""
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    """用户请求消息：包含已 tokenize 的 input_ids 和采样参数"""

    uid: int  # 用户请求 ID
    input_ids: torch.Tensor  # CPU 上的 1D int32 张量（已编码的 token ID 序列）
    sampling_params: SamplingParams  # 采样参数


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    """后端取消消息：通知 scheduler 取消某个请求"""

    uid: int  # 要取消的请求 ID
