from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from minisgl.core import SamplingParams

    from .prefill import ChunkedReq


@dataclass
class PendingReq:
    """等待调度的请求：包含输入 token、采样参数和分块 prefill 信息"""

    uid: int  # 请求 ID
    input_ids: torch.Tensor  # 输入 token ID 张量
    sampling_params: SamplingParams  # 采样参数
    chunked_req: ChunkedReq | None = None  # 分块 prefill 请求（正在处理中时为非空）

    @property
    def input_len(self) -> int:
        """输入序列长度"""
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        """最大输出长度"""
        return self.sampling_params.max_tokens


@dataclass
class ScheduleResult:
    """调度结果：包含本轮需要处理的请求和对应的输出位置索引"""

    reqs: List[PendingReq]  # 本轮调度选中的请求列表
    output_indices: List[torch.Tensor]  # 每个请求的输出位置索引列表
