from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from minisgl.core import SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    """所有请求已完成，用于退出 run_forever 循环的信号"""
    pass


@dataclass
class RequestStatus:
    """单个请求的处理状态"""

    uid: int  # 请求 ID
    input_ids: List[int]  # 输入 token ID 列表
    output_ids: List[int]  # 已生成的输出 token ID 列表


class LLM(Scheduler):
    """高级 LLM 接口封装，继承自 Scheduler，提供 generate 方法直接调用"""

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),  # 默认单卡运行
            dtype=dtype,
            offline_mode=True,  # 离线模式：不依赖 ZMQ 通信
            **kwargs,
        )
        super().__init__(config)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []  # 待处理请求队列
        self.status_map: Dict[int, RequestStatus] = {}  # uid -> 请求状态
        self.counter = 0  # 请求 ID 递增计数器

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """将单个 prompt 编码为 int32 张量"""
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")  # 已是 token ID 列表，直接转张量

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式接收消息：从 pending_requests 中取出请求并构造 UserMsg"""
        if blocking and len(self.pending_requests) == 0:
            raise RequestAllFinished()  # 所有请求处理完毕，通知调度器退出
        results: List[BaseBackendMsg] = []
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                break  # 不超过 prefill 预算
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            uid, added = self.counter + added, added + 1
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        self.counter += added
        self.pending_requests = self.pending_requests[added:]  # 移除已分配的请求
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式接收结果：将 detokenize 结果记录到 status_map"""
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                status.output_ids.append(msg.next_token)  # 记录输出的 token ID

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        """
        同步生成接口：对 prompt 列表执行推理，返回每个请求的文本和 token ID。

        Args:
            prompts: 输入 prompt 列表，可以是字符串或已编码的 token ID 列表
            sampling_params: 采样参数，可以是单个（共享）或列表（逐个指定）

        Returns:
            每个请求的结果字典，包含 "text"（文本）和 "token_ids"（输出 token ID）
        """
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            sampling_params = [sampling_params] * len(prompts)  # 共享参数时复制为等长列表
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            self.run_forever()  # 进入 scheduler 主循环
        except RequestAllFinished:
            pass  # 所有请求完成时正常退出
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            status = self.status_map[i]
            output_text = self.tokenizer.decode(status.output_ids)  # 将输出 token ID 解码为文本
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
