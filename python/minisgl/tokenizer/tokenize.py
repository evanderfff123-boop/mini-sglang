from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    """文本编码管理器：将用户输入的文本转换为 token ID 张量"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer  # HuggingFace tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """批量将 TokenizeMsg 中的文本编码为 1D int32 张量"""
        results: List[torch.Tensor] = []
        # TODO: 支持批量 tokenization 优化
        for msg in msgs:
            if isinstance(msg.text, list):
                # 聊天消息列表：使用 chat_template 格式化为一个字符串
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,  # 先不编码，只格式化
                    add_generation_prompt=True,  # 添加生成提示（如 "<|assistant|>"）
                )
                assert isinstance(prompt, str)
            else:
                prompt = msg.text  # 普通字符串直接使用
            input_ids: torch.Tensor = (  # type: ignore
                self.tokenizer.encode(prompt, return_tensors="pt")  # 编码为 PyTorch 张量
            )
            results.append(input_ids.view(-1).to(torch.int32))  # 展平并转为 int32
        return results
