from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from minisgl.core import SamplingParams

from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    """tokenizer 消息的基类（前端 FastAPI -> tokenizer 的消息）"""

    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    """批量 tokenizer 消息"""

    data: List[BaseTokenizerMsg]  # 消息列表


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    """解 tokenize 消息：将 scheduler 产出的 token ID 解码为文本"""

    uid: int  # 用户请求 ID
    next_token: int  # 新生成的 token ID
    finished: bool  # 是否已完成全部生成


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    """tokenize 消息：将用户文本编码为 token ID 序列"""

    uid: int  # 用户请求 ID
    text: str | List[Dict[str, str]]  # 输入文本，或聊天消息列表（含角色信息）
    sampling_params: SamplingParams  # 采样参数（随消息传递到后端）


@dataclass
class AbortMsg(BaseTokenizerMsg):
    """取消消息：通知 tokenizer 取消某个请求"""

    uid: int  # 要取消的请求 ID
