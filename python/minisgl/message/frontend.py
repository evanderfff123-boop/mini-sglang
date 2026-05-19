from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    """前端消息的基类（scheduler -> 前端 FastAPI 的消息）"""

    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    """批量前端消息：将多条消息打包在一起传输"""

    data: List[BaseFrontendMsg]  # 消息列表


@dataclass
class UserReply(BaseFrontendMsg):
    """用户回复消息：包含增量输出和完成状态"""

    uid: int  # 用户请求 ID
    incremental_output: str  # 本轮增量生成的文本
    finished: bool  # 是否已完成全部生成
