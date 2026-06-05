# 引入延迟类型注解功能，允许在类定义内部将当前类自身作为类型声明使用（支持向前引用）
from __future__ import annotations

# 从标准库导入 dataclass 装饰器，用于自动生成类的基础方法（如构造函数、表示方法等）
from dataclasses import dataclass
# 从 typing 模块导入 Dict 和 List，以便在代码中进行静态类型注解
from typing import Dict, List

# 从当前目录下的 utils 模块导入通用的类型反序列化与序列化工具函数
from .utils import deserialize_type, serialize_type


@dataclass
# 定义前端消息基类 BaseFrontendMsg，专门用于调度器向前端 FastAPI 服务发送交互数据
class BaseFrontendMsg:
    """前端消息的基类（scheduler -> 前端 FastAPI 的消息）"""

    # 声明一个静态方法，使该函数不需要实例化对象即可直接调用
    @staticmethod
    # 定义编码方法，接收一个 BaseFrontendMsg 实例并将其转换为字典形式
    def encoder(msg: BaseFrontendMsg) -> Dict:
        # 调用自定义的序列化函数，将复杂的数据类对象转换为标准的 Python 字典
        return serialize_type(msg)

    # 声明一个静态方法，使该函数不需要实例化对象即可直接调用
    @staticmethod
    # 定义解码方法，接收字典类型的 JSON 数据并还原为 BaseFrontendMsg 实例
    def decoder(json: Dict) -> BaseFrontendMsg:
        # 结合当前的全局命名空间 globals() 以及输入字典，动态反序列化回具体的消息数据类对象
        return deserialize_type(globals(), json)


@dataclass
# 定义批量消息打包类 BatchFrontendMsg，继承自 BaseFrontendMsg
class BatchFrontendMsg(BaseFrontendMsg):
    """批量前端消息：将多条消息打包在一起传输"""

    # 声明一个名为 data 的字段，其类型为 BaseFrontendMsg 对象的列表，用于批量化网络传输
    data: List[BaseFrontendMsg]  # 消息列表


@dataclass
# 定义用户单次推理回复消息类 UserReply，继承自 BaseFrontendMsg
class UserReply(BaseFrontendMsg):
    """用户回复消息：包含增量输出和完成状态"""

    uid: int  # 用户请求 ID
    incremental_output: str  # 本轮增量生成的文本
    finished: bool  # 是否已完成全部生成
