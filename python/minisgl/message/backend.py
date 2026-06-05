from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 导入 PyTorch 框架，用于处理输入序列的张量（Tensor）表示
import torch
# 从当前包的 core 模块中引入采样参数类 SamplingParams，控制生成策略
from minisgl.core import SamplingParams

# 从本地 utils 模块中引入通用的类型转换和序列化工具函数
from .utils import deserialize_type, serialize_type

# 使用数据类装饰器修饰此类
@dataclass
# 定义后端消息的基类 BaseBackendMsg，主要在分词器（Tokenizer）向调度器（Scheduler）传递消息时使用
class BaseBackendMsg:
    """后端消息的基类（tokenizer -> scheduler 的消息）"""

    # 定义实例的编码方法，将消息实例自身打包成字典结构
    def encoder(self) -> Dict:
        # 调用自定义的序列化逻辑，把自身属性转换为 Python 原生字典类型
        return serialize_type(self)

    # 声明一个静态方法，使外部可以直接通过类名直接调用解码功能
    @staticmethod
    # 定义静态解码方法，接收包含原始字段的字典并反序列化回消息对象
    def decoder(json: Dict) -> BaseBackendMsg:
        # 传入当前的全局变量命名空间 globals()，帮助解析并恢复具体的子消息类型
        return deserialize_type(globals(), json)


@dataclass
# 定义批量消息打包类 BatchBackendMsg，继承自 BaseBackendMsg
class BatchBackendMsg(BaseBackendMsg):
    """批量后端消息"""

    # 声明消息列表字段 data，用于在一个数据包中打包多个后端消息
    data: List[BaseBackendMsg]  # 消息列表


@dataclass
# 定义退出消息类 ExitMsg，继承自 BaseBackendMsg
class ExitMsg(BaseBackendMsg):
    """退出消息：通知 scheduler 进程结束"""
    # 占位符，该消息作为通知信号使用，不需要携带额外的属性成员
    pass


@dataclass
# 定义用户请求类 UserMsg，继承自 BaseBackendMsg
class UserMsg(BaseBackendMsg):
    """用户请求消息：包含已 tokenize 的 input_ids 和采样参数"""

    uid: int  # 用户请求 ID
    input_ids: torch.Tensor  # CPU 上的 1D int32 张量（已编码的 token ID 序列）
    sampling_params: SamplingParams  # 采样参数


@dataclass
# 定义取消请求消息类 AbortBackendMsg，继承自 BaseBackendMsg
class AbortBackendMsg(BaseBackendMsg):
    """后端取消消息：通知 scheduler 取消某个请求"""

    uid: int  # 要取消的请求 ID
