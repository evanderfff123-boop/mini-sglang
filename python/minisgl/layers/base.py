from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# 类型别名：state_dict 是一个从字符串到 Tensor 的字典
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    """拼接前缀和参数名，如果前缀为空则直接返回参数名"""
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    """所有 OP 的基类，提供 state_dict 的递归收集和加载功能"""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """递归收集所有参数到 state_dict 字典中"""
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):  # 跳过私有属性（以下划线开头）
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param  # 将 Tensor 参数加入结果字典
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)  # 递归处理子 OP

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """从 state_dict 中加载参数到当前对象"""
        for name, param in self.__dict__.items():
            if name.startswith("_"):  # 跳过私有属性
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))  # 弹出对应的权重
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype  # 校验形状和数据类型
                setattr(self, name, item)  # 替换当前对象的参数
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    """无状态的 OP，没有可保存的参数"""

    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """无状态 OP 只检查 state_dict 是否为空的额外键"""
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """无状态 OP 返回空字典"""
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    """OP 的列表容器，支持批量递归处理 state_dict"""

    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops  # 子 OP 列表

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """收集列表中所有 OP 的 state_dict"""
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)  # 按索引作为前缀
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        """加载 state_dict 到列表中的所有 OP"""
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
