from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    """递归序列化任意值（支持 dict/list/tuple/基本类型/自定义 dataclass）"""
    if isinstance(value, dict):
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        return value  # 基本类型直接返回
    else:
        return serialize_type(value)  # 自定义对象递归序列化


def serialize_type(self) -> Dict:
    """将任意对象序列化为字典（记录 __type__ 字段用于反序列化还原类型）"""
    # 找出所有成员变量
    serialized = {}

    if isinstance(self, torch.Tensor):
        assert self.dim() == 1, "we can only serialize 1D tensor for now"  # 目前只支持 1D 张量
        serialized["__type__"] = "Tensor"
        serialized["buffer"] = self.numpy().tobytes()  # 张量数据转为 bytes
        serialized["dtype"] = str(self.dtype)  # 记录 dtype 信息
        return serialized

    # 普通 dataclass 对象
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """递归反序列化任意值"""
    if isinstance(data, dict):
        if "__type__" in data:
            return deserialize_type(cls_map, data)  # 含类型标记的递归还原
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        return data
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """根据 __type__ 字段将字典反序列化为对应类型的对象"""
    type_name = data["__type__"]
    # 目前只支持 1D 张量的反序列化
    if type_name == "Tensor":
        buffer = data["buffer"]
        dtype_str = data["dtype"].replace("torch.", "")  # 去掉 "torch." 前缀
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)  # 从 bytes 恢复 numpy 张量
        return torch.from_numpy(np_tensor.copy())  # 转为 PyTorch 张量

    cls = cls_map[type_name]  # 从类型映射中查找对应的类
    kwargs = {}
    for k, v in data.items():
        if k == "__type__":
            continue  # 跳过类型标记
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)  # 使用构造参数重建对象
