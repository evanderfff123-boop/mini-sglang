from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from minisgl.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        """PyNCCL 通信器接口定义（仅类型检查时使用）"""
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...  # 全局归约
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...  # 全局收集
        @abstractmethod
        def get_buffer(self) -> int: ...  # 获取内部缓冲区的指针地址

else:
    PyNCCLCommunicator = Any  # 运行时使用 Any 类型绕过类型检查


@functools.cache
def _load_nccl_module() -> Module:
    """加载 PyNCCL 的 CUDA 模块，链接 NCCL 库"""
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache
def _get_pynccl_wrapper_cls():
    """获取 PyNCCL 包装器类（tvm_ffi 注册对象）"""
    import tvm_ffi

    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        """PyNCCL 的 tvm_ffi 实现，通过 __ffi_init__ 调用 C++ 构造函数"""
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,                              # 当前设备的 tensor parallel 排名
    tp_size: int,                              # tensor parallel 总大小
    tp_cpu_group: torch.distributed.ProcessGroup,  # CPU 端的进程组，用于广播 NCCL ID
    max_size_bytes: int = 0,                   # 最大缓冲区大小（字节）
) -> PyNCCLCommunicator:
    """初始化 PyNCCL 通信器，在 TP 组内建立 NCCL 通信"""
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)  # 受环境变量限制

    module = _load_nccl_module()  # 加载 NCCL 模块
    cls = _get_pynccl_wrapper_cls()  # 获取包装器类

    # rank 0 创建 NCCL 唯一 ID，并通过分布式广播给所有 rank
    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]  # 创建 NCCL UID
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,  # 通过 CPU 进程组广播
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # 绕过 FFI 对象的类型检查
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
