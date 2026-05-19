from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """带张量并行的线性层的实际实现基类"""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize  # 全局输入维度（未切分）
        self.full_output_size = full_osize  # 全局输出维度（未切分）
        self.local_input_size = local_isize  # 当前 TP rank 上的输入维度
        self.local_output_size = local_osize  # 当前 TP rank 上的输出维度
        self.weight = torch.empty(local_osize, local_isize)  # 当前 rank 的权重分片
        self.bias = torch.empty(local_osize) if has_bias else None  # 当前 rank 的偏置（可选）

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行标准线性变换 y = xW^T + b"""
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    权重在所有 TP rank 上完整复制的线性层（不做切分）。
    每个 GPU 持有完整的权重矩阵。
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,  # 本地等于全局，不切分
            local_osize=output_size,  # 本地等于全局，不切分
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    """按列并行的合并线性层（常用于 fused QKV / GateUp 投影）"""

    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # 确保所有输出维度都能被 TP 大小整除
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]  # 每个子部分的本地输出维度
        output_size = sum(output_sizes)  # 全局总输出维度
        tp_output_size = sum(tp_output_sizes)  # 本地总输出维度
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    """QKV 合并投影的线性层，按头数对 TP 进行切分"""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        # 计算当前 rank 上本地的 Q/KV 头数，KV 头允许复制（GQA 场景下不均匀切分）
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim  # 全局 Q + K + V 总维度
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim  # 本地 Q + K + V 总维度
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    """输出投影层（注意力后的 O 投影），需要 all-reduce 汇总各 TP rank 的结果"""

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)  # 输入按 TP 切分
        local_osize = output_size  # 输出不切分
        self._comm = DistributedCommunicator()  # 分布式通信器
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """线性变换后在 TP rank 之间做 all-reduce 求和"""
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)  # 跨 rank 求和得到完整输出
        return y


class LinearRowParallel(_LinearTPImpl):
    """按行并行的线性层，输入沿最后一维切分，输出需 all-reduce"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)  # 输入沿最后一维切分
        local_output_size = output_size  # 输出不切分
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """线性变换后在 TP rank 之间做 all-reduce 求和"""
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y
