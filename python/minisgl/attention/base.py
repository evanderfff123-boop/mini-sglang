from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


@dataclass
class BaseAttnMetadata(ABC):
    """注意力后端的元数据基类，存储 cu_seqlens、page_table 等信息"""

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...  # 获取每个序列最后一个 token 的索引


class BaseAttnBackend(ABC):
    """注意力后端的抽象基类，定义了所有注意力计算后端必须实现的接口"""

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...  # 执行注意力计算前向传播

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...  # 为当前 batch 准备注意力元数据

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...  # 初始化 CUDA Graph 捕获

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...  # 准备捕获 CUDA Graph 时的元数据

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...  # 准备重放 CUDA Graph 时的元数据


class HybridBackend(BaseAttnBackend):
    """混合注意力后端：prefill 和 decode 阶段使用不同的后端实现"""

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,  # prefill 阶段使用的后端
        decode_backend: BaseAttnBackend,    # decode 阶段使用的后端
    ) -> None:
        self.prefill_backend = prefill_backend
        self.decode_backend = decode_backend

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 根据 batch 的 phase 选择对应的后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)

    def prepare_metadata(self, batch: Batch) -> None:
        # 根据 batch 的 phase 选择对应的后端准备元数据
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 只有 decode 后端需要 CUDA Graph
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        self.decode_backend.prepare_for_replay(batch)
