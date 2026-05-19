from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from minisgl.distributed import DistributedInfo
from minisgl.utils import cached_load_hf_config

if TYPE_CHECKING:
    from minisgl.models import ModelConfig


@dataclass(frozen=True)
class EngineConfig:
    """推理引擎的配置类，包含模型路径、分布式信息、运行时参数等所有配置项。"""
    model_path: str  # 模型文件的路径
    tp_info: DistributedInfo  # 张量并行的rank和world size信息
    dtype: torch.dtype  # 模型权重和计算的数据类型
    max_running_req: int = 256  # 最大同时处理的请求数
    attention_backend: str = "auto"  # attention后端（auto/trtllm/flashinfer等）
    moe_backend: str = "auto"  # MoE后端（auto/fused等）
    cuda_graph_bs: List[int] | None = None  # CUDA图捕获的batch size列表
    cuda_graph_max_bs: int | None = None  # CUDA图的最大batch size
    page_size: int = 1  # KV cache的页大小（token数）
    memory_ratio: float = 0.9  # 可用于KV cache的显存比例
    distributed_timeout: float = 60.0  # 分布式初始化的超时时间（秒）
    use_dummy_weight: bool = False  # 是否使用随机dummy权重（用于调试）
    use_pynccl: bool = True  # 是否使用PyNCCL加速通信
    max_seq_len_override: int | None = None  # 手动覆盖最大序列长度
    num_page_override: int | None = None  # 手动覆盖KV cache页数（不为None则覆盖自动计算）

    @cached_property
    def hf_config(self):
        """缓存并返回HuggingFace模型配置。"""
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_config(self) -> ModelConfig:
        """将HuggingFace配置转换为内部ModelConfig格式。"""
        from minisgl.models import ModelConfig

        return ModelConfig.from_hf(self.hf_config)

    @property
    def max_seq_len(self) -> int:
        """获取最大序列长度（优先使用手动覆盖值，否则使用模型RoPE配置）。"""
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        """前向传播的最大长度，与max_seq_len相同。"""
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        """分布式初始化的master地址（固定为本地TCP地址）。"""
        return "tcp://127.0.0.1:2333"
