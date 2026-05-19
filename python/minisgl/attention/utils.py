from dataclasses import dataclass

import torch


@dataclass
class BaseCaptureData:
    """CUDA Graph 捕获所需的基础数据缓冲区"""
    seq_lens: torch.Tensor       # 每个序列的长度
    positions: torch.Tensor      # 每个序列的位置编码
    cu_seqlens_k: torch.Tensor   # Key 侧累积序列长度（用于 flash attention）
    cu_seqlens_q: torch.Tensor   # Query 侧累积序列长度
    page_table: torch.Tensor     # 页表，记录逻辑页到物理页的映射

    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        """创建 CUDA Graph 捕获数据缓冲区，用占位值初始化"""
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),       # 初始化为全 1
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),      # 初始化为全 0
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),  # [0,1,2,...,max_bs]
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),  # [0,1,2,...,max_bs]
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),  # 页表初始化为 0
            **kwargs,
        )
