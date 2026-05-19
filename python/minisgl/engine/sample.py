from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
from minisgl.utils import is_sm90_supported, nvtx_annotate

if TYPE_CHECKING:
    from minisgl.core import Batch


@dataclass
class BatchSamplingArgs:
    """一个batch的采样参数，包含温度、top-k和top-p值。"""
    temperatures: torch.Tensor | None  # 每个请求的温度参数，None表示贪心解码
    top_k: torch.Tensor | None = None  # 每个请求的top-k值
    top_p: torch.Tensor | None = None  # 每个请求的top-p值


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """创建固定内存上的tensor，再异步拷贝到GPU上（减少主线程阻塞）。"""
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    """执行实际的采样操作：先softmax，再根据top-k/top-p策略采样。"""
    import flashinfer.sampling as sampling

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    if top_k is None and top_p is None:
        return sampling.sampling_from_probs(probs)  # 无约束采样

    if top_p is None:
        assert top_k is not None
        return sampling.top_k_sampling_from_probs(probs, top_k)  # 只使用top-k

    if top_k is None:
        assert top_p is not None
        return sampling.top_p_sampling_from_probs(probs, top_p)  # 只使用top-p

    assert top_k is not None and top_p is not None
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)  # top-k和top-p同时使用


@dataclass
class Sampler:
    """采样器，负责从logits中采样生成下一个token。"""
    device: torch.device  # 采样操作所在的GPU设备
    vocab_size: int  # 词表大小，用于top-k的默认值

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """从batch中提取所有请求的采样参数，组装为BatchSamplingArgs。"""
        params = [r.sampling_params for r in batch.reqs]
        if all(p.is_greedy for p in params):
            return BatchSamplingArgs(temperatures=None)

        MIN_P = MIN_T = 1e-6
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        top_k, top_p = None, None
        if any(k != self.vocab_size for k in top_ks):
            top_k = make_device_tensor(top_ks, torch.int32, self.device)  # 存在非默认top_k才创建tensor
        if any(p < 1.0 for p in top_ps):
            top_p = make_device_tensor(top_ps, torch.float32, self.device)  # 存在非默认top_p才创建tensor
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        """根据采样参数从logits中采样生成下一个token。"""
        with torch.cuda.nvtx.range("Sampler"):
            if args.temperatures is None:  # 贪心解码，直接取最大概率
                return torch.argmax(logits, dim=-1)
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
