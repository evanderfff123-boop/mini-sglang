from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """
    融合的 SiLU 激活 + 逐元素乘法操作。
    用于 GLU（门控线性单元）变体，等价于 silu(x) * x。
    利用 flashinfer 的融合 kernel 以节省显存和访存。
    """
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """
    融合的 GELU 激活 + 逐元素乘法操作。
    用于 GELU 门控变体，等价于 gelu(x) * x。
    利用 flashinfer 的融合 kernel 以节省显存和访存。
    """
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
