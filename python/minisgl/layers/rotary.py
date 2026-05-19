from __future__ import annotations

import functools
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    """旋转位置编码（RoPE）：对 Q 和 K 施加位置相关的旋转变换"""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size  # 注意力头的维度
        assert rotary_dim == head_size
        # 计算逆频率：base^(-2i/d)，i 从 0 到 rotary_dim/2 - 1
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            inv_freq = post_process(inv_freq)  # 对逆频率做后处理（如 rope scaling）
        # 预计算所有位置的正余弦值并缓存
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # [max_pos, rotary_dim//2]
        cos = freqs.cos()
        sin = freqs.sin()
        # buffer（不参与序列化保存/加载）
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)  # [max_pos, rotary_dim]
        assert self.head_size in [64, 128, 256, 512]

        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        # flashinfer 的 RoPE 原地应用函数
        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据 positions 中的位置索引，从缓存中查表并对 Q/K 施加 RoPE
        返回旋转后的 query 和 key
        """
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    """
    根据配置创建 RotaryEmbedding 对象，支持多种 rope scaling 策略
    """
    if rope_scaling is None:
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # 根据 rope_type 选择不同的缩放策略
    match rope_scaling["rope_type"]:
        case "default":
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            """Llama3 风格的 RoPE 缩放：高频不变，低频拉伸"""
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 当 low_freq_factor == high_freq_factor 时不做平滑过渡
                wave_len = 2 * math.pi / inv_freq  # 波长
                if low_freq_factor == high_freq_factor:
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,  # 高频部分保持不变
                        inv_freq / scaling_factor,  # 低频部分除以缩放因子（拉伸）
                    )

                # 平滑过渡：低频拉伸，高频不变
                delta = high_freq_factor - low_freq_factor
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            """YaRN 风格的 RoPE 缩放：使用斜坡（ramp）在高低频之间进行插值"""
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                """找到对应旋转次数的维度索引"""
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            # 计算高低频的边界维度
            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 创建从 0 到 1 的斜坡，低频维度被缩放，高频维度保持不变
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                # 低频（ramp=0）: inv_freq / factor；高频（ramp=1）: inv_freq
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    raise ValueError(f"Unsupported {rope_scaling = }")


_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    """设置 RoPE 的创建设备（用于 meta device 场景）"""
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    """获取 RoPE 对象的工厂函数，带有缓存机制"""
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # 不能在 meta device 上创建 RoPE，需切换到真实设备
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]