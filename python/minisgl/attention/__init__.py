from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from minisgl.utils import Registry, init_logger

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

if TYPE_CHECKING:
    from minisgl.models import ModelConfig

logger = init_logger(__name__)


class BackendCreator(Protocol):
    """后端创建器协议，接受 ModelConfig 并返回注意力后端实例"""
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")  # 支持的注意力后端注册表


@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config: ModelConfig):
    """创建 TensorRT-LLM 注意力后端"""
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig):
    """创建 FlashInfer 注意力后端"""
    from .fi import FlashInferBackend

    return FlashInferBackend(config)


@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config: ModelConfig):
    """创建 FlashAttention 注意力后端"""
    from .fa import FlashAttentionBackend

    return FlashAttentionBackend(config)


def validate_attn_backend(backend: str, allow_auto: bool = True):
    """验证注意力后端名称是否在支持的列表中"""
    if backend != "auto":
        required_backends = backend.split(",") if "," in backend else [backend]  # 支持逗号分隔的混合后端名
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        assert allow_auto, "auto is not allowed here"
    return backend


def create_attention_backend(
    backend: str,      # 后端名称，支持 "fi", "fa", "trtllm" 或 "prefill,decode" 混合
    config: ModelConfig,
) -> BaseAttnBackend:
    """工厂函数：根据后端名称创建对应的注意力后端实例"""
    validate_attn_backend(backend, allow_auto=False)
    if "," in backend:
        # 处理混合后端，格式为 "prefill_backend,decode_backend"
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            p_backend = create_attention_backend(p_backend, config)  # 递归创建 prefill 后端
            d_backend = create_attention_backend(d_backend, config)  # 递归创建 decode 后端
            return HybridBackend(p_backend, d_backend)
        backend = p_backend  # 如果前后端相同，退化为单后端模式
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    return SUPPORTED_ATTENTION_BACKENDS[backend](config)  # 从注册表创建后端


__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
