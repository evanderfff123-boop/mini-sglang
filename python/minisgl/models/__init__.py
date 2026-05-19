from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """工厂方法：根据模型配置创建对应的 LLM 模型实例。"""
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
