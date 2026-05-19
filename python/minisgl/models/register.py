import importlib

from .config import ModelConfig

# 模型注册表：架构名称 -> (模块路径, 类名)
_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    # Mistral3 条件生成模型复用 MistralForCausalLM 的实现
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    """根据 HuggingFace 架构名称动态导入并实例化对应的模型类。"""
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"模型架构 {model_architecture} 不受支持")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    # 动态导入模型模块
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]
