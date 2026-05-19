import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase


class DisabledTqdm(tqdm):
    """禁用进度条显示的 tqdm 包装器"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True  # 强制关闭进度条
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """从本地路径或 HuggingFace hub 加载 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 某些 Mistral 模型将 chat_template 存储在单独的 JSON 文件中
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass  # 加载失败时忽略（没有 chat_template 也可以正常工作）
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    """缓存模型配置加载结果（避免重复从网络加载）"""
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    """加载并返回模型的 HuggingFace 配置的深拷贝"""
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())  # 返回深拷贝，防止意外修改缓存


def download_hf_weight(model_path: str) -> str:
    """下载或返回模型权重路径"""
    if os.path.isdir(model_path):
        return model_path  # 本地目录直接返回
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],  # 只下载 safetensors 格式
            tqdm_class=DisabledTqdm,  # 禁用进度条
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
