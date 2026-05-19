from __future__ import annotations

import glob
import re
from typing import Dict, Iterator, Tuple

import safetensors
import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
from tqdm import tqdm

# 张量并行中按 dim=0 切分的投影层名称列表
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
# 张量并行中按 dim=1 切分的投影层名称列表
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# 融合组：将多个独立的投影层融合为单个合并投影层
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
# 每个独立投影在融合投影中的 slot 名称
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
# 正则表达式：匹配 MoE 专家权重的键名，提取前缀、专家索引和权重名称
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """提取 rank r 的张量分片。返回连续的副本。"""
    # dim=0 切分：q_proj, k_proj, v_proj, gate_proj, up_proj
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        # GQA 场景：KV 头数少于总 TP 数时，需要按头粒度切分
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            head_dim = value.shape[0] // num_kv_heads
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        return value.chunk(n, dim=0)[r].clone()
    # dim=1 切分：o_proj, down_proj
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        return value.chunk(n, dim=1)[r].clone()
    # 词嵌入和 LM Head：按词表维度（dim=0）均匀切分
    elif key.count("lm_head") or key.count("embed_tokens"):
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        return value  # 其余权重（如 norm 参数）各 rank 完整复制


def _get_merge_info(key: str):
    """若键属于融合组，返回 (融合后的键名, slot, 所有 slots)。否则返回 None。"""
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """将检查点中每个专家单独的键映射为运行时打包后的键名和专家索引。"""
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """流式权重加载器。产出 (名称, 张量) 对，已分片、融合并放置在目标设备上。
    峰值 CPU 内存：一个完整张量 + 一个小型融合缓冲区。"""
    from .config import ModelConfig

    model_folder = download_hf_weight(model_path)  # 下载/缓存模型权重
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))  # 加载配置
    # 查找所有 safetensors 权重文件，排除 consolidated 文件
    files = glob.glob(f"{model_folder}/*.safetensors")
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()  # 获取张量并行信息（rank, size）

    # 融合组缓冲区：merged_key -> {slot: tensor}
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    # MoE 专家缓冲区：packed_key -> {expert_idx: tensor}
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # 跳过多模态视觉塔和投影器的权重
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                raw = f.get_tensor(name)
                # 移除 language_model. 前缀（多模态模型中嵌套使用）
                name = name.removeprefix("language_model.")
                # 根据张量并行策略对权重进行分片
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                del raw  # 释放原始完整张量

                if (info := _get_merge_info(name)) is None:
                    out = (name, tensor)  # 不需要融合，直接产出
                else:
                    # 需要融合：将当前 tensor 存入缓冲区
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    # 等待同一融合组的所有 slot 到齐后再 concat
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        continue
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                # 处理 MoE 专家权重：收集所有专家后 stack 为单一张量
                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    # 等到所有专家权重收集完毕再 yield
                    if len(slots) != config.num_experts:
                        continue
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # 普通稠密模型，直接产出
                    yield out[0], out[1]

    # 断言所有融合和专家收集都已完成
    assert not merge_buf, f"检查点中存在不完整的融合组: {list(merge_buf.keys())}"
    assert not expert_buf, f"检查点中存在不完整的专家张量: {list(expert_buf.keys())}"
