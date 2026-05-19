from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    """生成基于 PID 的唯一后缀，用于区分不同进程的 ZMQ 地址"""
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    """Scheduler 配置（继承自 EngineConfig，追加调度和网络相关配置）"""

    max_extend_tokens: int = 8192  # Chunk Prefill 的最大 chunk 大小（以 token 计）
    cache_type: str = "radix"  # KV cache 管理策略（默认 radix tree）
    offline_mode: bool = False  # 是否离线模式（不依赖 ZMQ 通信）

    # 网络配置
    _unique_suffix: str = field(default_factory=_get_pid_suffix)  # 每个进程的唯一后缀

    @property
    def zmq_backend_addr(self) -> str:
        """scheduler（后端）接收 tokenizer 消息的 ZMQ 地址"""
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        """scheduler 发送结果到 detokenizer 的 ZMQ 地址"""
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        """多 rank 模式下主 rank 广播消息给其他 rank 的 ZMQ 地址"""
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        """最大前向传播长度（等于 max_extend_tokens）"""
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """scheduler 是否需要创建到 detokenizer 的 ZMQ 链接"""
        return True
