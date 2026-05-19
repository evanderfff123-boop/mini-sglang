from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


class SchedulerIOMixin:
    """
    Scheduler I/O 操作的混入类。

    负责 scheduler 与 tokenizer 之间的通信。
    支持单 rank 和多 rank（张量并行）两种模式。

    公共接口：
        receive_msg: 从 tokenizer 接收消息。
        send_result: 将结果发回 tokenizer。
        sync_all_ranks: 同步所有 TP rank 的 CPU 侧操作。
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        tp_info = config.tp_info
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            # 离线模式：使用自定义的收发方法（不依赖 ZMQ）
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # 提前退出

        if tp_info.is_primary():
            # 主 rank 需要创建到 tokenizer 的收发链接
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,  # 主 rank 负责绑定地址
                decoder=BaseBackendMsg.decoder,
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank  # 单 rank 接收策略
        send = self._reply_tokenizer_rank0  # 发送策略（仅主 rank）
        if tp_info.size > 1:
            if tp_info.is_primary():
                # 多 rank 模式：主 rank 接收后通过 PUB 广播给其他 rank
                recv = self._recv_msg_multi_rank0
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 非主 rank：从 SUB 队列接收广播
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1  # 非主 rank 不回复
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,  # 非主 rank 只连接不绑定
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        """空闲时执行的任务（由子类实现）"""
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线模式接收消息（由 LLM 子类实现）"""
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线模式发送结果（由 LLM 子类实现）"""
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        """通过 CPU 进程组同步所有 TP rank"""
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """单 rank 模式：直接从 tokenizer 拉取消息"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()  # 空闲时执行一些逻辑再阻塞等待
            pending_msgs.append(self._recv_from_tokenizer.get())
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 模式（主 rank）：接收消息并通过 PUB 广播给其他 rank"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            raw = self._recv_from_tokenizer.get_raw()  # 获取原始字节
            self._send_into_ranks.put_raw(raw)  # 广播给其他 rank
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # 广播原始消息数量给所有 rank
        src_tensor = torch.tensor(len(pending_raw_msgs))
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            self._send_into_ranks.put_raw(raw)  # 逐个广播
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """多 rank 模式（非主 rank）：从 SUB 队列接收主 rank 广播的消息"""
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # 确保所有 rank 有相同数量的原始消息
        dst_tensor = torch.tensor(-1)
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()  # 接收主 rank 广播的消息数
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        """主 rank 将结果发回 tokenizer（单条或批量打包）"""
        num_reply = len(reply)
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            self._send_into_tokenizer.put(reply[0])  # 单条直接发送
        elif num_reply > 1:
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore  # 多条打包发送

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        """非主 rank 不回复 tokenizer（消息已由主 rank 发送）"""
        _ = reply  # do nothing for non-primary ranks
