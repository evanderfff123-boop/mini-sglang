from __future__ import annotations

from typing import TYPE_CHECKING, Final, List

import torch
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

if TYPE_CHECKING:
    from .config import SchedulerConfig

logger = init_logger(__name__)


# 定义 SchedulerIOMixin 类，作为调度器输入输出操作的混入（Mixin）类
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

    # 初始化方法，接收调度器配置以及分布式计算中的 CPU 进程组
    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        # 从配置对象中提取张量并行（Tensor Parallelism）的相关信息
        tp_info = config.tp_info
        # 将传入的分布式 CPU 进程组保存为常量属性，用于多 rank 间的同步控制
        self.tp_cpu_group: Final = tp_cpu_group
        # 判断当前是否处于离线测试或运行模式
        if config.offline_mode:
            # 离线模式：使用自定义的收发方法（不依赖 ZMQ）
            # 将接收消息的方法绑定为离线环境下的模拟接收函数
            self.receive_msg = self.offline_receive_msg
            # 将发送结果的方法绑定为离线环境下的模拟发送函数
            self.send_result = self.offline_send_result
            # 离线模式下无需初始化 ZMQ 网络队列，因此提前结束并退出初始化流程
            return

        # 判断当前进程是否为张量并行组中的主 rank（通常为 Rank 0）
        if tp_info.is_primary():
            # 主 rank 需要创建到 tokenizer 的收发链接
            # 初始化一个 ZMQ 拉取队列（Pull Mode），用于从 Tokenizer 接收请求数据
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                # 传入网络通信所需的后端绑定地址
                config.zmq_backend_addr,
                # 设置 create=True，表明由主 rank 来绑定（bind）该端口地址
                create=True,  # 主 rank 负责绑定地址
                # 指定消息的解码器，将原始字节数据反序列化为消息对象
                decoder=BaseBackendMsg.decoder,
            )
            # 初始化一个 ZMQ 推送队列（Push Mode），用于向 Tokenizer 发送生成的结果
            self._send_into_tokenizer: Final = ZmqPushQueue(
                # 传入反分词器（Detokenizer）的通信地址
                config.zmq_detokenizer_addr,
                # 根据配置参数决定是由当前端创建连接，还是仅作为客户端进行连接
                create=config.backend_create_detokenizer_link,
                # 指定消息的编码器，用于在发送前对数据进行序列化
                encoder=BaseTokenizerMsg.encoder,
            )

        recv = self._recv_msg_single_rank  # 单 rank 接收策略
        send = self._reply_tokenizer_rank0  # 发送策略（仅主 rank）
        # 如果张量并行的进程数大于 1，则启用多 rank 协同的通信逻辑
        if tp_info.size > 1:
            # 在多 rank 模式下，进一步区分主 rank 与非主 rank 的行为
            if tp_info.is_primary():
                # 多 rank 模式：主 rank 接收后通过 PUB 广播给其他 rank
                recv = self._recv_msg_multi_rank0
                # 将主 rank 的接收策略切换为多 rank 专属接收函数（接收后需要广播给其他 rank）
                self._send_into_ranks: Final = ZmqPubQueue(
                    # 使用调度器专属的广播地址，并设置 create=True 进行地址绑定
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            # 如果当前进程是多 rank 模式下的非主 rank（从属 rank）
            else:
                # 非主 rank：从 SUB 队列接收广播
                # 将非主 rank 的接收策略设置为多 rank SUB 接收函数（直接从主 rank 的广播中拉取）
                recv = self._recv_msg_multi_rank1
                # 非主 rank 不需要直接回复 Tokenizer，将其发送策略切换为对应的空操作或丢弃函数
                send = self._reply_tokenizer_rank1  # 非主 rank 不回复
                # 创建一个 ZMQ 订阅队列（Sub Mode），用于接收来自主 rank 广播的分发数据
                self._recv_from_rank0: Final = ZmqSubQueue(
                    # 传入相同的广播网络地址
                    config.zmq_scheduler_broadcast_addr,
                    # 设置 create=False，表示非主 rank 仅作为客户端进行连接而不进行绑定
                    create=False,  # 非主 rank 只连接不绑定
                    # 指定消息解码器，确保解析出与主 rank 一致的数据格式
                    decoder=BaseBackendMsg.decoder,
                )

        # 将最终根据运行模式（单/多 rank、主/非主 rank）筛选出的接收方法，赋值给公共接口 receive_msg
        self.receive_msg = recv
        # 将最终确定的发送方法，赋值给公共接口 send_result
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
