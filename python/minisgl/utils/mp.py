from __future__ import annotations

from typing import Callable, Dict, Generic, TypeVar

import msgpack
import zmq
import zmq.asyncio

T = TypeVar("T")


class ZmqPushQueue(Generic[T]):
    """ZMQ PUSH 同步队列（用于发送消息到其他进程）"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 创建者 bind，其他 connect
        self.encoder = encoder  # 编码器

    def put(self, obj: T):
        """将对象编码后发送"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        """关闭 socket 和 context"""
        self.socket.close()
        self.context.term()


class ZmqAsyncPushQueue(Generic[T]):
    """ZMQ PUSH 异步队列（协程安全版本）"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    async def put(self, obj: T):
        """异步发送消息"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        await self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """ZMQ PULL 同步队列（用于从其他进程接收消息）"""

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.decoder = decoder  # 解码器

    def get(self) -> T:
        """接收并解码一条消息"""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        """接收原始字节消息（不解码）"""
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        """解码原始字节为对象"""
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        """检查队列是否为空（非阻塞轮询）"""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()


# 定义异步 ZMQ PULL 队列类，继承自 Generic[T] 以支持类型参数化（泛型）
class ZmqAsyncPullQueue(Generic[T]):
    """ZMQ PULL 异步队列（协程安全版本）"""

    # 构造函数：初始化 ZMQ 异步上下文、套接字以及消息解码器
    def __init__(
        self,
        # 接收一个字符串形式的 ZMQ 通信地址（可以是 ipc:// 或 tcp:// 协议等）
        addr: str,
        # 接收一个布尔值，用于指明当前进程是负责绑定（bind）还是连接（connect）此通道
        create: bool,
        # 接收一个可调用对象（解码器函数），用于将字典数据反序列化为特定类型 T 的对象
        decoder: Callable[[Dict], T],
    ):
        # 创建一个支持 asyncio 的 PyZMQ 异步上下文对象，它是生成异步 Socket 的基石
        self.context = zmq.asyncio.Context()
        # 在该异步上下文中创建一个 ZMQ PULL 类型的异步套接字，专门用于单向接收数据
        self.socket = self.context.socket(zmq.PULL)
        # 根据 create 标志执行：若为 True 则调用 bind 绑定指定地址，若为 False 则调用 connect 连接指定地址
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 将传入的解码回调函数保存至成员变量 self.decoder 中
        self.decoder = decoder

    async def get(self) -> T:
        """异步接收并解码一条消息"""
        event = await self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """ZMQ PUB 发布队列（用于一对多广播）"""

    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        """广播原始字节"""
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        """编码后广播"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """ZMQ SUB 订阅队列（用于接收 PUB 广播）"""

    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.bind(addr) if create else self.socket.connect(addr)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅所有消息
        self.decoder = decoder

    def get(self) -> T:
        """接收并解码一条广播消息"""
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        """检查是否有待接收的消息"""
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        self.socket.close()
        self.context.term()
