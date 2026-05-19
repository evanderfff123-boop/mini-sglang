from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None  # 全局 FrontendManager 单例


def get_global_state() -> FrontendManager:
    """获取全局 FrontendManager 实例"""
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    """将可能被 BatchFrontendMsg 包裹的消息解包为 UserReply 列表"""
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    """生成请求的请求体模型"""

    prompt: str  # 输入提示文本
    max_tokens: int  # 最大生成长度
    ignore_eos: bool = False  # 是否忽略结束符


class Message(BaseModel):
    """对话消息的请求体模型"""

    role: Literal["system", "user", "assistant"]  # 消息角色
    content: str  # 消息内容


class OpenAICompletionRequest(BaseModel):
    """兼容 OpenAI API 格式的统一请求模型（支持 completions 和 chat-completions）"""

    model: str  # 模型名称

    prompt: str | None = None  # 原始 prompt（completions 接口）
    messages: List[Message] | None = None  # 聊天消息列表（chat-completions 接口）

    max_tokens: int = 16  # 最大生成 token 数
    temperature: float = 1.0  # 采样温度

    top_k: int = -1  # top-k 采样参数
    top_p: float = 1.0  # top-p 采样参数
    n: int = 1  # 生成候选数
    stream: bool = False  # 是否流式输出
    stop: List[str] = []  # 停止词列表
    presence_penalty: float = 0.0  # 存在惩罚
    frequency_penalty: float = 0.0  # 频率惩罚

    ignore_eos: bool = False  # 是否忽略结束符


class ModelCard(BaseModel):
    """模型卡片信息"""

    id: str  # 模型 ID
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间戳
    owned_by: str = "mini-sglang"  # 模型归属
    root: str  # 模型根路径


class ModelList(BaseModel):
    """模型列表响应"""

    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)  # 模型卡片列表


@dataclass
class FrontendManager:
    """前端管理器：管理用户请求的生命周期，包含 ZMQ 通信和事件映射"""

    config: ServerArgs  # 服务器配置
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]  # 向 tokenizer 进程发送消息的队列
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]  # 从 tokenizer 进程接收消息的队列
    uid_counter: int = 0  # 用户 ID 计数器，单调递增
    initialized: bool = False  # 是否已启动后台监听任务
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)  # uid -> 已收到的回复列表
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)  # uid -> 用于等待新回复的事件

    def new_user(self) -> int:
        """为新请求分配一个唯一的 uid，并初始化对应的 ack_map 和 event_map"""
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        """后台任务：持续从 tokenizer 接收回复，并按 uid 分发到对应的 ack_map 中"""
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                if msg.uid not in self.ack_map:
                    continue  # 忽略已取消的请求
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()  # 通知等待该 uid 的协程

    def _create_listener_once(self):
        """确保后台监听任务只被启动一次"""
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        """向 tokenizer 发送一条消息"""
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        """异步生成器：持续等待指定 uid 的回复，直到请求结束"""
        event = self.event_map[uid]

        while True:
            await event.wait()
            event.clear()  # 重置事件，准备等待下一批回复

            pending = self.ack_map[uid]
            self.ack_map[uid] = []  # 清空已取出的回复
            ack = None
            for ack in pending:
                yield ack  # 逐个产出回复
            if ack and ack.finished:
                break  # 请求结束时退出循环

        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        """流式生成：将回复包装为 SSE 格式（data: content）"""
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        """流式聊天补全：将回复包装为 OpenAI 兼容的 SSE chunk 格式"""
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"  # 第一个 chunk 中声明角色
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output

            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                break

        # 发送最终结束标记
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        """包装流式生成器：检测客户端断连时自动取消请求"""
        try:
            async for chunk in generator:
                # 检测客户端是否已断开连接
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)
                    raise asyncio.CancelledError
                yield chunk
        except asyncio.CancelledError:
            asyncio.create_task(self.abort_user(uid))  # 客户端断开后取消请求
            raise

    async def abort_user(self, uid: int):
        """取消指定 uid 的请求：清理映射并发送 Abort 消息"""
        await asyncio.sleep(0.1)  # 等待可能的最后一批数据到达
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        """关闭所有 ZMQ 队列"""
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """FastAPI 生命周期管理器：在应用关闭时执行清理"""
    yield
    # 关闭时的清理代码
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    """简化的文本生成端点，返回 SSE 流"""
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    """OpenAI API 根路径健康检查"""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """OpenAI 兼容的聊天补全端点，支持流式和非流式"""
    state = get_global_state()
    if req.messages:
        prompt = [msg.model_dump() for msg in req.messages]  # 将消息转为字典列表
    else:
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: 支持更多采样参数
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # 非流式模式：收集所有 chunk，合并为完整 JSON 响应
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        full_content += ack.incremental_output
        if ack.finished:
            break

    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    """获取可用模型列表"""
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    """交互式 Shell 的补全请求（使用 BackgroundTask 自动取消）"""
    state = get_global_state()
    assert req.messages is not None, "Shell completion only supports chat-completions"
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: 支持更多采样参数
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        await state.abort_user(uid)

    return StreamingResponse(
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )



async def shell():
    """交互式命令行 Shell：支持 /exit 退出和 /reset 重置对话历史"""
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)

    try:
        history: List[Tuple[str, str]] = []  # 对话历史（user_msg, assistant_msg）
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []  # 清空对话历史
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # 将当前消息和历史一起发送给服务器
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()  # type: ignore
                assert msg.startswith("data: "), msg
                msg = msg[6:]  # 去掉 "data: " 前缀
                assert msg.endswith("\n"), msg
                msg = msg[:-1]  # 去掉末尾换行
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))  # 保存到历史记录
    except EOFError:
        # 用户按下了 Ctrl-D
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        # 然后杀掉所有子进程
        import psutil

        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    启动前端 API 服务器（FastAPI + uvicorn），并通过 ZMQ 连接到 tokenizer 进程。

    Args:
        config: 服务器配置（host/port、ZMQ IPC 地址等）。
        start_backend: 启动后端工作进程（TP scheduler + tokenizer/detokenizer）的回调函数。
        run_shell: 如果为 True，则运行交互式终端 Shell 而非启动 uvicorn。
    """

    global _GLOBAL_STATE

    if run_shell:
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    host = config.server_host
    port = config.server_port

    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,  # 前端是第一个绑定该地址的进程
            decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,  # 根据配置决定是否创建 ZMQ 绑定
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # 在此处启动后端进程
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
