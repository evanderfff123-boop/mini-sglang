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


# @dataclass 装饰器，自动为该类生成构造函数（__init__）、属性表示（__repr__）等样板代码
@dataclass
class FrontendManager:
    """前端管理器：管理用户请求的生命周期，包含 ZMQ 通信和事件映射"""

    # 存储基础的服务器参数配置对象
    config: ServerArgs  # 服务器配置
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]  # 向 tokenizer 进程发送消息的队列
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]  # 从 tokenizer 进程接收消息的队列
    uid_counter: int = 0  # 用户 ID 计数器，单调递增
    initialized: bool = False  # 是否已启动后台监听任务
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)  # uid -> 已收到的回复列表
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)  # uid -> 用于等待新回复的事件

    # 定义分配新用户/请求的方法，返回生成的唯一 uid
    def new_user(self) -> int:
        """为新请求分配一个唯一的 uid，并初始化对应的 ack_map 和 event_map"""
        # 获取当前计数器对应的 uid 值
        uid = self.uid_counter
        # 计数器累加 1，为下一次新用户请求做准备
        self.uid_counter += 1
        # 在回复映射表中，为该 uid 创建一个空的回复列表
        self.ack_map[uid] = []
        # 为该 uid 创建并关联一个新的 asyncio.Event 事件，用于挂起/唤醒流式等待协程
        self.event_map[uid] = asyncio.Event()
        # 返回分配到的 uid
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

    # 异步处理取消和清理用户请求的逻辑
    async def abort_user(self, uid: int):
        """取消指定 uid 的请求：清理映射并发送 Abort 消息"""
        await asyncio.sleep(0.1)  # 等待可能的最后一批数据到达
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        await self.send_one(AbortMsg(uid=uid))
    
    # 关闭前端管理器连接并释放资源的方法
    def shutdown(self):
        """关闭所有 ZMQ 队列"""
        # 停止用于发送数据到 tokenizer 的 ZMQ PUSH 连接
        self.send_tokenizer.stop()
        # 停止用于接收来自 tokenizer 数据的 ZMQ PULL 连接
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


# 异步处理终端 Shell 模式补全请求的接口函数
async def shell_completion(req: OpenAICompletionRequest):
    """交互式 Shell 的补全请求（使用 BackgroundTask 自动取消）"""
    # 从全局环境中获取初始化好的 FrontendManager 实例（即控制连接与状态的 state 对象）
    state = get_global_state()
    # 断言确保请求中存在消息列表（messages 字段不为 None），因为交互式 Shell 只支持多轮对话补全模式
    assert req.messages is not None, "Shell completion only supports chat-completions"
    # 遍历请求中当前及历史的所有 Message 对象，调用其 model_dump() 将其转换为字典形式，构成提供给后端的 Prompt 上下文
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: 支持更多采样参数
    # 调用状态管理器的 new_user() 方法，为当前轮次的对话请求分配一个独一无二的 uid，并初始化缓存映射
    uid = state.new_user()
    # 异步通过 ZMQ 发送队列投递一条 TokenizeMsg 消息，驱动后端的 tokenizer 和推理调度器工作
    await state.send_one(
        TokenizeMsg(
            # 携带分配给当前请求的唯一标识 uid
            uid=uid,
            # 将刚刚整理、序列化好的对话历史和新提示词（prompt 列表）作为输入内容传给后端
            text=prompt,
             # 构造并配置模型推理时的采样参数
            sampling_params=SamplingParams(
                # 是否忽略结束符（EOS）
                ignore_eos=req.ignore_eos,
                # 设定单次生成的最大 Token 数量限制
                max_tokens=req.max_tokens,
                # 设置采样温度值，控制生成文本的多样性与发散程度
                temperature=req.temperature,
                # 设定 Top-K 采样参数
                top_k=req.top_k,
                # 设定 Top-P 采样参数
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


# 定义用于命令行交互式 shell 的异步函数，作为除 Web 模式之外的轻量终端测试环境
async def shell():
    """交互式命令行 Shell：支持 /exit 退出和 /reset 重置对话历史"""
    # 预设在交互提示符中可以支持的控制指令列表
    commands = ["/exit", "/reset"]
     # 使用 prompt_toolkit 提供的 WordCompleter 对指令列表进行封装，以便输入时提供补全建议
    completer = WordCompleter(commands)
    # 创建交互式话路 Session，设置提示符前缀为 "$ "，并将补全器注入其中
    session = PromptSession("$ ", completer=completer)

    # 尝试进入命令读取和处理逻辑，防止由于强行终止产生崩溃
    try:
        # 初始化一个元组列表，用于维护和累积当前会话下的多轮对话历史（[(用户输入, 模型回答)]）
        history: List[Tuple[str, str]] = []  # 对话历史（user_msg, assistant_msg）
        # 开启交互主循环，直至遇到退出信号
        while True:
            # 异步挂起并等待用户在终端输入内容，读取后自动剥离首尾的多余空白字符
            cmd = (await session.prompt_async()).strip()
            # 如果输入内容为空
            if cmd == "":
                continue
             # 判断输入的文本是否以斜杠 "/" 开头，如果是，说明这是一条系统控制指令
            if cmd.startswith("/"):
                # 如果控制指令是 "/exit"
                if cmd == "/exit":
                    return
                # 如果控制指令是 "/reset"
                if cmd == "/reset":
                    # 重置对话历史，将列表清空
                    history = []  # 清空对话历史
                    continue
                # 如果输入的是其他不认识的斜杠指令，则抛出值错误异常
                raise ValueError(f"Unknown command: {cmd}")
            # 初始化一个列表，用于存放组装好的、符合 API 格式的多轮对话消息对象
            history_messages: List[Message] = []
            # 遍历此前记录的历史对话数组，将多轮问答拆解为单独的消息对象
            # 每轮都想要重新append一下
            for user_msg, assistant_msg in history:
                # 构建代表用户的历史消息对象并追加至消息列表
                history_messages.append(Message(role="user", content=user_msg))
                # 构建代表模型助手的历史消息对象并追加至消息列表
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # 构建一个发送给推理服务器的 OpenAI 兼容补全请求对象
            req = OpenAICompletionRequest(
                # model 字段置空，本地单模型服务环境下一般无需显式指定具体模型名称
                model="",
                # 将前面整合的所有历史多轮对话消息，与用户当前键入的 cmd 合并为最终的消息列表
                messages=history_messages + [Message(role="user", content=cmd)],
                # 从系统环境变量中读取并填充最大 Token 生成限制
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                # 从系统环境变量中读取并填充 top_k 采样过滤参数
                top_k=ENV.SHELL_TOP_K.value,
                # 从系统环境变量中读取并填充 top_p 采样过滤参数
                top_p=ENV.SHELL_TOP_P.value,
                # 从系统环境变量中读取并填充采样温度值参数
                temperature=ENV.SHELL_TEMPERATURE.value,
                # 设置为流式生成，以便本地终端能够逐字打印模型的实时产出
                stream=True,
            )
            # 初始化一个字符串，用于暂存并累加当前这次对话收到的全部模型增量回复
            cur_msg = ""
            # 调用本地的 shell_completion 接口发送请求，并异步遍历返回数据包迭代器（body_iterator）
            async for chunk in (await shell_completion(req)).body_iterator:
                # 将接收到的二进制消息数据块解码为 Python 字符串
                msg = chunk.decode()  # type: ignore
                # 断言当前的数据块必须以 "data: " 前缀开头，验证其是否符合 SSE 规范
                assert msg.startswith("data: "), msg
                # 剥离前缀 "data: "（即切片去除前 6 个字符），得到实际的有效载荷字符串
                msg = msg[6:]  # 去掉 "data: " 前缀
                # 断言剥离前缀后的载荷文本是否是以换行符 "\n" 结尾
                assert msg.endswith("\n"), msg
                msg = msg[:-1]  # 去掉末尾换行
                # 如果接收到的有效载荷是流式完成标志 "[DONE]"
                if msg == "[DONE]":
                    # 说明模型已完成生成，跳过打印并等待数据流自然闭合
                    continue
                # 将每次解包出的一小段增量文本累加到 cur_msg 中
                cur_msg += msg
                # 立即向控制台打印这一段文字，end="" 避免自动换行，flush=True 强制刷写输出缓冲区以防延迟
                print(msg, end="", flush=True)
            # 整个流式推理结果输出完毕后，打印一个换行符并刷新终端显示
            print("", flush=True)
            # 将本次用户的提问与完整的模型增量应答作为二元组存入历史记录，以便下轮对话检索
            history.append((cmd, cur_msg))  # 保存到历史记录
    # 捕获 EOFError 异常，通常对应用户在终端交互中按下了 Ctrl-D
    except EOFError:
        # 用户按下了 Ctrl-D
        # 捕获后直接略过，进入后面的回收流程
        pass
    # 退出交互时必须调用的资源清理块
    finally:
        # 向终端打印提示信息，告知用户正在退出命令行 Shell 界面
        print("Exiting shell...")
        # 异步睡眠 0.1 秒，等待后台一些未尽的文件或套接字操作进行平滑收尾
        await asyncio.sleep(0.1)
        # 获取前端管理的全局状态对象，并主动调用其 shutdown 方法，安全中断 ZMQ 消息队列的收发连接
        get_global_state().shutdown()
        # 然后杀掉所有子进程
        # 导入系统与进程辅助控制库 psutil，用于管理底层的进程拓扑树
        import psutil

        # 获取当前的父进程对象（即运行当前 python api 服务的进程实例）
        parent = psutil.Process()
        # 递归遍历并检索当前进程下属的所有还在活跃的子进程（例如 tokenizer 进程或调度器工作进程）
        for child in parent.children(recursive=True):
            # 强制杀掉相关的子进程，从源头上杜绝孤儿进程对系统计算资源或端口的残留占用
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    启动前端 API 服务器（FastAPI + uvicorn），并通过 ZMQ 连接到 tokenizer 进程。

    Args:
        config: 服务器配置（host/port、ZMQ IPC 地址等）。
        start_backend: 启动后端工作进程（TP scheduler + tokenizer/detokenizer）的回调函数。
        run_shell: 如果为 True，则运行交互式终端 Shell 而非启动 uvicorn。
    """
    # ZMQ: 进程间收发信息的库，比TCP Socket简单，比管道灵活

    # 声明使用全局变量 _GLOBAL_STATE，以便在函数内对其进行修改与赋值
    global _GLOBAL_STATE

    # 判断是否需要运行交互式终端 Shell
    if run_shell:
        # 在 Shell 模式下，断言配置中不能使用虚拟权重（dummy weights），若使用了则抛出异常
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    # 从配置对象中读取服务器需要绑定的主机 IP 地址
    host = config.server_host
     # 从配置对象中读取服务器需要监听的端口号
    port = config.server_port

    # 断言全局状态 _GLOBAL_STATE 此时必须为 None，确保它没有被重复初始化
    assert _GLOBAL_STATE is None, "Global state is already initialized"
    # 实例化 FrontendManager，并将其赋值给全局变量 _GLOBAL_STATE，用于管理前端状态与连接
    _GLOBAL_STATE = FrontendManager(
        # 将配置参数传递给 FrontendManager 实例
        config=config,
        # 创建一个用于接收来自 tokenizer 消息的异步 PULL 队列
        recv_tokenizer=ZmqAsyncPullQueue(
            # 使用配置中定义的前端 ZMQ 接收地址
            config.zmq_frontend_addr,
            # 设置为 True，表示前端进程是第一个绑定（bind）该地址的进程，负责创建该通信通道
            create=True,  # 前端是第一个绑定该地址的进程
            # 传入解码器，用于将接收到的原始二进制字节数据反序列化为前端消息对象
            decoder=BaseFrontendMsg.decoder,
        ),
        # 创建一个用于发送消息给 tokenizer 的异步 PUSH 队列
        send_tokenizer=ZmqAsyncPushQueue(
            # 使用配置中定义的 tokenizer ZMQ 发送地址
            config.zmq_tokenizer_addr,
            # 根据配置决定是由前端进程创建（bind）还是仅仅连接（connect）到此 ZMQ 通道
            create=config.frontend_create_tokenizer_link,  # 根据配置决定是否创建 ZMQ 绑定
            # 传入编码器，用于将要发送的消息对象序列化为二进制字节数据
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # 在此处调用回调函数，启动后端工作进程（包括 TP 调度器和 tokenizer/detokenizer）
    start_backend()

    # 输出日志，提示 API 服务器已准备就绪，并展示具体的主机和端口号
    logger.info(f"API server is ready to serve on {host}:{port}")
    # 判断当前是否不需要运行终端 Shell 模式
    if not run_shell:
        # 启动 uvicorn 异步 Web 服务器，运行 app 应用，并监听指定的主机地址和端口
        uvicorn.run(app, host=host, port=port)
    # 如果需要运行终端 Shell 模式
    else:
        # 使用 asyncio 运行异步的交互式 shell 函数
        asyncio.run(shell())
