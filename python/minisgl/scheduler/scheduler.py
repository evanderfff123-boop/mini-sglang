from __future__ import annotations

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from minisgl.core import Batch, Req
from minisgl.env import ENV
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    """前向传播的输入，包含 batch、采样参数、输入映射和写回映射。"""
    batch: Batch  # 当前批次的请求
    sample_args: BatchSamplingArgs  # 采样参数
    input_tuple: Indice2D  # (token_mapping, positions)  从 token 池读取输入的位置
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)  将输出写回 token 池的位置


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# 定义 Scheduler 类，继承自处理输入输出交互的 SchedulerIOMixin 基类
class Scheduler(SchedulerIOMixin):
    """主调度器：管理 prefill/decode 调度、消息处理、重叠执行。"""

    # 初始化方法，接收一个 SchedulerConfig 配置对象
    def __init__(self, config: SchedulerConfig):
        # 局部导入 Engine 类，避免在文件顶部导入可能引起的循环引用
        from minisgl.engine import Engine

        # 实例化底层执行引擎，该引擎负责模型前向传播、注意力机制后端及采样器等核心计算
        self.engine = Engine(config)  # 底层引擎，包含模型、注意力后端、采样器等

        # 使用另一个 CUDA 流，以便将元数据/调度相关的处理与底层的 GPU 计算进行重叠（Overlap）
        # 获取底层引擎所使用的 GPU 设备
        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device  # GPU 设备
        # 创建一个专用于当前调度器任务的 PyTorch CUDA 流
        self.stream = torch.cuda.Stream(device=self.device)  # 调度器专用 CUDA 流
        # 获取底层引擎 CUDA 流的上下文管理器，以便在需要时切换到引擎流
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # 引擎 CUDA 流的上下文管理器
        # 将当前线程的默认 CUDA 流设置为调度器专用的 stream，使后续操作默认在该流上发射
        torch.cuda.set_stream(self.stream)

        # 初始化其他辅助管理组件
        # 实例化表管理器，用于维护最大运行请求数以及引擎物理页表的映射关系
        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)  # 行表管理器
        # 实例化缓存管理器，负责管理物理页分配、释放、逐出以及前缀缓存（Prefix Caching）
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )  # 缓存管理器（页面分配/逐出/前缀缓存）
        # 实例化解码管理器，用于维护和处理处于 Decode 阶段的请求信息
        self.decode_manager = DecodeManager(config.page_size)  # Decode 管理器
        # 实例化预填充管理器，协调缓存、表映射以及解码器来实现 Prefill 阶段的管理
        self.prefill_manager = PrefillManager(
            # 传入缓存管理器、表管理器和解码管理器作为其协同工作的依赖
            self.cache_manager, self.table_manager, self.decode_manager
        )  # Prefill 管理器

        # 定义一些常用变量的引用或别名，以便于后续逻辑快速访问
        # 初始化一个集合，用于记录当前调度轮次中已经执行完毕的请求对象
        # some alias for easy access
        self.finished_reqs: Set[Req] = set()  # 本轮已完成的请求集合
        # 根据配置中的模型路径加载对应的分词器（Tokenizer）
        self.tokenizer = load_tokenizer(config.model_path)  # tokenizer
        # 获取并保存分词器的结束符 Token ID，用于判断生成序列是否结束
        self.eos_token_id = self.tokenizer.eos_token_id  # EOS token ID
        # 创建对表管理器中 Token 池的引用，便于直接进行 Token 数据的存取
        self.token_pool = self.table_manager.token_pool  # token 池（快捷引用）
        # 设定单次 Prefill 调度能容纳的最大新 Token 预算限制
        self.prefill_budget = config.max_extend_tokens  # 每批 prefill 的最大 token 预算
        # self.config = config

        # 初始化 I/O 混入（Mixin）父类
        # 调用父类 SchedulerIOMixin 的初始化方法，传入配置对象和引擎的张量并行（TP）CPU 通信组
        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()  # 空闲时检查缓存完整性

    # --- 重叠调度主循环 ---
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        for msg in self.receive_msg(blocking=blocking):  # 接收并处理消息
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()  # 调度下一批
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)  # 同步调度流到引擎流
                ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(last_data)  # 处理上一批的结果（与当前前向传播重叠）
        return ongoing_data

    # --- 非重叠调度主循环 ---
    def normal_loop(self) -> None:
        """普通的非重叠调度循环，串行处理消息、调度、前向传播、结果。"""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        """主循环入口：根据配置选择重叠或普通模式。"""
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        """关闭调度器，同步设备并关闭引擎。"""
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    # --- 处理上一批的结果 ---
    def _process_last_data(self, last_data: ForwardData | None) -> None:
        """处理上一轮前向传播的输出（detokenize、释放资源、缓存等）。"""
        if last_data is None:
            return

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()  # 等待 CPU 拷贝完成
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():  # 延迟释放页面，减少碎片
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    continue  # 分块请求不参与 decode，不采样
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))  # 将新 token 追加到请求中
                next_token = int(next_token.item())
                finished = not req.can_decode  # 达到最大输出长度则完成
                if not req.sampling_params.ignore_eos:
                    finished |= next_token == self.eos_token_id  # 遇到 EOS 也完成
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:  # 请求结束
                    self.decode_manager.remove_req(req)  # 从 decode 集合移除
                    self._free_req_resources(req)  # 释放资源
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    self.cache_manager.cache_req(req, finished=False)  # 缓存 prefill 结果

        self.finished_reqs = new_finished_reqs
        self.send_result(reply)  # 发送 detokenize 结果给前端

    # --- 消息处理 ---
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理单个后端消息（用户请求、中止、退出等）。"""
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:  # 批量消息递归处理
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt  # 退出信号
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:  # 输入太长，丢弃请求
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:  # 调整 max_tokens 避免溢出
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)  # 加入 prefill 队列
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)  # 从 prefill 队列中止
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)  # 或从 decode 中止
            if req_to_free is not None:
                self._free_req_resources(req_to_free)  # 释放资源
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    # --- 释放请求资源 ---
    def _free_req_resources(self, req: Req) -> None:
        """释放请求的行表槽位和缓存资源。"""
        self.table_manager.free(req.table_idx)  # 归还行表槽位
        self.cache_manager.cache_req(req, finished=True)  # 缓存（或释放）请求的 KV cache

    # --- 准备 batch 前向传播 ---
    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        """对一个 batch 执行填充、分页分配、位置计算和元数据准备。"""
        self.engine.graph_runner.pad_batch(batch)  # 填充 batch 到对齐大小
        self.cache_manager.allocate_paged(batch.reqs)  # 分配物理页面
        batch.positions = _make_positions(batch, self.device)  # 生成位置编码
        input_mapping = _make_input_tuple(batch, self.device)  # 输入映射（token 池读取位置）
        write_mapping = _make_write_tuple(batch, self.device)  # 写回映射（输出写回位置）
        batch.out_loc = self.engine.page_table[input_mapping]  # 输出位置（页表地址）
        self.engine.attn_backend.prepare_metadata(batch)  # 准备注意力后端元数据
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),  # 准备采样参数
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    # --- 调度下一批 ---
    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)  # 优先调度 prefill
            or self.decode_manager.schedule_next_batch()  # 无 prefill 则调度 decode
        )
        return self._prepare_batch(batch) if batch else None

    # --- 前向传播 ---
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """执行一次前向传播：读取 token 池 -> 模型推理 -> 写回 token 池。"""
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]  # 从 token 池读取输入
        forward_output = self.engine.forward_batch(batch, sample_args)  # 模型前向
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu  # 将输出写回 token 池
        self.decode_manager.filter_reqs(forward_input.batch.reqs)  # 更新 decode 集合
        return forward_output


# --- 辅助函数：生成位置索引 ---
def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """为 batch 中每个请求的扩展部分生成位置编码索引（从 cached_len 到 device_len）。"""
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset: offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


# --- 辅助函数：生成输入映射 ---
def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """生成 (req_mapping, positions) 元组，用于从 token 池中读取输入 token。"""
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset: offset + length].fill_(req.table_idx)  # 每个 token 对应哪个请求
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


# --- 辅助函数：生成写回映射 ---
def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    """生成 (req_mapping, sequence_lens) 元组，用于将输出 token 写回 token 池。"""
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
