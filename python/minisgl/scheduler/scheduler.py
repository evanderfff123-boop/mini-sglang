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
    # 定义重叠调度循环函数，接收上一轮正在计算的数据对象，并返回本轮新启动计算的数据对象
    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # 计算当前接收消息是否需要阻塞等待
        blocking = not (
            # 如果上一轮启动计算的数据尚在流中等待处理，则当前不应发生阻塞，应尽快往下执行以重叠时间
            last_data is not None  # don't block if we have a batch to be processed
            # 或者当前 Prefill 队列中有就绪请求，也不应阻塞
            or self.prefill_manager.runnable
            # 或者当前 Decode 队列中有就绪请求，也不应阻塞
            or self.decode_manager.runnable
        )
        # 循环读取并处理接收到的网络/系统消息，根据计算出的 blocking 状态决定是否在通道无数据时挂起
        for msg in self.receive_msg(blocking=blocking):  # 接收并处理消息
            # 逐个处理接收到的用户消息或取消指令
            self._process_one_msg(msg)

        # 调度下一批待执行的请求（此时 GPU 可能仍在计算上一批，利用 CPU 空闲间隙并行做调度逻辑）
        forward_input = self._schedule_next_batch()
        # 初始化本轮推理的数据记录对象
        ongoing_data = None
        # 如果新调度出了可以执行的输入批次
        if forward_input is not None:
            # 进入底层引擎执行流的上下文管理器（将后续的前向计算命令发射到引擎流中）
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                # 让引擎计算流同步等待调度流，确保下一批次所需的元数据或张量已经搬移准备完毕
                self.engine.stream.wait_stream(self.stream)  # 同步调度流到引擎流
                # 在引擎流上异步启动当前批次的前向传播计算（非阻塞立即返回），并封装为 ongoing_data
                ongoing_data = (forward_input, self._forward(forward_input))

        # 核心重叠操作：在 GPU 执行当前批次前向推理的同时，CPU 同步处理并释放上一批 last_data 的采样和元数据
        self._process_last_data(last_data)  # 处理上一批的结果（与当前前向传播重叠）
        # 返回本轮发射但尚未在 CPU 侧处理结果的 ongoing_data，它将在下一次循环中作为 last_data 传入
        return ongoing_data

    # --- 非重叠调度主循环 ---
    def normal_loop(self) -> None:
        """普通的非重叠调度循环，串行处理消息、调度、前向传播、结果。"""
        # 判断当前接收消息时是否需要阻塞等待：如果当前 Prefill 和 Decode 管理器均无可运行的请求，则设为 True，否则为 False
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        # 遍历从接收消息通道中获取的全部最新消息（根据计算得到的阻塞状态等待）
        for msg in self.receive_msg(blocking=blocking):
            # 逐一解析并处理接收到的单条消息（如新请求加入、请求取消等）
            self._process_one_msg(msg)

        # 调用调度算法，对队列中的就绪请求进行打包，生成下一批待前向计算的输入数据
        forward_input = self._schedule_next_batch()
        # 初始化当前轮次的执行数据对象为 None
        ongoing_data = None
        # 如果成功调度出有效的批次输入数据
        if forward_input is not None:
            # 串行触发模型前向传播计算，并将输入和计算结果（logits/Future）绑定为元组存入 ongoing_data
            ongoing_data = (forward_input, self._forward(forward_input))

        # 处理本轮刚刚计算完的推理数据（串行地进行采样、Token 释放、状态更新以及结果发回）
        self._process_last_data(ongoing_data)

    # 这里的重叠指的是CPU和GPU是否是并行执行
    # 使用 PyTorch 的推理模式装饰器，禁用梯度计算并减少不必要的张量历史记录，以优化内存和推理速度
    @torch.inference_mode()
    # 定义调度器的主循环运行方法，返回类型为 NoReturn（表示该函数为无限循环，正常情况下不会主动退出）
    def run_forever(self) -> NoReturn:
        """主循环入口：根据配置选择重叠或普通模式。"""
        # 判断全局环境配置中是否禁用了重叠（Overlap）调度模式
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # 如果禁用了重叠调度，则通过上下文管理器切入到底层推理引擎的 CUDA 流环境中
            with self.engine_stream_ctx:
                # 阻塞引擎流，使其等待调度流中当前排队的所有操作执行完毕，确保数据一致性
                self.engine.stream.wait_stream(self.stream)
                # 开启一个无限循环，以串行、同步的方式持续处理请求
                while True:
                    # 调用普通模式的循环函数，串行执行单次调度与前向计算
                    self.normal_loop()
        # 如果未禁用重叠调度（即启用高吞吐的重叠流水线模式）
        else:
            # 断言确保当前线程中 PyTorch 默认活跃的 CUDA 流确实是调度器专属流
            assert torch.cuda.current_stream() == self.stream
            # 初始化一个过渡变量 data 为 None，用于在两次异步迭代之间传递调度元数据和状态
            data = None
            # 开启一个无限循环，以流水线重叠的方式持续进行并发调度与计算
            while True:
                # 执行单次重叠模式的循环逻辑，更新并在迭代间循环传递 data 数据状态
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
        if batch.is_decode:
            self._decode_count = getattr(self, '_decode_count', 0) + 1
        else:
            print(f"[RESULT] Prefill done, moving req to decode manager")
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
        if new_finished_reqs:
            print(f"[DONE]  Request finished after {self._decode_count} decode steps")
        self.send_result(reply)  # 发送 detokenize 结果给前端

    # --- 消息处理 ---
    # 定义内部方法 _process_one_msg，用于分发和处理单个后台传入的消息对象
    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        """处理单个后端消息（用户请求、中止、退出等）。"""
        # 判断消息类型是否为批量消息包（BatchBackendMsg）
        if isinstance(msg, BatchBackendMsg):
            # 遍历批量包内所封装的每一条具体的子消息对象
            for msg in msg.data:
                # 递归调用本方法，分别解析并处理每一个子消息
                self._process_one_msg(msg)
        # 判断消息类型是否为系统退出信号（ExitMsg）
        elif isinstance(msg, ExitMsg):
            # 抛出 KeyboardInterrupt 异常，以此中断外层的 while True 推理主循环，实现优雅停机
            raise KeyboardInterrupt 
        # 判断消息类型是否为新进来的用户推理请求消息（UserMsg）
        elif isinstance(msg, UserMsg):
            print(f"\n{'='*60}")
            print(f"[STATE] NEW REQUEST: uid={msg.uid}, input_len={len(msg.input_ids)}")
            print(f"[STATE]   → Entering PREFILL queue ({len(self.prefill_manager.pending_list)} pending)")
            print(f"{'='*60}\n")
            # 仅在主进程（Rank 0）上打印调试日志，记录收到用户消息的详细信息
            logger.debug_rank0("Received user msg: %s", msg)
            # 获取用户输入 prompt 的 Token 序列长度，以及当前推理引擎能支持的最大序列长度
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            # 计算该请求理论上最大能用于生成新 Token 的可用剩余空间长度
            max_output_len = max_seq_len - input_len
            # 如果可用剩余长度小于等于 0，表明用户输入的 prompt 长度本身就已经超过了模型的最大承载能力
            if max_output_len <= 0: # 输入太长，丢弃请求
                # 打印严重警告日志，并直接返回，不将该请求加入调度队列
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            # 如果用户设置的生成最大 Token 数（max_tokens）超过了当前剩余的最大空间边界
            if msg.sampling_params.max_tokens > max_output_len:  # 调整 max_tokens 避免溢出
                # 将该请求的目标生成长度强制限制收缩为安全的边界最大值，防止显存越界或计算溢出
                msg.sampling_params.max_tokens = max_output_len
                # 在主进程上打印警告日志，提示由于物理上限，生成长度已被自适应调整
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 校验并调整完参数后，将请求加入到 Prefill 管理器的等待队列中
            self.prefill_manager.add_one_req(msg)
        # 判断消息类型是否为中途中止当前请求的控制消息（AbortBackendMsg）
        elif isinstance(msg, AbortBackendMsg):
            # 在主进程上打印调试日志，记录准备强制终止的请求 ID
            logger.debug_rank0("Aborting request %d", msg.uid)
            # 尝试调用 Prefill 管理器的中止接口，寻找并移除该请求，成功则返回对应请求对象
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            # 如果在 Prefill 队列没找到该请求，则继续尝试在 Decode 管理器中查找并移出该请求
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            # 如果在以上任意阶段成功定位并拦截到了待释放的请求
            if req_to_free is not None:
                # 调用显存与表槽位回收接口，立即释放该请求之前占用的所有物理显存页和表索引资源
                self._free_req_resources(req_to_free)
        # 如果收到其他未知或未定义的消息类型
        else:
            # 记录严重错误日志，并输出未知消息的实际类名称
            logger.error(f"Unknown message type: {type(msg)}")
            # 抛出未实现异常，阻止程序带故障继续运行
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
    # 定义内部方法 _schedule_next_batch，用于生成下一个要在 GPU 上执行的批次输入，可返回 ForwardInput 或 Non
    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        # 待办事项：未来支持其他的调度策略，例如优先调度已经处于 DECODE 状态下的请求
        batch = (
            # 默认优先调度 Prefill 管理器中的请求，并限制单次计算的最大 Token 预算
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            # 如果本轮没有符合条件的 Prefill 请求，则顺延调度处于 Decode 管理器中的就绪请求
            or self.decode_manager.schedule_next_batch()
        )
        if batch and batch.is_prefill:
            print(f"[BATCH] ★ PREFILL: {len(batch.reqs)} requests, "
                  f"pending={len(self.prefill_manager.pending_list)}")
        # 如果成功组合出了待推理的批次对象，则调用 _prepare_batch 进行页表和张量打包并返回，否则返回 None
        return self._prepare_batch(batch) if batch else None

    # --- 前向传播 ---
    # 定义前向传播处理方法 _forward，接收准备好的输入包，并返回推理引擎输出结果对象
    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        """执行一次前向传播：读取 token 池 -> 模型推理 -> 写回 token 池。"""
        # 解构输入包，分别提取当前批次对象、采样超参数、输入内存映射索引和输出内存映射索引
        batch, sample_args, input_mapping, output_mapping = forward_input
        self._decode_count = getattr(self, '_decode_count', 0)
        if batch.is_prefill:
            n_tokens = sum(r.extend_len for r in batch.padded_reqs)
            print(f"\n[PREFILL] {batch.size} reqs, {n_tokens} tokens → entering DECODE phase")
            self._decode_count = 0
        # 通过输入索引映射，从物理 Token 池中精准提取本轮推理所需的全部 Token ID
        batch.input_ids = self.token_pool[input_mapping]
        # 调用底层推理引擎的 forward_batch 方法，驱动模型在 GPU 上进行前向计算和 Token 采样
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 将本次推理新采样出的 Token 数据，根据输出索引映射写回到物理 Token 池对应的槽位上
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu 
        # 通知并更新 Decode 管理器的状态，筛选出本次推理后依然处于解码生命周期中的请求
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        # 将包含推理结果和生成的 forward_output 对象返回给调用者
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
