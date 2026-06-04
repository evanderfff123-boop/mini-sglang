from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from minisgl.distributed import DistributedInfo
from minisgl.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs

# 定义在子进程中运行调度器主循环的辅助函数，接收当前卡特化的参数配置和用于发送就绪反馈的多进程队列
def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """在子进程中运行 scheduler 主循环"""
    # 局部导入 PyTorch 库，以便执行模型计算及显存操作
    import torch
    # 从调度器模块导入核心控制器 Scheduler 类
    from minisgl.scheduler import Scheduler

    # 启用 PyTorch 的推理模式（Inference Mode）上下文管理器，该模式比 no_grad 更加高效，能最大限度降低内存和计算开销
    with torch.inference_mode():
        # 实例化调度器对象，传入为其定制好的 ServerArgs 配置
        scheduler = Scheduler(args)
        # 阻塞并等待参与张量并行（TP）计算的所有卡（Rank）完成握手，确保所有计算卡状态对齐、完成同步就绪
        scheduler.sync_all_ranks()  # 等待所有 TP rank 就绪

        # 判断当前卡对应的分布式角色是否为主卡（Primary Rank，通常为 Rank 0）
        if args.tp_info.is_primary():
            # 只有主卡需要向 ack_queue 队列发送确认信号，告知启动进程调度器已经初始化并就绪
            ack_queue.put("Scheduler is ready")  # 主 rank 通知启动器 scheduler 已准备就绪

        # 判断参数中是否开启了静默输出模式
        if args.silent_output:
            # 如果开启了静默模式，则禁用日志框架中所有 INFO 级别（含）以下的常规输出，减少终端日志冗余
            logging.disable(logging.INFO)  # 静默模式下关闭 INFO 日志

        # 开启异常捕获块，准备驱动推理主循环，并拦截用户手动终止信号
        try:
            # 调用 scheduler 的常驻运行方法，使该子进程进入无限循环，不断处理来自 ZMQ 的推理请求并调度 GPU 计算
            scheduler.run_forever()  # 进入主循环，持续处理请求
        # 捕获键盘中断信号（通常对应终端用户按下了 Ctrl+C 组合键）
        except KeyboardInterrupt:
            # 初始化本地日志记录器，以便输出退出相关的信息
            logger = init_logger(__name__)
            # 判断当前卡是否为主卡
            if args.tp_info.is_primary():
                # 如果是主卡，在控制台额外打印一个空行，避免中断输出与后续控制台提示混在一行，使输出格式更整洁
                print()  # 在 ^C 后输出一个空行保持格式整洁
                # 打印日志，记录调度器正在执行安全退出的动作
                logger.info("Scheduler exiting gracefully...")
            # 调用调度器的 shutdown() 方法，安全释放和回收占用的 GPU 显存资源并正常关闭通信套接字
            scheduler.shutdown()

# 定义启动完整推理服务的入口函数，提供一个可选参数指示是否运行交互式终端 Shell 模式（默认为 False）
def launch_server(run_shell: bool = False) -> None:
    """启动完整的推理服务：包括 scheduler 进程、tokenizer 进程和前端 API 服务器"""
    # 局部导入 API 服务器运行入口 run_api_server，避免在模块初始化时发生循环引用
    from .api_server import run_api_server
    # 局部导入命令行参数解析函数 parse_args
    from .args import parse_args

    # 解析传入的命令行参数（排除 Python 脚本名称），并获取更新后的 server_args 配置对象和最终的 run_shell 标志
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    # 初始化本模块的日志记录器，设置其名称为 "initializer"
    logger = init_logger(__name__, "initializer")

    # 定义一个局部回调函数，用于在 API 服务器启动准备完毕时，拉起所有的多进程后台工作子进程
    def start_subprocess() -> None:
        """启动 scheduler 和 tokenizer 子进程"""
        # 导入 Python 的多进程包 multiprocessing，并简写为 mp
        import multiprocessing as mp

        # 从 tokenizer 模块导入负责执行分词/反分词工作的子进程入口函数 tokenize_worker
        from minisgl.tokenizer import tokenize_worker

        # 强制将子进程的启动方法设置为 "spawn"，对于 CUDA 并发运行以及多卡分布式环境，这能有效防止死锁并保证上下文的干净
        mp.set_start_method("spawn", force=True)  # 使用 spawn 方式启动子进程（兼容 CUDA）

        # 从配置中读取张量并行（Tensor Parallelism）的规模大小，即参与分布式计算的显卡（Rank）总数
        world_size = server_args.tp_info.size
        # 创建一个多进程安全的共享队列，专门用来收集各后台子进程初始化完毕后投递的就绪确认消息（ACK）
        ack_queue: mp.Queue[str] = mp.Queue()

        # 根据显卡数量进行循环，为每一个张量并行卡（Rank）分别启动一个独立的调度器（scheduler）子进程
        for i in range(world_size):
            # 拷贝当前的基础配置，并将其中的 tp_info 属性替换为具体到当前显卡 rank 索引 `i` 对应的分布式信息
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),  # 每个 rank 有独立的 tp_info
            )
            # 实例化并启动一个 scheduler 运行进程
            mp.Process(
                # 将后台进程的运行入口指向 _run_scheduler 函数
                target=_run_scheduler,
                # 将定制后的 rank 配置对象以及用来发送就绪确认的 ack_queue 队列作为参数传入
                args=(new_args, ack_queue),
                # 设置为非守护进程（daemon=False），确保即使主进程由于某种异常终止，子进程也有机会完成完整的资源释放操作
                daemon=False,
                # 为该子进程设定一个易于辨识和跟踪的名称
                name=f"minisgl-TP{i}-scheduler",
            ).start()  # 调用 start()，使进程开始在后台并行运行

        # 从配置中读取需要独立拉起的分词器（tokenizer）并行进程的数量
        num_tokenizers = server_args.num_tokenizer
        # 实例化并拉起负责反分词（detokenizer，即 ID 变文本）的工作进程（全系统仅运行 1 个）
        mp.Process(
            # 入口统一为 tokenize_worker 函数
            target=tokenize_worker,
            # 通过关键字参数字典的形式，向函数传入其运行所需要的路径与 ZMQ 连接端口
            kwargs={
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            # 设为非守护进程
            daemon=False,
            # 指定 detokenizer 的进程名称
            name="minisgl-detokenizer-0",
        ).start() # 调用 start() 异步启动反分词进程
        # 启动独立的 tokenizer 进程（如果 num_tokenizer > 0）

        # 循环 num_tokenizers 次，为系统拉起指定数量的、用于支持前端请求快速并发分词的独立工作进程
        for i in range(num_tokenizers):
            # 实例化第 `i` 个分词器工作子进程
            mp.Process(
                # 执行目标函数依旧为 tokenize_worker
                target=tokenize_worker,
                # 通过关键字参数注入特定分词器的配置信息
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                # 设为非守护进程
                daemon=False,
                # 指定分词器的进程名称
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # 等待所有工作进程发送就绪确认：
        # - world_size 个 scheduler（但只有主 rank 发送确认）
        # - num_tokenizers 个 tokenizer
        # - 1 个 detokenizer
        # 总共确认数：1 + num_tokenizers + 1 = num_tokenizers + 2
        for _ in range(num_tokenizers + 2):
            # ack_queue.get() 会处于阻塞状态，直到有子进程把它的就绪信息（如 "tokenizer 0 is ready"）丢进队列
            # 拿到就绪字符串后，将其输出为日志
            logger.info(ack_queue.get())

    # 调用之前第一轮里实现的 run_api_server 入口
    # 将解析好的参数、刚刚定义的子进程拉起逻辑（start_subprocess 闭包）和 run_shell 标志一并传给它，进入 Web 或 Shell 服务主循环
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
